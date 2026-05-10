"""
DICE Inversion + Reconstruction for Paella.

This script performs DICE inversion on images using the Paella masked generative model,
then reconstructs them using the recorded noise sequences and mask patterns.
Used for evaluating reconstruction fidelity (Table 1 in the paper).

Usage:
    python inversion.py --cfg 8.0 --lamb 0.9 --max_t 0.75 --gpu 0
"""

import os
import sys
import time
import math
import json
import argparse

import torch
import torchvision
import requests
import open_clip
import tqdm
from PIL import Image
from io import BytesIO
from open_clip import tokenizer
from transformers import AutoTokenizer, T5EncoderModel
from torchvision.utils import save_image

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(__file__))
from models.paella import Paella
from utils.alter_attention import replace_attention_layers
from utils.sampling import sample_gumbel, generate_mask_schedule, sample_noised_tokens


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
os.environ["OPENMP_NUM_THREADS"] = "1"
torch.set_num_threads(4)

parser = argparse.ArgumentParser(description="DICE Inversion + Reconstruction with Paella")
parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
parser.add_argument("--max_t", type=float, default=0.75, help="Maximum noise level (t_max)")
parser.add_argument("--cfg", type=float, default=8.0, help="Classifier-free guidance scale")
parser.add_argument("--lamb", type=float, default=0.9, help="Noise interpolation weight (lambda)")
parser.add_argument("--steps", type=int, default=32, help="Number of diffusion steps")
parser.add_argument("--use_saved_inputs", action="store_true", default=True,
                    help="Use pre-cached text/CLIP embeddings")
parser.add_argument("--cache_inputs", action="store_true", default=False,
                    help="Cache text/CLIP embeddings and exit")
parser.add_argument("--model_path", type=str, default="checkpoints/paella",
                    help="Path to model checkpoints")
parser.add_argument("--dataset_file", type=str, default="dataset/mapping_file.json",
                    help="Path to PIE-Bench mapping file")
parser.add_argument("--image_dir", type=str, default="dataset/annotation_images",
                    help="Path to source images")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (auto-generated if not specified)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# DICE Inversion (Forward Process)
# ---------------------------------------------------------------------------
def dice_invert(model, x0, model_inputs, latent_shape, unconditional_inputs,
                init_noise=None, steps=32, temperature=(0.2, 1.2),
                cfg=(8.0, 8.0), t_start=0.0, t_end=0.75, attn_weights=None):
    """
    DICE forward inversion process.
    
    Records noise residuals (zs) and mask schedule during the forward noising
    process. These are replayed in reverse during reconstruction/editing.
    
    Args:
        model: Paella model.
        x0: Clean token map (batch, H, W).
        model_inputs: Conditioning inputs (byt5, clip, clip_image).
        latent_shape: Shape of the latent space.
        unconditional_inputs: Unconditional conditioning for CFG.
        init_noise: Fixed noise tensor for reproducibility.
        steps: Number of diffusion steps.
        temperature: (start, end) temperature schedule.
        cfg: (start, end) CFG scale schedule.
        t_start: Start time (0.0 for inversion).
        t_end: End time / max noise level.
        attn_weights: Attention reweighting tensor.
    
    Returns:
        sampled_tokens: Final noised token map.
        zs: List of noise residual tensors.
        mask_schedule: List of binary masks used.
        noised_tokens: List of intermediate noised token maps.
    """
    dev = unconditional_inputs["byt5"].device
    
    with torch.inference_mode():
        t_list = torch.linspace(t_start, t_end, steps + 1)
        temperatures = torch.linspace(temperature[0], temperature[1], steps)
        cfgs = torch.linspace(cfg[0], cfg[1], steps)
        
        # Get model prediction for clean input
        x0_logits = model(x0, torch.ones(latent_shape[0], device=dev) * 0.0,
                          **model_inputs, attn_weights=attn_weights)
        
        # Generate mask schedule and noised tokens
        mask_schedule = generate_mask_schedule(steps, latent_shape, 
                                               max_t=max(t_end, t_start), device=dev)
        noised_tokens = sample_noised_tokens(x0, mask_schedule, latent_shape,
                                             init_noise=init_noise, device=dev)
        
        # Compute noise residuals
        zs = []
        current = None
        
        for i, tv in enumerate(t_list[1:steps]):
            t = torch.ones(latent_shape[0], device=dev) * tv
            i = i + 1
            current, idx = noised_tokens[i]
            
            # Model prediction from noised tokens
            x_0_hat = model(current.to(dev).long(), t, **model_inputs, attn_weights=attn_weights)
            
            # Apply classifier-free guidance
            x_0_hat = x_0_hat * cfgs[i] + model(
                current.to(dev).long(), t, **unconditional_inputs) * (1 - cfgs[i])
            
            # Compute noise residual: z = x0_logits - predicted_logits
            z = x0_logits - x_0_hat.div(temperatures[i])
            zs.append(z)
    
    return current, zs, mask_schedule, noised_tokens


# ---------------------------------------------------------------------------
# DICE Reconstruction (Reverse Process)
# ---------------------------------------------------------------------------
def dice_reconstruct(model, model_inputs, latent_shape, zs, mask_schedule,
                     init_noise=None, unconditional_inputs=None, init_x=None,
                     steps=32, temperature=(1.2, 0.2), cfg=(8.0, 8.0),
                     t_start=0.75, t_end=0.0, lamb=0.9, attn_weights=None):
    """
    DICE reverse reconstruction/editing process.
    
    Replays the recorded noise residuals and mask schedule in reverse
    to reconstruct or edit the image.
    
    Args:
        model: Paella model.
        model_inputs: Conditioning inputs (can be different prompt for editing).
        latent_shape: Shape of the latent space.
        zs: Noise residuals from inversion (reversed order).
        mask_schedule: Mask schedule from inversion (reversed order).
        init_noise: Fixed noise tensor.
        unconditional_inputs: Unconditional conditioning for CFG.
        init_x: Initial noised token map from inversion.
        steps: Number of diffusion steps.
        temperature: (start, end) temperature schedule.
        cfg: (start, end) CFG scale schedule.
        t_start: Start time (t_max for reconstruction).
        t_end: End time (0.0 for reconstruction).
        lamb: Noise interpolation weight.
        attn_weights: Attention reweighting tensor.
    
    Returns:
        sampled: Reconstructed token map.
    """
    dev = unconditional_inputs["byt5"].device
    noise = init_noise
    sampled = init_x
    
    with torch.inference_mode():
        t_list = torch.linspace(t_start, t_end, steps + 1)
        temperatures = torch.linspace(temperature[0], temperature[1], steps)
        cfgs = torch.linspace(cfg[0], cfg[1], steps)
        
        for i, tv in enumerate(t_list[1:steps]):
            t = torch.ones(latent_shape[0], device=dev) * tv
            
            # Model prediction
            logits = model(sampled, t, **model_inputs, attn_weights=attn_weights)
            logits = logits * cfgs[i] + model(
                sampled, t, **unconditional_inputs) * (1 - cfgs[i])
            
            # Apply noise residual with lambda interpolation
            logits = (logits.div(temperatures[i]) 
                      + zs[i].to(dev) * lamb 
                      + sample_gumbel(logits.shape, device=dev) * (1 - lamb))
            sampled = logits.argmax(dim=1)
            
            # Re-noise with mask schedule
            if i + 2 < steps:
                sampled = (sampled * (1 - mask_schedule[i + 2].to(dev)) 
                           + noise * mask_schedule[i + 2].to(dev)).long()
    
    return sampled


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
batch_size = 1

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
])


def encode(x):
    return vqmodel.encode(x, quantize=True)[2]


def decode(img_seq):
    return vqmodel.decode_indices(img_seq)


def embed_t5(text, t5_tokenizer, t5_model, device="cuda"):
    t5_tokens = t5_tokenizer(text, padding="longest", return_tensors="pt",
                              max_length=768, truncation=True).input_ids.to(device)
    return t5_model(input_ids=t5_tokens).last_hidden_state


def load_image(path):
    if path.startswith("http"):
        response = requests.get(path)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(path)
    return preprocess(img).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)[:, :3]


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
latent_shape = (batch_size, 64, 64)  # 256x256 resolution
negative_caption = "low quality, low resolution, bad image, blurry, blur"

# Setup output directory
cfg = args.cfg
lamb = args.lamb
t_end = args.max_t
output_dir = args.output_dir or f"outputs_inv/cfg-{cfg}_lamb-{lamb}_t_end-{t_end}"
os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(args.seed)

# Load dataset
with open(args.dataset_file, "r") as f:
    editing_instruction = json.load(f)

# ---------------------------------------------------------------------------
# Prepare Text/CLIP Embeddings
# ---------------------------------------------------------------------------
all_inputs = []

if not args.use_saved_inputs:
    from arroz import Diffuzz, PriorModel
    
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k")
    clip_model = clip_model.to(device).eval().requires_grad_(False)
    
    t5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl")
    t5_model = T5EncoderModel.from_pretrained("google/byt5-xl").to(device).requires_grad_(False)
    
    prior_ckpt = torch.load(os.path.join(args.model_path, "prior_v1.pt"), map_location=device)
    prior = PriorModel().to(device)
    prior.load_state_dict(prior_ckpt)
    prior.eval().requires_grad_(False)
    diffuzz = Diffuzz(device=device)
    del prior_ckpt
    
    for key, item in tqdm.tqdm(editing_instruction.items(), desc="Encoding prompts"):
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(args.image_dir, item["image_path"])
        
        caption = original_prompt
        text = tokenizer.tokenize([caption] * latent_shape[0]).to(device)
        
        with torch.inference_mode():
            clip_text_tokens_uncond = tokenizer.tokenize([negative_caption] * len(text)).to(device)
            t5_embeddings_uncond = embed_t5([negative_caption] * len(text), t5_tokenizer, t5_model, device=device)
            t5_embeddings = embed_t5([caption] * latent_shape[0], t5_tokenizer, t5_model, device=device)
            clip_text_embeddings = clip_model.encode_text(text)
            clip_text_embeddings_uncond = clip_model.encode_text(clip_text_tokens_uncond)
        
        model_inputs = {
            "byt5": t5_embeddings,
            "clip": clip_text_embeddings,
            "clip_image": None
        }
        unconditional_inputs = {
            "byt5": t5_embeddings_uncond,
            "clip": clip_text_embeddings_uncond,
            "clip_image": None
        }
        all_inputs.append((model_inputs, unconditional_inputs, model_inputs, unconditional_inputs, image_path))
    
    if args.cache_inputs:
        torch.save(all_inputs, "all_inputs.pt")
        print("Cached embeddings to all_inputs.pt")
        exit()
    
    del clip_model, t5_model, prior, diffuzz
else:
    all_inputs = torch.load("all_inputs.pt", map_location=device)

torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------------------
from src.vqgan import VQModel

vqmodel = VQModel().to(device)
vqmodel.load_state_dict(torch.load(os.path.join(args.model_path, "vqgan_f4.pt"), map_location=device))
vqmodel.eval().requires_grad_(False)

state_dict = torch.load(os.path.join(args.model_path, "paella_v3.pt"), map_location=device)
model = Paella(byt5_embd=2560).to(device)
model.load_state_dict(state_dict)
model.eval()
replace_attention_layers(model)
model.to(device)
del state_dict

# ---------------------------------------------------------------------------
# Run Inversion + Reconstruction
# ---------------------------------------------------------------------------
print(f"\nRunning DICE Inversion + Reconstruction")
print(f"  cfg={cfg}, lamb={lamb}, t_end={t_end}, steps={args.steps}")
print(f"  Output: {output_dir}\n")

for model_inputs_1, uncond_1, _, _, image_path in tqdm.tqdm(all_inputs, desc="Processing"):
    torch.manual_seed(args.seed)
    
    # For reconstruction, use the same prompt for both inversion and generation
    model_inputs_2 = model_inputs_1
    uncond_2 = uncond_1
    
    with torch.inference_mode():
        images = [load_image(image_path)]
        latents = encode(images[0])
        
        attn_weights = torch.ones((model_inputs_1["byt5"].shape[1])).to(device)
        attn_weights[-4:] = 0.4
        attn_weights[:-4] = 1.2
        
        steps = args.steps
        
        with torch.autocast(device_type="cuda"):
            noise = torch.randint(0, 8192, size=latent_shape, device=device)
            
            # Step 1: DICE Inversion
            sampled_tokens, zs, mask_schedule, _ = dice_invert(
                model, x0=latents, init_noise=noise,
                model_inputs=model_inputs_1,
                unconditional_inputs=uncond_1,
                temperature=(0.2, 1.2),
                cfg=(cfg, cfg), steps=steps,
                latent_shape=latent_shape,
                t_start=0.0, t_end=t_end,
                attn_weights=attn_weights)
            
            sampled_tokens = sampled_tokens.to(device).long()
            zs = [z.to(device) for z in zs]
        
        with torch.autocast(device_type="cuda"):
            # Step 2: DICE Reconstruction
            sampled_tokens = dice_reconstruct(
                model, zs=zs[::-1],
                mask_schedule=mask_schedule[::-1],
                init_noise=noise,
                model_inputs=model_inputs_2,
                unconditional_inputs=uncond_2,
                init_x=sampled_tokens.clone(),
                temperature=(1.2, 0.2),
                cfg=(cfg, cfg), steps=steps,
                latent_shape=latent_shape,
                t_start=t_end, t_end=0.0,
                lamb=lamb, attn_weights=attn_weights)
        
        sampled = decode(sampled_tokens)
    
    # Save output
    save_path = image_path.replace(args.image_dir, output_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(sampled.float(), save_path, normalize=True, value_range=(0, 1))

print(f"\nDone! Results saved to {output_dir}")
