"""
DICE Inversion + Editing for Paella.

This script performs DICE inversion using a source prompt, then generates
an edited image using a target prompt. This is the main editing pipeline
used for Tables 2-3 in the paper.

Usage:
    python editing.py --cfg 8.0 --lamb 0.9 --max_t 0.75 --gpu 0 --resolution 512
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
from utils.sampling import sample_gumbel, generate_mask_schedule, sample_noised_tokens, get_lambda_schedule


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
os.environ["OPENMP_NUM_THREADS"] = "1"
torch.set_num_threads(4)

parser = argparse.ArgumentParser(description="DICE Inversion + Editing with Paella")
parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
parser.add_argument("--max_t", type=float, default=0.75, help="Maximum noise level (t_max)")
parser.add_argument("--cfg", type=float, default=8.0, help="Classifier-free guidance scale")
parser.add_argument("--lamb", type=float, default=0.9, help="Noise interpolation weight (lambda)")
parser.add_argument("--steps", type=int, default=32, help="Number of diffusion steps")
parser.add_argument("--renoise_steps", type=int, default=26, help="Number of re-noising steps")
parser.add_argument("--resolution", type=int, default=512, choices=[256, 512],
                    help="Image resolution (256 or 512)")
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
parser.add_argument("--lamb_schedule", type=str, default="None", choices=["None", "linear", "exp"],
                    help="Lambda schedule type for noise interpolation")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()
print(args)

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
    process. These are replayed in reverse during editing.
    
    Returns:
        sampled_tokens: Final noised token map.
        zs: List of noise residual tensors.
        mask_schedule: List of binary masks used.
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
            
            x_0_hat = model(current.to(dev).long(), t, **model_inputs, attn_weights=attn_weights)
            x_0_hat = x_0_hat * cfgs[i] + model(
                current.to(dev).long(), t, **unconditional_inputs) * (1 - cfgs[i])
            
            z = x0_logits - x_0_hat.div(temperatures[i])
            zs.append(z)
    
    return current, zs, mask_schedule


# ---------------------------------------------------------------------------
# DICE Editing (Reverse Process with Target Prompt)
# ---------------------------------------------------------------------------
def dice_edit(model, model_inputs, latent_shape, zs, mask_schedule,
              init_noise=None, unconditional_inputs=None, init_x=None,
              steps=32, temperature=(1.2, 0.2), cfg=(8.0, 8.0),
              t_start=0.75, t_end=0.0, lamb=0.9, attn_weights=None,
              lamb_schedule_type="None"):
    """
    DICE reverse editing process with target prompt conditioning.
    
    Replays the recorded noise residuals and mask schedule in reverse
    while conditioning on a new (target) text prompt for editing.
    """
    dev = unconditional_inputs["byt5"].device
    noise = init_noise
    sampled = init_x
    
    lamb_schedule = get_lambda_schedule(lamb, steps, lamb_schedule_type)
    
    with torch.inference_mode():
        t_list = torch.linspace(t_start, t_end, steps + 1)
        temperatures = torch.linspace(temperature[0], temperature[1], steps)
        cfgs = torch.linspace(cfg[0], cfg[1], steps)
        
        for i, tv in enumerate(t_list[1:steps]):
            t = torch.ones(latent_shape[0], device=dev) * tv
            
            logits = model(sampled, t, **model_inputs, attn_weights=attn_weights)
            logits = logits * cfgs[i] + model(
                sampled, t, **unconditional_inputs) * (1 - cfgs[i])
            
            logits = (logits.div(temperatures[i])
                      + zs[i].to(dev) * lamb_schedule[i]
                      + sample_gumbel(logits.shape, device=dev) * (1 - lamb_schedule[i]))
            sampled = logits.argmax(dim=1)
            
            if i + 2 < steps:
                sampled = (sampled * (1 - mask_schedule[i + 2].to(dev))
                           + noise * mask_schedule[i + 2].to(dev)).long()
    
    return sampled


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
batch_size = 1

if args.resolution == 512:
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
    ])
    latent_shape = (batch_size, 128, 128)
else:
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
    ])
    latent_shape = (batch_size, 64, 64)


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
negative_caption = "low quality, low resolution, bad image, blurry, blur"
t5, clip_text, clip_image = True, True, False
use_prior = True

cfg = args.cfg
lamb = args.lamb
t_end = args.max_t
output_dir = args.output_dir or f"outputs_editing/cfg-{cfg}_lamb-{lamb}_t_end-{t_end}"
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
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(args.image_dir, item["image_path"])
        
        # Encode source prompt
        text_src = tokenizer.tokenize([original_prompt] * latent_shape[0]).to(device)
        with torch.inference_mode():
            t5_emb_uncond = embed_t5([negative_caption] * batch_size, t5_tokenizer, t5_model, device=device)
            clip_uncond = clip_model.encode_text(
                tokenizer.tokenize([negative_caption] * batch_size).to(device))
            
            t5_emb_src = embed_t5([original_prompt] * batch_size, t5_tokenizer, t5_model, device=device)
            clip_src = clip_model.encode_text(text_src)
        
        model_inputs_src = {"byt5": t5_emb_src, "clip": clip_src, "clip_image": None}
        uncond_src = {"byt5": t5_emb_uncond, "clip": clip_uncond, "clip_image": None}
        
        # Encode target prompt
        text_tgt = tokenizer.tokenize([editing_prompt] * latent_shape[0]).to(device)
        with torch.inference_mode():
            t5_emb_tgt = embed_t5([editing_prompt] * batch_size, t5_tokenizer, t5_model, device=device)
            clip_tgt = clip_model.encode_text(text_tgt)
            t5_emb_uncond_2 = embed_t5([negative_caption] * batch_size, t5_tokenizer, t5_model, device=device)
            clip_uncond_2 = clip_model.encode_text(
                tokenizer.tokenize([negative_caption] * batch_size).to(device))
        
        model_inputs_tgt = {"byt5": t5_emb_tgt, "clip": clip_tgt, "clip_image": None}
        uncond_tgt = {"byt5": t5_emb_uncond_2, "clip": clip_uncond_2, "clip_image": None}
        
        all_inputs.append((model_inputs_src, uncond_src, model_inputs_tgt, uncond_tgt, image_path))
    
    if args.cache_inputs:
        cache_name = f"all_inputs_{args.resolution}.pt"
        torch.save(all_inputs, cache_name)
        print(f"Cached embeddings to {cache_name}")
        exit()
    
    del clip_model, t5_model, prior, diffuzz
else:
    cache_name = f"all_inputs_{args.resolution}.pt"
    print(f"Loading cached embeddings from {cache_name}")
    all_inputs = torch.load(cache_name, map_location=device)

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
# Run Inversion + Editing
# ---------------------------------------------------------------------------
print(f"\nRunning DICE Inversion + Editing")
print(f"  cfg={cfg}, lamb={lamb}, t_end={t_end}, steps={args.steps}")
print(f"  lamb_schedule={args.lamb_schedule}")
print(f"  resolution={args.resolution}, latent_shape={latent_shape}")
print(f"  Output: {output_dir}\n")

for model_inputs_src, uncond_src, model_inputs_tgt, uncond_tgt, image_path in tqdm.tqdm(all_inputs, desc="Editing"):
    torch.manual_seed(args.seed)
    
    with torch.inference_mode():
        images = [load_image(image_path)]
        latents = encode(images[0])
        
        # Attention weights for source prompt
        attn_weights_src = torch.ones((model_inputs_src["byt5"].shape[1])).to(device)
        attn_weights_src[-4:] = 0.4
        attn_weights_src[:-4] = 1.2
        
        steps = args.steps
        
        with torch.autocast(device_type="cuda"), torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True):
            noise = torch.randint(0, 8192, size=latent_shape, device=device)
            
            # Step 1: DICE Inversion with source prompt
            sampled_tokens, zs, mask_schedule = dice_invert(
                model, x0=latents, init_noise=noise,
                model_inputs=model_inputs_src,
                unconditional_inputs=uncond_src,
                temperature=(0.2, 1.2),
                cfg=(cfg, cfg), steps=steps,
                latent_shape=latent_shape,
                t_start=0.0, t_end=t_end,
                attn_weights=attn_weights_src)
            
            sampled_tokens = sampled_tokens.to(device).long()
            zs = [z.to(device) for z in zs]
        
        # Attention weights for target prompt
        attn_weights_tgt = torch.ones((model_inputs_tgt["byt5"].shape[1])).to(device)
        attn_weights_tgt[-4:] = 0.4
        attn_weights_tgt[:-4] = 1.2
        
        with torch.autocast(device_type="cuda"):
            # Step 2: DICE Editing with target prompt
            sampled_tokens = dice_edit(
                model, zs=zs[::-1],
                mask_schedule=mask_schedule[::-1],
                init_noise=noise,
                model_inputs=model_inputs_tgt,
                unconditional_inputs=uncond_tgt,
                init_x=sampled_tokens.clone(),
                temperature=(1.2, 0.2),
                cfg=(cfg, cfg), steps=steps,
                latent_shape=latent_shape,
                t_start=t_end, t_end=0.0,
                lamb=lamb, attn_weights=attn_weights_tgt,
                lamb_schedule_type=args.lamb_schedule)
        
        sampled = decode(sampled_tokens)
        del zs, mask_schedule
        torch.cuda.empty_cache()
    
    # Save output
    save_path = image_path.replace(args.image_dir, output_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(sampled.float(), save_path, normalize=True, value_range=(0, 1))

print(f"\nDone! Edited results saved to {output_dir}")
