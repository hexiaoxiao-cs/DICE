"""
DICE Inversion + Editing for RoBERTa.

Applies DICE to RoBERTa (masked language model) for sentiment-controlled
text editing. Given a negative sentiment sentence, the model inverts it
and generates a positive sentiment version while preserving structure.

Corresponds to Tables 4-5 in the paper.

Usage:
    python editing.py --gpu 0 --task 0 1 2 3 4 5 6 7 8 9
"""

import sys
import argparse
import json
import itertools

import torch
import torch.nn.functional as F
import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DICE Text Editing with RoBERTa")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
parser.add_argument("--task", nargs="+", type=int, default=list(range(10)),
                    help="Task IDs to process")
parser.add_argument("--max_t_range", nargs=2, type=int, default=[70, 100],
                    help="max_t sweep range (percentage, e.g., 70 100)")
parser.add_argument("--lamb_range", nargs=2, type=int, default=[20, 90],
                    help="Lambda sweep range (percentage, e.g., 20 90)")
parser.add_argument("--step_size", type=int, default=5,
                    help="Step size for hyperparameter sweep (percentage)")
parser.add_argument("--dataset", type=str, default="dataset/text_editing_dataset.json",
                    help="Path to sentiment dataset")
parser.add_argument("--output_dir", type=str, default="results_roberta",
                    help="Output directory")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def sample_gumbel(shape, eps=1e-20, device=device):
    """Sample from the Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def generate_mask_schedule(timesteps, latent_shape=(64,), max_t=0.75):
    """Generate a linear mask schedule for 1D token sequences."""
    t = torch.linspace(0, max_t, timesteps)
    mask_schedule = (torch.rand(latent_shape) <= t[:, None, None]).long()
    return list(mask_schedule)


# ---------------------------------------------------------------------------
# Load Model & Dataset
# ---------------------------------------------------------------------------
print(f"Loading RoBERTa-large on {device}...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMaskedLM.from_pretrained("roberta-large").to(device)

print(f"Loading dataset from {args.dataset}...")
dataset = json.load(open(args.dataset, "r"))

# ---------------------------------------------------------------------------
# Prepare Inputs
# ---------------------------------------------------------------------------
all_inputs = []
for i in args.task:
    i = str(i)
    pos = dataset[i]["positive_sentiment"]
    neg = dataset[i]["negative_sentiment"]
    
    # Direction: negative -> positive (invert negative, edit to positive)
    text_1 = neg[0]       # Negative context sentence
    text_2 = neg[1]       # Negative content sentence (to be edited)
    prompt = pos[0]       # Positive context sentence (target)
    
    inputs_1 = tokenizer(text_1, text_2, return_tensors="pt").to(device)
    only_text_2 = tokenizer(text_2, return_tensors="pt").to(device)
    inputs_forward = tokenizer(prompt, text_2, return_tensors="pt").to(device)
    
    all_inputs.append((i, inputs_1, only_text_2, inputs_forward))

# ---------------------------------------------------------------------------
# Run DICE Inversion + Editing
# ---------------------------------------------------------------------------
import os
os.makedirs(args.output_dir, exist_ok=True)

for current_idx, inputs_1, only_text_2, inputs_forward in all_inputs:
    curr_outputs = []
    timesteps = len(only_text_2["input_ids"][0]) - 2
    
    # Hyperparameter sweep
    param_grid = list(itertools.product(
        [i / 100 for i in range(args.max_t_range[0], args.max_t_range[1], args.step_size)],
        [j / 100 for j in range(args.lamb_range[0], args.lamb_range[1], args.step_size)]
    ))
    pbar = tqdm.tqdm(param_grid, desc=f"Task {current_idx}")
    
    for max_t, lamb in pbar:
        torch.manual_seed(args.seed)
        
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            # Locate text_2 within the concatenated input
            loc_of_text_2_start = torch.where(inputs_1["input_ids"] == 2)[1][-2] + 1
            
            # Generate mask schedule
            masks = generate_mask_schedule(
                timesteps, latent_shape=(len(only_text_2["input_ids"][0]) - 2,), max_t=max_t)
            
            # Expand masks to full input size
            masks_only = masks.copy()
            new_masks = []
            for mask in masks:
                new_mask = torch.zeros_like(inputs_1["input_ids"])
                new_mask[:, loc_of_text_2_start:-1] = mask
                new_masks.append(new_mask)
            masks = new_masks
            
            # ---- DICE Inversion (Forward) ----
            inputs = inputs_1
            zs = []
            T0 = inputs["input_ids"][0]
            noise_size = T0.size()[0]
            
            x0 = model(**inputs).logits
            
            # Random noise tokens
            current_noise = torch.randint(4, 50264, [noise_size]).to(device)
            
            for i in range(timesteps):
                T = masks[i] * current_noise + (1 - masks[i]) * T0
                curr_input = inputs.copy()
                curr_input["input_ids"] = T
                x0_hat = model(**curr_input).logits
                
                # Noise residual
                z = x0[:, loc_of_text_2_start:] - x0_hat[:, loc_of_text_2_start:]
                zs.append(z)
            
            T_T = T.clone()
            
            # ---- DICE Editing (Reverse with target prompt) ----
            inputs = inputs_forward
            loc_of_text_2_start_new = torch.where(inputs["input_ids"] == 2)[1][-2] + 1
            old_T = inputs["input_ids"].clone()
            
            # Transfer noised tokens to target input
            inputs["input_ids"][0, loc_of_text_2_start_new:] = T_T[0, loc_of_text_2_start:]
            T = inputs["input_ids"]
            noise_size = T.size()[0]
            
            # Remap masks for target input
            new_masks = []
            for mask in masks_only:
                new_mask = torch.zeros_like(inputs["input_ids"])
                new_mask[:, loc_of_text_2_start_new:-1] = mask
                new_masks.append(new_mask)
            masks = new_masks
            
            for i in reversed(range(timesteps)):
                curr_input = inputs.copy()
                curr_input["input_ids"] = T
                z = zs[i]
                x0_hat = model(**curr_input).logits
                
                # Apply noise residual with lambda interpolation + Gumbel noise
                x0_hat[0, loc_of_text_2_start_new:] = (
                    x0_hat[0, loc_of_text_2_start_new:] 
                    + z * lamb 
                    + sample_gumbel(x0_hat[0, loc_of_text_2_start_new:].shape) * (1 - lamb)
                )
                T = x0_hat.argmax(axis=-1)
                T[:, :loc_of_text_2_start_new] = old_T[:, :loc_of_text_2_start_new]
                
                # Re-noise
                if i > 0:
                    T[:, loc_of_text_2_start_new:] = (
                        masks[i - 1][:, loc_of_text_2_start_new:] * current_noise[loc_of_text_2_start:]
                        + (1 - masks[i - 1][:, loc_of_text_2_start_new:]) * T[:, loc_of_text_2_start_new:]
                    )
        
        result = tokenizer.decode(T[0])
        curr_outputs.append(f"lamb: {lamb}, t_end:{max_t},sentence:\t{result}")
        torch.cuda.empty_cache()
    
    # Save results
    output_file = os.path.join(args.output_dir, f"output_{current_idx}.txt")
    with open(output_file, "w") as f:
        for line in curr_outputs:
            f.write(f"{line}\n")
    print(f"Saved task {current_idx} results to {output_file}")

print(f"\nDone! All results saved to {args.output_dir}")
