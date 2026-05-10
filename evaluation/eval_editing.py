"""
Image Editing Evaluation.

Evaluates editing results using multiple metrics:
  - Structure Distance (DINO features)
  - CLIP Similarity (text-image alignment)
  - Background Preservation (PSNR, LPIPS, MSE, SSIM on unedited regions)

Corresponds to Tables 2-3 in the paper.

Usage:
    python eval_editing.py \
        --tgt_image_folders path/to/edited/images \
        --result_path results/editing.csv \
        --dataset_file dataset/mapping_file.json
"""

import os
import sys
import json
import argparse
import csv

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from metrics import MetricEvaluator


def mask_decode(encoded_mask, image_shape=(512, 512)):
    """Decode a run-length encoded mask."""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    return mask_array


def evaluate_editing(tgt_folders, dataset_file, image_dir, result_path,
                     device="cuda:0", image_size=512):
    """Run editing evaluation across all target folders."""
    
    # Initialize metrics
    evaluator = MetricEvaluator(device=device)
    
    # Load dataset
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    all_results = []
    
    for tgt_folder in tgt_folders:
        print(f"\nEvaluating: {tgt_folder}")
        
        metrics_accumulator = {
            "structure_distance": [],
            "clip_similarity_source": [],
            "clip_similarity_target": [],
            "psnr_unedit": [],
            "lpips_unedit": [],
            "mse_unedit": [],
            "ssim_unedit": [],
        }
        
        for key, item in dataset.items():
            image_path = os.path.join(image_dir, item["image_path"])
            edited_path = image_path.replace(image_dir, tgt_folder)
            
            if not os.path.exists(edited_path):
                continue
            
            original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            
            # Load images
            src_img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
            tgt_img = Image.open(edited_path).convert("RGB").resize((image_size, image_size))
            
            src_tensor = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
            tgt_tensor = transforms.ToTensor()(tgt_img).unsqueeze(0).to(device)
            
            # Structure distance (DINO)
            sd = evaluator.calculate_metric("structure_distance", src_tensor, tgt_tensor)
            metrics_accumulator["structure_distance"].append(sd)
            
            # CLIP similarity
            clip_src = evaluator.calculate_metric("clip_similarity", tgt_tensor, original_prompt)
            clip_tgt = evaluator.calculate_metric("clip_similarity", tgt_tensor, editing_prompt)
            metrics_accumulator["clip_similarity_source"].append(clip_src)
            metrics_accumulator["clip_similarity_target"].append(clip_tgt)
            
            # Background preservation with mask
            if "mask" in item:
                mask = mask_decode(item["mask"])
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
                mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(image_size, image_size))
                
                # Mask out edited region (keep background)
                bg_mask = 1 - mask_tensor
                src_bg = src_tensor * bg_mask
                tgt_bg = tgt_tensor * bg_mask
                
                metrics_accumulator["psnr_unedit"].append(
                    evaluator.calculate_metric("psnr", src_bg, tgt_bg))
                metrics_accumulator["lpips_unedit"].append(
                    evaluator.calculate_metric("lpips", src_bg, tgt_bg))
                metrics_accumulator["mse_unedit"].append(
                    evaluator.calculate_metric("mse", src_bg, tgt_bg))
                metrics_accumulator["ssim_unedit"].append(
                    evaluator.calculate_metric("ssim", src_bg, tgt_bg))
        
        # Aggregate results
        result_row = {"folder": tgt_folder}
        for metric, values in metrics_accumulator.items():
            if values:
                result_row[f"{metric}_mean"] = np.mean(values)
                result_row[f"{metric}_std"] = np.std(values)
        
        all_results.append(result_row)
        
        print(f"  Structure Distance: {result_row.get('structure_distance_mean', 'N/A'):.4f}")
        print(f"  CLIP Sim (target):  {result_row.get('clip_similarity_target_mean', 'N/A'):.4f}")
        print(f"  PSNR (background):  {result_row.get('psnr_unedit_mean', 'N/A'):.2f}")
    
    # Save to CSV
    if all_results:
        os.makedirs(os.path.dirname(result_path) if os.path.dirname(result_path) else ".", exist_ok=True)
        with open(result_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {result_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Editing Evaluation")
    parser.add_argument("--tgt_image_folders", nargs="+", required=True,
                        help="Target edited image folder(s)")
    parser.add_argument("--dataset_file", type=str, default="dataset/mapping_file.json",
                        help="PIE-Bench mapping file")
    parser.add_argument("--image_dir", type=str, default="dataset/annotation_images",
                        help="Source image directory")
    parser.add_argument("--result_path", type=str, default="results/editing_results.csv",
                        help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--image_size", type=int, default=512, help="Image resolution")
    args = parser.parse_args()
    
    evaluate_editing(
        args.tgt_image_folders, args.dataset_file, args.image_dir,
        args.result_path, args.device, args.image_size)
