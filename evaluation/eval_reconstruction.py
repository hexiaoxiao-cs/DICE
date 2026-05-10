"""
Image Reconstruction Evaluation.

Evaluates reconstruction quality by comparing reconstructed images
against source images (decoded through VQ-GAN for fair comparison).

Metrics: PSNR, LPIPS, MSE, SSIM, Structure Distance (DINO)

Corresponds to Table 1 in the paper.

Usage:
    python eval_reconstruction.py \
        --src_image_folder path/to/decoded/originals \
        --tgt_image_folders path/to/reconstructed/images \
        --result_path results/reconstruction.csv
"""

import os
import sys
import json
import csv
import argparse

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from metrics import MetricEvaluator


def evaluate_reconstruction(src_folder, tgt_folders, dataset_file, image_dir,
                             result_path, device="cuda:0", image_size=512):
    """Run reconstruction evaluation comparing decoded originals vs reconstructions."""
    
    evaluator = MetricEvaluator(device=device)
    
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    all_results = []
    
    for tgt_folder in tgt_folders:
        print(f"\nEvaluating: {tgt_folder}")
        
        metrics_accumulator = {
            "psnr": [],
            "lpips": [],
            "mse": [],
            "ssim": [],
            "structure_distance": [],
        }
        
        count = 0
        for key, item in dataset.items():
            src_path = os.path.join(image_dir, item["image_path"]).replace(image_dir, src_folder)
            tgt_path = os.path.join(image_dir, item["image_path"]).replace(image_dir, tgt_folder)
            
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                continue
            
            src_img = Image.open(src_path).convert("RGB").resize((image_size, image_size))
            tgt_img = Image.open(tgt_path).convert("RGB").resize((image_size, image_size))
            
            src_tensor = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
            tgt_tensor = transforms.ToTensor()(tgt_img).unsqueeze(0).to(device)
            
            for metric in metrics_accumulator:
                val = evaluator.calculate_metric(metric, src_tensor, tgt_tensor)
                metrics_accumulator[metric].append(val)
            
            count += 1
        
        result_row = {"folder": tgt_folder, "count": count}
        for metric, values in metrics_accumulator.items():
            if values:
                result_row[f"{metric}_mean"] = np.mean(values)
                result_row[f"{metric}_std"] = np.std(values)
        
        all_results.append(result_row)
        
        print(f"  Evaluated {count} images")
        print(f"  PSNR:  {result_row.get('psnr_mean', 'N/A'):.2f}")
        print(f"  LPIPS: {result_row.get('lpips_mean', 'N/A'):.4f}")
        print(f"  SSIM:  {result_row.get('ssim_mean', 'N/A'):.4f}")
        print(f"  Struct Dist: {result_row.get('structure_distance_mean', 'N/A'):.4f}")
    
    if all_results:
        os.makedirs(os.path.dirname(result_path) if os.path.dirname(result_path) else ".", exist_ok=True)
        with open(result_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {result_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Reconstruction Evaluation")
    parser.add_argument("--src_image_folder", type=str, required=True,
                        help="Source images (decoded originals)")
    parser.add_argument("--tgt_image_folders", nargs="+", required=True,
                        help="Target reconstructed image folder(s)")
    parser.add_argument("--dataset_file", type=str, default="dataset/mapping_file.json",
                        help="PIE-Bench mapping file")
    parser.add_argument("--image_dir", type=str, default="dataset/annotation_images",
                        help="Source image directory path in mapping file")
    parser.add_argument("--result_path", type=str, default="results/reconstruction_results.csv",
                        help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--image_size", type=int, default=512, help="Image resolution")
    args = parser.parse_args()
    
    evaluate_reconstruction(
        args.src_image_folder, args.tgt_image_folders,
        args.dataset_file, args.image_dir,
        args.result_path, args.device, args.image_size)
