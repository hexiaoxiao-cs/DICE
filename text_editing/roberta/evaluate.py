"""
Evaluation for DICE Text Editing (RoBERTa).

Two evaluation modes:
  1. Reconstruction accuracy: checks if inverted text exactly matches the original.
  2. Cosine similarity: measures semantic similarity between source, baseline, and DICE outputs.

Usage:
    python evaluate.py --mode reconstruction --results_dir results_roberta
    python evaluate.py --mode similarity --results_dir results_roberta --baseline_dir results_baseline
"""

import os
import re
import glob
import json
import argparse

import torch
from sentence_transformers import SentenceTransformer


def extract_text_between_tags(text):
    """Extract text between </s></s> and the last </s>."""
    pattern = r"</s></s>(.*?)</s>$"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def extract_text_after_dot_space(text):
    """Extract text after the first '. ' (period followed by space)."""
    pattern = r"\.\s(.*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def evaluate_reconstruction(results_dir, dataset):
    """Evaluate reconstruction accuracy (exact match after inversion + reconstruction)."""
    files = glob.glob(os.path.join(results_dir, "*.txt"))
    count = 0
    total = len(files)
    
    for file_path in files:
        bn = os.path.basename(file_path)
        task_id = bn.split("_")[1].split(".")[0]
        ref = dataset[task_id]["negative_sentiment"][1]
        
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        text = extract_text_between_tags(lines[0])
        if text and text.strip() == ref.strip():
            count += 1
    
    accuracy = count / total if total > 0 else 0
    print(f"Reconstruction Accuracy: {accuracy:.4f} ({count}/{total})")
    return accuracy


def evaluate_similarity(results_dir, baseline_dir, dataset, model_name="sentence-transformers/all-roberta-large-v1"):
    """Evaluate cosine similarity between reference, baseline, and DICE outputs."""
    model = SentenceTransformer(model_name)
    
    output_lines = []
    total_sim_baseline = 0
    total_sim_dice = 0
    count = 0
    
    for i in range(765):
        file_dice = os.path.join(results_dir, f"output_{i}.txt")
        file_baseline = os.path.join(baseline_dir, f"output_{i}.txt")
        
        if not os.path.exists(file_dice) or not os.path.exists(file_baseline):
            continue
        
        with open(file_dice, "r") as f:
            lines_dice = f.readlines()
        with open(file_baseline, "r") as f:
            lines_baseline = f.readlines()
        
        text_baseline = extract_text_between_tags(lines_baseline[0])
        text_dice = extract_text_between_tags(lines_dice[0])
        
        if text_baseline is None or text_dice is None:
            continue
        
        task_id = str(i)
        ref = dataset[task_id]["negative_sentiment"][1]
        
        similarities = model.similarity(
            model.encode([ref.strip(), text_baseline.strip(), text_dice.strip()]),
            model.encode([ref.strip(), text_baseline.strip(), text_dice.strip()])
        )
        
        sim_baseline = float(abs(similarities[0][1]))
        sim_dice = float(similarities[0][2])
        
        output_lines.append(f"{i},{sim_baseline},{sim_dice}")
        total_sim_baseline += sim_baseline
        total_sim_dice += sim_dice
        count += 1
    
    if count > 0:
        print(f"Average Cosine Similarity:")
        print(f"  Baseline: {total_sim_baseline / count:.4f}")
        print(f"  DICE:     {total_sim_dice / count:.4f}")
    
    return output_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DICE Text Editing")
    parser.add_argument("--mode", type=str, choices=["reconstruction", "similarity"],
                        default="reconstruction", help="Evaluation mode")
    parser.add_argument("--results_dir", type=str, default="results_roberta",
                        help="DICE results directory")
    parser.add_argument("--baseline_dir", type=str, default="results_baseline",
                        help="Baseline results directory (for similarity mode)")
    parser.add_argument("--dataset", type=str, default="dataset/text_editing_dataset.json",
                        help="Path to sentiment dataset")
    parser.add_argument("--output_file", type=str, default="similarity_scores.txt",
                        help="Output file for similarity scores")
    args = parser.parse_args()
    
    dataset = json.load(open(args.dataset, "r"))
    
    if args.mode == "reconstruction":
        evaluate_reconstruction(args.results_dir, dataset)
    elif args.mode == "similarity":
        scores = evaluate_similarity(args.results_dir, args.baseline_dir, dataset)
        with open(args.output_file, "w") as f:
            for line in scores:
                f.write(f"{line}\n")
        print(f"Scores saved to {args.output_file}")
