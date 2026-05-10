"""
AI-based Evaluation for DICE Text Editing (LLaDA / RoBERTa).

Uses GPT-4o-mini to evaluate whether edited sentences:
  1. Preserve the structure of the original sentence
  2. Successfully flip the sentiment

Requires the OPENAI_API_KEY environment variable to be set.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python evaluate.py --results_dir results_llada
"""

import os
import re
import glob
import json
import argparse

import openai


def get_sample_lines(file_path, line_indices=(3, -23)):
    """Get specific lines from a results file for evaluation.
    
    Args:
        file_path: Path to results file.
        line_indices: Tuple of line indices to extract.
    
    Returns:
        Tuple of extracted lines.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    return tuple(lines[idx] for idx in line_indices)


def extract_text_after_dot_space(text):
    """Extract text after the first '. ' in a line."""
    pattern = r"\.\s(.*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def evaluate_with_gpt(results_dir, dataset, model="gpt-4o-mini",
                       source_sentiment="negative", line_indices=(3, -23)):
    """
    Evaluate text editing results using GPT.
    
    For each result file, extracts two edited sentences and asks GPT to
    check structure preservation and sentiment change.
    
    Args:
        results_dir: Directory containing result files.
        dataset: Loaded dataset dictionary.
        model: OpenAI model to use.
        source_sentiment: Sentiment of the source text ("positive" or "negative").
        line_indices: Which lines in each file to evaluate.
    
    Returns:
        List of evaluation results.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. "
                         "Set it with: export OPENAI_API_KEY='your-key'")
    
    openai.api_key = api_key
    
    files = sorted(glob.glob(os.path.join(results_dir, "*")))
    responses = []
    
    target_sentiment = "positive" if source_sentiment == "negative" else "negative"
    
    for file_path in files:
        sample_lines = get_sample_lines(file_path, line_indices)
        
        extracted_text = []
        for line in sample_lines:
            text = extract_text_after_dot_space(line.split("\t")[1].strip())
            extracted_text.append(text)
        
        bn = os.path.basename(file_path)
        task_id = bn.split("_")[1].split(".")[0]
        
        sentence_template = dataset[task_id][f"{source_sentiment}_sentiment"][1].strip()
        
        prompt = (
            f"Given three sentences, confirm that the second and third sentence "
            f"is roughly the same sentence structure as the first sentence, then "
            f"confirm that any of the second and third sentence have {target_sentiment} "
            f"sentiment. Output only two numbers with each number indicating whether "
            f"the corresponding criteria is satisfied. Use 1 for satisfied and 0 for "
            f"not satisfied. The sentences are given below:\n"
            f"{sentence_template}\n"
            f"{extracted_text[0]}\n"
            f"{extracted_text[1]}\n"
        )
        
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=100,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        print(f"Task {task_id}: {result}")
        responses.append(f"{task_id}, {result}")
    
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Evaluation for DICE Text Editing")
    parser.add_argument("--results_dir", type=str, default="results_llada",
                        help="Directory containing result files")
    parser.add_argument("--dataset", type=str, default="dataset/text_editing_dataset.json",
                        help="Path to sentiment dataset")
    parser.add_argument("--output_file", type=str, default="ai_evaluation_results.txt",
                        help="Output file for evaluation results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model for evaluation")
    parser.add_argument("--source_sentiment", type=str, default="negative",
                        choices=["positive", "negative"],
                        help="Sentiment of the source text")
    args = parser.parse_args()
    
    dataset = json.load(open(args.dataset, "r"))
    
    results = evaluate_with_gpt(
        args.results_dir, dataset,
        model=args.model,
        source_sentiment=args.source_sentiment
    )
    
    with open(args.output_file, "w") as f:
        for line in results:
            f.write(f"{line}\n")
    
    print(f"\nResults saved to {args.output_file}")
