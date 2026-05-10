#!/bin/bash
# Run all evaluations on generated results

set -e

DEVICE=${1:-"cuda:0"}

echo "=== Running All Evaluations ==="

# Image editing evaluation
echo ""
echo "1. Image Editing Evaluation..."
python evaluation/eval_editing.py \
    --tgt_image_folders outputs_editing/cfg-8.0_lamb-0.9_t_end-0.75 \
    --result_path results/editing_results.csv \
    --device ${DEVICE}

# Image reconstruction evaluation
echo ""
echo "2. Image Reconstruction Evaluation..."
python evaluation/eval_reconstruction.py \
    --src_image_folder outputs_decoded \
    --tgt_image_folders outputs_inv/cfg-8.0_lamb-0.9_t_end-0.75 \
    --result_path results/reconstruction_results.csv \
    --device ${DEVICE}

# Text editing evaluation
echo ""
echo "3. Text Editing Evaluation (RoBERTa)..."
python text_editing/roberta/evaluate.py \
    --mode reconstruction \
    --results_dir results_roberta

echo ""
echo "All evaluations complete! Check results/ for output files."
