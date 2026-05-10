#!/bin/bash
# Reproduce RoBERTa text editing results from Tables 4-5

set -e

GPU=${1:-0}

echo "=== DICE Text Editing with RoBERTa ==="
echo "GPU: ${GPU}"

# Run editing across all tasks (0-9 per batch)
# Adjust task ranges for full evaluation
for start in $(seq 0 10 760); do
    end=$((start + 10))
    tasks=""
    for t in $(seq $start $((end - 1))); do
        tasks="$tasks $t"
    done
    echo "Processing tasks: $tasks"
    python text_editing/roberta/editing.py \
        --gpu ${GPU} \
        --task $tasks
done

echo ""
echo "Evaluating reconstruction accuracy..."
python text_editing/roberta/evaluate.py \
    --mode reconstruction \
    --results_dir results_roberta

echo ""
echo "Done!"
