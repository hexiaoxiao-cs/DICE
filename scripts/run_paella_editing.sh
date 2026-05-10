#!/bin/bash
# Reproduce Paella editing results from Table 2
# Default hyperparameters: cfg=8.0, lamb=0.9, max_t=0.75

set -e

GPU=${1:-0}
RESOLUTION=${2:-512}

echo "=== DICE Editing with Paella (${RESOLUTION}x${RESOLUTION}) ==="
echo "GPU: ${GPU}"

# Step 1: Cache embeddings (only needed once)
echo ""
echo "Step 1: Caching text/CLIP embeddings..."
python image_editing/paella/editing.py \
    --cache_inputs \
    --gpu ${GPU} \
    --resolution ${RESOLUTION}

# Step 2: Run editing with paper's best hyperparameters
echo ""
echo "Step 2: Running DICE editing (cfg=8.0, lamb=0.9, max_t=0.75)..."
python image_editing/paella/editing.py \
    --cfg 8.0 \
    --lamb 0.9 \
    --max_t 0.75 \
    --gpu ${GPU} \
    --resolution ${RESOLUTION}

echo ""
echo "Done! Check outputs_editing/ for results."
