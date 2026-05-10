#!/bin/bash
# Reproduce Paella inversion/reconstruction results from Table 1

set -e

GPU=${1:-0}

echo "=== DICE Inversion + Reconstruction with Paella ==="
echo "GPU: ${GPU}"

# Run reconstruction (same source prompt for both inversion and reconstruction)
python image_editing/paella/inversion.py \
    --cfg 8.0 \
    --lamb 0.9 \
    --max_t 0.75 \
    --gpu ${GPU}

echo ""
echo "Done! Check outputs_inv/ for results."
