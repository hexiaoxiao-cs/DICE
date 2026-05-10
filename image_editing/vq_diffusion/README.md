# VQ-Diffusion Setup for DICE

## Prerequisites

DICE with VQ-Diffusion requires the [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion) codebase.

## Setup Steps

### 1. Clone VQ-Diffusion

```bash
git clone https://github.com/microsoft/VQ-Diffusion.git
cd VQ-Diffusion
pip install -e .
```

### 2. Download Checkpoints

Download the ITHQ model checkpoint:

```bash
mkdir -p checkpoints/vq_diffusion
# Download ithq_learnable.pth from the VQ-Diffusion repository
# Place at: checkpoints/vq_diffusion/ithq_learnable.pth
```

### 3. Run DICE Editing

```bash
python image_editing/vq_diffusion/inference.py \
    --lamb 0.5 \
    --config VQ-Diffusion/configs/ithq.yaml \
    --checkpoint checkpoints/vq_diffusion/ithq_learnable.pth \
    --gpu 0
```

## Notes

- The DICE modifications to VQ-Diffusion involve adding a `edit_content_dps` method to the VQ-Diffusion transformer that supports noise residual injection during the reverse process.
- The `lamb` parameter controls the strength of the noise residual signal (higher = more faithful to source structure).
- VQ-Diffusion uses `guidance_scale=5.0` by default (equivalent to the CFG parameter in Paella).
- Results marked with `†` in Table 2 of the paper indicate VQ-Diffusion experiments (trained on ITHQ dataset).
