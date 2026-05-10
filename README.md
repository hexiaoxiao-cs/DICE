# 🎲 DICE: Discrete Inversion for Controllable Editing

[![Paper](https://img.shields.io/badge/Paper-WACV%202026-blue)](https://openaccess.thecvf.com/content/WACV2026/html/He_DICE_Discrete_Inversion_Enabling_Controllable_Editing_for_Masked_Generative_Models_WACV_2026_paper.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)

**D**iscrete **I**nversion for **C**ontrollable **E**diting (DICE) is the first inversion framework for discrete diffusion models, enabling precise reconstruction and controllable editing of both images and text.

## 🎯 Overview

DICE pioneers inversion capabilities for discrete diffusion models, including:
- **Masked Generative Models** (e.g., Paella, MaskGIT)
- **Multinomial Diffusion Models** (e.g., VQ-Diffusion)
- **Discrete Language Models** (e.g., RoBERTa, LLaDA)

### Key Features
✅ **Accurate Reconstruction**: Near-perfect reconstruction with PSNR ∞ (VQ-GAN latent space)  
✅ **Controllable Editing**: Fine-grained control over editing strength via noise injection  
✅ **Structure Preservation**: Superior structure preservation compared to continuous diffusion  
✅ **Multi-Modal**: Works for both image and text editing  
✅ **Simple & Efficient**: No complex attention manipulation required  

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DICE.git
cd DICE

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

Download pretrained model weights:

| Model | Size | Download |
|-------|------|----------|
| Paella v3 | 4.0 GB | [Google Drive](link) |
| Prior v1 | 3.6 GB | [Google Drive](link) |
| VQGAN f4 | 73 MB | [Google Drive](link) |

```bash
# Place downloaded weights in models/ directory
mkdir -p models/paella
# Move downloaded files to models/paella/
```

## 🚀 Quick Start

### Image Editing with Paella

```bash
# Best configuration from paper (Table 2)
python scripts/image_editing.py \
    --config configs/paella_editing.yaml \
    --cfg 10.0 \
    --lambda_1 0.7 \
    --lambda_2 0.3 \
    --tau 0.9
```

### Image Reconstruction

```bash
# Perfect reconstruction (Table 1)
python scripts/image_editing.py \
    --config configs/paella_reconstruction.yaml \
    --lambda_1 1.0 \
    --lambda_2 0.0 \
    --tau 1.0
```

### Text Editing with RoBERTa

```bash
# Sentiment editing (Tables 4-5)
python scripts/text_editing_roberta.py \
    --config configs/roberta_editing.yaml \
    --lambda_1 0.2 \
    --lambda_2 0.8 \
    --tau 0.7
```

## 📊 Reproducing Paper Results

### Main Results

**Image Editing (Paella)**
```bash
# Best configuration from paper (Table 2)
python scripts/image_editing.py \
    --config configs/paella_editing.yaml \
    --cfg 10.0 \
    --lambda_1 0.7 \
    --lambda_2 0.3 \
    --tau 0.9
```

**Image Reconstruction**
```bash
# Perfect reconstruction (Table 1)
python scripts/image_editing.py \
    --config configs/paella_reconstruction.yaml \
    --lambda_1 1.0 \
    --lambda_2 0.0 \
    --tau 1.0
```

**Text Editing (RoBERTa)**
```bash
# Sentiment editing (Tables 4-5)
python scripts/text_editing_roberta.py \
    --config configs/roberta_editing.yaml \
    --lambda_1 0.2 \
    --lambda_2 0.8 \
    --tau 0.7
```

### Evaluation

```bash
# Evaluate image editing
python evaluation/eval_editing.py \
    --output_dir outputs/editing/ \
    --metrics structure_distance psnr lpips ssim clip_similarity

# Evaluate text editing
python evaluation/eval_text.py \
    --output_dir outputs/roberta_editing/ \
    --use_chatgpt
```

## 🔧 Hyperparameter Guide

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `λ₁` (lambda_1) | Weight for recorded noise | 0.1 - 0.9 |
| `λ₂` (lambda_2) | Weight for random Gumbel noise | 1.0 - λ₁ |
| `τ` (tau/max_t) | Maximum masking ratio | 0.5 - 1.0 |
| `CFG` | Classifier-free guidance scale | 5.0 - 16.0 |
| `steps` | Number of sampling steps | 12 - 32 |

### Trade-offs
- **Higher λ₁**: Stronger structure preservation, less editing flexibility
- **Higher τ**: More editing flexibility, less structure preservation
- **Higher CFG**: Better text alignment, but may reduce structural quality

## 📁 Project Structure

```
DICE/
├── dice/                   # Core DICE algorithms
│   ├── inversion.py        # Main inversion functions
│   ├── sampling.py         # Sampling utilities
│   ├── noise_injection.py  # Noise injection strategies
│   └── utils.py            # Helper functions
├── models/                 # Model implementations
│   ├── paella/            # Paella model
│   ├── vqgan/             # VQ-GAN for tokenization
│   └── arroz/             # Bundled arroz package
├── scripts/               # Executable scripts
│   ├── image_editing.py
│   ├── image_reconstruction.py
│   └── text_editing_roberta.py
├── configs/               # Configuration files
├── evaluation/            # Evaluation scripts
├── notebooks/             # Tutorial notebooks
├── ablation/              # Ablation study scripts
└── data/                  # Sample data
```

## 📚 Documentation

For detailed usage examples and tutorials, see the `notebooks/` directory.

## 🎨 Results

### Image Editing

| Method | Structure Distance ↓ | CLIP Similarity ↑ | Background PSNR ↑ |
|--------|---------------------|-------------------|------------------|
| DDIM + SD1.4 | 69.43 | 25.01 | 17.87 |
| Null-Text + SD1.4 | 13.44 | 24.75 | - |
| **DICE + Paella** | **11.34** | 23.79 | **27.29** |
| **DICE + VQ-Diffusion** | **12.70** | 23.85 | - |

### Image Reconstruction

| Method | PSNR ↑ | LPIPS ↓ | SSIM ↑ |
|--------|--------|---------|--------|
| Inpainting | 10.50 | 0.565 | 0.301 |
| **DICE** | **30.91** | **0.040** | **0.902** |
| **DICE†** | **∞** | **0.000** | **1.000** |

† VQ-GAN latent space reconstruction

## 🧪 Ablation Studies

Run ablation studies to understand parameter effects:

```bash
# Lambda sweep
python ablation/ablation_lambda.py --cfg 10.0 --tau 0.9

# Tau sweep  
python ablation/ablation_tau.py --cfg 10.0 --lambda 0.7

# CFG sweep
python ablation/ablation_cfg.py --lambda 0.7 --tau 0.9

# Noise injection functions
python ablation/ablation_noise_injection.py
```

## 📖 Citation

If you use DICE in your research, please cite:

```bibtex
@inproceedings{he2026dice,
  title={DICE: Discrete Inversion for Controllable Editing of Masked Generative Models},
  author={He, Xiaoxiao and Dao, Quan and Han, Ligong and Wen, Song and Bai, Minhao and Liu, Di and Zhang, Han and Juefei-Xu, Felix and Tan, Chaowei and Liu, Bo and Min, Martin Renqiang and Li, Kang and Ahmed, Faez and Srivastava, Akash and Li, Hongdong and Huang, Junzhou and Metaxas, Dimitris N.},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## 🙏 Acknowledgments

- [Paella](https://github.com/dome272/Paella) for the base model
- [Arroz-Con-Cosas](https://github.com/pabloppp/Arroz-Con-Cosas) for diffusion utilities
- [VQ-GAN](https://github.com/CompVis/taming-transformers) for image tokenization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📧 Contact

For questions and issues:
- Open an issue on GitHub
- Email: [your-email]

---

**[Project Website](https://hexiaoxiao-cs.github.io/DICE/)** | **[Paper](https://arxiv.org)** | **[Demo](https://huggingface.co/spaces/yourusername/DICE)**
