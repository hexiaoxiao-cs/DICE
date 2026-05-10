"""
Utility functions for DICE
"""

import torch
from typing import List
import numpy as np


__all__ = ['compute_mutual_information', 'visualize_mask_schedule', 'load_pretrained_model']


def compute_mutual_information(
    beta_t: float,
    alpha_bar_t_minus_1: float,
    alpha_t: float,
    D: int = 1
) -> float:
    """
    Compute mutual information I(z_t; x_0) for DDPM inversion.
    
    From the paper: For a simple Gaussian DDPM with x_0 ~ N(0, I),
    the mutual information between z_t and x_0 can be computed in closed form.
    
    Args:
        beta_t: Beta at timestep t
        alpha_bar_t_minus_1: Cumulative alpha at timestep t-1
        alpha_t: Alpha at timestep t
        D: Dimensionality (default 1)
        
    Returns:
        Mutual information value
        
    Reference:
        Remark 1 in the paper
    """
    numerator = beta_t**2 * alpha_bar_t_minus_1 + 1 - alpha_bar_t_minus_1 + alpha_t * (1 - alpha_t)
    denominator = 1 - alpha_bar_t_minus_1 + alpha_t * (1 - alpha_t)
    
    mi = (D / 2) * np.log(numerator / denominator)
    return mi


def visualize_mask_schedule(
    mask_schedule: List[torch.Tensor],
    save_path: str = None
) -> None:
    """
    Visualize mask schedule.
    
    Args:
        mask_schedule: List of mask tensors
        save_path: Path to save visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        n_masks = len(mask_schedule)
        n_cols = min(8, n_masks)
        n_rows = (n_masks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, mask in enumerate(mask_schedule[:n_cols * n_rows]):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(mask.cpu().numpy(), cmap='gray')
            axes[row, col].set_title(f't={i}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_masks, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


def load_pretrained_model(model_path: str, model_class, device: torch.device = None):
    """
    Load pretrained model weights.
    
    Args:
        model_path: Path to model weights
        model_class: Model class to instantiate
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model = model_class()
    model.load_state_dict(state_dict)
    model.eval()
    
    if device:
        model = model.to(device)
    
    return model
