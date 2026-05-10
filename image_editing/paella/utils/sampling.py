"""
DICE Sampling Utilities.

Contains Gumbel noise sampling and mask schedule generation functions
used by the DICE inversion algorithm for discrete diffusion models.
"""

import torch
import math


def sample_gumbel(shape, eps=1e-20, device="cuda"):
    """Sample from the Gumbel(0, 1) distribution.
    
    Used as the noise source in the discrete diffusion process,
    following the Gumbel-Max reparameterization trick.
    
    Args:
        shape: Shape of the output tensor.
        eps: Small constant for numerical stability.
        device: Device to create the tensor on.
    
    Returns:
        Tensor of Gumbel noise samples.
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def generate_mask_schedule(timesteps, latent_shape, max_t=0.75, device="cuda"):
    """Generate a linear mask schedule for discrete diffusion.
    
    Creates a sequence of binary masks where each mask indicates which
    tokens are replaced by noise at each timestep. The masking ratio
    increases linearly from 0 to max_t.
    
    Args:
        timesteps: Number of diffusion timesteps.
        latent_shape: Shape of the latent token map, e.g. (batch, H, W).
        max_t: Maximum masking ratio (0 to 1).
        device: Device to create tensors on.
    
    Returns:
        List of binary mask tensors, one per timestep.
    """
    t = torch.linspace(0, max_t, timesteps)
    mask_schedule = (torch.rand(latent_shape) <= t[:, None, None, None]).long().to(device)
    return list(mask_schedule)


def sample_noised_tokens(x0, mask_schedule, latent_shape, init_noise=None, 
                         num_labels=8192, device="cuda"):
    """Sample noised token sequences from clean tokens x0.
    
    For each timestep's mask, replaces masked positions with random tokens
    (or provided noise tokens) while keeping unmasked positions as x0.
    
    This implements the forward noising process in token space:
        T_t = x0 * (1 - mask_t) + noise * mask_t
    
    Args:
        x0: Clean token map of shape (batch, H, W).
        mask_schedule: List of binary masks from generate_mask_schedule.
        latent_shape: Shape of the latent space.
        init_noise: Optional fixed noise tensor. If None, random noise is used.
        num_labels: Number of token labels in the vocabulary.
        device: Device for tensor creation.
    
    Returns:
        List of (noised_tokens, timestep_index) tuples.
    """
    noised_tokens = [(x0, -1)]
    
    for idx, mask in enumerate(mask_schedule):
        if init_noise is not None:
            noise = init_noise
        else:
            noise = torch.randint(low=0, high=num_labels, size=latent_shape, device=device)
        xt = x0 * (1 - mask) + noise * mask
        noised_tokens.append((xt, idx))
    
    return noised_tokens


def get_lambda_schedule(lamb, steps, schedule_type="None"):
    """Generate a lambda schedule for noise interpolation during reconstruction.
    
    Args:
        lamb: Base lambda value (noise interpolation weight).
        steps: Number of diffusion steps.
        schedule_type: Type of schedule:
            - "None": Constant lambda
            - "linear": Linear decay from lamb to 0
            - "exp": Exponential decay from lamb to 1
    
    Returns:
        Tensor of lambda values, one per step.
    """
    if schedule_type == "None":
        return torch.ones(steps, dtype=torch.float32) * lamb
    elif schedule_type == "linear":
        return torch.linspace(lamb, 0, steps)
    elif schedule_type == "exp":
        return torch.exp(torch.linspace(math.log(lamb + 1e-8), 0, steps))
    else:
        raise ValueError(f"Unknown lambda schedule type: {schedule_type}")
