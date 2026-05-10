"""
Core inversion algorithms for DICE (Discrete Inversion for Controllable Editing)
"""

import torch
from typing import List, Tuple, Optional, Dict
import numpy as np
import random
from itertools import accumulate
import copy


def sample_gumbel(shape: tuple, eps: float = 1e-20, device: torch.device = None) -> torch.Tensor:
    """
    Sample from Gumbel distribution.
    
    Args:
        shape: Shape of the output tensor
        eps: Small constant for numerical stability
        device: Torch device
        
    Returns:
        Gumbel samples
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def generate_mask_schedule(
    timesteps: int,
    latent_shape: tuple = (64, 64),
    max_t: float = 0.75,
    schedule_type: str = "linear",
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Generate mask schedule for inversion.
    
    Args:
        timesteps: Number of timesteps
        latent_shape: Shape of latent space (H, W)
        max_t: Maximum time for masking
        schedule_type: Type of schedule ("linear", "sqrt", "cosine")
        device: Torch device
        
    Returns:
        List of mask tensors for each timestep
    """
    if schedule_type == "linear":
        t = torch.linspace(0, max_t, timesteps, device=device)
        mask_schedule = (torch.rand(latent_shape, device=device) <= t[:, None, None, None]).long()
    elif schedule_type == "sqrt":
        t = torch.linspace(0, max_t, timesteps, device=device)
        mask_schedule = (torch.rand(latent_shape, device=device) <= torch.sqrt(t)[:, None, None, None]).long()
    elif schedule_type == "cosine":
        t = torch.linspace(0, max_t, timesteps, device=device)
        mask_schedule = (torch.rand(latent_shape, device=device) <= (1 - torch.cos(t * 3.14159 / 2))[:, None, None, None]).long()
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return list(mask_schedule)


def sample_Tts_from_x0_mask(
    x0: torch.Tensor,
    mask_schedule: List[torch.Tensor],
    latent_shape: tuple,
    init_noise: Optional[torch.Tensor] = None,
    num_labels: int = 8192,
    device: torch.device = None
) -> List[Tuple[torch.Tensor, int]]:
    """
    Sample noisy latents from clean image tokens.
    
    Args:
        x0: Clean token map [B, H, W]
        mask_schedule: List of masks for each timestep
        latent_shape: Shape of latent space
        init_noise: Initial noise tokens (optional)
        num_labels: Number of token labels
        device: Torch device
        
    Returns:
        List of (noisy_tokens, timestep) tuples
    """
    xts_master_list = [(x0, -1)]
    
    for idx, mask in enumerate(mask_schedule):
        if init_noise is not None:
            noise = init_noise
        else:
            noise = torch.randint(low=0, high=num_labels, size=latent_shape, device=device)
        
        xt_token = x0 * (1 - mask) + noise * mask
        xts_master_list.append((xt_token, idx))
    
    return xts_master_list


def invert_new(
    model,
    x0: torch.Tensor,
    model_inputs: Dict,
    latent_shape: tuple,
    init_noise: Optional[torch.Tensor] = None,
    unconditional_inputs: Optional[Dict] = None,
    steps: int = 32,
    renoise_steps: Optional[int] = None,
    temperature: Tuple[float, float] = (0.2, 1.2),
    cfg: Tuple[float, float] = (8.0, 8.0),
    t_start: float = 0.0,
    t_end: float = 0.75,
    attn_weights: Optional[torch.Tensor] = None,
    device: torch.device = None
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List]:
    """
    Perform discrete inversion to record noise sequence.
    
    Args:
        model: Discrete diffusion model
        x0: Clean token map [B, H, W]
        model_inputs: Model conditioning inputs
        latent_shape: Shape of latent space
        init_noise: Initial noise tokens
        unconditional_inputs: Unconditional inputs for CFG
        steps: Number of inversion steps
        renoise_steps: Number of renoising steps
        temperature: Temperature schedule (start, end)
        cfg: Classifier-free guidance scale (start, end)
        t_start: Start time (0.0 for inversion)
        t_end: End time (max_t for inversion)
        attn_weights: Attention weights for conditioning
        device: Torch device
        
    Returns:
        Tuple of (final_noisy_tokens, intermediate_images, zs, mask_schedule, Tts)
    """
    if renoise_steps is None:
        renoise_steps = steps - 5
    
    if unconditional_inputs is None:
        unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
    
    intermediate_images = []
    
    with torch.inference_mode():
        t_list = torch.linspace(t_start, t_end, steps + 1, device=device)
        temperatures = torch.linspace(temperature[0], temperature[1], steps, device=device)
        cfgs = torch.linspace(cfg[0], cfg[1], steps, device=device)
        
        x0_logits = model(x0, torch.ones(latent_shape[0], device=device) * 0.0, 
                         **model_inputs, attn_weights=attn_weights)
        
        mask_schedule = generate_mask_schedule(steps, latent_shape[-2:], max_t=max(t_end, t_start), device=device)
        Tts = sample_Tts_from_x0_mask(x0, mask_schedule, latent_shape, init_noise=init_noise, 
                                       device=device)
        
        zs = []
        
        for i, tv in enumerate(t_list[1:steps]):
            t = torch.ones(latent_shape[0], device=device) * tv
            
            i = i + 1
            current, idx = Tts[i]
            
            x_0_hat = model(current.long(), t, **model_inputs, attn_weights=attn_weights)
            
            if cfg is not None:
                x_0_hat = x_0_hat * cfgs[i] + model(current.long(), t, **unconditional_inputs) * (1 - cfgs[i])
            
            T_0_hat = x_0_hat.div(temperatures[i]).argmax(dim=1)
            
            z = x0_logits - x_0_hat.div(temperatures[i])
            zs.append(z)
            
            intermediate_images.append(T_0_hat)
    
    return current, intermediate_images, zs, mask_schedule, Tts


def sample_ddpm_inverse_new(
    model,
    model_inputs: Dict,
    latent_shape: tuple,
    zs: List[torch.Tensor],
    mask_schedule: List[torch.Tensor],
    init_noise: torch.Tensor,
    unconditional_inputs: Optional[Dict] = None,
    init_x: Optional[torch.Tensor] = None,
    steps: int = 32,
    renoise_steps: Optional[int] = None,
    temperature: Tuple[float, float] = (1.2, 0.2),
    cfg: Tuple[float, float] = (8.0, 8.0),
    t_start: float = 0.75,
    t_end: float = 0.0,
    lamb: float = 0.7,
    attn_weights: Optional[torch.Tensor] = None,
    noise_injection_fn: callable = None,
    device: torch.device = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Sample using discrete inversion for editing.
    
    Args:
        model: Discrete diffusion model
        model_inputs: Model conditioning inputs
        latent_shape: Shape of latent space
        zs: Recorded noise sequence from inversion
        mask_schedule: Mask schedule from inversion
        init_noise: Initial noise tokens
        unconditional_inputs: Unconditional inputs for CFG
        init_x: Initial tokens to start from
        steps: Number of sampling steps
        renoise_steps: Number of renoising steps
        temperature: Temperature schedule (start, end)
        cfg: Classifier-free guidance scale (start, end)
        t_start: Start time (should match inversion t_end)
        t_end: End time (0.0 for sampling)
        lamb: Noise injection weight (lambda_1)
        attn_weights: Attention weights for conditioning
        noise_injection_fn: Custom noise injection function
        device: Torch device
        
    Returns:
        Tuple of (final_tokens, intermediate_images)
    """
    if renoise_steps is None:
        renoise_steps = steps - 5
    
    if unconditional_inputs is None:
        unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
    
    if noise_injection_fn is None:
        def noise_injection_fn(logits, z, gumbel_noise, lambda_1, lambda_2):
            return logits + lambda_1 * z + lambda_2 * gumbel_noise
    
    intermediate_images = []
    
    with torch.inference_mode():
        sampled = init_x
        noise = init_noise
        
        t_list = torch.linspace(t_start, t_end, steps + 1, device=device)
        temperatures = torch.linspace(temperature[0], temperature[1], steps, device=device)
        cfgs = torch.linspace(cfg[0], cfg[1], steps, device=device)
        
        for i, tv in enumerate(t_list[1:steps]):
            t = torch.ones(latent_shape[0], device=device) * tv
            
            logits = model(sampled, t, **model_inputs, attn_weights=attn_weights)
            
            if cfg is not None:
                logits = logits * cfgs[i] + model(sampled, t, **unconditional_inputs) * (1 - cfgs[i])
            
            lambda_2 = 1.0 - lamb
            gumbel_noise = sample_gumbel(logits.shape, device=device)
            
            logits = noise_injection_fn(logits.div(temperatures[i]), zs[i].to(logits.device), 
                                        gumbel_noise, lamb, lambda_2)
            
            sampled = logits.argmax(dim=1)
            intermediate_images.append(sampled)
            
            if i + 2 < steps:
                sampled = (sampled * (1 - mask_schedule[i + 2].to(device)) + 
                          noise * mask_schedule[i + 2].to(device)).long()
                intermediate_images.append(sampled)
    
    return sampled, intermediate_images
