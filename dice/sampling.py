"""
Sampling functions for discrete diffusion models
"""

import torch
from typing import Optional, Dict, Tuple
from .inversion import sample_gumbel


def sample(
    model,
    model_inputs: Dict,
    latent_shape: tuple,
    unconditional_inputs: Optional[Dict] = None,
    init_x: Optional[torch.Tensor] = None,
    steps: int = 12,
    renoise_steps: Optional[int] = None,
    temperature: Tuple[float, float] = (0.7, 0.3),
    cfg: Tuple[float, float] = (8.0, 8.0),
    t_start: float = 1.0,
    t_end: float = 0.0,
    attn_weights: Optional[torch.Tensor] = None,
    fixed_noise: bool = True,
    device: torch.device = None
) -> Tuple[torch.Tensor, list]:
    """
    Sample from discrete diffusion model (standard sampling without inversion).
    
    Args:
        model: Discrete diffusion model
        model_inputs: Model conditioning inputs
        latent_shape: Shape of latent space (B, H, W)
        unconditional_inputs: Unconditional inputs for CFG
        init_x: Initial tokens (optional)
        steps: Number of sampling steps
        renoise_steps: Number of renoising steps
        temperature: Temperature schedule (start, end)
        cfg: Classifier-free guidance scale (start, end)
        t_start: Start time (1.0 for standard sampling)
        t_end: End time (0.0 for standard sampling)
        attn_weights: Attention weights for conditioning
        fixed_noise: Whether to use fixed noise
        device: Torch device
        
    Returns:
        Tuple of (sampled_tokens, intermediate_images)
    """
    if renoise_steps is None:
        renoise_steps = steps - 1
    
    if unconditional_inputs is None:
        unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
    
    intermediate_images = []
    
    with torch.inference_mode():
        init_noise = torch.randint(0, model.num_labels, size=latent_shape, device=device)
        
        if init_x is not None:
            sampled = init_x
        else:
            sampled = init_noise.clone()
        
        t_list = torch.linspace(t_start, t_end, steps + 1, device=device)
        temperatures = torch.linspace(temperature[0], temperature[1], steps, device=device)
        cfgs = torch.linspace(cfg[0], cfg[1], steps, device=device)
        
        for i, tv in enumerate(t_list[:steps]):
            t = torch.ones(latent_shape[0], device=device) * tv
            
            logits = model(sampled, t, **model_inputs, attn_weights=attn_weights)
            
            if cfg is not None:
                logits = logits * cfgs[i] + model(sampled, t, **unconditional_inputs) * (1 - cfgs[i])
            
            logits = logits.div(temperatures[i]).softmax(dim=1) + sample_gumbel(logits.shape, device=device).softmax(dim=1)
            sampled = logits.argmax(dim=1)
            
            intermediate_images.append(sampled)
            
            if i < renoise_steps:
                t_next = torch.ones(latent_shape[0], device=device) * t_list[i + 1]
                sampled = model.add_noise(sampled, t_next, random_x=init_noise if fixed_noise else None)[0]
                intermediate_images.append(sampled)
    
    return sampled, intermediate_images
