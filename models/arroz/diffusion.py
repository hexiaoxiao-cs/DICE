"""
Diffusion utilities with cosine schedule
Original source: https://github.com/pabloppp/Arroz-Con-Cosas/blob/master/arroz/diffusion.py
"""

import torch
from typing import Optional, Dict, Callable


class Diffuzz:
    """
    Custom simplified forward/backward diffusion using cosine schedule.
    
    Args:
        s: Cosine schedule parameter (default: 0.008)
        device: Device for computations
    """
    
    def __init__(self, s: float = 0.008, device: str = "cpu"):
        self.device = device
        self.s = torch.tensor([s]).to(device)
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def _alpha_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Compute cumulative product of alphas at time t."""
        alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod
        return alpha_cumprod.clamp(0.0001, 0.9999) 

    def diffuse(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> tuple:
        """
        Add noise to x at timestep t.
        
        Args:
            x: Input tensor
            t: Timestep in [0, 1]
            noise: Optional noise tensor
            
        Returns:
            Tuple of (noisy_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x)
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        return alpha_cumprod.sqrt() * x + (1-alpha_cumprod).sqrt() * noise, noise

    def undiffuse(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, 
                  noise: torch.Tensor, sampler=None) -> torch.Tensor:
        """
        Remove noise from x at timestep t to t_prev.
        
        Args:
            x: Noisy input
            t: Current timestep
            t_prev: Previous timestep
            noise: Predicted noise
            sampler: Sampler instance
            
        Returns:
            Denoised tensor
        """
        if sampler is None:
            sampler = DDPMSampler(self)
        return sampler(x, t, t_prev, noise)
        
    def sample(
        self,
        model: Callable,
        model_inputs: Dict,
        shape: tuple,
        t_start: float = 1.0,
        t_end: float = 0.0,
        timesteps: int = 20,
        x_init: Optional[torch.Tensor] = None,
        cfg: float = 3.0,
        unconditional_inputs: Optional[Dict] = None,
        sampler: str = 'ddpm'
    ) -> list:
        """
        Sample from the diffusion model.
        
        Args:
            model: Diffusion model
            model_inputs: Model conditioning inputs
            shape: Shape of samples to generate
            t_start: Start time (default: 1.0)
            t_end: End time (default: 0.0)
            timesteps: Number of timesteps
            x_init: Initial tensor (optional)
            cfg: Classifier-free guidance scale
            unconditional_inputs: Unconditional inputs for CFG
            sampler: Sampler type ('ddpm' or 'ddim')
            
        Returns:
            List of predictions at each timestep
        """
        r_range = torch.linspace(t_start, t_end, timesteps+2)[1:][:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        
        # Select sampler
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self)
            else:
                raise ValueError(f"Unknown sampler: {sampler}. Available: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler = sampler(self)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object")
        
        preds = []
        x = sampler.init_x(shape) if x_init is None else x_init.clone()
        
        for i in range(0, timesteps):
            pred_noise = model(x, r_range[i], **model_inputs)
            if cfg is not None:
                if unconditional_inputs is None:
                    unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
                pred_noise_unconditional = model(x, r_range[i], **unconditional_inputs)
                pred_noise = torch.lerp(pred_noise_unconditional, pred_noise, cfg)
            x = self.undiffuse(x, r_range[i], r_range[i+1], pred_noise, sampler=sampler)
            preds.append(x)
        return preds
        
    def p2_weight(self, t: torch.Tensor, k: float = 1.0, gamma: float = 1.0) -> torch.Tensor:
        """Compute p2 weighting for training."""
        alpha_cumprod = self._alpha_cumprod(t)
        return (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma


class SimpleSampler:
    """Base sampler class."""
    
    def __init__(self, diffuzz: Diffuzz):
        self.current_step = -1
        self.diffuzz = diffuzz

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape: tuple) -> torch.Tensor:
        return torch.randn(*shape, device=self.diffuzz.device)

    def step(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Override the 'step' function")


class DDPMSampler(SimpleSampler):
    """DDPM sampler."""
    
    def step(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        mu = (1.0 / alpha).sqrt() * (x - (1-alpha) * noise / (1-alpha_cumprod).sqrt())
        std = ((1-alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * torch.randn_like(mu)
        return mu + std * (t_prev != 0).float().view(t_prev.size(0), *[1 for _ in x.shape[1:]])


class DDIMSampler(SimpleSampler):
    """DDIM sampler."""
    
    def step(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
        dp_xt = (1 - alpha_cumprod_prev).sqrt()
        return (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise


sampler_dict = {
    'ddpm': DDPMSampler,
    'ddim': DDIMSampler,
}
