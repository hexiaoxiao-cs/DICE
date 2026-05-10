"""
Neural network modules for diffusion models
Original source: https://github.com/pabloppp/Arroz-Con-Cosas/blob/master/arroz/modules.py
"""

import torch
from torch import nn
import math


class PriorModel(nn.Module):
    """
    Prior model for diffusion-based generation.
    
    Used for generating CLIP image embeddings from text embeddings
    in the Paella model.
    
    Args:
        clip_r: CLIP embedding dimension (default: 1024)
        c_hidden: Hidden dimension (default: 1280)
        c_r: Time embedding dimension (default: 64)
        num_blocks: Number of transformer blocks (default: 48)
    """
    
    def __init__(self, clip_r: int = 1024, c_hidden: int = 1280, c_r: int = 64, num_blocks: int = 48):
        super().__init__()
        self.c_r = c_r
        
        self.blocks = nn.ModuleList([
            nn.Linear(clip_r, c_hidden),
        ])
        
        for i in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(clip_r + c_hidden + c_r, c_hidden * 4, bias=False),
                    nn.LayerNorm(c_hidden * 4),
                    nn.GELU(),
                    nn.Linear(c_hidden * 4, c_hidden, bias=False),
                    nn.LayerNorm(c_hidden),
                )
            )
            self.blocks[-1][0].weight.data *= math.sqrt(1 / num_blocks)
        
        self.blocks.append(nn.Linear(c_hidden, clip_r))
        
    def gen_r_embedding(self, r: torch.Tensor, max_positions: int = 10000) -> torch.Tensor:
        """
        Generate time step embedding.
        
        Args:
            r: Time step in [0, 1]
            max_positions: Maximum positions for sinusoidal encoding
            
        Returns:
            Time embedding tensor
        """
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
        
    def forward(self, x: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, clip_r]
            r: Time step [B]
            c: Conditioning [B, clip_r]
            
        Returns:
            Output tensor [B, clip_r]
        """
        r = self.gen_r_embedding(r)
        x_prev = None
        
        for i, block in enumerate(self.blocks):
            if 0 < i < len(self.blocks) - 1:
                x = torch.cat([x, c, r], dim=1)
            x = block(x) 
            if x_prev is not None and i < len(self.blocks) - 1:
                x = x + x_prev
            x_prev = x
        
        return x

    def update_weights_ema(self, src_model: nn.Module, beta: float = 0.999):
        """
        Update weights with exponential moving average.
        
        Args:
            src_model: Source model to copy weights from
            beta: EMA decay rate
        """
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1 - beta)
