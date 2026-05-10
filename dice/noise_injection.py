"""
Noise injection strategies for DICE
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional


class BaseNoiseInjection(ABC):
    """Base class for noise injection strategies"""
    
    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        gumbel_noise: torch.Tensor,
        lambda_1: float,
        lambda_2: float
    ) -> torch.Tensor:
        """
        Apply noise injection to logits.
        
        Args:
            logits: Model output logits
            z: Recorded noise from inversion
            gumbel_noise: Random Gumbel noise
            lambda_1: Weight for recorded noise
            lambda_2: Weight for random noise
            
        Returns:
            Modified logits
        """
        pass


class AdditionNoiseInjection(BaseNoiseInjection):
    """
    Addition-based noise injection (default).
    
    Formula: logits + lambda_1 * z + lambda_2 * gumbel_noise
    """
    
    def __call__(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        gumbel_noise: torch.Tensor,
        lambda_1: float,
        lambda_2: float
    ) -> torch.Tensor:
        """
        Addition-based noise injection.
        
        Args:
            logits: Model output logits [B, C, H, W]
            z: Recorded noise from inversion [B, C, H, W]
            gumbel_noise: Random Gumbel noise [B, C, H, W]
            lambda_1: Weight for recorded noise
            lambda_2: Weight for random noise
            
        Returns:
            Modified logits
        """
        return logits + lambda_1 * z + lambda_2 * gumbel_noise


class VariancePreservingNoiseInjection(BaseNoiseInjection):
    """
    Variance-preserving noise injection.
    
    Formula: logits + sqrt(lambda_1) * z + sqrt(lambda_2) * gumbel_noise
    
    This ensures lambda_1 + lambda_2 = 1 for variance preservation.
    """
    
    def __call__(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        gumbel_noise: torch.Tensor,
        lambda_1: float,
        lambda_2: float
    ) -> torch.Tensor:
        """
        Variance-preserving noise injection.
        
        Args:
            logits: Model output logits [B, C, H, W]
            z: Recorded noise from inversion [B, C, H, W]
            gumbel_noise: Random Gumbel noise [B, C, H, W]
            lambda_1: Weight for recorded noise (will be sqrt'd)
            lambda_2: Weight for random noise (will be sqrt'd)
            
        Returns:
            Modified logits
        """
        import math
        return logits + math.sqrt(lambda_1) * z + math.sqrt(lambda_2) * gumbel_noise


class MaxNoiseInjection(BaseNoiseInjection):
    """
    Max-based noise injection.
    
    Formula: logits + max(lambda_1 * z, lambda_2 * gumbel_noise)
    
    Inspired by Gumbel distribution property: max of two Gumbels is Gumbel.
    """
    
    def __call__(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        gumbel_noise: torch.Tensor,
        lambda_1: float,
        lambda_2: float
    ) -> torch.Tensor:
        """
        Max-based noise injection.
        
        Args:
            logits: Model output logits [B, C, H, W]
            z: Recorded noise from inversion [B, C, H, W]
            gumbel_noise: Random Gumbel noise [B, C, H, W]
            lambda_1: Weight for recorded noise
            lambda_2: Weight for random noise
            
        Returns:
            Modified logits
        """
        return logits + torch.max(lambda_1 * z, lambda_2 * gumbel_noise)


def get_noise_injection(strategy: str = "addition") -> BaseNoiseInjection:
    """
    Get noise injection strategy by name.
    
    Args:
        strategy: Name of strategy ("addition", "variance_preserving", "max")
        
    Returns:
        Noise injection instance
        
    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "addition": AdditionNoiseInjection,
        "variance_preserving": VariancePreservingNoiseInjection,
        "max": MaxNoiseInjection,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown noise injection strategy: {strategy}. "
                        f"Available: {list(strategies.keys())}")
    
    return strategies[strategy]()
