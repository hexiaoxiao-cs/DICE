"""
Bundled arroz package for DICE
Original source: https://github.com/pabloppp/Arroz-Con-Cosas

This package includes essential classes for Paella prior model:
- Diffuzz: Custom simplified forward/backward diffusion with cosine schedule
- PriorModel: Neural network for diffusion prior modeling
"""

from .diffusion import Diffuzz
from .modules import PriorModel

__version__ = "0.1.0"

__all__ = ["Diffuzz", "PriorModel"]
