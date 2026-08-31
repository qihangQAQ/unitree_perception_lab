"""Project-wide extensions for the external :mod:`rsl_rl` package.

The upstream library remains the training backend.  This package contains only
the custom HIM, cross-attention, and MoE components shared by Unitree tasks.
"""

from .algorithms import FootholdPPO, HimMoePPO
from .modules import FootholdImagination, HimMoeActorCritic
from .runners import FootholdRunner, HimMoeRunner

__all__ = [
    "FootholdImagination",
    "FootholdPPO",
    "FootholdRunner",
    "HimMoeActorCritic",
    "HimMoePPO",
    "HimMoeRunner",
]
