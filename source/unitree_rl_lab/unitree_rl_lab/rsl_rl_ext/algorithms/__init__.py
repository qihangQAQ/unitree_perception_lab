"""Custom algorithms built on top of RSL-RL."""

from .foothold_ppo import FootholdPPO
from .him_moe_ppo import HimMoePPO

__all__ = ["FootholdPPO", "HimMoePPO"]
