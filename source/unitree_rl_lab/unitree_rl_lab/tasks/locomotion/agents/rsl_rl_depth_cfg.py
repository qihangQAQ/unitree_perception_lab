"""RSL-RL configuration for the G1 depth-image HIM-MoE task."""

from isaaclab.utils import configclass

from .rsl_rl_him_moe_cfg import (
    RslRlHimMoeActorCriticCfg,
    RslRlHimMoeAlgorithmCfg,
    RslRlHimMoeRunnerCfg,
)


@configclass
class RslRlDepthProActorCriticCfg(RslRlHimMoeActorCriticCfg):
    """Select the channels-last 16x24 depth-image adapter."""

    exteroception_group: str = "depth"
    exteroception_shape: tuple[int, int] = (16, 24)
    exteroception_input_mode: str = "image"
    exteroception_first_conv_channels: int = 8
    exteroception_first_conv_stride: int = 2


# Backwards-compatible configuration import; both tasks now use the same algorithm class.
RslRlDepthProAlgorithmCfg = RslRlHimMoeAlgorithmCfg


@configclass
class UnitreeDepthRunnerCfg(RslRlHimMoeRunnerCfg):
    experiment_name = "Unitree-Velocity-depth-pro"
    obs_groups = {"policy": ["policy", "depth"], "critic": ["critic"]}

    policy = RslRlDepthProActorCriticCfg()
