"""RSL-RL configuration for the G1 height-map HIM-MoE task."""

from isaaclab.utils import configclass

from .rsl_rl_him_moe_cfg import (
    RslRlHimMoeActorCriticCfg,
    RslRlHimMoeAlgorithmCfg,
    RslRlHimMoeRunnerCfg,
)


@configclass
class RslRlPerceptionProActorCriticCfg(RslRlHimMoeActorCriticCfg):
    """Select the flattened 11x17 height-map adapter."""

    exteroception_group: str = "height_map"
    # GridPatternCfg(ordering="xy") flattens the 17 x-points inside the 11 y-rows.
    exteroception_shape: tuple[int, int] = (11, 17)
    exteroception_input_mode: str = "flat"
    exteroception_first_conv_channels: int = 16
    exteroception_first_conv_stride: int = 1


# Backwards-compatible configuration import; both tasks now use the same algorithm class.
RslRlPerceptionProAlgorithmCfg = RslRlHimMoeAlgorithmCfg


@configclass
class UnitreePerceptionProRunnerCfg(RslRlHimMoeRunnerCfg):
    experiment_name = "Unitree-perception-pro"
    obs_groups = {"policy": ["policy", "height_map"], "critic": ["critic"]}

    policy = RslRlPerceptionProActorCriticCfg()
