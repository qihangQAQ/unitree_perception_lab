"""Perception baseline with SSR foothold guidance and foot-edge penetration."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_perception_ablation_env_cfg import (
    Exp2EventCfg,
    Exp2RewardsCfg,
    Exp2RobotSceneCfg,
    _configure_play,
)
from .velocity_perception_env_cfg import RobotEnvCfg as PerceptionRobotEnvCfg


@configclass
class RewardsCfg(Exp2RewardsCfg):
    """Perception + Exp2 penetration + predictive sole-support guidance."""

    imagined_foothold_guidance = RewTerm(
        func=mdp.imagined_foothold_guidance,
        weight=0.25,
        params={
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "patch_size": (0.225, 0.10),
            "patch_resolution": 0.05,
            "support_height_tolerance": 0.03,
            "sole_offset": 0.06,
            "reward_sigma": 0.25,
            "curriculum_level_threshold": 3,
            "terrain_types": ("stairs_up", "stairs_down"),
            "downstairs_terrain_types": ("stairs_down",),
            "downstairs_scale": 1.5,
        },
    )


@configclass
class RobotEnvCfg(PerceptionRobotEnvCfg):
    """Train configuration retaining the original perception actor interface."""

    scene: Exp2RobotSceneCfg = Exp2RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfg = RewardsCfg()
    events: Exp2EventCfg = Exp2EventCfg()

    # None follows SSR's generic dynamics supervision and collects valid
    # touchdown labels on every terrain. Guidance itself remains stairs-only.
    foothold_collection_terrain_types: tuple[str, ...] | None = None


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Deterministic play configuration for the prediction task."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)
