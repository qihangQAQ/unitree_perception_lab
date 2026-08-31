"""Terrain, fall-handling, and yaw-tracking upgrade for the depth task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.terrains import UPGRADE_TERRAIN2

from .velocity_depth_env_cfg import (
    CROSS_STEPPING_STONES_TERRAIN,
    RewardsCfg as DepthRewardsCfg,
    RobotEnvCfg as DepthRobotEnvCfg,
    RobotPlayEnvCfg as DepthRobotPlayEnvCfg,
    TerminationsCfg as DepthTerminationsCfg,
)


@configclass
class RewardsCfg(DepthRewardsCfg):
    """Remove the global termination cost and penalize large yaw-rate errors."""

    termination_penalty = None
    delta_yaw = RewTerm(func=mdp.delta_yaw_reward, weight=-0.5)


@configclass
class TerminationsCfg(DepthTerminationsCfg):
    """Use orientation and terrain-specific fall detection instead of base contact."""

    base_contact = None
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    fall_down_in_cross_stepping_stones = DoneTerm(
        func=mdp.fall_down_in_some_terrain,
        params={
            "minimum_height": 0.5,
            "terrain_types": (CROSS_STEPPING_STONES_TERRAIN,),
        },
    )


@configclass
class RobotEnvCfg(DepthRobotEnvCfg):
    """Training configuration for the depth upgrade."""

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.scene.terrain.terrain_generator = UPGRADE_TERRAIN2
        super().__post_init__()


@configclass
class RobotPlayEnvCfg(DepthRobotPlayEnvCfg):
    """Depth play configuration with the upgrade objectives and terrain."""

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.scene.terrain.terrain_generator = UPGRADE_TERRAIN2
        super().__post_init__()
