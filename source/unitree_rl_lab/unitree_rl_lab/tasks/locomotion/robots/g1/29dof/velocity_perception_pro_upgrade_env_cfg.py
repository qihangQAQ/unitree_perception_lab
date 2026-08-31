"""Yaw-tracking and stepping-stone upgrade for the perception-pro task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_perception_pro_env_cfg import (
    CROSS_STEPPING_STONES_TERRAIN,
    RewardsCfg as PerceptionProRewardsCfg,
    RobotEnvCfg as PerceptionProRobotEnvCfg,
    TerminationsCfg as PerceptionProTerminationsCfg,
)


@configclass
class RewardsCfg(PerceptionProRewardsCfg):
    """Remove the global termination cost and penalize large yaw-rate errors."""

    termination_penalty = None
    delta_yaw = RewTerm(func=mdp.delta_yaw_reward, weight=-0.5)


@configclass
class TerminationsCfg(PerceptionProTerminationsCfg):
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
class RobotEnvCfg(PerceptionProRobotEnvCfg):
    """Training configuration for the perception-pro upgrade."""

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Deterministic play configuration for the perception-pro upgrade."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.terminations.time_out = None
        self.scene.num_envs = 16
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.terrain_specific_ranges = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.observations.height_map.enable_corruption = False
