"""Upgraded terrain-perception velocity task for the extra G1 model."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG
from unitree_rl_lab.sensors import Grid3dPointsGeneratorCfg, VolumePointsCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.terrains import (
    UPGRADE_TERRAIN1,
    EdgeCylinderCfg,
    TerrainImporterCfg,
)

from .velocity_perception_env_cfg import EventCfg as PerceptionEventCfg
from .velocity_perception_env_cfg import ObservationsCfg as PerceptionObservationsCfg
from .velocity_perception_env_cfg import RewardsCfg as PerceptionRewardsCfg
from .velocity_perception_env_cfg import RobotEnvCfg as PerceptionRobotEnvCfg
from .velocity_perception_env_cfg import RobotSceneCfg as PerceptionRobotSceneCfg


@configclass
class RobotSceneCfg(PerceptionRobotSceneCfg):
    """Perception scene with the upgrade terrain and virtual edge inflation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UPGRADE_TERRAIN1,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": EdgeCylinderCfg(
                cylinder_radius=0.05,
                min_points=2,
            )
        },
    )
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        update_period=0.02,
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=-0.025,
            x_max=0.12,
            x_num=10,
            y_min=-0.03,
            y_max=0.03,
            y_num=5,
            z_min=-0.04,
            z_max=0.0,
            z_num=2,
        ),
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    """Direct velocity commands with terrain-specific forward speed and independent zeroing."""

    base_velocity = mdp.TerrainAwareUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            zero_prob=(0.4, 0.8, 0.4),
        ),
        terrain_specific_ranges={
            "flat": mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.6, 2.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=(0.4, 0.8, 0.4),
            )
        },
    )


@configclass
class ObservationsCfg(PerceptionObservationsCfg):
    """Perception observations with Noetix-style policy corruption."""

    @configclass
    class PolicyCfg(PerceptionObservationsCfg.PolicyCfg):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1.0,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.025, n_max=0.025),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=1.0,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(PerceptionRewardsCfg):
    """Base perception rewards plus yaw-error and virtual-inflation penalties."""

    delta_yaw = RewTerm(func=mdp.delta_yaw_reward, weight=-0.5)
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            "terrain_types": ("stairs_up", "stairs_down", "discrete_obstacles"),
        },
    )


@configclass
class EventCfg(PerceptionEventCfg):
    """Base randomization events plus virtual-obstacle registration."""

    register_virtual_obstacles = EventTerm(
        func=mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={"sensor_cfgs": SceneEntityCfg("leg_volume_points")},
    )


@configclass
class RobotEnvCfg(PerceptionRobotEnvCfg):
    """Training configuration for the upgraded perception task."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Deterministic play configuration for the upgraded perception task."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.terminations.time_out = None

        self.scene.num_envs = 16
        # Keep the complete 10x12 training grid so --terrain/--level address the original cells.
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.terrain_specific_ranges = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)
