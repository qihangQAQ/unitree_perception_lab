"""G1 rollout environment for collecting first-stage height-map FDM data."""

from __future__ import annotations

import importlib

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp as locomotion_mdp

from . import mdp
from .terrain_importer import SplitAwareUsdTerrainImporterCfg

_perception_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_perception_env_cfg"
)
_predict_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_perception_predict_env_cfg"
)
PerceptionRobotSceneCfg = _perception_module.RobotSceneCfg
PredictRobotEnvCfg = _predict_module.RobotEnvCfg

DEFAULT_TERRAIN_USD = (
    "/home/qihang/code/fdm/exts/fdm/data/Terrains/"
    "navigation_terrain_wall_usd_merge_large_single_object_maze.usd"
)

NAVIGATION_BODY_NAMES = [
    "torso_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]


@configclass
class FDMRobotSceneCfg(PerceptionRobotSceneCfg):
    terrain = SplitAwareUsdTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=DEFAULT_TERRAIN_USD,
        usd_uniform_env_spacing=10.0,
        active_split="train",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=6, track_air_time=True
    )
    fdm_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(1.75, 0.0, 4.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(4.5, 5.9)),
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.ExternalVelocityCommandCfg()


@configclass
class EmptyRewardsCfg:
    pass


@configclass
class EmptyCurriculumCfg:
    pass


@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=mdp.TerrainAnalysisRootReset(
            cfg=mdp.TerrainAnalysisSpawnCfg(
                raycaster_sensor="fdm_height_scanner",
                sample_points=30_000,
                grid_resolution=0.05,
                wall_height=2.25,
                robot_height=0.6,
                robot_buffer_spawn=0.7,
                door_filtering=True,
                door_height_threshold=1.2,
                safety_margin=0.3,
            ),
            robot_dim=0.6,
        ),
        mode="reset",
        params={
            "xy_jitter": 3.5,
            "yaw_range": (-3.141592653589793, 3.141592653589793),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=locomotion_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=locomotion_mdp.time_out, time_out=True)
    navigation_collision = DoneTerm(
        func=mdp.navigation_contact_delayed,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NAVIGATION_BODY_NAMES, preserve_order=True),
            "threshold": 1.0,
        },
    )


@configclass
class RobotEnvCfg(PredictRobotEnvCfg):
    """The frozen actor interface is inherited unchanged; learning managers are empty."""

    scene: FDMRobotSceneCfg = FDMRobotSceneCfg(num_envs=256, env_spacing=10.0)
    commands: CommandsCfg = CommandsCfg()
    rewards: EmptyRewardsCfg = EmptyRewardsCfg()
    curriculum: EmptyCurriculumCfg = EmptyCurriculumCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 80.0
        self.observations.policy.enable_corruption = True
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 0.1
        self.scene.fdm_height_scanner.update_period = 0.5
        self.scene.left_foot_height_scanner.update_period = self.sim.dt
        self.scene.right_foot_height_scanner.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
