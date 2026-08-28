"""Independent ablations for the terrain-perception velocity task.

Each experiment inherits directly from the perception baseline and enables one
additional mechanism:

* Exp1: give the actor the same full observation set as the critic.
* Exp2: add virtual terrain-edge inflation and the penetration penalty.
* Exp3: add the two stairs-down progress rewards from perception-pro.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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


DOWNHILL_TERRAIN_TYPES = ("stairs_down",)


def _configure_play(cfg) -> None:
    """Apply deterministic play settings shared by all ablations."""

    cfg.events.push_robot = None
    cfg.terminations.time_out = None
    cfg.scene.num_envs = 16
    cfg.curriculum.terrain_levels = None

    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.terrain_specific_ranges = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)


@configclass
class Exp1ObservationsCfg(PerceptionObservationsCfg):
    """Use the critic's complete clean observation set for both networks."""

    @configclass
    class PolicyCfg(PerceptionObservationsCfg.CriticCfg):
        pass

    policy: PolicyCfg = PolicyCfg()


@configclass
class Exp1RobotEnvCfg(PerceptionRobotEnvCfg):
    """Perception baseline with identical actor and critic observations."""

    observations: Exp1ObservationsCfg = Exp1ObservationsCfg()


@configclass
class Exp1RobotPlayEnvCfg(Exp1RobotEnvCfg):
    """Deterministic play configuration for Exp1."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class Exp2RobotSceneCfg(PerceptionRobotSceneCfg):
    """Perception scene with analytical terrain-edge inflation."""

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
class Exp2RewardsCfg(PerceptionRewardsCfg):
    """Perception rewards plus the virtual-inflation penetration penalty."""

    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            "terrain_types": ("stairs_up", "stairs_down", "discrete_obstacles"),
        },
    )


@configclass
class Exp2EventCfg(PerceptionEventCfg):
    """Register inflated terrain edges with the ankle-volume sensor."""

    register_virtual_obstacles = EventTerm(
        func=mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={"sensor_cfgs": SceneEntityCfg("leg_volume_points")},
    )


@configclass
class Exp2RobotEnvCfg(PerceptionRobotEnvCfg):
    """Perception baseline with only the penetration mechanism enabled."""

    scene: Exp2RobotSceneCfg = Exp2RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: Exp2RewardsCfg = Exp2RewardsCfg()
    events: Exp2EventCfg = Exp2EventCfg()


@configclass
class Exp2RobotPlayEnvCfg(Exp2RobotEnvCfg):
    """Deterministic play configuration for Exp2."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class Exp3RewardsCfg(PerceptionRewardsCfg):
    """Perception rewards plus the two perception-pro stairs-down terms."""

    stairs_down_safe_progress = RewTerm(
        func=mdp.stairs_outward_progress_reward,
        weight=0.5,
        params={
            "terrain_types": DOWNHILL_TERRAIN_TYPES,
            "min_forward_command": 0.1,
            "cap_by_command": False,
            "max_outward_speed": 0.5,
        },
    )
    stairs_down_stall = RewTerm(
        func=mdp.stairs_stall_penalty,
        weight=-1.0,
        params={
            "terrain_types": DOWNHILL_TERRAIN_TYPES,
            "min_forward_command": 0.1,
            "min_outward_speed": 0.05,
            "grace_steps": 50,
        },
    )


@configclass
class Exp3RobotEnvCfg(PerceptionRobotEnvCfg):
    """Perception baseline with only the two stairs-down rewards enabled."""

    rewards: Exp3RewardsCfg = Exp3RewardsCfg()


@configclass
class Exp3RobotPlayEnvCfg(Exp3RobotEnvCfg):
    """Deterministic play configuration for Exp3."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)
