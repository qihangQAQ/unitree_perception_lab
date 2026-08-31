"""Standalone Pro terrain-perception velocity task for the 29-DoF G1."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG
from unitree_rl_lab.sensors import Grid3dPointsGeneratorCfg, VolumePointsCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.terrains import UPGRADE_TERRAIN1,UPGRADE_TERRAIN2, EdgeCylinderCfg, TerrainImporterCfg

CROSS_STEPPING_STONES_TERRAIN = "cross_stepping_stones"
STEPPING_STONES_TERRAIN = "stepping_stones"
GAP_TERRAIN = "gap"
DOWNHILL_TERRAIN_TYPES = ("stairs_down",)
PRECISE_FOOTHOLD_TERRAINS = (
    CROSS_STEPPING_STONES_TERRAIN,
    STEPPING_STONES_TERRAIN,
    GAP_TERRAIN,
)
STEPPING_STONE_TERRAINS = (
    CROSS_STEPPING_STONES_TERRAIN,
    STEPPING_STONES_TERRAIN,
)
CARDINAL_YAWS = (0.0, 1.57, 3.14, -1.57)
ZERO_ROOT_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Terrain, robot, and sensors owned by the perception-pro task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UPGRADE_TERRAIN2,
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
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    left_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0.0, -0.05)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.20, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0.0, -0.05)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.20, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class EventCfg:
    """Domain randomization, reset, disturbance, and virtual-obstacle events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    randomize_actuator_effort_limit = EventTerm(
        func=mdp.randomize_actuator_effort_limit,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "effort_limit_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    clear_privileged_property_cache = EventTerm(
        func=mdp.clear_privileged_property_cache,
        mode="startup",
    )
    register_virtual_obstacles = EventTerm(
        func=mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={"sensor_cfgs": SceneEntityCfg("leg_volume_points")},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_terrain_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "terrain_specific_pos_range": {
                CROSS_STEPPING_STONES_TERRAIN: {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                },
                STEPPING_STONES_TERRAIN: {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
                GAP_TERRAIN: {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
            "terrain_specific_yaw": {
                terrain_type: CARDINAL_YAWS
                for terrain_type in PRECISE_FOOTHOLD_TERRAINS
            },
            "terrain_specific_velocity_range": {
                terrain_type: ZERO_ROOT_VELOCITY_RANGE
                for terrain_type in PRECISE_FOOTHOLD_TERRAINS
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_terrain_specific,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "terrain_specific_velocity_range": {
                terrain_type: ZERO_ROOT_VELOCITY_RANGE
                for terrain_type in PRECISE_FOOTHOLD_TERRAINS
            },
        },
    )


@configclass
class CommandsCfg:
    """Terrain-aware velocity-command sampling."""

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
            ),
            CROSS_STEPPING_STONES_TERRAIN: mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.8, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=(0.0, 1.0, 1.0),
            ),
            STEPPING_STONES_TERRAIN: mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.8, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=(0.0, 1.0, 1.0),
            ),
            GAP_TERRAIN: mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.8, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=(0.0, 1.0, 1.0),
            ),
        },
    )


@configclass
class ActionsCfg:
    """Joint-position actions for all 29 actuated joints."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Policy, height-map, critic, and auxiliary observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Five noisy proprioceptive frames, ordered from oldest to newest."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1.0,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-18.0, 18.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.025, n_max=0.025),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-18.0, 18.0),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-360.0, 360.0),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-18.0, 18.0))

        def __post_init__(self):
            self.history_length = 5
            self.flatten_history_dim = False
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class HeightMapCfg(ObsGroup):
        """Current noisy 17x11 terrain-height scan."""

        height_scanner = ObsTerm(
            func=mdp.height_scan_hpc,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            scale=1.0,
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    height_map: HeightMapCfg = HeightMapCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Clean privileged observations for the value network."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-18.0, 18.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, clip=(-18.0, 18.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-18.0, 18.0))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-360.0, 360.0))
        last_action = ObsTerm(func=mdp.last_action, clip=(-18.0, 18.0))
        payload = ObsTerm(
            func=mdp.payload,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            scale=0.2,
        )
        left_foot_material = ObsTerm(
            func=mdp.material_properties,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="left_ankle_roll_link"
                ),
            },
        )
        right_foot_material = ObsTerm(
            func=mdp.material_properties,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="right_ankle_roll_link"
                ),
            },
        )
        kp_params = ObsTerm(func=mdp.kp_params, scale=0.005)
        kd_params = ObsTerm(func=mdp.kd_params, scale=0.05)
        effort_limit = ObsTerm(func=mdp.effort_limit, scale=0.01)
        feet_contact = ObsTerm(
            func=mdp.feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "threshold": 0.5,
            },
        )
        feet_stumble = ObsTerm(
            func=mdp.current_feet_stumble,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*ankle_roll.*"
                ),
            },
        )
        left_foot_height_scan = ObsTerm(
            func=mdp.height_scan_hpc,
            params={
                "sensor_cfg": SceneEntityCfg("left_foot_height_scanner"),
                "offset": 0.06,
            },
            clip=(-1.0, 1.0),
        )
        right_foot_height_scan = ObsTerm(
            func=mdp.height_scan_hpc,
            params={
                "sensor_cfg": SceneEntityCfg("right_foot_height_scanner"),
                "offset": 0.06,
            },
            clip=(-1.0, 1.0),
        )
        height_scanner = ObsTerm(
            func=mdp.height_scan_hpc,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            scale=1.0,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class AuxiliaryCfg(ObsGroup):
        """Clean 96-D physical state for the Old-HIM target encoder."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, clip=(-18.0, 18.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-18.0, 18.0))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-360.0, 360.0))
        last_action = ObsTerm(func=mdp.last_action, clip=(-18.0, 18.0))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-18.0, 18.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    auxiliary: AuxiliaryCfg = AuxiliaryCfg()


@configclass
class RewardsCfg:
    """Velocity tracking, regularization, gait, and obstacle rewards."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw.*",
                    ".*_hip_roll.*",
                    ".*_shoulder_pitch.*",
                    ".*_elbow.*",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"],
            )
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_unhold = RewTerm(
        func=mdp.feet_unhold_reward,
        weight=-1.0,
        params={
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll.*"
            ),
            "left_foot_height_scanner_cfg": SceneEntityCfg(
                "left_foot_height_scanner"
            ),
            "right_foot_height_scanner_cfg": SceneEntityCfg(
                "right_foot_height_scanner"
            ),
            "terrain_types": STEPPING_STONE_TERRAINS,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # delta_yaw = RewTerm(func=mdp.delta_yaw_reward, weight=-0.5)
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            "terrain_types": ("stairs_up", "stairs_down", "discrete_obstacles"),
        },
    )
    # stairs_down_safe_progress = RewTerm(
    #     func=mdp.stairs_outward_progress_reward,
    #     weight=0.5,
    #     params={
    #         "terrain_types": DOWNHILL_TERRAIN_TYPES,
    #         "min_forward_command": 0.1,
    #         "cap_by_command": False,
    #         "max_outward_speed": 0.5,
    #     },
    # )
    # stairs_down_stall = RewTerm(
    #     func=mdp.stairs_stall_penalty,
    #     weight=-1.0,
    #     params={
    #         "terrain_types": DOWNHILL_TERRAIN_TYPES,
    #         "min_forward_command": 0.1,
    #         "min_outward_speed": 0.05,
    #         "grace_steps": 50,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Episode termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"),
            "threshold": 1.0,
        },
    )
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Termination-aware terrain curriculum for independently zeroed commands."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_with_termination)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone training configuration for perception-pro."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**28

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 0.1
        self.scene.left_foot_height_scanner.update_period = self.sim.dt
        self.scene.right_foot_height_scanner.update_period = self.sim.dt

        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.curriculum = self.curriculum.terrain_levels is not None


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Deterministic play configuration without observation corruption."""

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
