"""Standalone depth-perception velocity task for the 29-DoF G1."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.sensors import (
    Grid3dPointsGeneratorCfg,
    MultiMeshRayCasterCameraCfg,
    VolumePointsCfg,
)
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.terrains import (
    UPGRADE_TERRAIN1,
    UPGRADE_TERRAIN2,
    EdgeCylinderCfg,
    TerrainImporterCfg,
)

CROSS_STEPPING_STONES_TERRAIN = "cross_stepping_stones"
STEPPING_STONES_TERRAIN = "stepping_stones"
GAP_TERRAIN = "gap"
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

# G1 visual links used by InstinctLab's perceptive tasks. Keeping the robot
# meshes in the ray-cast targets reproduces self-occlusion in the depth image.
# The camera is attached to torso_link and its origin lies inside that visual
# mesh, so torso_link itself must be excluded or every ray immediately self-hits.
G1_29DOF_LINKS = (
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
)


def _g1_visual_raycast_targets() -> list[MultiMeshRayCasterCameraCfg.RaycastTargetCfg]:
    return [
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr=f"{{ENV_REGEX_NS}}/Robot/{link_name}/visuals",
            is_shared=True,
        )
        for link_name in G1_29DOF_LINKS
    ]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

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
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
    raycaster_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=["/World/ground", *_g1_visual_raycast_targets()],
        # InstinctLab G1 head-camera nominal pose.
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0487988662332928, 0.01, 0.4378029937970051),
            rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        # Keep the TRAIN_E1ObstacleRace image contract for the later depth encoder.
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                241.418 * 0.1,
                0.0,
                (242.5 - 120.0) * 0.1,
                0.0,
                241.418 * 0.1,
                (130.0 - 110.0) * 0.1,
                0.0,
                0.0,
                1.0,
            ],
            width=24,
            height=16,
            focal_length=24.0,
        ),
        update_period=1.0 / 60.0,
        depth_clipping_behavior="zero",
        max_distance=4.0,
        debug_vis=False,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/RayCasterCamera",
            markers={
                **{
                    f"attn_{index}": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(index / 9.0, 0.0, 1.0 - index / 9.0),
                        ),
                    )
                    for index in range(10)
                }
            },
        ),
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
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain randomization, reset, disturbance, and virtual-obstacle events."""

    # startup
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
    randomize_camera_mount_orientation = EventTerm(
        func=mdp.randomize_camera_mount_orientation,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("raycaster_camera"),
            "orientation_range": {
                "roll": (-math.radians(3.0), math.radians(3.0)),
                "pitch": (-math.radians(3.0), math.radians(3.0)),
                "yaw": (-math.radians(3.0), math.radians(3.0)),
            },
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
    """Terrain-aware velocity commands with independently zeroed components."""

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
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Policy history, depth, clean critic, and Old-HIM target groups."""

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
    class CriticCfg(ObsGroup):
        """Clean privileged observations aligned with perception-pro."""

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
        """Clean 96-D physical state used as the Old-HIM target."""

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
    class DepthCfg(ObsGroup):
        """Noisy channels-last depth images consumed by cross-attention."""

        depths = ObsTerm(
            func=mdp.raycaster_depth,
            params={
                "sensor_cfg": SceneEntityCfg("raycaster_camera"),
                "distance_noise": 0.01,
                "gaussian_std": 0.005,
                "salt_pepper_prob": 0.01,
                "stripe_prob": 0.005,
                "edge_drag_prob": 0.01,
                "min_range": 0.3,
                "max_range": 2.0,
                "high_reflection_prob": 0.01,
                "high_reflection_num_patches": (1, 2),
                "high_reflection_rect_ratio": (0.05, 0.1),
                "high_reflection_coefficient_range": (1.1, 1.3),
                "high_reflection_persist_steps": 10,
                "mask_prob": 0.03,
                "mask_num_masks": (1, 3),
                "mask_rect_ratio": (0.1, 0.2),
                "mask_persist_steps": 20,
                "full_image_mask_prob": 0.005,
                "full_image_mask_persist_steps": 10,
                "full_image_random_prob": 0.001,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
            self.history_length = 0

    depth: DepthCfg = DepthCfg()


@configclass
class RewardsCfg:
    """Rewards aligned with the mature perception-pro task."""

    # ==========================================
    # 1. Task & Survival
    # ==========================================

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

    # ==========================================
    # 2. Base & Posture
    # ==========================================

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

    # ==========================================
    # 3. Joints & Regularization
    # ==========================================

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

    # ==========================================
    # 4. Feet & Gait
    # ==========================================

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
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            "terrain_types": ("stairs_up", "stairs_down", "discrete_obstacles"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Termination-aware curriculum for independently zeroed commands."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_with_termination)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**28

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 0.1
        self.scene.left_foot_height_scanner.update_period = self.sim.dt
        self.scene.right_foot_height_scanner.update_period = self.sim.dt
        self.scene.raycaster_camera.update_period = 3 * self.sim.dt

        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.curriculum = self.curriculum.terrain_levels is not None


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.terminations.time_out = None

        self.scene.num_envs = 16
        self.scene.raycaster_camera.debug_vis = True
        self.scene.height_scanner.debug_vis = False
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.terrain_specific_ranges = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)

        self.observations.policy.enable_corruption = False
        depth_params = self.observations.depth.depths.params
        for parameter_name in (
            "distance_noise",
            "gaussian_std",
            "salt_pepper_prob",
            "stripe_prob",
            "edge_drag_prob",
            "high_reflection_prob",
            "mask_prob",
            "full_image_mask_prob",
            "full_image_random_prob",
        ):
            depth_params[parameter_name] = 0.0
