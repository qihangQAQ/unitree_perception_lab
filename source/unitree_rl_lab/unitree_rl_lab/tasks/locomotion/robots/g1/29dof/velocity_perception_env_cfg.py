import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.fdm.mdp.terrains import (
    MeshPillarTerrainCfg,
    SingleObjectTerrainCfg,
    StairsRampEvalTerrainCfg,
)
from unitree_rl_lab.tasks.fdm.mdp.terrains.single_object import cross_object_pattern
from unitree_rl_lab.terrains import UPGRADE_TERRAIN1

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)

ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 1. 倒金字塔台阶——4种步宽 (28/30/32/34cm)，模仿下山/下坡地形
        "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 2. 随机网格盒子——地面上随机散布高低不一的方块
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.0, 0.15),
            platform_width=2.0,
        ),
        # 3. 不规则凹凸地面——随机噪声生成坑洼地面
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(-0.02, 0.04),
            noise_step=0.02,
            border_width=0.25,
        ),
        # 4. 波浪地形——模拟起伏路面
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.0, 0.2),
            num_waves=5.0,
        ),
        # 5. 高台/坑洞——下陷平台 (double_pit 双向坑)
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15,
            pit_depth_range=(0.0, 0.3),
            platform_width=2.0,
            double_pit=True,
        ),
    },
)
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
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
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # 定义扫描范围 宽度1m * 长度1.6m,分辨率0.1m(两条射线之间的间隔是 0.1 米)
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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

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

    clear_privileged_property_cache = EventTerm(
        func=mdp.clear_privileged_property_cache,
        mode="startup",
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
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

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CommandsCfg:
    """Terrain-aware velocity commands with independent per-axis zeroing."""

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
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # 机身角速度（3）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, clip=(-18.0, 18.0))
        # 重力向量（3）、
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 指令根节点线速度（3）
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置（29）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-18.0, 18.0))
        # 关节速度（29）
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-360.0, 360.0))
        # 上一帧动作（29）
        last_action = ObsTerm(func=mdp.last_action, clip=(-18.0, 18.0))
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        # 高度扫描（187）
        height_scanner = ObsTerm(
            func=mdp.height_scan_hpc,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,  # 与 LeggedLab 对齐
            },
            scale=1.0,
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 与 LeggedLab noise_scale=0.1 对齐
        )

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Clean privileged observations for the value network."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-18.0, 18.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, clip=(-18.0, 18.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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

        # 高度扫描（187）
        height_scanner = ObsTerm(
            func=mdp.height_scan_hpc,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,  # 与 LeggedLab 对齐
            },
            scale=1.0,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP (LeggedLab G1Rough-aligned)."""

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

    # Ablation-only rewards; intentionally disabled in the perception baseline.
    # delta_yaw = RewTerm(func=mdp.delta_yaw_reward, weight=-0.5)
    # volume_points_penetration = RewTerm(func=mdp.volume_points_penetration, weight=-1.0)


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
    """Termination-aware terrain curriculum for independently zeroed commands."""

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
        # The 4096-env upgrade terrain can exceed PhysX's default 64 MiB collision stack.
        self.sim.physx.gpu_collision_stack_size = 2**28

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 0.1
        self.scene.left_foot_height_scanner.update_period = self.sim.dt
        self.scene.right_foot_height_scanner.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.terminations.time_out = None

        self.scene.num_envs = 16
        # Generate the complete training terrain grid, but do not move between levels during play.
        self.curriculum.terrain_levels = None

        # PLAY 使用固定前进速度 + 固定朝向，方便观察避障行为
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.terrain_specific_ranges = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)
