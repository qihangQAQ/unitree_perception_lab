"""G1_CFG wall painting, with episode-fixed paths and compact policy inputs."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide

from unitree_rl_lab.assets.robots.unitree import G1_CFG
from unitree_rl_lab.painting.trajectories import PaintingPathCfg
from unitree_rl_lab.tasks.locomotion.mdp.rewards import energy

from .mdp import observations, rewards, terminations
from .mdp.actions import SafeJointPositionActionCfg
from .mdp.painting_command import PaintingCommandCfg


@configclass
class PaintingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        debug_vis=False,
    )
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CommandsCfg:
    painting_command = PaintingCommandCfg(
        # Stage A: standing plus balanced left/right lateral stepping.
        task_mode="side_step",
        duration_range=(8.0, 12.0),
        speed_range=(0.2, 0.4),
        distance_range=(0.05, 0.15),
        base_speed_range=(0.10, 0.25),
        standing_probability=0.20,
        side_step_ramp_time=0.75,
        right_probability=0.5,
        prepare_time=1.0,
        catchup_time=1.5,
        lookahead_times=(0.0, 0.1, 0.2, 0.4, 0.8),
        # Path generation.
        path=PaintingPathCfg(
            spacing=0.01,
            height_range=(1.1, 1.4),
            probabilities=(0.30, 0.20, 0.25, 0.25),
            connector_length=(0.4, 0.7),
            straight_length=(0.35, 0.8),
            wave_length=(0.6, 1.2),
            shape_height=(0.18, 0.28),
            samples_per_piece=193,
        ),
        # Training keeps markers disabled; PaintingPlayEnvCfg enables them.
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    JointPositionAction = SafeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True,
        scale={".*_hip_.*": 0.25, ".*_knee_joint": 0.25, ".*_ankle_.*": 0.25,
               "waist_.*": 0.2, ".*_shoulder_.*": 0.5, ".*_elbow_joint": 0.5, ".*_wrist_.*": 0.4},
    )


@configclass
class ProprioCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.1, n_max=0.1))
    projected_gravity = ObsTerm(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
    joint_pos_rel = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel_rel = ObsTerm(func=base_mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5))
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.concatenate_terms = True
        self.history_length = 5
        self.flatten_history_dim = False
        self.enable_corruption = True


@configclass
class CommandObservationCfg(ObsGroup):
    targets = ObsTerm(func=observations.painting_targets)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class CriticCfg(ProprioCfg):
    targets = ObsTerm(func=observations.painting_targets)
    velocities = ObsTerm(func=observations.velocity_targets)
    task_time = ObsTerm(func=observations.task_time)

    def __post_init__(self):
        self.concatenate_terms = True
        self.history_length = 0
        self.enable_corruption = False


@configclass
class VelocityTargetsCfg(ObsGroup):
    velocities = ObsTerm(func=observations.velocity_targets)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy: ProprioCfg = ProprioCfg()
    painting: CommandObservationCfg = CommandObservationCfg()
    critic: CriticCfg = CriticCfg()
    velocity_targets: VelocityTargetsCfg = VelocityTargetsCfg()


@configclass
class EventsCfg:
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.015, 0.015), "velocity_range": (0.0, 0.0)},
    )
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.0), "dynamic_friction_range": (0.5, 0.6),
                "restitution_range": (0.0, 0.0), "num_buckets": 32},
    )
    push_robot = None


@configclass
class RewardsCfg:
    # Stage-A task signal. TCP fields remain in the observation contract but
    # are explicitly inactive until the horizontal coordination stage.
    base_velocity = RewTerm(func=rewards.base_velocity_tracking, weight=2.5, params={"std": 0.15})
    base_yaw_rate = RewTerm(func=rewards.base_yaw_rate_tracking, weight=0.5, params={"std": 0.25})
    base_anchor = RewTerm(func=rewards.base_anchor_tracking, weight=0.25, params={"std": 0.25})
    facing_wall = RewTerm(func=rewards.facing_wall, weight=0.25)
    alive = RewTerm(func=base_mdp.is_alive, weight=0.15)

    # Physical posture terms are kept separate so a large error in one term
    # cannot hide all other posture information inside a single exponential.
    base_height = RewTerm(func=rewards.base_height_outside_band, weight=-1.0)
    pelvis_upright = RewTerm(func=rewards.pelvis_upright_outside_tolerance, weight=-0.5)
    torso_upright = RewTerm(func=rewards.torso_upright_outside_tolerance, weight=-0.5)
    vertical_velocity = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-1.0)
    roll_pitch_rate = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.1)
    leg_lateral_posture = RewTerm(
        func=rewards.joint_deviation_outside_tolerance,
        weight=-0.2,
        params={
            "tolerance": 0.15,
            "normalization": 0.35,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]),
        },
    )
    waist_posture = RewTerm(
        func=rewards.joint_deviation_outside_tolerance,
        weight=-0.1,
        params={
            "tolerance": 0.15,
            "normalization": 0.35,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*"]),
        },
    )
    left_arm_posture = RewTerm(
        func=rewards.joint_deviation_outside_tolerance,
        weight=-0.03,
        params={
            "tolerance": 0.25,
            "normalization": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_.*", "left_elbow_joint", "left_wrist_.*"]
            ),
        },
    )
    right_arm_posture = RewTerm(
        func=rewards.joint_deviation_outside_tolerance,
        weight=-0.03,
        params={
            "tolerance": 0.25,
            "normalization": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"]
            ),
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.02)
    joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=energy, weight=-1e-4)
    torque_saturation = RewTerm(func=rewards.torque_saturation, weight=-0.1)
    feet_slide = RewTerm(
        func=feet_slide, weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")},
    )
    undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*")},
    )
    success = RewTerm(func=rewards.terminal_event, weight=5.0, params={"event": "success"})
    bad_posture = RewTerm(func=rewards.terminal_event, weight=-10.0, params={"event": "bad"})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    bad_posture = DoneTerm(
        func=terminations.bad_posture,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
            "threshold": 1.0,
            "debounce_time": 0.08,
        },
    )
    success = DoneTerm(func=terminations.success, time_out=False)


@configclass
class PaintingEnvCfg(ManagerBasedRLEnvCfg):
    scene: PaintingSceneCfg = PaintingSceneCfg(num_envs=1024, env_spacing=28.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: dict = {}

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = self.commands.painting_command.duration_range[1]
        self.is_finite_horizon = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        # G1's forearm points along local +x at elbow zero; +pi/2 points it down.
        joints = dict(self.scene.robot.init_state.joint_pos)
        joints.pop(".*_elbow_joint", None)
        joints.update({
            "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
            "left_elbow_joint": 1.5708,
            # Static IK at TCP (0.28, -0.18, 1.25), pelvis z=0.8, spray +x.
            "right_shoulder_pitch_joint": -1.24824, "right_shoulder_roll_joint": -1.18576,
            "right_shoulder_yaw_joint": 0.14842, "right_elbow_joint": -0.24625,
            "right_wrist_roll_joint": -0.96081, "right_wrist_pitch_joint": -0.59741,
            "right_wrist_yaw_joint": 1.23998,
        })
        self.scene.robot.init_state.joint_pos = joints
        self.viewer.eye = (-3.0, -3.0, 2.3)
        self.viewer.lookat = (0.0, 0.0, 1.0)


@configclass
class PaintingPlayEnvCfg(PaintingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.commands.painting_command.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.push_robot = None
