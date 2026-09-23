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

from unitree_rl_lab.assets.robots.unitree import G1_CFG
from unitree_rl_lab.tasks.locomotion.mdp.rewards import energy, feet_slide

from .mdp import observations, rewards, terminations
from .mdp.painting_command import PaintingCommandCfg


@configclass
class PaintingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        debug_vis=False,
    )
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 26.0, 2.0), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.68, 0.72)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.05, 0.0, 1.0)),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CommandsCfg:
    painting_command = PaintingCommandCfg()


@configclass
class ActionsCfg:
    JointPositionAction = base_mdp.JointPositionActionCfg(
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
    tcp_position = RewTerm(func=rewards.position_tracking, weight=5.0)
    spray_axis = RewTerm(func=rewards.axis_tracking, weight=2.0)
    tcp_velocity = RewTerm(func=rewards.velocity_tracking, weight=1.0)
    facing_wall = RewTerm(func=rewards.facing_wall, weight=1.0)
    left_arm_down = RewTerm(func=rewards.left_arm_down, weight=1.0)
    body_stability = RewTerm(func=rewards.body_stability, weight=0.5)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    energy = RewTerm(func=energy, weight=-1e-4)
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
    bad_orientation = RewTerm(func=rewards.terminal_event, weight=-5.0, params={"event": "bad"})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    bad_orientation = DoneTerm(func=terminations.bad_orientation)
    success = DoneTerm(func=terminations.success, time_out=True)


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
