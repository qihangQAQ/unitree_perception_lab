from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from .commands.velocity_command import terrain_type_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from unitree_rl_lab.sensors import VolumePoints


def delta_yaw_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize large yaw-rate tracking errors using the Noetix thresholding rule."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    speed_yaw = asset.data.root_ang_vel_w[:, 2]
    command_speed_yaw = command[:, 2]
    delta_speed = torch.abs(speed_yaw - command_speed_yaw)
    absolute_command = torch.abs(command_speed_yaw)
    punish = delta_speed * (delta_speed > 0.1) * (delta_speed > 0.5 * absolute_command)
    return punish


def volume_points_penetration(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
    terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Penalize moving body-volume points that penetrate inflated terrain edges."""
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    penetration = volume_sensor.data.penetration_offset.flatten(1, 2)
    penetration_depth = torch.norm(penetration, dim=-1)
    in_obstacle = (penetration_depth > tolerance).float()
    point_speed = torch.norm(volume_sensor.data.points_vel_w.flatten(1, 2), dim=-1)
    penalty = torch.sum(in_obstacle * (point_speed + 1.0e-6) * penetration_depth, dim=-1)

    if terrain_types:
        penalty *= terrain_type_mask(env, terrain_types)
    return penalty


def stairs_outward_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    min_forward_command: float = 0.1,
    cap_by_command: bool = True,
    max_outward_speed: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward planar velocity directed away from a stair tile's center platform."""
    if min_forward_command < 0.0:
        raise ValueError(
            "min_forward_command must be non-negative, got "
            f"{min_forward_command}."
        )
    if max_outward_speed <= 0.0:
        raise ValueError(
            f"max_outward_speed must be positive, got {max_outward_speed}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    active = terrain_type_mask(env, terrain_types) & (
        command[:, 0] > float(min_forward_command)
    )

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    radial_distance = torch.linalg.vector_norm(local_xy, dim=1, keepdim=True)
    outward_direction = local_xy / radial_distance.clamp_min(1.0e-6)
    outward_speed = torch.sum(
        asset.data.root_lin_vel_w[:, :2] * outward_direction, dim=1
    ).clamp_min(0.0)

    speed_cap = torch.full_like(outward_speed, float(max_outward_speed))
    if cap_by_command:
        speed_cap = torch.minimum(speed_cap, command[:, 0].clamp_min(0.0))
    reward = torch.minimum(outward_speed, speed_cap)
    return reward * active.float()


def stairs_stall_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    min_forward_command: float = 0.1,
    min_outward_speed: float = 0.05,
    grace_steps: int = 50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize complete stalls on stairs without command-speed tracking."""
    if min_forward_command < 0.0:
        raise ValueError(
            "min_forward_command must be non-negative, got "
            f"{min_forward_command}."
        )
    if min_outward_speed < 0.0:
        raise ValueError(
            f"min_outward_speed must be non-negative, got {min_outward_speed}."
        )
    if grace_steps < 0:
        raise ValueError(f"grace_steps must be non-negative, got {grace_steps}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    outward_direction = local_xy / torch.linalg.vector_norm(
        local_xy, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    outward_speed = torch.sum(
        asset.data.root_lin_vel_w[:, :2] * outward_direction, dim=1
    )
    active = (
        terrain_type_mask(env, terrain_types)
        & (command[:, 0] > float(min_forward_command))
        & (env.episode_length_buf >= int(grace_steps))
    )
    return (active & (outward_speed < float(min_outward_speed))).float()


"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_unhold_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    offset: float = 0.06,
    terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Penalize a contacting foot when its scanned sole is not fully supported."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[
        :, :, contact_sensor_cfg.body_ids, :
    ]
    feet_contact = torch.max(
        torch.linalg.vector_norm(contact_forces, dim=-1), dim=1
    ).values > 0.5

    left_scanner: RayCaster = env.scene.sensors[
        left_foot_height_scanner_cfg.name
    ]
    left_unsupported = torch.abs(
        left_scanner.data.pos_w[:, 2].unsqueeze(1)
        - left_scanner.data.ray_hits_w[..., 2]
        - offset
    ) > 0.03
    left_penalty = left_unsupported.float().mean(dim=-1) * feet_contact[:, 0]

    right_scanner: RayCaster = env.scene.sensors[
        right_foot_height_scanner_cfg.name
    ]
    right_unsupported = torch.abs(
        right_scanner.data.pos_w[:, 2].unsqueeze(1)
        - right_scanner.data.ray_hits_w[..., 2]
        - offset
    ) > 0.03
    right_penalty = right_unsupported.float().mean(dim=-1) * feet_contact[:, 1]

    penalty = left_penalty + right_penalty
    if terrain_types:
        penalty *= terrain_type_mask(env, terrain_types)
    return penalty


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


# ==============================================================================
# LeggedLab 对齐：步态、姿态、安全约束
# ==============================================================================


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long steps for bipeds, with rotation-aware command masking.

    Compared to ``feet_air_time_biped``, this version:
    - Rewards single-stance time up to *threshold* instead of clamping air time.
    - Masks reward using ``norm(lin_vel) + abs(ang_vel)``, so pure rotation also
      produces a gait reward (the original only checked linear velocity).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    cmd = env.command_manager.get_command(command_name)
    reward *= (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > 0.1
    return reward


def fly(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize both feet being off the ground simultaneously."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
    )[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def body_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize torso tilt away from upright (gravity-aligned)."""
    from isaaclab.utils.math import quat_apply_inverse

    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :],
        asset.data.GRAVITY_VEC_W,
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def body_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500,
    max_reward: float = 400,
) -> torch.Tensor:
    """Penalize large vertical contact forces on specified bodies (e.g. feet)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
    max_excess: float = 400.0,
) -> torch.Tensor:
    """Penalize excessive foot contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1
    )
    excess = torch.clamp(force_mag - threshold, min=0.0, max=max_excess)
    return torch.sum(excess, dim=1) / max_excess


def feet_too_near_humanoid(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize feet that are too close to each other (biped-specific, expects exactly 2 feet)."""
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)
