"""TCP tracking, painting posture and single-transition event rewards."""

import math

import torch

from isaaclab.managers import SceneEntityCfg


def _command(env, command_name):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command


def position_tracking(env, command_name="painting_command", broad=0.08, narrow=0.03):
    error = _command(env, command_name).position_error
    return 0.5 * (torch.exp(-(error / broad).square()) + torch.exp(-(error / narrow).square()))


def axis_tracking(env, command_name="painting_command", std=math.radians(15)):
    return torch.exp(-(_command(env, command_name).axis_error / std).square())


def velocity_tracking(env, command_name="painting_command", std=0.1):
    command = _command(env, command_name)
    return position_tracking(env, command_name) * axis_tracking(env, command_name) * torch.exp(
        -(command.velocity_error / std).square()
    )


def base_velocity_tracking(env, command_name="painting_command", std=0.15):
    error = _command(env, command_name).base_velocity_error
    return torch.exp(-(error / std).square())


def base_yaw_rate_tracking(env, command_name="painting_command", std=0.25):
    error = _command(env, command_name).base_yaw_rate_error
    return torch.exp(-(error / std).square())


def base_anchor_tracking(env, command_name="painting_command", std=0.25):
    error = torch.linalg.vector_norm(_command(env, command_name).base_anchor_relative, dim=-1)
    return torch.exp(-(error / std).square())


def facing_wall(env, command_name="painting_command", std=math.radians(15)):
    error = _command(env, command_name).facing_error
    return torch.exp(-(error / std).square()).mean(-1)


def left_arm_down(env, command_name="painting_command", std=0.08):
    command = _command(env, command_name)
    # Both segments point down; the hand also stays to the left of the pelvis.
    left_side = torch.sigmoid((command.left_hand_base[:, 1] - 0.07) * 40)
    return torch.exp(-command.left_arm_error / std) * left_side


def base_height_outside_band(
    env, command_name="painting_command", target=0.80, tolerance=0.035, normalization=0.05
):
    error = (_command(env, command_name).base_height - target).abs()
    return (torch.relu(error - tolerance) / normalization).square()


def pelvis_upright_outside_tolerance(
    env, command_name="painting_command", tolerance=math.radians(10), normalization=math.radians(15)
):
    error = _command(env, command_name).tilt
    return (torch.relu(error - tolerance) / normalization).square()


def torso_upright_outside_tolerance(
    env, command_name="painting_command", tolerance=math.radians(12), normalization=math.radians(15)
):
    error = _command(env, command_name).torso_tilt
    return (torch.relu(error - tolerance) / normalization).square()


def joint_deviation_outside_tolerance(
    env,
    tolerance,
    normalization,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ).abs()
    return ((torch.relu(error - tolerance) / normalization).square()).sum(-1)


def torque_saturation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1e-6)
    ratio = asset.data.applied_torque[:, asset_cfg.joint_ids].abs() / limits
    return torch.relu(ratio - 0.95).square().sum(-1)


def terminal_event(env, event, command_name="painting_command"):
    if event not in ("success", "bad"):
        raise ValueError(f"Unknown painting reward event: {event}")
    # RewardManager multiplies by dt: weight is the actual one-time reward.
    return getattr(_command(env, command_name), event).float() / env.step_dt
