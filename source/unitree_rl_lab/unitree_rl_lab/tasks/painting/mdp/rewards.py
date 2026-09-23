"""TCP tracking, painting posture and single-transition event rewards."""

import math

import torch


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


def facing_wall(env, command_name="painting_command", std=math.radians(15)):
    error = _command(env, command_name).facing_error
    return torch.exp(-(error / std).square()).mean(-1)


def left_arm_down(env, command_name="painting_command", std=0.08):
    command = _command(env, command_name)
    # Both segments point down; the hand also stays to the left of the pelvis.
    left_side = torch.sigmoid((command.left_hand_base[:, 1] - 0.07) * 40)
    return torch.exp(-command.left_arm_error / std) * left_side


def body_stability(env, command_name="painting_command"):
    command = _command(env, command_name)
    data = command.robot.data
    error = (command.tilt / 0.3).square() + (data.root_link_lin_vel_w[:, 2] / 0.3).square()
    error += (data.root_link_ang_vel_b[:, :2] / 0.8).square().sum(-1)
    return torch.exp(-error)


def terminal_event(env, event, command_name="painting_command"):
    if event not in ("success", "bad"):
        raise ValueError(f"Unknown painting reward event: {event}")
    # RewardManager multiplies by dt: weight is the actual one-time reward.
    return getattr(_command(env, command_name), event).float() / env.step_dt
