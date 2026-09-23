"""Nineteen task inputs and training-only velocity labels."""

import torch


def painting_targets(env, command_name="painting_command"):
    return env.command_manager.get_term(command_name).command


def velocity_targets(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command.velocity_targets


def task_time(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return torch.stack(
        ((command.deadline - command.elapsed).clamp_min(0) / command.cfg.duration_range[1],
         command.progress / command.lengths), -1
    )
