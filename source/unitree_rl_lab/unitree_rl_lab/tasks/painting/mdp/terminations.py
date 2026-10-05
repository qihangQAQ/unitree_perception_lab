"""Stage-aware painting ending conditions."""

import torch


def time_out(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command.timeout


def bad_posture(
    env,
    sensor_cfg,
    threshold=1.0,
    debounce_time=0.08,
    command_name="painting_command",
):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    maximum_force = torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))
    return command.update_bad_contact(maximum_force, threshold, debounce_time)


def success(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command.success
