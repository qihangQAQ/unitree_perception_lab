from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel_with_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_command_threshold: float = 0.2,
    curriculum_half_extent_m: float | None = None,
    no_forward_move_down_term: str | None = None,
) -> torch.Tensor:
    """Update terrain levels while handling non-forward commands explicitly.

    Forward commands use the usual distance-based curriculum. For standing,
    turning, lateral, and backward commands, a normal timeout keeps the current
    level while a genuine early termination moves the environment down. This
    prevents zeroed command components from being evaluated as failed forward
    traversal attempts.
    """

    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    terrain = env.scene.terrain
    if env_ids_tensor.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        raise RuntimeError("The termination-aware terrain curriculum requires generated terrain.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")[env_ids_tensor]
    distance = torch.norm(
        asset.data.root_pos_w[env_ids_tensor, :2] - env.scene.env_origins[env_ids_tensor, :2],
        dim=1,
    )
    move_up_distance = (
        float(curriculum_half_extent_m)
        if curriculum_half_extent_m is not None
        else float(terrain_generator.size[0]) / 2.0
    )

    command_x = command[:, 0]
    has_forward_command = command_x > float(forward_command_threshold)
    move_up = distance > move_up_distance
    move_down = torch.zeros_like(move_up)

    if has_forward_command.any():
        forward_indices = torch.where(has_forward_command)[0]
        move_down[forward_indices] = (
            distance[forward_indices]
            < command_x[forward_indices] * env.max_episode_length_s * 0.5
        ) & ~move_up[forward_indices]

    no_forward_command = ~has_forward_command
    if no_forward_command.any():
        no_forward_indices = torch.where(no_forward_command)[0]
        no_forward_env_ids = env_ids_tensor[no_forward_indices]
        if no_forward_move_down_term is None:
            failed = (
                env.termination_manager.terminated[no_forward_env_ids]
                & ~env.termination_manager.time_outs[no_forward_env_ids]
            )
        else:
            failed = env.termination_manager.get_term(no_forward_move_down_term)[no_forward_env_ids]
        move_down[no_forward_indices] = failed & ~move_up[no_forward_indices]

    terrain.update_env_origins(env_ids_tensor, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
