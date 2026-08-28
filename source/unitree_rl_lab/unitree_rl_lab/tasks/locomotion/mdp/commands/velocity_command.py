from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def terrain_type_mask(
    env: ManagerBasedEnv,
    terrain_type_names: tuple[str, ...],
    env_ids: torch.Tensor | None = None,
    use_assigned_terrain: bool = False,
) -> torch.Tensor:
    """Return a mask for robots located on or assigned to requested terrain types."""
    terrain = env.scene.terrain
    generator_cfg = terrain.cfg.terrain_generator
    if generator_cfg is None or not generator_cfg.curriculum:
        raise RuntimeError(
            "Terrain-specific commands require a curriculum terrain generator."
        )

    cache_key = (
        id(generator_cfg),
        generator_cfg.num_cols,
        tuple(generator_cfg.sub_terrains),
        tuple(float(cfg.proportion) for cfg in generator_cfg.sub_terrains.values()),
    )
    if getattr(terrain, "_unitree_column_type_cache_key", None) != cache_key:
        proportions = [
            float(cfg.proportion) for cfg in generator_cfg.sub_terrains.values()
        ]
        total_proportion = sum(proportions)
        if total_proportion <= 0.0:
            raise ValueError("Terrain proportions must sum to a positive value.")

        cumulative_proportions = []
        cumulative = 0.0
        for proportion in proportions:
            cumulative += proportion / total_proportion
            cumulative_proportions.append(cumulative)

        column_type_indices = []
        for column_idx in range(generator_cfg.num_cols):
            column_position = column_idx / generator_cfg.num_cols + 0.001
            terrain_type_idx = next(
                (
                    idx
                    for idx, cumulative_proportion in enumerate(cumulative_proportions)
                    if column_position < cumulative_proportion
                ),
                len(cumulative_proportions) - 1,
            )
            column_type_indices.append(terrain_type_idx)

        terrain._unitree_column_type_cache_key = cache_key
        terrain._unitree_column_type_indices = torch.tensor(
            column_type_indices, device=env.device, dtype=torch.long
        )

    requested_type_indices = [
        idx
        for idx, name in enumerate(generator_cfg.sub_terrains)
        if name in terrain_type_names
    ]
    if not requested_type_indices:
        return torch.zeros(
            env.num_envs if env_ids is None else env_ids.numel(),
            dtype=torch.bool,
            device=env.device,
        )

    if use_assigned_terrain:
        terrain_columns = terrain.terrain_types
        if env_ids is not None:
            terrain_columns = terrain_columns[env_ids]
        in_bounds = (terrain_columns >= 0) & (
            terrain_columns < generator_cfg.num_cols
        )
    else:
        robot_positions_w = env.scene["robot"].data.root_pos_w
        if env_ids is not None:
            robot_positions_w = robot_positions_w[env_ids]

        terrain_size_x, terrain_size_y = generator_cfg.size
        grid_x = (
            robot_positions_w[:, 0]
            + generator_cfg.num_rows * terrain_size_x / 2.0
        ) / terrain_size_x
        grid_y = (
            robot_positions_w[:, 1]
            + generator_cfg.num_cols * terrain_size_y / 2.0
        ) / terrain_size_y
        terrain_columns = torch.floor(grid_y).to(torch.long)
        in_bounds = (
            (grid_x >= 0)
            & (grid_x < generator_cfg.num_rows)
            & (grid_y >= 0)
            & (grid_y < generator_cfg.num_cols)
        )
    terrain_columns = terrain_columns.clamp(0, generator_cfg.num_cols - 1)
    terrain_type_indices = terrain._unitree_column_type_indices[terrain_columns]
    requested = torch.tensor(
        requested_type_indices, device=env.device, dtype=torch.long
    )
    return in_bounds & torch.isin(terrain_type_indices, requested)


class TerrainAwareUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with terrain overrides and independent zero probabilities."""

    cfg: TerrainAwareUniformVelocityCommandCfg

    def __init__(
        self, cfg: TerrainAwareUniformVelocityCommandCfg, env: ManagerBasedEnv
    ):
        super().__init__(cfg, env)
        if self.cfg.heading_command:
            raise ValueError(
                "TerrainAwareUniformVelocityCommand samples yaw rate directly and does "
                "not support heading commands."
            )

        configured_ranges = {"default": self.cfg.ranges}
        configured_ranges.update(self.cfg.terrain_specific_ranges or {})
        for terrain_type, ranges in configured_ranges.items():
            for component_name, limits in (
                ("lin_vel_x", ranges.lin_vel_x),
                ("lin_vel_y", ranges.lin_vel_y),
                ("ang_vel_z", ranges.ang_vel_z),
            ):
                if float(limits[0]) > float(limits[1]):
                    raise ValueError(
                        f"Invalid {component_name} range for {terrain_type}: {limits}."
                    )
            if len(ranges.zero_prob) != 3 or any(
                not 0.0 <= float(probability) <= 1.0
                for probability in ranges.zero_prob
            ):
                raise ValueError(
                    f"Invalid zero probabilities for {terrain_type}: "
                    f"{ranges.zero_prob}."
                )

        self.is_zero_vel_x_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._resample_command(env_ids)

        zero_prob = torch.tensor(self.cfg.ranges.zero_prob, device=self.device).repeat(
            env_ids.numel(), 1
        )
        for terrain_type, ranges in (self.cfg.terrain_specific_ranges or {}).items():
            mask = terrain_type_mask(
                self._env,
                (terrain_type,),
                env_ids,
                use_assigned_terrain=True,
            )
            if not torch.any(mask):
                continue

            count = int(mask.sum().item())
            sample = torch.empty(count, device=self.device)
            selected_env_ids = env_ids[mask]
            self.vel_command_b[selected_env_ids, 0] = sample.uniform_(*ranges.lin_vel_x)
            self.vel_command_b[selected_env_ids, 1] = sample.uniform_(*ranges.lin_vel_y)
            self.vel_command_b[selected_env_ids, 2] = sample.uniform_(*ranges.ang_vel_z)
            zero_prob[mask] = torch.tensor(ranges.zero_prob, device=self.device)

        random_values = torch.rand(env_ids.numel(), 3, device=self.device)
        self.is_zero_vel_x_env[env_ids] = random_values[:, 0] < zero_prob[:, 0]
        self.is_zero_vel_y_env[env_ids] = random_values[:, 1] < zero_prob[:, 1]
        self.is_zero_vel_yaw_env[env_ids] = random_values[:, 2] < zero_prob[:, 2]

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[self.is_zero_vel_x_env, 0] = 0.0
        self.vel_command_b[self.is_zero_vel_y_env, 1] = 0.0
        self.vel_command_b[self.is_zero_vel_yaw_env, 2] = 0.0


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class TerrainAwareUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for terrain-aware velocity sampling with per-axis zeroing."""

    class_type: type = TerrainAwareUniformVelocityCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        zero_prob: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ranges: Ranges = MISSING
    terrain_specific_ranges: dict[str, Ranges] | None = None
