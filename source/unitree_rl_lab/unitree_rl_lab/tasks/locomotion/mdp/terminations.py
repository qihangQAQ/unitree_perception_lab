"""Task-specific termination functions for locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .commands.velocity_command import terrain_type_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fall_down_in_some_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
    terrain_types: tuple[str, ...] | list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate low-root states only while the robot is on selected terrains."""
    asset: Articulation = env.scene[asset_cfg.name]
    below_minimum = asset.data.root_link_pos_w[:, 2] < float(minimum_height)
    if not terrain_types:
        return below_minimum

    on_selected_terrain = terrain_type_mask(env, tuple(terrain_types))
    return below_minimum & on_selected_terrain
