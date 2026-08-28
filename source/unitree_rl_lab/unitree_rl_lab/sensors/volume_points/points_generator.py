"""Point-pattern generators for volume sensors."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .points_generator_cfg import Grid3dPointsGeneratorCfg


def grid3d_points_generator(cfg: Grid3dPointsGeneratorCfg) -> torch.Tensor:
    """Generate a dense XYZ grid in the attached body's local frame."""
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.x_num)
    y = torch.linspace(cfg.y_min, cfg.y_max, cfg.y_num)
    z = torch.linspace(cfg.z_min, cfg.z_max, cfg.z_num)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
