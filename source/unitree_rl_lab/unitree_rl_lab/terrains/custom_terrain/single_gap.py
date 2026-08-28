"""Single square-ring gap height-field terrain."""

from __future__ import annotations

import numpy as np
from isaaclab.terrains import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def single_gap_terrain(
    difficulty: float, cfg: HfSingleGapTerrainCfg
) -> np.ndarray:
    """Generate one square-ring gap around a flat center platform."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    heights = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    gap_depth = cfg.gap_depth_range[0] + difficulty * (
        cfg.gap_depth_range[1] - cfg.gap_depth_range[0]
    )
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )

    if cfg.gap_width_pixels_range is not None:
        gap_width_pixels = int(
            cfg.gap_width_pixels_range[0]
            + difficulty
            * (
                cfg.gap_width_pixels_range[1]
                - cfg.gap_width_pixels_range[0]
            )
        )
        gap_width = gap_width_pixels * cfg.horizontal_scale
    elif cfg.gap_width_range is not None:
        gap_width = cfg.gap_width_range[0] + difficulty * (
            cfg.gap_width_range[1] - cfg.gap_width_range[0]
        )
        gap_width_pixels = int(gap_width / cfg.horizontal_scale)
    else:
        raise ValueError(
            "Either gap_width_pixels_range or gap_width_range must be provided."
        )

    ground_ring_width = 1.0 - gap_width
    gap_depth_pixels = int(gap_depth / cfg.vertical_scale)
    step_height_pixels = int(step_height / cfg.vertical_scale)
    platform_width_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    ground_ring_width_pixels = int(ground_ring_width / cfg.horizontal_scale)

    center_x = int(cfg.size[0] / 2.0 / cfg.horizontal_scale)
    center_y = int(cfg.size[1] / 2.0 / cfg.horizontal_scale)

    current_inner_size = platform_width_pixels
    current_inner_half = current_inner_size // 2
    platform_start_x = max(0, center_x - current_inner_half)
    platform_end_x = min(width_pixels, center_x + current_inner_half)
    platform_start_y = max(0, center_y - current_inner_half)
    platform_end_y = min(length_pixels, center_y + current_inner_half)
    ground_ring_height = step_height_pixels
    heights[
        platform_start_x:platform_end_x,
        platform_start_y:platform_end_y,
    ] = ground_ring_height

    previous_inner_half = current_inner_half
    current_inner_size += 2 * ground_ring_width_pixels
    current_inner_half = current_inner_size // 2
    ground_start_x = max(0, center_x - current_inner_half)
    ground_end_x = min(width_pixels, center_x + current_inner_half)
    ground_start_y = max(0, center_y - current_inner_half)
    ground_end_y = min(length_pixels, center_y + current_inner_half)
    if ground_start_x < ground_end_x and ground_start_y < ground_end_y:
        x_grid, y_grid = np.meshgrid(
            np.arange(ground_start_x, ground_end_x),
            np.arange(ground_start_y, ground_end_y),
            indexing="ij",
        )
        max_distance = np.maximum(
            np.abs(x_grid - center_x), np.abs(y_grid - center_y)
        )
        ground_ring_mask = (max_distance >= previous_inner_half) & (
            max_distance < current_inner_half
        )
        heights[
            ground_start_x:ground_end_x,
            ground_start_y:ground_end_y,
        ][ground_ring_mask] = ground_ring_height

    previous_inner_half = current_inner_half
    current_inner_size += 2 * gap_width_pixels
    current_inner_half = current_inner_size // 2
    gap_start_x = max(0, center_x - current_inner_half)
    gap_end_x = min(width_pixels, center_x + current_inner_half)
    gap_start_y = max(0, center_y - current_inner_half)
    gap_end_y = min(length_pixels, center_y + current_inner_half)
    gap_bottom_height = ground_ring_height - gap_depth_pixels
    if gap_start_x < gap_end_x and gap_start_y < gap_end_y:
        x_grid, y_grid = np.meshgrid(
            np.arange(gap_start_x, gap_end_x),
            np.arange(gap_start_y, gap_end_y),
            indexing="ij",
        )
        max_distance = np.maximum(
            np.abs(x_grid - center_x), np.abs(y_grid - center_y)
        )
        gap_ring_mask = (max_distance >= previous_inner_half) & (
            max_distance < current_inner_half
        )
        heights[gap_start_x:gap_end_x, gap_start_y:gap_end_y][
            gap_ring_mask
        ] = gap_bottom_height

    return np.rint(heights).astype(np.int16)


@configclass
class HfSingleGapTerrainCfg(HfTerrainBaseCfg):
    """Configuration for one square-ring gap around a center platform."""

    function = single_gap_terrain
    platform_width: float = 1.0
    gap_width_range: tuple[float, float] | None = None
    gap_width_pixels_range: tuple[int, int] | None = (2, 8)
    gap_depth_range: tuple[float, float] = (0.1, 0.5)
    step_height_range: tuple[float, float] = (0.0, 0.2)
