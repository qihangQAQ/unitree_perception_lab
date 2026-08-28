"""Cross-shaped stepping-stone height-field terrain."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from isaaclab.terrains import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def cross_stepping_stones_terrain(
    difficulty: float, cfg: HfCrossSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate stepping stones along intersecting horizontal and vertical branches."""
    stone_width = cfg.stone_width_range[1] - difficulty * (
        cfg.stone_width_range[1] - cfg.stone_width_range[0]
    )
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    holes_depth = cfg.holes_depth[0] + difficulty * (
        cfg.holes_depth[1] - cfg.holes_depth[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_distance_pixels = int(stone_distance / cfg.horizontal_scale)
    stone_width_pixels = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    holes_depth_pixels = int(holes_depth / cfg.vertical_scale)
    platform_width_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    border_width_pixels = int(cfg.border_width / cfg.horizontal_scale)
    stone_height_range = np.arange(
        -stone_height_max - 1, stone_height_max, step=1
    )

    heights = np.full(
        (width_pixels, length_pixels), holes_depth_pixels, dtype=np.int16
    )
    center_x = width_pixels // 2
    center_y = length_pixels // 2
    num_rows = cfg.num_stone_rows
    stone_stride = stone_width_pixels + stone_distance_pixels

    for row_index in range(num_rows):
        if num_rows == 1:
            row_y = center_y - stone_width_pixels // 2
        else:
            row_offset = (row_index - (num_rows - 1) / 2) * stone_stride
            row_y = center_y - stone_width_pixels // 2 + int(row_offset)
        row_y = max(0, min(length_pixels - stone_width_pixels, row_y))

        start_x = 0
        while start_x < width_pixels:
            stop_x = min(width_pixels, start_x + stone_width_pixels)
            if 0 <= row_y and row_y + stone_width_pixels <= length_pixels:
                heights[
                    start_x:stop_x,
                    row_y : row_y + stone_width_pixels,
                ] = np.random.choice(stone_height_range)
            start_x += stone_stride

    for column_index in range(num_rows):
        if num_rows == 1:
            column_x = center_x - stone_width_pixels // 2
        else:
            column_offset = (
                column_index - (num_rows - 1) / 2
            ) * stone_stride
            column_x = center_x - stone_width_pixels // 2 + int(column_offset)
        column_x = max(0, min(width_pixels - stone_width_pixels, column_x))

        start_y = 0
        while start_y < length_pixels:
            stop_y = min(length_pixels, start_y + stone_width_pixels)
            if 0 <= column_x and column_x + stone_width_pixels <= width_pixels:
                heights[
                    column_x : column_x + stone_width_pixels,
                    start_y:stop_y,
                ] = np.random.choice(stone_height_range)
            start_y += stone_stride

    x1 = (width_pixels - platform_width_pixels) // 2
    x2 = (width_pixels + platform_width_pixels) // 2
    y1 = (length_pixels - platform_width_pixels) // 2
    y2 = (length_pixels + platform_width_pixels) // 2
    heights[x1:x2, y1:y2] = 0

    if border_width_pixels > 0:
        heights[:, :border_width_pixels] = 0
        heights[:, length_pixels - border_width_pixels :] = 0
        heights[:border_width_pixels, :] = 0
        heights[width_pixels - border_width_pixels :, :] = 0

    return np.rint(heights).astype(np.int16)


@configclass
class HfCrossSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for cross-shaped stepping stones over a recessed floor."""

    function = cross_stepping_stones_terrain
    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: tuple[float, float] = (-0.3, -0.5)
    platform_width: float = 1.0
    border_width: float = 0.5
    num_stone_rows: int = 2
