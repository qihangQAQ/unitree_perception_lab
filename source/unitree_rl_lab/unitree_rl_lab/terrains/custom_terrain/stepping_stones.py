"""Full-field stepping-stone height-field terrain."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from isaaclab.terrains import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def stepping_stones_terrain(
    difficulty: float, cfg: HfSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate stepping stones across the full terrain with a center platform."""
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
    stone_height_range = np.arange(
        -stone_height_max - 1, stone_height_max, step=1
    )

    heights = np.full(
        (width_pixels, length_pixels), holes_depth_pixels, dtype=np.int16
    )
    start_x, start_y = 0, 0

    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            stop_y = min(length_pixels, start_y + stone_width_pixels)
            start_x = np.random.randint(0, stone_width_pixels)
            stop_x = max(0, start_x - stone_distance_pixels)
            heights[0:stop_x, start_y:stop_y] = np.random.choice(
                stone_height_range
            )
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width_pixels)
                heights[start_x:stop_x, start_y:stop_y] = np.random.choice(
                    stone_height_range
                )
                start_x += stone_width_pixels + stone_distance_pixels
            start_y += stone_width_pixels + stone_distance_pixels
    else:
        while start_x < width_pixels:
            stop_x = min(width_pixels, start_x + stone_width_pixels)
            start_y = np.random.randint(0, stone_width_pixels)
            stop_y = max(0, start_y - stone_distance_pixels)
            heights[start_x:stop_x, 0:stop_y] = np.random.choice(
                stone_height_range
            )
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width_pixels)
                heights[start_x:stop_x, start_y:stop_y] = np.random.choice(
                    stone_height_range
                )
                start_y += stone_width_pixels + stone_distance_pixels
            start_x += stone_width_pixels + stone_distance_pixels

    x1 = (width_pixels - platform_width_pixels) // 2
    x2 = (width_pixels + platform_width_pixels) // 2
    y1 = (length_pixels - platform_width_pixels) // 2
    y2 = (length_pixels + platform_width_pixels) // 2
    heights[x1:x2, y1:y2] = 0
    return np.rint(heights).astype(np.int16)


@configclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for full-field stepping stones over a recessed floor."""

    function = stepping_stones_terrain
    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: tuple[float, float] = (-0.3, -0.5)
    platform_width: float = 1.0
