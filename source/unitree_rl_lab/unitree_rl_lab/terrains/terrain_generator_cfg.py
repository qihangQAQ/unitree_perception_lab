"""Reusable terrain-generator configurations for Unitree RL Lab tasks."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.terrains as terrain_gen
import numpy as np
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .custom_terrain import (
    HfCrossSteppingStonesTerrainCfg,
    HfNoisyInvertedPyramidSlopedTerrainCfg,
    HfNoisyPyramidSlopedTerrainCfg,
    HfSingleGapTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfVariableInvertedPyramidStairsTerrainCfg,
    HfVariablePyramidStairsTerrainCfg,
)
@height_field_to_mesh
def positive_discrete_obstacles_terrain(
    difficulty: float, cfg: PositiveDiscreteObstaclesTerrainCfg
) -> np.ndarray:
    """Generate randomly placed low cuboid pillars with positive heights only."""
    max_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    min_width = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    max_width = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    max_height_units = max(1, int(round(max_height / cfg.vertical_scale)))
    min_height_units = max(1, int(round(0.5 * max_height_units)))

    obstacle_widths = np.arange(min_width, max_width + 1, 4)
    obstacle_x = np.arange(0, width_pixels, 4)
    obstacle_y = np.arange(0, length_pixels, 4)
    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    for _ in range(cfg.num_obstacles):
        obstacle_width = int(np.random.choice(obstacle_widths))
        obstacle_length = int(np.random.choice(obstacle_widths))
        x_start = min(int(np.random.choice(obstacle_x)), width_pixels - obstacle_width)
        y_start = min(
            int(np.random.choice(obstacle_y)), length_pixels - obstacle_length
        )
        obstacle_height = np.random.randint(min_height_units, max_height_units + 1)
        heights[
            x_start : x_start + obstacle_width,
            y_start : y_start + obstacle_length,
        ] = obstacle_height

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    heights[x1:x2, y1:y2] = 0
    return heights


@configclass
class PositiveDiscreteObstaclesTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    """Configuration for curriculum-driven low cuboid pillars."""

    function = positive_discrete_obstacles_terrain
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0


UPGRADE_TERRAIN1 = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0 / 6.0),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0 / 6.0,
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=30,
        ),
        "stairs_up": HfVariableInvertedPyramidStairsTerrainCfg(
            proportion=1.0 / 6.0,
            step_height_range=(0.05, 0.25),
            step_width_pixels_range=(10, 5),
            platform_width=2.0,
            border_width=0.5,
        ),
        "stairs_down": HfVariablePyramidStairsTerrainCfg(
            proportion=1.0 / 6.0,
            step_height_range=(0.05, 0.25),
            step_width_pixels_range=(10, 5),
            platform_width=2.0,
            border_width=0.5,
        ),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=1.0 / 6.0,
            slope_range=(0.15, 0.4),
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            platform_width=2.0,
        ),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=1.0 / 6.0,
            slope_range=(0.1, 0.4),
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            platform_width=2.0,
        ),
    },
)


UPGRADE_TERRAIN2 = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=2.0 / 16.0),
        "stairs_up": HfVariableInvertedPyramidStairsTerrainCfg(
            proportion=2.0 / 16.0,
            step_height_range=(0.05, 0.25),
            step_width_pixels_range=(10, 5),
            platform_width=2.0,
            border_width=0.5,
        ),
        "stairs_down": HfVariablePyramidStairsTerrainCfg(
            proportion=2.0 / 16.0,
            step_height_range=(0.05, 0.25),
            step_width_pixels_range=(10, 5),
            platform_width=2.0,
            border_width=0.5,
        ),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=1.0 / 16.0,
            slope_range=(0.1, 0.4),
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            platform_width=2.0,
        ),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=1.0 / 16.0,
            slope_range=(0.15, 0.4),
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            platform_width=2.0,
        ),
        "cross_stepping_stones": HfCrossSteppingStonesTerrainCfg(
            proportion=3.0 / 16.0,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            platform_width=1.5,
            border_width=0.5,
            num_stone_rows=2,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=3.0 / 16.0,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            border_width=0.5,
            platform_width=2.0,
        ),
        "gap": HfSingleGapTerrainCfg(
            proportion=1.0 / 16.0,
            platform_width=2.0,
            gap_width_pixels_range=(3, 10),
            gap_depth_range=(0.2, 1.0),
            step_height_range=(0.03, 0.1),
        ),
        "discrete_obstacles": PositiveDiscreteObstaclesTerrainCfg(
            proportion=1.0 / 16.0,
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=30,
            platform_width=2.0,
        ),
    },
)
