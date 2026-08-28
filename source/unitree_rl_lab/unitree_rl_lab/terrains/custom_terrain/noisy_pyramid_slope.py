"""Pyramid slopes with curriculum-controlled inclination and surface noise."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from isaaclab.terrains import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def noisy_pyramid_sloped_terrain(
    difficulty: float, cfg: HfNoisyPyramidSlopedTerrainCfg
) -> np.ndarray:
    """Generate a truncated pyramid slope and add quantized height noise."""
    slope = cfg.slope_range[0] + difficulty * (
        cfg.slope_range[1] - cfg.slope_range[0]
    )
    if cfg.inverted:
        slope *= -1.0

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2.0 / cfg.vertical_scale)

    half_width = cfg.size[0] / 2.0
    half_length = cfg.size[1] / 2.0
    x = (np.arange(width_pixels, dtype=np.float64)[:, None] + 0.5) * (
        cfg.horizontal_scale
    ) - half_width
    y = (np.arange(length_pixels, dtype=np.float64)[None, :] + 0.5) * (
        cfg.horizontal_scale
    ) - half_length
    normalized_x = np.clip((half_width - np.abs(x)) / half_width, 0.0, 1.0)
    normalized_y = np.clip(
        (half_length - np.abs(y)) / half_length, 0.0, 1.0
    )
    heights = height_max * normalized_x * normalized_y

    platform_half_width = int(cfg.platform_width / cfg.horizontal_scale / 2.0)
    platform_x = width_pixels // 2 - platform_half_width
    platform_y = length_pixels // 2 - platform_half_width
    platform_height = heights[platform_x, platform_y]
    heights = np.clip(
        heights, min(0.0, platform_height), max(0.0, platform_height)
    )

    noise_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    noise_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    noise_step = int(cfg.noise_step / cfg.vertical_scale)
    if noise_step <= 0:
        raise ValueError(
            f"Noise step must be at least one vertical-scale unit, got {cfg.noise_step}."
        )
    noise_values = np.arange(noise_min, noise_max + noise_step, noise_step)
    heights += np.random.choice(noise_values, size=heights.shape)

    return np.rint(heights).astype(np.int16)


@configclass
class HfNoisyPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a noisy pyramid slope."""

    function = noisy_pyramid_sloped_terrain
    slope_range: tuple[float, float] = MISSING
    noise_range: tuple[float, float] = (0.0, 0.02)
    noise_step: float = 0.005
    platform_width: float = 1.0
    inverted: bool = False


@configclass
class HfNoisyInvertedPyramidSlopedTerrainCfg(HfNoisyPyramidSlopedTerrainCfg):
    """Configuration for a noisy inverted pyramid slope."""

    inverted: bool = True
