"""Standard pyramid stairs with curriculum-controlled tread depth."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from isaaclab.terrains import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def variable_pyramid_stairs_terrain(
    difficulty: float, cfg: HfVariablePyramidStairsTerrainCfg
) -> np.ndarray:
    """Generate concentric square stairs whose height rises and tread depth shrinks with difficulty."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    if cfg.inverted:
        step_height *= -1.0

    step_width_pixels = int(
        cfg.step_width_pixels_range[0]
        + difficulty * (cfg.step_width_pixels_range[1] - cfg.step_width_pixels_range[0])
    )
    if step_width_pixels <= 0:
        raise ValueError(
            f"Resolved stair tread depth must be positive, got {step_width_pixels} pixels."
        )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    platform_width_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    step_height_units = int(step_height / cfg.vertical_scale)

    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    current_step_height = 0
    start_x = start_y = 0
    stop_x, stop_y = width_pixels, length_pixels
    while (
        stop_x - start_x > platform_width_pixels
        and stop_y - start_y > platform_width_pixels
    ):
        start_x += step_width_pixels
        stop_x -= step_width_pixels
        start_y += step_width_pixels
        stop_y -= step_width_pixels
        current_step_height += step_height_units
        heights[start_x:stop_x, start_y:stop_y] = current_step_height

    return heights


@configclass
class HfVariablePyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for standard pyramid stairs with variable tread depth."""

    function = variable_pyramid_stairs_terrain
    step_height_range: tuple[float, float] = MISSING
    step_width_pixels_range: tuple[int, int] = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@configclass
class HfVariableInvertedPyramidStairsTerrainCfg(HfVariablePyramidStairsTerrainCfg):
    """Inverted standard pyramid stairs used for center-to-edge ascent."""

    inverted: bool = True
