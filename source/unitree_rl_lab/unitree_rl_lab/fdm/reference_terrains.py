# Copyright (c) 2025, The Nav-Suite Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""One reference FDM planning terrain, kept separate from navigation benchmarks.

Adapted from nav-suite's terrains/single_object.py (cross_object_pattern) and
FDM's PLANNER_EVAL_2D_CFG.box_cross_pattern. This port centers the coordinates,
uses a local seeded RNG, and returns our simulator-independent scene spec.
The original license is in fdm/licenses/nav-suite-Apache-2.0.txt.
"""

import math

import numpy as np

from .benchmarks import BenchmarkScene, BoxObstacle


def reference_box_cross_scene(difficulty: float = .5, seed: int = 0) -> BenchmarkScene:
    """Reproduce one 10 m reference tile, including its start and +4 m goal."""
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Difficulty must be between 0 and 1.")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be a nonnegative 32-bit integer.")
    size = (10., 10.)
    # Original cross_object_pattern: center and four diagonals at +/-20% size.
    dx, dy = .2 * size[0], .2 * size[1]
    centers = ((0., 0.), (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy))
    width = .5 + difficulty * (1. - .5)
    rng = np.random.RandomState(seed)
    obstacles = tuple(BoxObstacle(center, (width, width, float(rng.uniform(.5, 1.5))))
                      for center in centers)
    start = (-.25 * size[0], 0., 0.)
    return BenchmarkScene(name="reference_box_cross", size=size, start=start,
                          goal=(start[0] + 4., 0.), obstacles=obstacles)
