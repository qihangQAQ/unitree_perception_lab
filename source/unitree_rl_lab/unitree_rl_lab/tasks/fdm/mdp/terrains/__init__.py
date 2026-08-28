# Copyright (c) 2025, The Nav-Suite Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FDM terrain generators — ported from nav-suite.

Provides navigational obstacle terrains (pillars, single objects, stairs/ramps/walls)
used to train the Forward Dynamics Model to distinguish traversable from non-traversable
terrain.
"""

from .pillar_terrain_cfg import (
    MeshPillarPlannerTestTerrainCfg,
    MeshPillarTerrainCfg,
    MeshPillarTerrainDeterministicCfg,
)
from .single_object import center_object_pattern, cross_object_pattern, extended_cross_object_pattern
from .single_object_cfg import SingleObjectTerrainCfg
from .stairs_ramp_terrain_cfg import StairsRampEvalTerrainCfg, StairsRampTerrainCfg, StairsRampUpDownTerrainCfg
