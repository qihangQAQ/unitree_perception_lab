"""Terrain importer configuration with virtual-obstacle support."""

from __future__ import annotations

from dataclasses import field

from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgBase
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter
from .virtual_obstacle import VirtualObstacleCfg


@configclass
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    virtual_obstacles: dict[str, VirtualObstacleCfg] = field(default_factory=dict)
