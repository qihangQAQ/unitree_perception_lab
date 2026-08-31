"""Terrain importer that derives virtual obstacles from the generated terrain mesh."""

from __future__ import annotations

from typing import TYPE_CHECKING

import trimesh
from isaaclab.terrains import TerrainImporter as TerrainImporterBase
from isaaclab.utils.timer import Timer

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainImporterBase):
    """Generate virtual obstacles before importing the combined terrain mesh."""

    def __init__(self, cfg: TerrainImporterCfg):
        self._virtual_obstacles = {
            name: obstacle_cfg.class_type(obstacle_cfg)
            for name, obstacle_cfg in cfg.virtual_obstacles.items()
            if obstacle_cfg is not None
        }
        super().__init__(cfg)

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        return self._virtual_obstacles.copy()

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()

        for obstacle_name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {obstacle_name}"):
                virtual_obstacle.generate(mesh, device=self.device)

        super().import_mesh(name, mesh)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        result = super().set_debug_vis(debug_vis)
        for virtual_obstacle in self._virtual_obstacles.values():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()
        return result
