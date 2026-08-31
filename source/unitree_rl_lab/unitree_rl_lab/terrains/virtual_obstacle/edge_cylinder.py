"""Sharp-edge extraction and cylindrical virtual inflation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from isaaclab.markers import VisualizationMarkers

from unitree_rl_lab.utils.warp import CylinderSpatialGrid

from .virtual_obstacle_base import VirtualObstacleBase

if TYPE_CHECKING:
    from .edge_cylinder_cfg import EdgeCylinderCfg


class EdgeCylinder(VirtualObstacleBase):
    """Represent connected sharp mesh edges as virtually inflated cylinders."""

    cfg: EdgeCylinderCfg

    def __init__(self, cfg: EdgeCylinderCfg):
        super().__init__(cfg)
        self.device = torch.device("cpu")
        self.cylinders: CylinderSpatialGrid | None = None
        self.edges = torch.empty((0, 6), dtype=torch.float32)

    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu"):
        edge_coordinates = self._extract_sharp_edges(mesh)
        edge_coordinates = self._merge_connected_edges(edge_coordinates)
        self._initialize_cylinders(edge_coordinates, device)

    def _extract_sharp_edges(self, mesh: trimesh.Trimesh) -> np.ndarray:
        threshold = np.deg2rad(self.cfg.angle_threshold)
        sharp_mask = mesh.face_adjacency_angles > threshold
        if np.any(sharp_mask):
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]
            vertices = mesh.vertices
            edge_coordinates = np.hstack(
                [vertices[sharp_edges[:, 0]], vertices[sharp_edges[:, 1]]]
            )
            return np.asarray(edge_coordinates, dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    def _initialize_cylinders(
        self,
        edge_coordinates: np.ndarray,
        device: torch.device | str,
    ):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.num_edges = len(edge_coordinates)
        self.edges = torch.as_tensor(
            edge_coordinates, dtype=torch.float32, device=self.device
        )
        if edge_coordinates.size == 0:
            self.cylinders = None
            return

        cylinders = np.concatenate(
            [
                edge_coordinates,
                np.full(
                    (len(edge_coordinates), 1),
                    self.cfg.cylinder_radius,
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )
        self.cylinders = CylinderSpatialGrid(
            cylinders=cylinders,
            num_grid_cells=self.cfg.num_grid_cells,
            device=self.device,
        )

    def visualize(self):
        """Draw the analytical edge cylinders used by penetration queries."""
        if self.edges.numel() == 0:
            self.disable_visualizer()
            return

        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)

        direction = self.edges[:, 3:6] - self.edges[:, :3]
        lengths = torch.linalg.vector_norm(direction, dim=-1)
        valid = lengths > 1.0e-6
        if not torch.any(valid):
            self.disable_visualizer()
            return

        direction = direction[valid]
        lengths = lengths[valid]
        translations = 0.5 * (self.edges[valid, :3] + self.edges[valid, 3:6])
        orientations = self._orient_z_axis(direction / lengths.unsqueeze(-1))
        scales = torch.empty((lengths.shape[0], 3), device=self.device)
        scales[:, :2] = float(self.cfg.cylinder_radius)
        scales[:, 2] = lengths

        self._cylinder_visualizer.set_visibility(True)
        self._cylinder_visualizer.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )

    def disable_visualizer(self):
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)

    @staticmethod
    def _orient_z_axis(direction: torch.Tensor) -> torch.Tensor:
        """Return quaternions rotating a marker's +Z axis onto ``direction``."""
        cross = torch.stack(
            (-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])),
            dim=-1,
        )
        quaternions = torch.cat(((1.0 + direction[:, 2]).unsqueeze(-1), cross), dim=-1)

        antiparallel = direction[:, 2] < -1.0 + 1.0e-6
        quaternions[antiparallel] = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], device=direction.device, dtype=direction.dtype
        )
        return torch.nn.functional.normalize(quaternions, dim=-1)

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        if self.cylinders is None:
            return torch.zeros_like(points)
        return self.cylinders.get_points_penetration_offset(points)

    def _merge_connected_edges(self, edge_coordinates: np.ndarray) -> np.ndarray:
        """Greedily concatenate nearly collinear adjacent mesh edges."""
        line_points = edge_coordinates.reshape(-1, 3)
        vertices, inverse_indices = np.unique(line_points, axis=0, return_inverse=True)
        edge_pairs = inverse_indices.reshape(-1, 2)
        adjacency = {index: set() for index in range(vertices.shape[0])}
        for start, end in edge_pairs:
            if start != end:
                adjacency[start].add(end)
                adjacency[end].add(start)

        remaining_degree = np.asarray(
            [len(adjacency[index]) for index in range(len(vertices))]
        )
        available = set(np.where(remaining_degree > 0)[0])
        cosine_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))
        merged_edges: list[np.ndarray] = []

        def consume_edge(start: int, end: int):
            adjacency[start].remove(end)
            adjacency[end].remove(start)
            for vertex in (start, end):
                remaining_degree[vertex] -= 1
                if remaining_degree[vertex] == 0:
                    available.discard(vertex)

        def straight_neighbor(endpoint: int, previous: int) -> int | None:
            neighbors = sorted(adjacency[endpoint])
            if not neighbors:
                return None
            directions = vertices[neighbors] - vertices[endpoint]
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            reference = vertices[endpoint] - vertices[previous]
            reference /= np.linalg.norm(reference)
            candidates = np.where(directions @ reference > cosine_threshold)[0]
            return neighbors[int(candidates[0])] if candidates.size else None

        while available:
            selected = min(available)
            if not adjacency[selected]:
                available.discard(selected)
                continue
            neighbor = min(adjacency[selected])
            chain = [selected, neighbor]
            consume_edge(selected, neighbor)

            while True:
                changed = False
                start_neighbor = straight_neighbor(chain[0], chain[1])
                if start_neighbor is not None:
                    consume_edge(chain[0], start_neighbor)
                    chain.insert(0, start_neighbor)
                    changed = True

                end_neighbor = straight_neighbor(chain[-1], chain[-2])
                if end_neighbor is not None:
                    consume_edge(chain[-1], end_neighbor)
                    chain.append(end_neighbor)
                    changed = True
                if not changed:
                    break

            if len(chain) >= self.cfg.min_points:
                merged_edges.append(
                    np.concatenate([vertices[chain[0]], vertices[chain[-1]]])
                )

        if not merged_edges:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(merged_edges, dtype=np.float32)
