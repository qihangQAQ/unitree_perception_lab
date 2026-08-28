"""Sharp-edge extraction and cylindrical virtual inflation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

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

    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu"):
        threshold = np.deg2rad(self.cfg.angle_threshold)
        sharp_mask = mesh.face_adjacency_angles > threshold
        if np.any(sharp_mask):
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]
            vertices = mesh.vertices
            edge_coordinates = np.hstack(
                [vertices[sharp_edges[:, 0]], vertices[sharp_edges[:, 1]]]
            )
            edge_coordinates = self._merge_connected_edges(edge_coordinates)
        else:
            edge_coordinates = np.empty((0, 6), dtype=np.float32)

        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.num_edges = len(edge_coordinates)
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
