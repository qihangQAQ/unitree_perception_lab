"""GPU-accelerated penetration queries for collections of finite cylinders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import warp as wp

from .kernels import points_penetrate_cylinder_kernel


class CylinderSpatialGrid:
    """Index finite cylinders in a regular grid for batched point queries."""

    def __init__(
        self,
        cylinders: torch.Tensor | np.ndarray,
        num_grid_cells: int = 64**3,
        device: str | torch.device = "cuda",
    ):
        self.cylinders_np = (
            cylinders if isinstance(cylinders, np.ndarray) else cylinders.cpu().numpy()
        )
        if self.cylinders_np.ndim != 2 or self.cylinders_np.shape[1] != 7:
            raise ValueError("Cylinders must have shape (N, 7).")

        self.num_grid_cells = num_grid_cells
        self.device = device
        self._compute_bounding_box()
        self._create_grid()

    def _compute_bounding_box(self):
        xyz = np.concatenate(
            [self.cylinders_np[:, :3], self.cylinders_np[:, 3:6]], axis=0
        )
        radius_max = self.cylinders_np[:, 6].max()
        self.bbox_min = xyz.min(axis=0) - radius_max
        self.bbox_max = xyz.max(axis=0) + radius_max
        extent = np.maximum(self.bbox_max - self.bbox_min, 1.0e-6)

        scale = extent / extent.max()
        grid_resolution = np.maximum(
            np.round(scale * (self.num_grid_cells ** (1 / 3))), 1
        ).astype(int)
        self.grid_res = grid_resolution
        self.total_num_cells = int(np.prod(grid_resolution))
        self.cell_size = extent / grid_resolution

    def _get_flat_grid_index(self, x_index: int, y_index: int, z_index: int) -> int:
        if not (
            0 <= x_index < self.grid_res[0]
            and 0 <= y_index < self.grid_res[1]
            and 0 <= z_index < self.grid_res[2]
        ):
            return -1
        return (
            x_index * self.grid_res[1] * self.grid_res[2]
            + y_index * self.grid_res[2]
            + z_index
        )

    def _create_grid(self):
        grid: defaultdict[int, list[int]] = defaultdict(list)
        for cylinder_index, cylinder in enumerate(self.cylinders_np):
            radius = cylinder[6]
            bounds_min = np.minimum(cylinder[:3], cylinder[3:6]) - radius
            bounds_max = np.maximum(cylinder[:3], cylinder[3:6]) + radius
            min_cell = np.floor((bounds_min - self.bbox_min) / self.cell_size).astype(
                int
            )
            max_cell = np.floor((bounds_max - self.bbox_min) / self.cell_size).astype(
                int
            )

            for x_index in range(min_cell[0], max_cell[0] + 1):
                for y_index in range(min_cell[1], max_cell[1] + 1):
                    for z_index in range(min_cell[2], max_cell[2] + 1):
                        flat_index = self._get_flat_grid_index(
                            x_index, y_index, z_index
                        )
                        if flat_index >= 0:
                            grid[flat_index].append(cylinder_index)

        cell_offsets = np.zeros(self.total_num_cells + 1, dtype=np.int32)
        cell_indices: list[int] = []
        for cell_index in range(self.total_num_cells):
            cell_offsets[cell_index] = len(cell_indices)
            cell_indices.extend(grid[cell_index])
        cell_offsets[-1] = len(cell_indices)

        device = str(self.device)
        self.cell_offsets_wp = wp.array(cell_offsets, dtype=wp.int32, device=device)
        self.cell_indices_wp = wp.array(
            np.asarray(cell_indices, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.cell_size_wp = wp.vec3(*self.cell_size)
        self.bbox_min_wp = wp.vec3(*self.bbox_min)
        self.grid_res_wp = wp.vec3i(*self.grid_res)
        self.cylinder_start_wp = wp.array(
            self.cylinders_np[:, :3], dtype=wp.vec3, device=device
        )
        self.cylinder_end_wp = wp.array(
            self.cylinders_np[:, 3:6], dtype=wp.vec3, device=device
        )
        self.cylinder_radius_wp = wp.array(
            self.cylinders_np[:, 6], dtype=wp.float32, device=device
        )

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        """Return the deepest cylinder penetration offset for each point."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3).")

        penetration_offset = torch.zeros_like(points)
        wp.launch(
            points_penetrate_cylinder_kernel,
            dim=points.shape[0],
            inputs=[
                wp.from_torch(points, dtype=wp.vec3),
                self.cylinder_start_wp,
                self.cylinder_end_wp,
                self.cylinder_radius_wp,
                self.cell_offsets_wp,
                self.cell_indices_wp,
                self.grid_res_wp,
                self.bbox_min_wp,
                self.cell_size_wp,
                wp.from_torch(penetration_offset, dtype=wp.vec3),
            ],
            device=str(points.device),
        )
        return penetration_offset
