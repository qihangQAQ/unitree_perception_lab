"""Choose safe demo endpoints from the actual pillar mesh, without planning a route."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull


class PillarScene:
    """CPU-only setup geometry. Never supplied to the FDM or MPPI cost function."""

    def __init__(self, mesh, *, clearance: float = 0.35, resolution: float = 0.1):
        self.clearance, self.resolution = clearance, resolution
        self.minimum, self.maximum = np.asarray(mesh.bounds[:, :2], dtype=float)
        self.polygons = []
        for part in mesh.split(only_watertight=False):
            if part.bounds[1, 2] < 0.25:  # The floor's top is z=0.
                continue
            xy = np.unique(part.vertices[:, :2], axis=0)
            self.polygons.append(xy[ConvexHull(xy).vertices])
        if not self.polygons:
            raise ValueError("The mixed-pillar demo requires pillar obstacles.")

        self.axes = [np.arange(lo + resolution / 2, hi, resolution)
                     for lo, hi in zip(self.minimum, self.maximum)]
        grid = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1)
        self.grid = grid
        # Every square cell is free, including its corners. Four-neighbor links
        # cannot cut diagonally across an obstacle or a narrow gap.
        free_cells = self.free(grid.reshape(-1, 2), clearance + resolution / math.sqrt(2))
        self.labels, _ = label(free_cells.reshape(grid.shape[:2]))
        endpoint_mask = self.free(grid.reshape(-1, 2), 0.75).reshape(grid.shape[:2])
        self.endpoints = grid[endpoint_mask & (self.labels > 0)]

    def contains(self, xy):
        xy = np.asarray(xy)
        return ((xy >= self.minimum + self.clearance)
                & (xy <= self.maximum - self.clearance)).all(axis=-1)

    def free(self, xy, clearance=None):
        points = np.atleast_2d(np.asarray(xy, dtype=float))
        clearance = self.clearance if clearance is None else clearance
        valid = (np.isfinite(points).all(axis=1)
                 & (points >= self.minimum + clearance).all(axis=1)
                 & (points <= self.maximum - clearance).all(axis=1))
        for polygon in self.polygons:
            edges = np.roll(polygon, -1, axis=0) - polygon
            delta = points[:, None] - polygon[None]
            inside = (edges[None, :, 0] * delta[..., 1]
                      - edges[None, :, 1] * delta[..., 0] >= -1e-9).all(axis=1)
            t = ((delta * edges).sum(axis=-1) / (edges * edges).sum(axis=-1)).clip(0, 1)
            distances = np.linalg.norm(delta - t[..., None] * edges, axis=-1).min(axis=1)
            valid &= ~inside & (distances > clearance)
        return valid

    def component(self, xy):
        points = np.atleast_2d(np.asarray(xy, dtype=float))
        index = np.rint((points - np.array([axis[0] for axis in self.axes])) / self.resolution).astype(int)
        valid = ((index >= 0) & (index < np.array(self.labels.shape))).all(axis=1)
        result = np.zeros(len(points), dtype=int)
        result[valid] = self.labels[tuple(index[valid].T)]
        return result

    def direct_path_blocked(self, a, b):
        """Exact intersection of a line segment with any convex pillar footprint."""
        direction = np.asarray(b) - a
        for polygon in self.polygons:
            lower, upper = 0.0, 1.0
            for vertex, edge in zip(polygon, np.roll(polygon, -1, axis=0) - polygon):
                delta = np.asarray(a) - vertex
                start = edge[0] * delta[1] - edge[1] * delta[0]
                slope = edge[0] * direction[1] - edge[1] * direction[0]
                if abs(slope) < 1e-12:
                    if start < 0:
                        lower, upper = 1.0, 0.0
                        break
                elif slope > 0:
                    lower = max(lower, -start / slope)
                else:
                    upper = min(upper, -start / slope)
                if lower > upper:
                    break
            if lower <= upper:
                return True
        return False

    def choose_endpoints(self, *, seed, start=None, goal=None):
        """Explicit positions are validated, never silently moved to another cell."""
        for name, point in (("start", start), ("goal", goal)):
            if point is not None:
                if not self.free(point[:2], 0.75)[0] or self.component(point[:2])[0] == 0:
                    raise ValueError(f"--{name} needs 0.75 m of flat clearance from pillars and the floor edge.")
        if start is not None and goal is not None:
            a, b = np.asarray(start[:2]), np.asarray(goal)
            if self.component(a)[0] != self.component(b)[0]:
                raise ValueError("Start and goal have no connected passage with the required robot clearance.")
            if np.linalg.norm(b - a) < 1.0:
                raise ValueError("Start and goal must be at least 1 m apart.")
        else:
            if not len(self.endpoints):
                raise ValueError("No flat start/goal area has enough clearance in this layout.")
            rng = np.random.default_rng(seed)
            components = self.component(self.endpoints)
            for _ in range(2000):
                a = (self.endpoints[rng.integers(len(self.endpoints))]
                     if start is None else np.asarray(start[:2]))
                choices = self.endpoints if goal is None else np.asarray(goal)[None]
                groups = components if goal is None else self.component(goal)
                distances = np.linalg.norm(choices - a, axis=1)
                candidates = np.flatnonzero((distances >= 5.0) & (distances <= 7.0)
                                           & (groups == self.component(a)[0]))
                for index in rng.permutation(candidates)[:64]:
                    b = choices[index]
                    if self.direct_path_blocked(a, b):
                        break
                else:
                    continue
                break
            else:
                raise ValueError("No connected 5–7 m start/goal pair with a pillar in between. "
                                 "Try another --seed or specify both --start and --goal.")
        yaw = float(start[2]) if start is not None else math.atan2(b[1] - a[1], b[0] - a[0])
        return (float(a[0]), float(a[1]), yaw), (float(b[0]), float(b[1]))
