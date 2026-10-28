"""Height-aware setup for the atlas demo; never used for MPPI route planning."""

from __future__ import annotations

import math

import numpy as np

from .terrain_atlas import SIZE, build_atlas, sample_mesh_heights


class AtlasScene:
    """Validate endpoints on actual collision geometry without blocking stairs.

    Starts need a level standing patch. Goals may be on a continuous gentle
    slope or a level platform. These local support checks do not classify
    routes, create waypoints, or supply a global map to the FDM planner.
    """

    def __init__(self, mesh, *, seed=0, clearance=0.35):
        self.minimum, self.maximum = np.asarray(mesh.bounds[:, :2], dtype=float)
        expected = np.array(SIZE) / 2
        if not (np.allclose(self.minimum, -expected) and np.allclose(self.maximum, expected)):
            raise ValueError("Atlas demo requires the world-centred 48 x 36 m atlas mesh.")
        self.clearance = clearance
        self.meshes = list(mesh.split(only_watertight=False))
        reference = build_atlas(seed)
        self.zones = {z["id"]: z for z in reference.zones}
        # Semantic exclusions for random goal placement only. Route planning
        # still receives only its existing live observations and final goal.
        self._obstacle_meshes = [p.mesh for p in reference.parts if p.kind in ("wall", "pillar")]

    def contains(self, xy):
        points = np.asarray(xy, dtype=float)
        return (np.isfinite(points).all(axis=-1)
                & (points >= self.minimum + self.clearance).all(axis=-1)
                & (points <= self.maximum - self.clearance).all(axis=-1))

    def height_at(self, xy):
        return sample_mesh_heights(self.meshes, np.atleast_2d(np.asarray(xy, dtype=float)))

    def validate_goal(self, xy, *, reject_obstacle_tops=False):
        """Return goal surface Z, or raise if the local support is unsuitable."""
        z = self._validate_endpoint(xy, start=False)
        if reject_obstacle_tops:
            obstacle_z = sample_mesh_heights(self._obstacle_meshes, np.asarray(xy)[None])[0]
            if np.isfinite(obstacle_z):
                raise ValueError("Random goals cannot be placed on pillars or walls.")
        return z

    def zone_at(self, xy):
        x, y = xy
        col = min(3, max(0, int((x + 24) // 12)))
        row = min(2, max(0, int((18 - y) // 12)))
        return f"{'ABC'[row]}{col + 1}"

    def _validate_endpoint(self, point, *, start):
        name = "start" if start else "goal"
        radius = 0.55 if start else 0.5
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not self.contains(point):
            raise ValueError(f"--{name} must be a finite XY point inside the atlas boundary.")
        # A dense disk and its perimeter catch stair risers, thin walls and
        # platform edges. Sampling is only for setup, never navigation costs.
        axis = np.arange(-radius, radius + 0.001, 0.1)
        grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        disk = grid[np.linalg.norm(grid, axis=1) <= radius]
        angles = np.linspace(0, math.tau, 48, endpoint=False)
        ring = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        offsets = np.vstack((np.zeros((1, 2)), disk, ring))
        samples = point + offsets
        heights = self.height_at(samples)
        if not self.contains(samples).all() or not np.isfinite(heights).all():
            raise ValueError(f"--{name} needs a supported {radius:.2f} m radius patch away from the map edge.")
        if start:
            if np.ptp(heights) > 0.015:
                raise ValueError("--start needs a level 0.55 m radius standing patch; "
                                 "choose flat ground or a wide platform, not a ramp, stair or wall edge.")
        else:
            design = np.column_stack((offsets, np.ones(len(offsets))))
            plane, *_ = np.linalg.lstsq(design, heights, rcond=None)
            residual = np.max(np.abs(design @ plane - heights))
            if residual > 0.025 or np.linalg.norm(plane[:2]) > math.tan(math.radians(30)):
                raise ValueError("--goal needs a continuous 0.50 m radius surface on flat ground, "
                                 "a platform or a slope <= 30 degrees; avoid walls and stair/drop edges.")
        return float(heights[0])

    def choose_endpoints(self, *, zone=None, start=None, goal=None):
        """Use a zone preset for missing endpoints; never move explicit points.

        With only one explicit endpoint, infer its zone unless --zone is given.
        With neither, B1 tests a stair bridge against its flat bypass.
        """
        for name, point, size in (("start", start, 3), ("goal", goal, 2)):
            if point is not None:
                values = np.asarray(point, dtype=float)
                if values.shape != (size,) or not np.isfinite(values).all():
                    raise ValueError(f"--{name} needs {size} finite coordinates.")
                self._validate_endpoint(values[:2], start=name == "start")
        if zone is not None and zone not in self.zones:
            raise ValueError(f"Unknown atlas zone {zone!r}; choose A1 through C4.")
        anchor = goal if goal is not None else (None if start is None else start[:2])
        zone = zone or ("B1" if anchor is None else self.zone_at(anchor))
        self.selected_zone = zone
        preset = self.zones[zone]["test"]
        a = np.array(preset["start_ground_xyz"][:2] if start is None else start[:2], dtype=float)
        b = np.array(preset["goal_ground_xyz"][:2] if goal is None else goal, dtype=float)
        self.start_height = self._validate_endpoint(a, start=True)
        self.goal_height = self._validate_endpoint(b, start=False)
        if np.linalg.norm(b - a) < 1.0:
            raise ValueError("Start and goal must be at least 1 m apart; specify --start/--goal explicitly.")
        yaw = float(start[2]) if start is not None else math.atan2(b[1] - a[1], b[0] - a[0])
        return (float(a[0]), float(a[1]), yaw), (float(b[0]), float(b[1]))

    def choose_spawn(self, *, zone=None, start=None):
        """Initialize an interactive run without creating an unused fixed goal."""
        if zone is not None and zone not in self.zones:
            raise ValueError(f"Unknown atlas zone {zone!r}.")
        if start is not None:
            values = np.asarray(start, dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError("--start needs three finite coordinates: X Y YAW.")
        zone = zone or ("B1" if start is None else self.zone_at(start[:2]))
        point = self.zones[zone]["test"]["start_ground_xyz"][:2] if start is None else start[:2]
        self.start_height = self._validate_endpoint(point, start=True)
        if abs(self.start_height) > 0.015:
            raise ValueError("Interactive mode must initially spawn on the flat z=0 floor.")
        self.selected_zone = zone
        return (float(point[0]), float(point[1]), 0.0 if start is None else float(start[2]))
