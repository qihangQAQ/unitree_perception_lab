"""Terrain-analysis based safe reset locations for the G1 FDM task.

This is a focused port of Nav-Suite's ``TerrainAnalysisRootReset``.  It keeps
only the pieces needed for spawning: a terrain height map, wall/door filtering,
wall-clearance rays, and footprint-aware spawn heights.  Navigation graph
construction is intentionally not included.
"""

from __future__ import annotations

import math

import omni.log
import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform
from isaaclab.utils.warp import raycast_mesh


@configclass
class TerrainAnalysisSpawnCfg:
    """Configuration matching the spawn-related reference FDM terrain analysis."""

    raycaster_sensor: str = "fdm_height_scanner"
    sample_points: int = 30_000
    grid_resolution: float = 0.05
    wall_height: float = 2.25
    robot_height: float = 0.6
    robot_buffer_spawn: float = 0.7
    door_filtering: bool = True
    door_probe_height: float = 0.5
    door_height_threshold: float = 1.2
    safety_margin: float = 0.3
    wall_clearance_rays: int = 20
    raycast_batch_size: int = 250_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.sample_points < 1:
            raise ValueError("safe-spawn sample_points must be positive.")
        if self.grid_resolution <= 0.0:
            raise ValueError("safe-spawn grid_resolution must be positive.")
        if self.robot_buffer_spawn <= 0.0 or self.robot_height <= 0.0:
            raise ValueError("safe-spawn robot dimensions must be positive.")
        if self.wall_clearance_rays < 4:
            raise ValueError("safe-spawn wall_clearance_rays must be at least four.")
        if self.raycast_batch_size < self.wall_clearance_rays:
            raise ValueError("safe-spawn raycast_batch_size is too small.")


class TerrainAnalysisRootReset:
    """Reset the robot at a pre-analysed, split-local collision-free point.

    Analysis is delayed until the first reset, when the simulator and the
    ray-caster mesh are available.  Candidate points are generated in every USD
    origin cell, so switching from validation to train within one process does
    not require rebuilding the analysis.
    """

    def __init__(self, cfg: TerrainAnalysisSpawnCfg, robot_dim: float = 0.6) -> None:
        if robot_dim <= 0.0:
            raise ValueError("safe-spawn robot_dim must be positive.")
        self.cfg = cfg
        self.robot_dim = robot_dim
        self._complete = False

    def __name__(self) -> str:
        return "TerrainAnalysisRootReset"

    @property
    def complete(self) -> bool:
        return self._complete

    def _raycast(
        self,
        starts: torch.Tensor,
        directions: torch.Tensor,
        max_dist: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hit_positions, distances, _, _ = raycast_mesh(
            ray_starts=starts.unsqueeze(0),
            ray_directions=directions.unsqueeze(0),
            mesh=self._mesh,
            max_dist=max_dist,
            return_distance=True,
        )
        if distances is None:
            raise RuntimeError("Terrain ray-caster did not return hit distances.")
        return hit_positions.squeeze(0), distances.squeeze(0)

    def _door_filtered_downcast(
        self,
        xy: torch.Tensor,
        ray_start_z: float,
        max_dist: float,
    ) -> torch.Tensor:
        starts = torch.empty((xy.shape[0], 3), device=self.device, dtype=torch.float32)
        starts[:, :2] = xy
        starts[:, 2] = ray_start_z
        down = torch.zeros_like(starts)
        down[:, 2] = -1.0
        top_hit, _ = self._raycast(starts, down, max_dist)

        if not self.cfg.door_filtering or xy.shape[0] == 0:
            return top_hit

        probe = starts.clone()
        probe[:, 2] = self.cfg.door_probe_height
        hit_down, _ = self._raycast(probe, down, max_dist)
        up = -down
        hit_up, _ = self._raycast(probe, up, max_dist)
        use_lower_surface = (
            torch.isfinite(hit_up[:, 2])
            & torch.isfinite(hit_down[:, 2])
            & (hit_up[:, 2] < top_hit[:, 2] - 1.0e-3)
            & (hit_up[:, 2] - hit_down[:, 2] > self.cfg.door_height_threshold)
        )
        top_hit[use_lower_surface] = hit_down[use_lower_surface]
        return top_hit

    def _construct_local_max_height_map(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_min, y_min, z_min = self.bounds_min
        x_max, y_max, z_max = self.bounds_max
        resolution = self.cfg.grid_resolution
        grid_x = torch.arange(x_min, x_max + 0.5 * resolution, resolution, device=self.device)
        grid_y = torch.arange(y_min, y_max + 0.5 * resolution, resolution, device=self.device)
        total = grid_x.numel() * grid_y.numel()
        heights = torch.empty(total, device=self.device, dtype=torch.float32)
        ray_start_z = max(z_max + 1.0, self.cfg.wall_height + 1.0)
        max_dist = ray_start_z - z_min + 100.0
        y_count = grid_y.numel()

        omni.log.info(
            f"[FDM] Building safe-spawn height map ({grid_x.numel()} x {grid_y.numel()} = {total} rays)."
        )
        for start in range(0, total, self.cfg.raycast_batch_size):
            stop = min(start + self.cfg.raycast_batch_size, total)
            flat = torch.arange(start, stop, device=self.device)
            x_idx = torch.div(flat, y_count, rounding_mode="floor")
            y_idx = flat % y_count
            xy = torch.stack((grid_x[x_idx], grid_y[y_idx]), dim=-1)
            heights[start:stop] = self._door_filtered_downcast(xy, ray_start_z, max_dist)[:, 2]

        height_grid = heights.view(grid_x.numel(), grid_y.numel())
        radius = math.ceil(self.robot_dim / resolution)
        padded = F.pad(height_grid[None, None], (radius, radius, radius, radius), value=-torch.inf)
        local_max = F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1)[0, 0]
        return grid_x, grid_y, local_max

    def _sample_cell_candidates(self, xy_jitter: float, draw_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.terrain.all_usd_origins[:, :2]
        per_origin = math.ceil(draw_count / origins.shape[0])
        total = per_origin * origins.shape[0]
        unit = self._sobol.draw(total).to(device=self.device, dtype=torch.float32)
        origin_ids = torch.arange(origins.shape[0], device=self.device).repeat_interleave(per_origin)
        xy = origins[origin_ids] + (unit * 2.0 - 1.0) * xy_jitter
        return xy, origin_ids

    def _filter_wall_clearance(self, points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] == 0:
            return torch.zeros(0, device=self.device, dtype=torch.bool)
        angles = torch.arange(self.cfg.wall_clearance_rays, device=self.device, dtype=torch.float32)
        angles *= 2.0 * math.pi / self.cfg.wall_clearance_rays
        directions = torch.stack(
            (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
        )
        keep = torch.empty(points.shape[0], device=self.device, dtype=torch.bool)
        points_per_batch = max(1, self.cfg.raycast_batch_size // self.cfg.wall_clearance_rays)
        for start in range(0, points.shape[0], points_per_batch):
            stop = min(start + points_per_batch, points.shape[0])
            count = stop - start
            ray_starts = points[start:stop].repeat_interleave(self.cfg.wall_clearance_rays, dim=0)
            ray_starts[:, 2] += self.cfg.robot_height
            ray_directions = directions.repeat(count, 1)
            _, distances = self._raycast(ray_starts, ray_directions, self.cfg.robot_buffer_spawn)
            keep[start:stop] = torch.isinf(distances).view(count, self.cfg.wall_clearance_rays).all(dim=1)
        return keep

    def _analyse(self, env, xy_jitter: float) -> None:
        if xy_jitter <= 0.0:
            raise ValueError("safe-spawn xy_jitter must be positive.")
        self.terrain = env.scene.terrain
        if not hasattr(self.terrain, "all_usd_origins") or not hasattr(self.terrain, "origin_split_ids"):
            raise RuntimeError("TerrainAnalysisRootReset requires SplitAwareUsdTerrainImporter.")
        self.device = torch.device(env.device)
        sensor = env.scene.sensors[self.cfg.raycaster_sensor]
        if len(sensor.cfg.mesh_prim_paths) != 1:
            raise RuntimeError("Safe-spawn terrain analysis currently requires one ray-caster terrain mesh.")
        self._mesh = sensor.meshes[sensor.cfg.mesh_prim_paths[0]]
        self.bounds_min = tuple(float(value) for value in self.terrain.usd_bbox[0])
        self.bounds_max = tuple(float(value) for value in self.terrain.usd_bbox[1])
        self._sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=self.cfg.seed)
        self._generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        grid_x, grid_y, local_max_height = self._construct_local_max_height_map()
        safe_points: list[torch.Tensor] = []
        safe_origin_ids: list[torch.Tensor] = []
        safe_count = 0
        stalled_passes = 0
        ray_start_z = max(self.bounds_max[2] + 1.0, self.cfg.wall_height + 1.0)
        max_dist = ray_start_z - self.bounds_min[2] + 100.0

        omni.log.info(f"[FDM] Sampling at least {self.cfg.sample_points} split-aware safe spawn points.")
        while safe_count < self.cfg.sample_points:
            xy, origin_ids = self._sample_cell_candidates(xy_jitter, self.cfg.sample_points)
            hit_parts: list[torch.Tensor] = []
            for start in range(0, xy.shape[0], self.cfg.raycast_batch_size):
                stop = min(start + self.cfg.raycast_batch_size, xy.shape[0])
                hit_parts.append(self._door_filtered_downcast(xy[start:stop], ray_start_z, max_dist))
            hits = torch.cat(hit_parts)
            valid_surface = torch.isfinite(hits[:, 2]) & (hits[:, 2] < self.cfg.wall_height)
            points = hits[valid_surface]
            point_origin_ids = origin_ids[valid_surface]
            clearance = self._filter_wall_clearance(points)
            points = points[clearance]
            point_origin_ids = point_origin_ids[clearance]

            x_idx = torch.round((points[:, 0] - grid_x[0]) / self.cfg.grid_resolution).long()
            y_idx = torch.round((points[:, 1] - grid_y[0]) / self.cfg.grid_resolution).long()
            x_idx.clamp_(0, grid_x.numel() - 1)
            y_idx.clamp_(0, grid_y.numel() - 1)
            points[:, 2] = local_max_height[x_idx, y_idx]
            finite_local_height = torch.isfinite(points[:, 2])
            points = points[finite_local_height]
            point_origin_ids = point_origin_ids[finite_local_height]

            if points.shape[0] == 0:
                stalled_passes += 1
                if stalled_passes >= 3:
                    raise RuntimeError(
                        "Terrain analysis could not find safe spawn points after three complete sampling passes."
                    )
                continue
            stalled_passes = 0
            safe_points.append(points)
            safe_origin_ids.append(point_origin_ids)
            safe_count += points.shape[0]
            omni.log.info(f"[FDM] Safe-spawn analysis accepted {safe_count} points so far.")

        self.safe_points = torch.cat(safe_points)
        self.safe_origin_ids = torch.cat(safe_origin_ids)
        origin_count = self.terrain.all_usd_origins.shape[0]
        self._origin_pools = [
            torch.nonzero(self.safe_origin_ids == origin_id).flatten() for origin_id in range(origin_count)
        ]
        self._split_pools = {
            split_id: torch.nonzero(self.terrain.origin_split_ids[self.safe_origin_ids] == split_id).flatten()
            for split_id in range(3)
        }
        missing_origins = [index for index, pool in enumerate(self._origin_pools) if pool.numel() == 0]
        if missing_origins:
            omni.log.warn(
                f"[FDM] {len(missing_origins)} USD origin cells contain no safe point; "
                "resets assigned to them will use another cell in the same spatial split."
            )
        for split_id, pool in self._split_pools.items():
            if pool.numel() == 0:
                raise RuntimeError(f"Terrain analysis found no safe spawn point for spatial split {split_id}.")

        self.terrain.safe_spawn_points = self.safe_points
        self.terrain.safe_spawn_origin_ids = self.safe_origin_ids
        self.terrain.safe_spawn_counts_per_origin = torch.bincount(
            self.safe_origin_ids, minlength=origin_count
        )
        self._complete = True
        del local_max_height
        omni.log.info(f"[FDM] Terrain safe-spawn analysis complete with {self.safe_points.shape[0]} points.")

    def _select_points(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected: list[torch.Tensor] = []
        selected_origin_ids: list[int] = []
        for env_id in env_ids.tolist():
            origin_id = int(self.terrain.env_origin_ids[env_id])
            pool = self._origin_pools[origin_id]
            if pool.numel() == 0:
                split_id = int(self.terrain.origin_split_ids[origin_id])
                pool = self._split_pools[split_id]
            pool_index = torch.randint(
                pool.numel(), (1,), device=self.device, generator=self._generator
            )
            point_index = pool[pool_index]
            selected.append(point_index)
            selected_origin_ids.append(int(self.safe_origin_ids[point_index]))
        return torch.cat(selected), torch.tensor(selected_origin_ids, device=self.device, dtype=torch.long)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        xy_jitter: float,
        yaw_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        if not self._complete:
            self._analyse(env, xy_jitter)

        asset = env.scene[asset_cfg.name]
        root_state = asset.data.default_root_state[env_ids].clone()
        point_indices, origin_ids = self._select_points(env_ids)
        points = self.safe_points[point_indices]

        # Keep the importer metadata consistent when an empty cell falls back to
        # another safe cell from the same spatial split.
        self.terrain.env_origin_ids[env_ids] = origin_ids
        self.terrain.env_origins[env_ids] = self.terrain.all_usd_origins[origin_ids]

        root_state[:, :2] = points[:, :2]
        root_state[:, 2] = points[:, 2] + asset.data.default_root_state[env_ids, 2] + self.cfg.safety_margin
        yaw = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=asset.device)
        root_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        ).squeeze(1)
        root_state[:, 7:13] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
        if hasattr(env, "_fdm_previous_collision"):
            env._fdm_previous_collision[env_ids] = False
