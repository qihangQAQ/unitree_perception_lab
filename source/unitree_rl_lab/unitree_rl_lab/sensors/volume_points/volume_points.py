"""Sensor that tracks body-attached points and their virtual-obstacle penetration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import omni.physics.tensors.impl.api as physx
import torch
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase

from .volume_points_data import VolumePointsData

if TYPE_CHECKING:
    from .volume_points_cfg import VolumePointsCfg


class VolumePoints(SensorBase):
    """Attach a configurable grid of virtual points to one or more rigid bodies."""

    cfg: VolumePointsCfg

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)
        self._volume_points = None

    @property
    def data(self) -> VolumePointsData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        return self._body_physx_view

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]):
        self._virtual_obstacles.update(virtual_obstacles)

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(
            name_keys, self.body_names, preserve_order
        )

    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        leaf_regex = re.compile(f"^{leaf_pattern}$")
        body_names = [
            prim.GetName()
            for prim in sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: leaf_regex.match(prim.GetName()) is not None,
                depth=1,
            )
        ]
        if not body_names:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies."
            )

        body_names_regex = r"(" + "|".join(body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(
            body_names_regex.replace(".*", "*")
        )
        self._num_bodies = self.body_physx_view.count // self._num_envs
        if self._num_bodies != len(body_names):
            raise RuntimeError(
                "Failed to initialize volume-points sensor."
                f"\n\tInput prim path: {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_names_regex}"
            )

        self._volume_points_pattern = self.cfg.points_generator.func(
            self.cfg.points_generator
        ).to(self.device)
        self._data = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=self.device,
        )
        self._virtual_obstacles: dict[str, Any] = {}

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        self._refresh_volume_points(env_ids)
        self._refresh_penetration_offset(env_ids)

    def _refresh_volume_points(self, env_ids):
        body_poses = self.body_physx_view.get_transforms().view(-1, self.num_bodies, 7)[
            env_ids
        ]
        body_velocities = self.body_physx_view.get_velocities().view(
            -1, self.num_bodies, 6
        )[env_ids]
        self._data.pos_w[env_ids] = body_poses[..., :3]
        self._data.quat_w[env_ids] = math_utils.convert_quat(
            body_poses[..., 3:], to="wxyz"
        )
        self._data.vel_w[env_ids] = body_velocities[..., :3]
        self._data.ang_vel_w[env_ids] = body_velocities[..., 3:]

        num_body_instances = self._data.pos_w[env_ids].shape[0] * self.num_bodies
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(num_body_instances, -1, -1),
            self._data.pos_w[env_ids].flatten(0, 1),
            self._data.quat_w[env_ids].flatten(0, 1),
        ).reshape(
            *self._data.pos_w[env_ids].shape[:2], self._data.point_num_each_body, 3
        )
        self._data.points_pos_w[env_ids] = points_pos_w

        points_vel_w = (
            self._data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()
        )
        points_vel_w += torch.linalg.cross(
            self._data.ang_vel_w[env_ids].unsqueeze(-2),
            points_pos_w - self._data.pos_w[env_ids].unsqueeze(-2),
            dim=-1,
        )
        self._data.points_vel_w[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids):
        points_pos_w = self._data.points_pos_w[env_ids]
        penetration_offset = torch.zeros_like(points_pos_w)
        penetration_depth = torch.zeros_like(points_pos_w[..., 0])

        for virtual_obstacle in self._virtual_obstacles.values():
            candidate = virtual_obstacle.get_points_penetration_offset(
                points_pos_w.flatten(0, 2)
            )
            candidate = candidate.reshape(points_pos_w.shape)
            candidate_depth = torch.norm(candidate, dim=-1)
            deeper = candidate_depth > penetration_depth
            penetration_depth[deeper] = candidate_depth[deeper]
            penetration_offset[deeper] = candidate[deeper]

        self._data.penetration_offset[env_ids] = penetration_offset

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "points_visualizer"):
                self.points_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.points_visualizer.set_visibility(True)
        elif hasattr(self, "points_visualizer"):
            self.points_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.body_physx_view is None:
            return
        points = self._data.points_pos_w.view(-1, 3)
        penetrated = torch.norm(self._data.penetration_offset.view(-1, 3), dim=-1) > 0.0
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat(
                [penetrated, torch.tensor([True], device=self.device)], dim=0
            )
        self.points_visualizer.visualize(
            translations=points, marker_indices=penetrated.long()
        )

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        if hasattr(self, "points_visualizer"):
            del self.points_visualizer
        self._physics_sim_view = None
        self._body_physx_view = None
