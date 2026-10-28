"""Live scan points, executed commands and terrain-aligned demo goal markers."""

import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz
from isaacsim.util.debug_draw import _debug_draw

from ..utils.se2 import yaw_from_quaternion_xyzw


class PillarDemoVisualizer:
    def __init__(self, env, goal, arrival_radius, *, goal_height_at=None):
        self.env, self.robot = env, env.scene["robot"]
        self.goal, self.arrival_radius = goal, arrival_radius
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.blue = self._points("FdmScan", (0.0, 0.15, 1.0), 0.012)
        self.red = self._points("PolicyScan", (1.0, 0.0, 0.0), 0.022)
        self.arrow = VisualizationMarkers(
            GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/PillarDemo/CommandVelocity")
        )
        self.arrow.set_visibility(False)
        self.blue.set_visibility(False)
        self.red.set_visibility(False)
        self.goal_height_at = goal_height_at
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = goal
        self.goal_segments = ([] if goal is None else
                              self._goal_segments(goal, self.arrival_radius, self.goal_height_at))

    @staticmethod
    def _goal_segments(goal, radius, height_at=None):
        # Recompute only when the goal changes; the terrain is static.
        gx, gy = goal
        xy = [(gx + radius * math.cos(i * math.tau / 48),
               gy + radius * math.sin(i * math.tau / 48)) for i in range(49)]
        xy += [(gx - 0.1, gy), (gx + 0.1, gy), (gx, gy - 0.1), (gx, gy + 0.1)]
        heights = [0.0] * len(xy) if height_at is None else height_at(xy)
        points = [(x, y, float(z) + 0.035) for (x, y), z in zip(xy, heights)]
        return list(zip(points[:48], points[1:49])) + [(points[49], points[50]), (points[51], points[52])]

    @staticmethod
    def _points(name, color, radius):
        return VisualizationMarkers(VisualizationMarkersCfg(
            prim_path=f"/Visuals/PillarDemo/{name}",
            markers={"point": sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            )},
        ))

    @staticmethod
    def _show_points(marker, hits, lift, valid=None):
        mask = torch.isfinite(hits).all(dim=-1)
        if valid is not None:
            mask &= valid
        points = hits[mask].clone()
        marker.set_visibility(bool(len(points)))
        if len(points):
            points[:, 2] += lift  # Display only; neither sensor nor model inputs are changed.
            marker.visualize(translations=points)

    def update_fdm_scan(self, invalid=None):
        # These solid terrain obstacles have no overhangs: the door correction keeps
        # the same surface hits. Preserve the planning-time snapshot in world coordinates.
        valid = None if invalid is None else ~torch.flip(invalid[0, 0], dims=[0]).flatten()
        self._show_points(self.blue, self.env.scene["fdm_height_scanner"].data.ray_hits_w[0], 0.008, valid)

    def update_policy_scan(self):
        self._show_points(self.red, self.env.scene["height_scanner"].data.ray_hits_w[0], 0.025)

    def update_command(self, command):
        root = self.robot.data.root_pos_w[:1].clone()
        root[:, 2] += 0.6
        yaw = yaw_from_quaternion_xyzw(self.robot.data.root_quat_w[:1, (1, 2, 3, 0)])
        speed = command[:, :2].norm(dim=-1)
        moving = bool((speed > 1e-3).any())
        self.arrow.set_visibility(moving)
        if moving:
            zeros = torch.zeros_like(yaw)
            orientation = quat_from_euler_xyz(zeros, zeros, yaw + torch.atan2(command[:, 1], command[:, 0]))
            scale = torch.tensor((1.0, 0.1, 0.1), device=root.device).repeat(1, 1)
            scale[:, 0] = speed * 3.0
            self.arrow.visualize(root, orientation, scale)

        starts, ends, colors, widths = [], [], [], []

        def line(a, b, color, width=3.0):
            starts.append(tuple(a)); ends.append(tuple(b))
            colors.append(color); widths.append(width)

        for a, b in self.goal_segments:
            line(a, b, (1.0, 0.85, 0.0, 1.0))

        wz = float(command[0, 2])
        if abs(wz) > 1e-3:
            x, y, z = root[0].tolist()
            begin, sweep = float(yaw[0]), wz * 1.5
            arc = [(x + 0.45 * math.cos(begin + sweep * i / 20),
                    y + 0.45 * math.sin(begin + sweep * i / 20), z + 0.08) for i in range(21)]
            green = (0.0, 1.0, 0.0, 1.0)
            for a, b in zip(arc, arc[1:]):
                line(a, b, green, 4.0)
            angle = begin + sweep + math.copysign(math.pi / 2, wz)
            tip = arc[-1]
            for side in (-0.5, 0.5):
                tail = (tip[0] - 0.12 * math.cos(angle + side),
                        tip[1] - 0.12 * math.sin(angle + side), tip[2])
                line(tip, tail, green, 4.0)
        self.draw.clear_lines()
        self.draw.draw_lines(starts, ends, colors, widths)

    def close(self):
        self.draw.clear_lines()
        for marker in (self.blue, self.red, self.arrow):
            marker.set_visibility(False)
