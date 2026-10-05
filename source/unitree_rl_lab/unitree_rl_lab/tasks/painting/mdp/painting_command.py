"""Stage-aware painting command with a clock independent of episode buffers."""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

from unitree_rl_lab.painting.reference import (
    ending_masks,
    inverse_rotate,
    reference_at_time,
    rotate,
    trajectory_observation,
)
from unitree_rl_lab.painting.trajectories import PaintingPathCfg, import_surface_path, sample_paths

from .events import reset_painting_root


def _trapezoid_profile(elapsed, duration, ramp_time):
    """Return velocity scale and its time integral for a symmetric linear ramp."""
    ramp = torch.minimum(torch.full_like(duration, ramp_time), 0.5 * duration)
    t = elapsed.clamp_min(0).minimum(duration)
    accelerate = t < ramp
    decelerate = t > duration - ramp
    velocity_scale = torch.where(accelerate, t / ramp, torch.ones_like(t))
    velocity_scale = torch.where(decelerate, (duration - t) / ramp, velocity_scale).clamp(0, 1)
    integral = torch.where(accelerate, 0.5 * t.square() / ramp, t - 0.5 * ramp)
    final_segment = duration - ramp - 0.5 * (duration - t).square() / ramp
    integral = torch.where(decelerate, final_segment, integral)
    return velocity_scale, integral


class PaintingCommand(CommandTerm):
    """All consumers query one post-physics reference snapshot per control step.

    Isaac Lab increments common_step_counter before termination and reward
    evaluation, whereas CommandManager.compute runs afterwards. Reading the
    common counter here avoids that one-step delay without copying env.step().
    """

    def __init__(self, cfg: PaintingCommandCfg, env):
        cfg.path.validate()
        if cfg.task_mode not in ("side_step", "painting"):
            raise ValueError("task_mode must be 'side_step' or 'painting'.")
        if not 0 <= cfg.right_probability <= 1:
            raise ValueError("right_probability must be in [0,1].")
        if not 0 <= cfg.standing_probability <= 1:
            raise ValueError("standing_probability must be in [0,1].")
        for bounds in (cfg.duration_range, cfg.speed_range, cfg.distance_range, cfg.base_speed_range):
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"Invalid painting sampling range {bounds}.")
        if cfg.side_step_ramp_time <= 0:
            raise ValueError("side_step_ramp_time must be positive.")
        if cfg.prepare_time < 0 or cfg.catchup_time < 0:
            raise ValueError("Preparation and catchup times cannot be negative.")
        if cfg.duration_range[0] <= cfg.prepare_time + cfg.catchup_time:
            raise ValueError("Episode duration must exceed preparation and catchup times.")
        if len(cfg.lookahead_times) != 5 or cfg.lookahead_times[0] != 0 or any(
            b <= a for a, b in zip(cfg.lookahead_times, cfg.lookahead_times[1:])
        ):
            raise ValueError("Expected current point and four increasing lookahead times.")
        if not math.isclose(sum(x * x for x in cfg.spray_axis) ** 0.5, 1.0, abs_tol=1e-5):
            raise ValueError("spray_axis must be a unit vector in the TCP link frame.")
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        names = [cfg.tcp_body, "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link"]
        ids, resolved = self.robot.find_bodies(names, preserve_order=True)
        if resolved != names:
            raise ValueError(f"Painting body names do not match G1_CFG: {resolved}")
        self.tcp_id, self.torso_id, self.left_shoulder_id, self.left_elbow_id, self.left_hand_id = ids
        maximum = cfg.speed_range[1] * (cfg.duration_range[1] - cfg.prepare_time - cfg.catchup_time)
        self.capacity = math.ceil(maximum / cfg.path.spacing) + 2
        n, d = self.num_envs, self.device
        self.points = torch.zeros(n, self.capacity, 6, device=d)
        self.points[..., 2] = sum(cfg.path.height_range) / 2
        self.points[..., 3] = -1
        self.arc = (torch.arange(self.capacity, device=d) * cfg.path.spacing)[None].repeat(n, 1)
        self.lengths = self.arc[:, -1].clone()
        self.counts = torch.full((n,), self.capacity, dtype=torch.long, device=d)
        self.primitive_counts = torch.zeros(n, 4, dtype=torch.long, device=d)
        self.speed = torch.full((n,), cfg.speed_range[0], device=d)
        self.distance = torch.full((n,), sum(cfg.distance_range) / 2, device=d)
        self.deadline = torch.full((n,), cfg.duration_range[1], device=d)
        self.heading = -torch.ones(n, device=d)
        self.base_lateral_speed = torch.zeros(n, device=d)
        self.base_anchor_start = torch.zeros(n, 3, device=d)
        self.base_anchor_world = torch.zeros(n, 3, device=d)
        self.base_velocity_command = torch.zeros(n, 3, device=d)
        self.base_anchor_relative = torch.zeros(n, 2, device=d)
        self.end_effector_active = torch.zeros(n, device=d)
        self.start_step = torch.zeros(n, dtype=torch.long, device=d)
        self.lookahead = torch.tensor(cfg.lookahead_times, device=d)
        self.offset = torch.tensor(cfg.tcp_offset, device=d)
        self.local_axis = torch.tensor(cfg.spray_axis, device=d)
        self._snapshot_step = -1
        self._metric_step = torch.full((n,), -1, device=d, dtype=torch.long)
        self._started = torch.zeros(n, device=d, dtype=torch.bool)
        self._sums = torch.zeros(n, 3, device=d)
        self._samples = torch.zeros(n, device=d)
        self._base_sums = torch.zeros(n, 4, device=d)
        self._base_samples = torch.zeros(n, device=d)
        self._target_limit_hits = torch.zeros(n, device=d)
        self._torque_saturation = torch.zeros(n, device=d)
        self._bad_contact_steps = torch.zeros(n, device=d, dtype=torch.long)
        self._bad_contact_step = -1
        self._position_history = torch.full(
            (n, math.ceil(cfg.duration_range[1] / env.step_dt) + 2), float("nan"), device=d
        )
        self.refresh()

    @property
    def command(self):
        self.refresh()
        return self.goal_observation

    def refresh(self):
        step = self._env.common_step_counter
        if self._snapshot_step == step:
            return
        self._snapshot_step = step
        self.elapsed = (step - self.start_step).to(self.points.dtype) * self._env.step_dt
        if self.cfg.task_mode == "side_step":
            target = self.points[:, :1, :3].expand(-1, len(self.lookahead), -1).clone()
            target[..., 0] -= self.distance[:, None]
            velocity = torch.zeros(self.num_envs, 3, device=self.device)
            self.progress = torch.zeros_like(self.elapsed)
            self.command_speed = torch.zeros_like(self.elapsed)
            scale, displacement_time = _trapezoid_profile(
                self.elapsed, self.deadline, self.cfg.side_step_ramp_time
            )
            lateral_command = self.base_lateral_speed * scale
            self.base_anchor_world = self.base_anchor_start.clone()
            self.base_anchor_world[:, 1] += self.base_lateral_speed * displacement_time
            self.end_effector_active.zero_()
        else:
            target, velocity, self.progress, self.command_speed = reference_at_time(
                self.points, self.arc, self.lengths, self.speed, self.distance, self.elapsed,
                self.cfg.prepare_time, self.lookahead,
            )
            lateral_command = torch.zeros_like(self.elapsed)
            self.base_anchor_world = self.base_anchor_start
            self.end_effector_active = (self.elapsed >= self.cfg.prepare_time).to(self.points.dtype)
        origins = self._env.scene.env_origins
        self.target_world = target + origins[:, None]
        self.target_velocity = velocity
        data = self.robot.data
        self.base_pos = data.root_link_pos_w
        self.base_quat = data.root_link_quat_w
        link_pose = data.body_link_pose_w[:, self.tcp_id]
        link_velocity = data.body_link_vel_w[:, self.tcp_id]
        tcp_offset = rotate(link_pose[:, 3:7], self.offset.expand(self.num_envs, -1))
        self.tcp_position = link_pose[:, :3] + tcp_offset
        self.tcp_velocity = link_velocity[:, :3] + torch.cross(link_velocity[:, 3:6], tcp_offset, dim=-1)
        self.spray_axis = rotate(link_pose[:, 3:7], self.local_axis.expand(self.num_envs, -1))
        self.position_error = torch.linalg.vector_norm(self.tcp_position - self.target_world[:, 0], dim=-1)
        self.axis_error = torch.acos(self.spray_axis[:, 0].clamp(-1.0, 1.0))
        self.velocity_error = torch.linalg.vector_norm(self.tcp_velocity - velocity, dim=-1)
        self.tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        self.base_height = self.base_pos[:, 2] - origins[:, 2]
        endpoint = self.points[torch.arange(self.num_envs, device=self.device), self.counts - 1, :3].clone()
        endpoint[:, 0] -= self.distance
        self.endpoint_error = torch.linalg.vector_norm(self.tcp_position - (endpoint + origins), dim=-1)
        torso_quat = data.body_link_quat_w[:, self.torso_id]
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        self.torso_tilt = torch.acos(rotate(torso_quat, up)[:, 2].clamp(-1.0, 1.0))
        grace_over = self.elapsed >= self.cfg.fall_grace_time
        self.posture_bad = grace_over & (
            (self.tilt > self.cfg.bad_orientation_limit)
            | (self.torso_tilt > self.cfg.bad_torso_orientation_limit)
            | (self.base_height < self.cfg.minimum_base_height)
        )
        if self.cfg.task_mode == "side_step":
            self.success = torch.zeros_like(self.posture_bad)
            self.timeout = ~self.posture_bad & (self.elapsed >= self.deadline)
            self.bad = self.posture_bad.clone()
        else:
            self.bad, self.success, self.timeout = ending_masks(
                self.tilt, self.progress, self.lengths, self.endpoint_error, self.elapsed, self.deadline,
                self.cfg.bad_orientation_limit, self.cfg.success_tolerance,
            )
            self.bad |= self.posture_bad
            self.success &= ~self.bad
            self.timeout &= ~self.bad
        self.velocity_targets = torch.cat(
            (inverse_rotate(self.base_quat, data.root_link_lin_vel_w), inverse_rotate(self.base_quat, self.tcp_velocity)), -1
        )
        command_world = torch.zeros(self.num_envs, 3, device=self.device)
        command_world[:, 1] = lateral_command
        command_base = inverse_rotate(self.base_quat, command_world)
        self.base_velocity_command = torch.cat(
            (command_base[:, :2], torch.zeros(self.num_envs, 1, device=self.device)), -1
        )
        anchor_delta = inverse_rotate(self.base_quat, self.base_anchor_world - self.base_pos)
        self.base_anchor_relative = anchor_delta[:, :2]
        actual_base_velocity = self.velocity_targets[:, :3]
        self.base_velocity_error = torch.linalg.vector_norm(
            actual_base_velocity[:, :2] - self.base_velocity_command[:, :2], dim=-1
        )
        self.base_yaw_rate_error = (
            data.root_link_ang_vel_b[:, 2] - self.base_velocity_command[:, 2]
        ).abs()
        startup_duration = self.cfg.side_step_ramp_time if self.cfg.task_mode == "side_step" else self.cfg.prepare_time
        startup_countdown = ((startup_duration - self.elapsed).clamp_min(0) / max(startup_duration, 1e-6))[:, None]
        remaining_time = ((self.deadline - self.elapsed).clamp_min(0) / self.cfg.duration_range[1])[:, None]
        task_fields = torch.cat(
            (
                self.base_velocity_command,
                self.base_anchor_relative,
                self.end_effector_active[:, None],
                startup_countdown,
                remaining_time,
            ),
            -1,
        )
        self.goal_observation = torch.cat(
            (trajectory_observation(self.target_world, self.base_pos, self.base_quat, self.command_speed), task_fields), -1
        )
        forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        torso_forward, base_forward = rotate(torso_quat, forward), rotate(self.base_quat, forward)
        self.facing_error = torch.stack(
            (torch.atan2(base_forward[:, 1], base_forward[:, 0]),
             torch.atan2(torso_forward[:, 1], torso_forward[:, 0])), -1
        )
        arm = data.body_link_pos_w[:, [self.left_shoulder_id, self.left_elbow_id, self.left_hand_id]]
        directions = torch.diff(arm, dim=1)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-6)
        self.left_arm_error = (1 + directions[..., 2]).mean(-1)
        self.left_hand_base = inverse_rotate(self.base_quat, arm[:, -1] - self.base_pos)
        self._record_metrics(step)

    def _record_metrics(self, step):
        new_step = self._started & (self._metric_step != step)
        valid = new_step & self.end_effector_active.bool() & (self.command_speed > 0)
        values = torch.stack(
            (self.position_error, self.axis_error, self.velocity_error), -1
        )
        self._sums += torch.where(valid[:, None], values, 0.0)
        self._samples += valid
        base_values = torch.stack(
            (
                self.base_velocity_error,
                self.base_height,
                self.tilt,
                self.torso_tilt,
            ),
            -1,
        )
        self._base_sums += torch.where(new_step[:, None], base_values, 0.0)
        self._base_samples += new_step
        if hasattr(self._env, "action_manager"):
            action = self._env.action_manager.get_term("JointPositionAction")
            hit_rate = action.target_limit_hits.float().mean(-1)
            self._target_limit_hits += torch.where(new_step, hit_rate, 0.0)
        effort_limits = self.robot.data.joint_effort_limits.clamp_min(1e-6)
        torque_saturated = (self.robot.data.applied_torque.abs() >= 0.98 * effort_limits).any(-1)
        self._torque_saturation += new_step & torque_saturated
        rows = valid.nonzero().flatten()
        cols = (step - self.start_step[rows]).clamp_max(self._position_history.shape[1] - 1)
        self._position_history[rows, cols] = self.position_error[rows]
        self._metric_step[:] = step

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids if env_ids is not None else slice(None)]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.refresh()
        old = env_ids[self._started[env_ids]]
        logs = {}
        if len(old):
            base_means = self._base_sums[old] / self._base_samples[old, None].clamp_min(1)
            for i, name in enumerate(("base_velocity_error", "base_height", "pelvis_tilt", "torso_tilt")):
                logs[name] = base_means[:, i].mean().item()
            for name, values in (("time_limit", self.timeout), ("bad_posture", self.bad),
                                 ("duration", self.elapsed)):
                logs[name] = values[old].float().mean().item()
            denominator = self._base_samples[old].sum().clamp_min(1)
            logs["action/target_guard_rate"] = (self._target_limit_hits[old].sum() / denominator).item()
            logs["action/torque_saturation_rate"] = (
                self._torque_saturation[old].sum() / denominator
            ).item()
            for name, direction in (("right", -1), ("left", 1), ("stand", 0)):
                selected = old[self.heading[old] == direction]
                if len(selected):
                    directional = self._base_sums[selected, 0] / self._base_samples[selected].clamp_min(1)
                    logs[f"base_velocity_error_{name}"] = directional.mean().item()
            if self.cfg.task_mode == "painting":
                tracked = old[self._samples[old] > 0]
                if len(tracked):
                    means = self._sums[tracked] / self._samples[tracked, None]
                    for i, name in enumerate(("position_error", "axis_error", "velocity_error")):
                        logs[name] = means[:, i].mean().item()
                    p95 = torch.nanquantile(self._position_history[tracked], 0.95, dim=-1)
                    if torch.isfinite(p95).any():
                        logs["position_p95"] = p95[torch.isfinite(p95)].mean().item()
                logs["tracking_valid_count"] = self._samples[old].sum().item()
                for name, values in (
                    ("success", self.success),
                    ("path_length", self.lengths),
                    ("endpoint_error", self.endpoint_error),
                    ("reference_progress", self.progress / self.lengths),
                ):
                    logs[name] = values[old].float().mean().item()
        self._sums[env_ids] = 0
        self._samples[env_ids] = 0
        self._base_sums[env_ids] = 0
        self._base_samples[env_ids] = 0
        self._target_limit_hits[env_ids] = 0
        self._torque_saturation[env_ids] = 0
        self._bad_contact_steps[env_ids] = 0
        self._position_history[env_ids] = float("nan")
        self._metric_step[env_ids] = self._env.common_step_counter
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return logs

    def _resample_command(self, env_ids):
        n = len(env_ids)
        self.deadline[env_ids] = (
            torch.empty(n, device=self.device).uniform_(*self.cfg.duration_range) / self._env.step_dt
        ).round() * self._env.step_dt
        self.speed[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.speed_range)
        self.distance[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
        if self.cfg.task_mode == "side_step":
            moving = torch.rand(n, device=self.device) >= self.cfg.standing_probability
            magnitude = torch.empty(n, device=self.device).uniform_(*self.cfg.base_speed_range)
            direction = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
            self.base_lateral_speed[env_ids] = torch.where(moving, magnitude * direction, 0.0)
            self.heading[env_ids] = torch.sign(self.base_lateral_speed[env_ids])
            self.points[env_ids] = 0
            self.points[env_ids, :, 2] = self.cfg.static_tcp_height
            self.points[env_ids, :, 3] = -1
            self.arc[env_ids] = 0
            self.counts[env_ids] = 1
            self.lengths[env_ids] = 1.0
            self.primitive_counts[env_ids] = 0
        else:
            self.heading[env_ids] = torch.where(torch.rand(n, device=self.device) < self.cfg.right_probability, -1.0, 1.0)
            self.base_lateral_speed[env_ids] = 0
            length = self.speed[env_ids] * (self.deadline[env_ids] - self.cfg.prepare_time - self.cfg.catchup_time)
            paths = sample_paths(length, self.heading[env_ids], self.cfg.path, capacity=self.capacity)
            for name in ("points", "arc", "counts", "lengths", "primitive_counts"):
                getattr(self, name)[env_ids] = getattr(paths, name)
        self.start_step[env_ids] = self._env.common_step_counter
        self._started[env_ids] = True
        reset_painting_root(
            self.robot, env_ids, self._env.scene.env_origins, self.distance[env_ids],
            self.cfg.base_forward_reach, self.cfg.base_lateral_offset, self.cfg.initial_yaw_range,
        )
        origins = self._env.scene.env_origins
        self.base_anchor_start[env_ids, 0] = origins[env_ids, 0] - self.distance[env_ids] - self.cfg.base_forward_reach
        self.base_anchor_start[env_ids, 1] = origins[env_ids, 1] + self.cfg.base_lateral_offset
        self.base_anchor_start[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2] + origins[env_ids, 2]
        self._snapshot_step = -1

    def update_bad_contact(self, contact_force, threshold, debounce_time):
        """Merge debounced forbidden contact into the single bad-posture outcome."""
        step = self._env.common_step_counter
        if self._bad_contact_step != step:
            hit = (contact_force > threshold) & (self.elapsed >= self.cfg.fall_grace_time)
            self._bad_contact_steps = torch.where(hit, self._bad_contact_steps + 1, 0)
            self._bad_contact_step = step
        required = max(1, math.ceil(debounce_time / self._env.step_dt))
        self.bad = self.posture_bad | (self._bad_contact_steps >= required)
        self.success &= ~self.bad
        self.timeout &= ~self.bad
        return self.bad

    def set_external_path(self, env_id: int, surface_points: torch.Tensor, speed: float, distance: float):
        """Install a wall-frame path immediately after reset, preserving the time-budget rule."""
        if self.cfg.task_mode != "painting":
            raise ValueError("External paths are unavailable in the side-step curriculum stage.")
        if not 0 <= env_id < self.num_envs:
            raise ValueError("Invalid environment index.")
        if not self.cfg.speed_range[0] <= speed <= self.cfg.speed_range[1]:
            raise ValueError("External speed is outside the trained range.")
        if not self.cfg.distance_range[0] <= distance <= self.cfg.distance_range[1]:
            raise ValueError("External TCP distance is outside the trained range.")
        if self._env.common_step_counter != int(self.start_step[env_id]):
            raise ValueError("Install an external trajectory immediately after resetting that environment.")
        path = import_surface_path(surface_points.to(device=self.device, dtype=torch.float32), self.cfg.path, self.capacity)
        duration = float(path.lengths[0]) / speed + self.cfg.prepare_time + self.cfg.catchup_time
        if not self.cfg.duration_range[0] <= duration <= self.cfg.duration_range[1]:
            raise ValueError("External trajectory length and speed require a duration outside the trained range.")
        for name in ("points", "arc", "counts", "lengths", "primitive_counts"):
            getattr(self, name)[env_id] = getattr(path, name)[0]
        self.deadline[env_id], self.speed[env_id], self.distance[env_id] = duration, speed, distance
        self.heading[env_id] = 1 if surface_points[-1, 1] >= surface_points[0, 1] else -1
        self._snapshot_step = -1

    def compute(self, dt):
        # Episode-only resampling: CommandManager's periodic timer is deliberately unused.
        self.refresh()

    def _update_metrics(self):
        self.refresh()

    def _update_command(self):
        self.refresh()

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "visualizer"):
            self.visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Painting",
                    markers={
                        "trajectory": sim_utils.SphereCfg(
                            radius=0.006,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                        "target": sim_utils.SphereCfg(
                            radius=0.018,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        ),
                    },
                )
            )
        if hasattr(self, "visualizer"):
            self.visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "points"):
            return
        self.refresh()
        n = min(self.num_envs, 4)
        trajectory = self.points[:n, :, :3].clone()
        trajectory[..., 0] -= self.distance[:n, None]
        trajectory += self._env.scene.env_origins[:n, None]
        valid = torch.arange(self.capacity, device=self.device)[None] < self.counts[:n, None]
        trajectory = trajectory[valid]
        targets = self.target_world[:n].reshape(-1, 3)
        indices = torch.cat(
            (torch.zeros(len(trajectory), device=self.device), torch.ones(len(targets), device=self.device))
        ).long()
        self.visualizer.visualize(torch.cat((trajectory, targets)), marker_indices=indices)


@configclass
class PaintingCommandCfg(CommandTermCfg):
    class_type: type = PaintingCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    duration_range: tuple[float, float] = (20.0, 30.0)
    speed_range: tuple[float, float] = (0.2, 0.4)
    distance_range: tuple[float, float] = (0.05, 0.15)
    right_probability: float = 0.8
    task_mode: str = "painting"
    base_speed_range: tuple[float, float] = (0.10, 0.25)
    standing_probability: float = 0.20
    side_step_ramp_time: float = 0.75
    static_tcp_height: float = 1.25
    prepare_time: float = 1.0
    catchup_time: float = 1.5
    lookahead_times: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.8)
    tcp_body: str = "right_wrist_yaw_link"
    tcp_offset: tuple[float, float, float] = (0.10, 0.0, 0.0)
    spray_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    base_forward_reach: float = 0.28
    base_lateral_offset: float = 0.18
    initial_yaw_range: tuple[float, float] = (-0.03, 0.03)
    bad_orientation_limit: float = 0.8
    bad_torso_orientation_limit: float = 0.8
    minimum_base_height: float = 0.55
    fall_grace_time: float = 0.5
    success_tolerance: float = 0.03
    path: PaintingPathCfg = PaintingPathCfg()
