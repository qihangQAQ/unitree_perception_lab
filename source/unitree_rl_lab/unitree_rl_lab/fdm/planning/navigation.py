"""Fixed-goal local navigation configuration and progress tracking (no global planner)."""

from dataclasses import dataclass
import math

import torch

from ..utils.se2 import yaw_from_quaternion_xyzw


@dataclass
class NavigationCfg:
    timeout_s: float = 120.0
    initialization_s: float = 2.0
    arrival_radius: float = .4
    arrival_speed: float = .15
    arrival_yaw_rate: float = .2
    arrival_hold_s: float = .5
    stall_s: float = 5.0
    progress_m: float = .1
    heading_progress_rad: float = .15
    max_recoveries: int = 2
    align_enter_rad: float = math.pi / 3
    align_exit_rad: float = math.pi / 7.2
    forward_speed: float = .8
    yaw_rate: float = 1.0
    log_interval_s: float = 10.0

    def validate(self):
        for name in ('timeout_s', 'initialization_s', 'arrival_radius', 'arrival_speed', 'arrival_yaw_rate', 'arrival_hold_s',
                     'stall_s', 'progress_m', 'heading_progress_rad', 'forward_speed', 'yaw_rate'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.initialization_s < 1 or self.forward_speed > 1.2 or self.yaw_rate > 1:
            raise ValueError("Require initialization >= 1 s, vx <= 1.2 m/s and |wz| <= 1 rad/s.")
        if not 0 < self.align_exit_rad < self.align_enter_rad <= math.pi:
            raise ValueError("Invalid alignment hysteresis.")
        if self.max_recoveries < 0 or not math.isfinite(self.log_interval_s) or self.log_interval_s < 0:
            raise ValueError("Recovery count and log interval must be nonnegative.")


def local_goal(goal_world, raw_state):
    delta = goal_world - raw_state[:, :2]
    yaw = yaw_from_quaternion_xyzw(raw_state[:, 3:7])
    return torch.stack((yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1],
                        -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]), dim=-1)


class GoalProgress:
    """One immutable final goal per trial, measured from actual poses at 50 Hz."""

    def __init__(self, goals, state, cfg):
        cfg.validate()
        self.cfg = cfg
        self.goals = goals.clone()
        delta = local_goal(goals, state)
        self.best_distance = delta.norm(dim=-1)
        self.best_angle = torch.atan2(delta[:, 1], delta[:, 0]).abs()
        self.last_progress = torch.zeros_like(self.best_distance)
        self.hold_since = torch.full_like(self.best_distance, -1.)
        self.recoveries = torch.zeros_like(self.best_distance, dtype=torch.long)
        self.align = self.best_angle > cfg.align_enter_rad

    def update(self, state, speed, active, now, angular_speed=None):
        delta = local_goal(self.goals, state)
        distance = delta.norm(dim=-1)
        angle = torch.atan2(delta[:, 1], delta[:, 0]).abs()
        near = distance <= self.cfg.arrival_radius
        previous_align = self.align
        self.align = active & ~near & torch.where(self.align, angle > self.cfg.align_exit_rad,
                                                  angle > self.cfg.align_enter_rad)
        self.best_angle[self.align & ~previous_align] = angle[self.align & ~previous_align]
        advanced = distance < self.best_distance - self.cfg.progress_m
        turned = self.align & (angle < self.best_angle - self.cfg.heading_progress_rad)
        self.last_progress[active & (advanced | turned)] = now
        self.best_distance[active & advanced] = distance[active & advanced]
        self.best_angle[active & turned] = angle[active & turned]
        self.best_angle[active & ~self.align] = angle[active & ~self.align]
        held = active & near & (speed <= self.cfg.arrival_speed)
        if angular_speed is not None:
            held &= angular_speed.abs() <= self.cfg.arrival_yaw_rate
        self.hold_since = torch.where(held, torch.where(self.hold_since < 0, now, self.hold_since), -1.)
        reached = held & (now - self.hold_since >= self.cfg.arrival_hold_s)
        stalled = active & ~near & (now - self.last_progress >= self.cfg.stall_s)
        exhausted = stalled & (self.recoveries >= self.cfg.max_recoveries)
        recover = stalled & ~exhausted
        self.recoveries[recover] += 1
        self.last_progress[recover] = now
        self.best_distance[recover] = distance[recover]
        self.best_angle[recover] = angle[recover]
        return dict(distance=distance, near=near, reached=reached, recover=recover, exhausted=exhausted)
