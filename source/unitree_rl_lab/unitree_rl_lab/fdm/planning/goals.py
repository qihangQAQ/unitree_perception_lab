"""Split-local, connected free-space goals and navigation outcome accounting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch

from ..utils.se2 import yaw_from_quaternion_xyzw


class SpatialSplit:
    """Resolve actual world coordinates, not just the recorded spawn cell."""

    def __init__(self, terrain):
        origins = terrain.all_usd_origins
        self.spacing = terrain.cfg.usd_uniform_env_spacing
        self.minimum = origins[:, :2].amin(dim=0) - self.spacing / 2
        self.shape = (len(origins[:, 0].unique()), len(origins[:, 1].unique()))
        self.splits = terrain.origin_split_ids

    def lookup(self, xy):
        cell = torch.floor((xy - self.minimum) / self.spacing).long()
        inside = ((cell >= 0) & (cell < torch.tensor(self.shape, device=xy.device))).all(dim=-1)
        flat = cell[..., 0].clamp(0, self.shape[0] - 1) * self.shape[1] + cell[..., 1].clamp(0, self.shape[1] - 1)
        return torch.where(inside, self.splits[flat].long(), -1)


class NavigationGrid:
    def __init__(self, terrain):
        data = getattr(terrain, "fdm_navigation_grid", None)
        if data is None:
            raise ValueError("Stage two requires retain_navigation_grid=True before the first environment reset.")
        self.origin, self.resolution = data["origin"], data["resolution"]
        self.device = self.origin.device
        self.spatial = SpatialSplit(terrain)
        shape = data["walkable"].shape
        x, y = torch.meshgrid(torch.arange(shape[0], device=self.device),
                              torch.arange(shape[1], device=self.device), indexing="ij")
        xy = self.origin + torch.stack((x, y), dim=-1) * self.resolution
        splits = self.spatial.lookup(xy).cpu().numpy()
        free = data["walkable"].cpu().numpy() & (splits >= 0)
        labels = np.zeros(shape, dtype=np.int32)
        # One-time CPU flood fill. No CPU graph search in the planning loop.
        component = 0
        for i, j in zip(*np.nonzero(free)):
            if labels[i, j]:
                continue
            component += 1
            labels[i, j] = component
            queue = deque([(i, j)])
            while queue:
                a, b = queue.popleft()
                for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if (0 <= c < shape[0] and 0 <= d < shape[1] and free[c, d]
                            and labels[c, d] == 0 and splits[c, d] == splits[a, b]):
                        labels[c, d] = component
                        queue.append((c, d))
        self.labels = torch.from_numpy(labels).to(self.device)
        print(f"[FDM] Navigation grid: shape={tuple(shape)} resolution={self.resolution:g}m components={component}.", flush=True)

    def lookup(self, xy):
        ij = torch.round((xy - self.origin) / self.resolution).long()
        inside = ((ij >= 0) & (ij < torch.tensor(self.labels.shape, device=xy.device))).all(dim=-1)
        labels = self.labels[ij[..., 0].clamp(0, self.labels.shape[0] - 1),
                             ij[..., 1].clamp(0, self.labels.shape[1] - 1)]
        return torch.where(inside, labels, 0)


@dataclass
class GoalCfg:
    minimum_distance: float = 1.5
    maximum_distance: float = 3.0
    arrival_radius: float = 0.4
    timeout_s: float = 20.0
    stall_s: float = 5.0
    progress_m: float = 0.1
    candidates: int = 64
    terrain_clearance: float = 0.4

    def validate(self):
        values = (self.minimum_distance, self.maximum_distance, self.arrival_radius,
                  self.timeout_s, self.stall_s, self.progress_m, self.terrain_clearance)
        if any(not math.isfinite(v) or v <= 0 for v in values) or self.candidates < 1:
            raise ValueError("Invalid local goal configuration.")
        if self.minimum_distance > self.maximum_distance or self.arrival_radius >= self.minimum_distance:
            raise ValueError("Invalid goal distance/radius ordering.")


class LocalGoalManager:
    def __init__(self, grid, num_envs, *, cfg=None, seed=42, single_goal=False):
        self.grid, self.device = grid, grid.device
        self.cfg = cfg or GoalCfg()
        self.cfg.validate()
        self.single_goal = single_goal
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.goal = torch.zeros(num_envs, 2, device=self.device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.finished = self.active.clone()
        self.started = torch.zeros(num_envs, device=self.device)
        self.last_progress = self.started.clone()
        self.best_distance = self.started.clone()
        self.unavailable_since = torch.full_like(self.started, -1.)
        self.reset_statistics()

    def reset_statistics(self):
        self.counts = {key: 0 for key in ("started", "reached", "collision", "timeout", "stall", "split_exit",
                                         "watchdog", "goal_unavailable", "anchor_unavailable", "goal_unavailable_resets", "censored")}
        self.arrival_seconds = []
        self.scenarios = []
        self.outcomes = []

    def reset(self, ids=None, *, seed=None):
        ids = slice(None) if ids is None else ids
        self.active[ids] = False
        self.unavailable_since[ids] = -1
        if not self.single_goal or seed is not None:
            self.finished[ids] = False
        if seed is not None:
            self.generator.manual_seed(seed)
            self.reset_statistics()

    def finish(self, ids, outcome, now):
        ids = ids[self.active[ids]]
        if not len(ids):
            return
        self.counts[outcome] += len(ids)
        elapsed = (now - self.started[ids]).cpu().tolist()
        if outcome == "reached":
            self.arrival_seconds.extend(elapsed)
        if self.single_goal:
            self.outcomes.extend({"env": env, "outcome": outcome, "elapsed_s": seconds}
                                 for env, seconds in zip(ids.cpu().tolist(), elapsed))
        self.active[ids] = False
        if self.single_goal:
            self.finished[ids] = True

    def update(self, ids, raw_state, now):
        if not len(ids):
            return
        xy = raw_state[:, :2]
        distance = torch.linalg.vector_norm(self.goal[ids] - xy, dim=-1)
        active = self.active[ids]
        progress = active & (distance < self.best_distance[ids] - self.cfg.progress_m)
        self.best_distance[ids[progress]] = distance[progress]
        self.last_progress[ids[progress]] = now
        self.finish(ids[active & (distance <= self.cfg.arrival_radius)], "reached", now)
        self.finish(ids[self.active[ids] & (now - self.started[ids] >= self.cfg.timeout_s)], "timeout", now)
        self.finish(ids[self.active[ids] & (now - self.last_progress[ids] >= self.cfg.stall_s)], "stall", now)
        need = ~self.active[ids] & ~self.finished[ids]
        self._sample(ids[need], raw_state[need], now)

    def _sample(self, ids, raw_state, now):
        if not len(ids):
            return
        xy = raw_state[:, :2]
        unit = torch.rand(len(ids), self.cfg.candidates, 2, device=self.device, generator=self.generator)
        distance = self.cfg.minimum_distance + unit[..., 0] * (self.cfg.maximum_distance - self.cfg.minimum_distance)
        angle = unit[..., 1] * (2 * math.pi)
        candidates = xy[:, None] + distance[..., None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
        anchor = self.grid.lookup(xy)
        # If the base is outside eroded free space, stop and retry; never jump
        # across a nearby wall by snapping to the nearest connected component.
        labels = self.grid.lookup(candidates)
        eligible = (labels == anchor[:, None]) & (anchor[:, None] > 0)
        valid = eligible.any(dim=1)
        rows = torch.arange(len(ids), device=self.device)
        goals = candidates[rows, eligible.long().argmax(dim=1)]
        self.goal[ids] = torch.where(valid[:, None], goals, xy)
        self.active[ids] = valid
        since = self.unavailable_since[ids]
        self.unavailable_since[ids] = torch.where(valid, -1., torch.where(since < 0, now, since))
        self.started[ids] = self.last_progress[ids] = now
        self.best_distance[ids] = torch.linalg.vector_norm(self.goal[ids] - xy, dim=-1)
        self.counts["started"] += int(valid.sum())
        self.counts["goal_unavailable"] += int((~valid).sum())
        self.counts["anchor_unavailable"] += int((anchor <= 0).sum())
        if self.single_goal:
            self.finished[ids[~valid]] = True
            self.scenarios.extend({"env": int(env), "start": state, "goal": goal, "valid": bool(ok)}
                                  for env, state, goal, ok in zip(ids.cpu().tolist(), raw_state.cpu().tolist(),
                                                                  self.goal[ids].cpu().tolist(), valid.cpu().tolist()))

    def local(self, ids, raw_state):
        delta = self.goal[ids] - raw_state[:, :2]
        yaw = yaw_from_quaternion_xyzw(raw_state[:, 3:7])
        return torch.stack((yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1],
                            -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]), dim=-1)

    def statistics(self):
        count = max(1, self.counts["started"])
        return {**self.counts, "arrival_rate": self.counts["reached"] / count,
                "collision_rate": self.counts["collision"] / count,
                "timeout_rate": self.counts["timeout"] / count, "stall_rate": self.counts["stall"] / count,
                "mean_arrival_seconds": sum(self.arrival_seconds) / len(self.arrival_seconds) if self.arrival_seconds else None,
                **({"scenarios": self.scenarios, "outcomes": self.outcomes} if self.single_goal else {})}
