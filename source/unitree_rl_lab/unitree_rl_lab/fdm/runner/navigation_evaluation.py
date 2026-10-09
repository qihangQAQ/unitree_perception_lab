"""Matched-start, single-goal navigation trials without archiving training data."""

from __future__ import annotations

import math

import torch

from .stage2_collector import Stage2RolloutCollector


class _DiscardWriter:
    next_episode_id = 0
    write_seconds = 0.0

    def __init__(self):
        self.written_paths = []

    def append(self, episode):
        self.next_episode_id += 1

    def flush(self):
        pass


def evaluate_navigation(env, observations, policy, rollout_cfg, model, grid, *, mppi_cfg, goal_cfg,
                        seed, log_interval_s=10.0):
    collector = Stage2RolloutCollector(
        env, observations, policy, _DiscardWriter(), rollout_cfg, model=model, navigation_grid=grid,
        mppi_cfg=mppi_cfg, goal_cfg=goal_cfg, single_goal=True,
    )
    capacity = max(150, math.ceil(goal_cfg.timeout_s / rollout_cfg.command_timestep) + 20)
    result = collector.collect(capacity, seed=seed, min_fill=1, mppi_fraction=1, log_interval_s=log_interval_s)
    if result["stop_reason"] != "evaluation_complete":
        raise RuntimeError("Navigation evaluation exhausted its capacity before all single-goal trials ended.")
    return {"seed": seed, "navigation": result["navigation"], "planning": result["planning"],
            "elapsed_seconds": result["elapsed_seconds"]}


def compare_navigation(baseline, current, *, tolerance=1e-3):
    """Report comparability explicitly rather than claim mismatched trials improved."""
    a = {item["env"]: item for item in baseline["navigation"]["scenarios"]}
    b = {item["env"]: item for item in current["navigation"]["scenarios"]}
    matched = baseline["seed"] == current["seed"] and a.keys() == b.keys()
    mismatches = []
    for env in sorted(a.keys() & b.keys()):
        left, right = a[env], b[env]
        same = left["valid"] == right["valid"] and torch.allclose(
            torch.tensor(left["start"][:7] + left["goal"]), torch.tensor(right["start"][:7] + right["goal"]),
            atol=tolerance, rtol=0,
        )
        if not same:
            mismatches.append(env)
    matched = matched and not mismatches
    result = {"matched_scenarios": matched, "mismatched_envs": mismatches, "tolerance_m_quat": tolerance,
              "baseline": baseline, "current": current}
    if matched:
        result["delta"] = {key: current["navigation"][key] - baseline["navigation"][key]
                           for key in ("arrival_rate", "collision_rate", "timeout_rate", "stall_rate")}
    return result
