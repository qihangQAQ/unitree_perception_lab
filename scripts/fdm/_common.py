"""Shared helpers for FDM command-line entry points."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def configure_safe_spawn(env_cfg, rollout_cfg) -> None:
    """Copy recorded rollout settings into the stateful terrain-analysis reset term."""

    reset = env_cfg.events.reset_base
    reset.params["xy_jitter"] = rollout_cfg.spawn_xy_jitter
    analysis = reset.func
    analysis.robot_dim = rollout_cfg.safe_spawn_robot_dim
    analysis.cfg.sample_points = rollout_cfg.safe_spawn_sample_points
    analysis.cfg.grid_resolution = rollout_cfg.safe_spawn_grid_resolution
    analysis.cfg.wall_height = rollout_cfg.safe_spawn_wall_height
    analysis.cfg.robot_height = rollout_cfg.safe_spawn_robot_height
    analysis.cfg.robot_buffer_spawn = rollout_cfg.safe_spawn_robot_clearance
    analysis.cfg.safety_margin = rollout_cfg.safe_spawn_safety_margin
    analysis.cfg.door_filtering = rollout_cfg.safe_spawn_door_filtering
    analysis.cfg.door_height_threshold = rollout_cfg.safe_spawn_door_height_threshold
    analysis.cfg.wall_clearance_rays = rollout_cfg.safe_spawn_wall_clearance_rays
    analysis.cfg.raycast_batch_size = rollout_cfg.safe_spawn_raycast_batch_size
    analysis.cfg.seed = rollout_cfg.seed


def dataset_metadata(env, rollout_cfg) -> dict[str, Any]:
    importer = env.unwrapped.scene.terrain
    checkpoint = Path(rollout_cfg.policy_checkpoint).resolve()
    terrain = Path(rollout_cfg.terrain_usd_path).resolve()
    rollout_values = rollout_cfg.as_dict()
    rollout_values.pop("split", None)
    return {
        "task": rollout_cfg.task_name,
        "schema_semantics": "command-boundary frames, newest-first history, rolling command_plan",
        "git_commit": git_commit(),
        "rollout": rollout_values,
        "policy_checkpoint": str(checkpoint),
        "policy_checkpoint_sha256": sha256_file(checkpoint),
        "terrain_usd": str(terrain),
        "terrain_usd_sha256": sha256_file(terrain),
        "usd_bbox": importer.usd_bbox,
        "split_origin_ids": importer.split_origin_ids,
        "usd_origins": importer.all_usd_origins.detach().cpu().tolist(),
        "safe_spawn_counts_per_origin": importer.safe_spawn_counts_per_origin.detach().cpu().tolist(),
        "contact_sensor_body_names": env.unwrapped.scene.sensors["contact_forces"].body_names,
        "robot_joint_names": env.unwrapped.scene["robot"].joint_names,
        "simulator_versions": env.unwrapped.metadata,
        "policy_observation": {
            "dimension": 283,
            "layout": [
                ["base_angular_velocity", 3],
                ["projected_gravity", 3],
                ["velocity_command", 3],
                ["joint_position_relative", 29],
                ["joint_velocity_relative", 29],
                ["last_action", 29],
                ["local_height_map", 187],
            ],
            "joint_velocity_scale": 0.05,
            "local_height_clip": [-1.0, 1.0],
            "local_height_noise": [-0.05, 0.05],
            "corruption_enabled": rollout_cfg.policy_observation_corruption,
        },
        "fdm_observation": {
            "proprio_dimension": 96,
            "height_map_shape": [1, rollout_cfg.map_height, rollout_cfg.map_width],
            "height_map_clip": [rollout_cfg.map_clip_min, rollout_cfg.map_clip_max],
            "invalid_height_sentinel": rollout_cfg.invalid_height_sentinel,
            "height_map_processing": "reference_door_recognition",
            "door_probe_height": rollout_cfg.map_door_probe_height,
            "door_height_threshold": rollout_cfg.map_door_height_threshold,
        },
        "domain_randomization": {"physics": False, "push": False, "mass": False},
    }
