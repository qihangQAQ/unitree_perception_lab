"""Configuration objects shared by rollout, dataset and training code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TERRAIN_USD_FILENAME = "navigation_terrain_wall_usd_merge_large_single_object_maze.usd"
# This module lives at <project>/source/unitree_rl_lab/unitree_rl_lab/fdm/config.py.
DEFAULT_TERRAIN_USD = Path(__file__).resolve().parents[4] / "fdm" / "assets" / "terrains" / TERRAIN_USD_FILENAME


@dataclass
class CommandSamplingCfg:
    """Distribution of command sequences used during data collection."""

    min_forward_speed: float = 0.0
    max_forward_speed: float = 1.2
    min_lateral_speed: float = -0.2
    max_lateral_speed: float = 0.2
    max_yaw_rate: float = 1.0
    correlated_probability: float = 0.60
    straight_probability: float = 0.20
    turn_probability: float = 0.15
    stop_probability: float = 0.05
    correlation: float = 0.85
    straight_yaw_std: float = 0.06
    turn_forward_max: float = 0.25

    def validate(self) -> None:
        probabilities = (
            self.correlated_probability,
            self.straight_probability,
            self.turn_probability,
            self.stop_probability,
        )
        if abs(sum(probabilities) - 1.0) > 1.0e-6:
            raise ValueError(f"Command mode probabilities must sum to one, got {sum(probabilities):.6f}.")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("Command mode probabilities cannot be negative.")
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError("Command correlation must be in [0, 1).")
        if self.min_forward_speed < 0.0 or self.max_forward_speed < self.min_forward_speed:
            raise ValueError("Invalid forward speed range.")
        if self.max_lateral_speed < self.min_lateral_speed:
            raise ValueError("Invalid lateral speed range.")


@dataclass
class RolloutCfg:
    """Simulator-independent timing and data collection configuration."""

    task_name: str = "Unitree-G1-29dof-FDM-Rollout"
    policy_checkpoint: str = (
        "logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt"
    )
    terrain_usd_path: str = str(DEFAULT_TERRAIN_USD)
    dataset_root: str = "datasets/fdm_g1/default"
    num_envs: int = 256
    seed: int = 42
    physics_dt: float = 0.005
    policy_dt: float = 0.02
    command_timestep: float = 0.5
    history_timestep: float = 0.05
    history_length: int = 10
    prediction_horizon: int = 10
    max_episode_commands: int = 150
    warmup_commands: int = 1
    map_height: int = 60
    map_width: int = 46
    map_clip_min: float = -1.0
    map_clip_max: float = 1.5
    invalid_height_sentinel: float = 1.5
    map_door_probe_height: float = 0.5
    map_door_height_threshold: float = 1.25
    policy_observation_corruption: bool = True
    collision_force_threshold: float = 1.0
    collision_delay_policy_steps: int = 1
    collision_body_names: tuple[str, ...] = (
        "torso_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "left_rubber_hand",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "right_rubber_hand",
    )
    usd_origin_spacing: float = 10.0
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split: str = "train"
    spawn_xy_jitter: float = 3.5
    spawn_attempts: int = 12
    initial_settle_steps: int = 10
    safe_spawn_sample_points: int = 30_000
    safe_spawn_grid_resolution: float = 0.05
    safe_spawn_wall_height: float = 2.25
    safe_spawn_robot_height: float = 0.6
    safe_spawn_robot_clearance: float = 0.7
    safe_spawn_robot_dim: float = 0.6
    safe_spawn_safety_margin: float = 0.3
    safe_spawn_door_filtering: bool = True
    safe_spawn_door_height_threshold: float = 1.2
    safe_spawn_wall_clearance_rays: int = 20
    safe_spawn_raycast_batch_size: int = 250_000
    max_frames_per_shard: int = 20_000
    command: CommandSamplingCfg = field(default_factory=CommandSamplingCfg)

    @property
    def policy_steps_per_command(self) -> int:
        return round(self.command_timestep / self.policy_dt)

    @property
    def physics_steps_per_policy(self) -> int:
        return round(self.policy_dt / self.physics_dt)

    def validate(self) -> None:
        self.command.validate()
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {self.split!r}.")
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive.")
        if self.history_length < 1 or self.prediction_horizon < 1:
            raise ValueError("History length and prediction horizon must be positive.")
        if abs(self.policy_steps_per_command * self.policy_dt - self.command_timestep) > 1.0e-8:
            raise ValueError("command_timestep must be an integer number of policy steps.")
        if abs(self.physics_steps_per_policy * self.physics_dt - self.policy_dt) > 1.0e-8:
            raise ValueError("policy_dt must be an integer number of physics steps.")
        if abs(sum(self.split_ratios) - 1.0) > 1.0e-6 or any(value <= 0.0 for value in self.split_ratios):
            raise ValueError("split_ratios must be positive and sum to one.")
        if self.spawn_xy_jitter <= 0.0 or self.spawn_attempts < 1 or self.initial_settle_steps < 1:
            raise ValueError("Invalid safe-spawn reset settings.")
        if self.safe_spawn_sample_points < 1 or self.safe_spawn_grid_resolution <= 0.0:
            raise ValueError("Invalid safe-spawn terrain-analysis resolution or sample count.")
        if min(
            self.safe_spawn_robot_height,
            self.safe_spawn_robot_clearance,
            self.safe_spawn_robot_dim,
            self.safe_spawn_safety_margin,
        ) <= 0.0:
            raise ValueError("Safe-spawn robot dimensions and safety margin must be positive.")
        if self.safe_spawn_wall_clearance_rays < 4:
            raise ValueError("safe_spawn_wall_clearance_rays must be at least four.")
        if self.safe_spawn_raycast_batch_size < self.safe_spawn_wall_clearance_rays:
            raise ValueError("safe_spawn_raycast_batch_size is too small.")
        if self.map_height != 60 or self.map_width != 46:
            raise ValueError("The first-stage height model requires a [1, 60, 46] map.")
        if self.map_door_height_threshold <= 0.0:
            raise ValueError("map_door_height_threshold must be positive.")
        if not Path(self.policy_checkpoint).suffix == ".pt":
            raise ValueError("The frozen policy checkpoint must be a .pt file.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FDMModelCfg:
    """G1 height-map FDM architecture."""

    history_length: int = 10
    state_dim: int = 5
    proprio_dim: int = 96
    command_dim: int = 3
    horizon: int = 10
    state_hidden_dim: int = 64
    state_gru_layers: int = 2
    height_latent_dim: int = 512
    command_latent_dim: int = 16
    command_hidden_dim: int = 128
    command_gru_layers: int = 2
    command_timestep: float = 0.5

    def validate(self) -> None:
        if (self.history_length, self.state_dim, self.proprio_dim) != (10, 5, 96):
            raise ValueError("The first-stage G1 model expects history [10, 5+96].")
        if self.horizon != 10 or self.command_dim != 3:
            raise ValueError("The first-stage model expects a [10, 3] command plan.")


@dataclass
class TrainCfg:
    """Optimization and online collection schedule."""

    collection_rounds: int = 20
    epochs_per_round: int = 8
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    num_workers: int = 4
    collision_window_fraction: float = 0.35
    low_motion_fraction: float = 0.10
    position_weight: float = 1.7
    heading_weight: float = 1.7
    collision_weight: float = 2.0
    stop_weight: float = 1.0
    checkpoint_dir: str = "logs/fdm_g1"
    device: str = "cuda"

    def validate(self) -> None:
        if self.collection_rounds < 1 or self.epochs_per_round < 1 or self.batch_size < 1:
            raise ValueError("Training counts must be positive.")
        if not 0.0 <= self.collision_window_fraction <= 1.0:
            raise ValueError("collision_window_fraction must be in [0, 1].")
