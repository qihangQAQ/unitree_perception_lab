"""Streaming dataset diagnostics using the same windows as the training set."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields

import torch

from ..utils.se2 import integrate_body_twists, wrap_angle, yaw_from_quaternion_xyzw
from ..utils.statistics import RunningMoments, ratio
from .schema import EpisodeData, TerminationReason
from .window import window_targets

CONTACT_GROUPS = ("torso", "left_hand", "right_hand")


class DatasetStatistics:
    """Consume one episode at a time while its shard is already in memory."""

    def __init__(self, horizon: int, command_timestep: float, low_motion_threshold: float) -> None:
        self.horizon = horizon
        self.command_timestep = command_timestep
        self.low_motion_threshold = low_motion_threshold
        self.episodes = self.frames = self.windows = self.candidates = 0
        self.collision_episodes = self.collision_windows = self.low_motion_windows = 0
        self.invalid_pixels = self.pixels = self.invalid_dt = 0
        self.endings: Counter[str] = Counter()
        self.contacts: Counter[str] = Counter()
        self.nonfinite: Counter[str] = Counter()
        self.moments = {
            name: RunningMoments(width)
            for name, width in (
                ("episode_frames", 1), ("episode_duration_s", 1), ("endpoint_distance_m", 1),
                ("commands", 3), ("velocity", 3), ("acceleration", 3),
                ("baseline_position_error_m", 1), ("baseline_heading_error_deg", 1),
            )
        }
        self.histograms = {name: Counter() for name in ("distance_m", "abs_x_m", "abs_y_m")}

    def update_episode(self, episode: EpisodeData, indices: list) -> None:
        self.episodes += 1
        self.frames += episode.num_frames
        self.windows += len(indices)
        self.candidates += int((episode.has_outgoing_command & ~episode.collision_now).sum())
        collided = bool(episode.collision_now.any())
        self.collision_episodes += int(collided)
        self.collision_windows += sum(item.collision for item in indices)
        self.low_motion_windows += sum(item.low_motion for item in indices)
        reason = "collision" if collided else TerminationReason(int(episode.termination_reason[-1])).name.lower()
        self.endings[reason] += 1
        groups = (episode.contact_groups & episode.collision_now[:, None]).any(dim=0)
        self.contacts.update({name: int(value) for name, value in zip(CONTACT_GROUPS, groups)})
        self.invalid_pixels += int(episode.height_map_invalid.sum())
        self.pixels += episode.height_map_invalid.numel()
        for field in fields(episode):
            value = getattr(episode, field.name)
            if value.is_floating_point():
                self.nonfinite[field.name] += int((~torch.isfinite(value)).sum())
        self.moments["episode_frames"].update(torch.tensor([episode.num_frames]))
        self.moments["episode_duration_s"].update((episode.timestamp[-1] - episode.timestamp[0]).reshape(1))
        self._kinematics(episode)
        # Statistics describe natural windows, before the weighted training sampler.
        for offset in range(0, len(indices), 256):
            starts = torch.tensor([item.start for item in indices[offset : offset + 256]], dtype=torch.long)
            target = window_targets(episode, starts, self.horizon)
            valid = target["valid_mask"]
            has_endpoint = valid[:, -1]
            endpoint = target["future_pose"][has_endpoint, -1, :2]
            distance = torch.linalg.vector_norm(endpoint, dim=-1)
            self.moments["endpoint_distance_m"].update(distance)
            for name, values in zip(self.histograms, (distance, endpoint[:, 0].abs(), endpoint[:, 1].abs())):
                bins = torch.floor(values[torch.isfinite(values)]).long()
                bucket, count = torch.unique(bins, return_counts=True)
                self.histograms[name].update(dict(zip(bucket.tolist(), count.tolist())))
            self.moments["commands"].update(target["future_commands"])
            baseline = integrate_body_twists(target["future_commands"], self.command_timestep)
            error = torch.linalg.vector_norm(baseline[..., :2] - target["future_pose"][..., :2], dim=-1)
            baseline_yaw = torch.atan2(baseline[..., 2], baseline[..., 3])
            actual_yaw = torch.atan2(target["future_pose"][..., 2], target["future_pose"][..., 3])
            yaw_error = torch.rad2deg(wrap_angle(baseline_yaw - actual_yaw).abs())
            self.moments["baseline_position_error_m"].update(error[valid])
            self.moments["baseline_heading_error_deg"].update(yaw_error[valid])

    def _kinematics(self, episode: EpisodeData) -> None:
        state = episode.state_history_raw[:, 0].double()
        dt = episode.timestamp[1:] - episode.timestamp[:-1]
        yaw = yaw_from_quaternion_xyzw(state[:, 3:7])
        delta = state[1:, :2] - state[:-1, :2]
        cosine, sine = yaw[0].cos(), yaw[0].sin()
        displacement = torch.stack(
            (cosine * delta[:, 0] + sine * delta[:, 1], -sine * delta[:, 0] + cosine * delta[:, 1],
             wrap_angle(yaw[1:] - yaw[:-1])), dim=-1
        )
        good = torch.isfinite(dt) & (dt > 0) & torch.isfinite(displacement).all(dim=-1)
        self.invalid_dt += int((~torch.isfinite(dt) | (dt <= 0)).sum())
        velocity = displacement / dt.clamp_min(1.0e-12)[:, None]
        self.moments["velocity"].update(velocity[good])
        # Velocities represent intervals, so acceleration uses their midpoint separation.
        acceleration = (velocity[1:] - velocity[:-1]) / ((dt[1:] + dt[:-1]) / 2).clamp_min(1.0e-12)[:, None]
        self.moments["acceleration"].update(acceleration[good[1:] & good[:-1]])

    def result(self) -> dict:
        histograms = {}
        for name, counts in self.histograms.items():
            total = sum(counts.values())
            histograms[name] = {
                f"[{index},{index + 1})": {"count": counts[index], "fraction": ratio(counts[index], total)}
                for index in sorted(counts)
            }
        moments = {name: accumulator.result() for name, accumulator in self.moments.items()}
        for name in ("velocity", "acceleration"):
            moments[name]["max_abs"] = [
                max(abs(low), abs(high)) if low is not None else None
                for low, high in zip(moments[name]["min"], moments[name]["max"])
            ]
        return {
            "counts": {
                "episodes": self.episodes, "frames": self.frames, "candidate_windows": self.candidates,
                "valid_windows": self.windows, "excluded_windows": self.candidates - self.windows,
                "non_start_frames": self.frames - self.candidates,
                "collision_episodes": self.collision_episodes, "collision_windows": self.collision_windows,
                "low_motion_windows": self.low_motion_windows,
            },
            "collision_episode_fraction": ratio(self.collision_episodes, self.episodes),
            "collision_window_fraction": ratio(self.collision_windows, self.windows),
            "low_motion_window_fraction": ratio(self.low_motion_windows, self.windows),
            "termination_reasons": dict(self.endings),
            "collision_groups": {
                name: {"episodes": self.contacts[name], "episode_fraction": ratio(self.contacts[name], self.episodes)}
                for name in CONTACT_GROUPS
            },
            "quality": {
                "invalid_height_pixels": self.invalid_pixels, "height_pixels": self.pixels,
                "invalid_height_fraction": ratio(self.invalid_pixels, self.pixels),
                "nonfinite_values": dict(self.nonfinite), "invalid_time_intervals": self.invalid_dt,
            },
            "statistics": moments,
            "endpoint_histograms": histograms,
            "definitions": {
                "window_distribution": "natural valid windows before weighted sampling",
                "histograms": "1m half-open bins; only populated bins are listed",
                "std": "population standard deviation (ddof=0); null for empty data",
                "commands": "planned [vx, vy, wz] over window horizons; units [m/s, m/s, rad/s]",
                "kinematics": "raw episode frame transitions, episode-initial yaw frame, real timestamps; no padding",
                "acceleration": "signed velocity differences / interval midpoint separation; [m/s^2, m/s^2, rad/s^2]",
                "duration": "last recorded frame minus first recorded frame; excludes settling/warmup",
                "baseline": "command plan integrated at nominal command_timestep; collision targets freeze, baseline continues",
                "collision_groups": "episodes with a collision in each group; groups may overlap",
                "low_motion_threshold_m": self.low_motion_threshold,
                "horizon": self.horizon, "command_timestep_s": self.command_timestep,
            },
        }
