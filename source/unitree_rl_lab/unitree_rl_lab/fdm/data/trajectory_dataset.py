"""Lazy reset-safe slicing of rollout episodes into FDM samples."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..utils.se2 import relative_pose_sequence
from ..utils.progress import ProgressLogger
from .schema import EpisodeData
from .shard_writer import read_manifest
from .statistics import DatasetStatistics
from .window import window_targets


@dataclass(frozen=True)
class WindowIndex:
    shard: int
    episode: int
    start: int
    collision: bool
    low_motion: bool


class FDMWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Build horizon-ten samples without ever crossing an episode reset."""

    def __init__(
        self,
        root_or_manifest: str | Path,
        split: str,
        *,
        horizon: int = 10,
        cache_size: int = 2,
        include_incomplete_noncollision: bool = False,
        low_motion_threshold: float = 0.05,
        shard_paths: list[str | Path] | None = None,
        log_interval_s: float = 10.0,
        collect_statistics: bool = False,
        allow_empty: bool = False,
    ) -> None:
        self.root, manifest = read_manifest(root_or_manifest)
        allowed = None if shard_paths is None else {str(Path(path).resolve()) for path in shard_paths}
        self.shards = [
            self.root / item["path"]
            for item in manifest["shards"]
            if item["split"] == split
            and (allowed is None or str((self.root / item["path"]).resolve()) in allowed)
        ]
        if not self.shards and not allow_empty:
            raise ValueError(f"No {split!r} shards in {self.root / 'manifest.json'}.")
        self.split = split
        self.command_timestep = float(manifest["metadata"].get("rollout", {}).get("command_timestep", 0.5))
        self.statistics = None
        self._statistics_accumulator = (
            DatasetStatistics(horizon, self.command_timestep, low_motion_threshold) if collect_statistics else None
        )
        self.horizon = horizon
        self.cache_size = max(1, cache_size)
        self.include_incomplete_noncollision = include_incomplete_noncollision
        self.low_motion_threshold = low_motion_threshold
        self._cache: OrderedDict[int, list[EpisodeData]] = OrderedDict()
        progress = ProgressLogger(f"index split={split}", len(self.shards), unit="shards", interval_s=log_interval_s)
        self.indices = self._build_index(progress)
        if self._statistics_accumulator is not None:
            self.statistics = self._statistics_accumulator.result()
            self._statistics_accumulator = None
        if not self.indices and not allow_empty:
            raise ValueError(f"No valid horizon-{horizon} windows were found for split {split!r}.")

    def _load_shard(self, shard_index: int) -> list[EpisodeData]:
        cached = self._cache.pop(shard_index, None)
        if cached is None:
            payload = torch.load(self.shards[shard_index], map_location="cpu", weights_only=False)
            episodes = [EpisodeData.from_dict(data) for data in payload["episodes"]]
        else:
            episodes = cached
        self._cache[shard_index] = episodes
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episodes

    def _build_index(self, progress: ProgressLogger) -> list[WindowIndex]:
        output: list[WindowIndex] = []
        for shard_index in range(len(self.shards)):
            for episode_index, episode in enumerate(self._load_shard(shard_index)):
                first_index = len(output)
                current_xy = episode.state_history_raw[:, 0, :2]
                for start in range(episode.num_frames):
                    if not bool(episode.has_outgoing_command[start]) or bool(episode.collision_now[start]):
                        continue
                    stop = min(start + self.horizon + 1, episode.num_frames)
                    collision_indices = torch.nonzero(episode.collision_now[start + 1 : stop]).flatten()
                    collision = collision_indices.numel() > 0
                    has_full_future = start + self.horizon < episode.num_frames
                    if not collision and not has_full_future and not self.include_incomplete_noncollision:
                        continue
                    if not collision and has_full_future:
                        target_slice = slice(start + 1, start + self.horizon + 1)
                        if torch.any(episode.terminated[target_slice] | episode.truncated[target_slice]):
                            continue
                    final = min(start + self.horizon, episode.num_frames - 1)
                    displacement = torch.linalg.vector_norm(current_xy[final] - current_xy[start]).item()
                    output.append(
                        WindowIndex(
                            shard_index,
                            episode_index,
                            start,
                            collision,
                            displacement < self.low_motion_threshold,
                        )
                    )
                if self._statistics_accumulator is not None:
                    self._statistics_accumulator.update_episode(episode, output[first_index:])
                progress.update(shard_index, detail=lambda: f"windows={len(output)}")
            progress.update(shard_index + 1, detail=lambda: f"windows={len(output)}")
        self._cache.clear()
        progress.update(len(self.shards), detail=lambda: f"windows={len(output)}", force=True)
        return output

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.indices[index]
        episode = self._load_shard(item.shard)[item.episode]
        start = item.start
        state_history = episode.state_history_raw[start]
        history_pose = relative_pose_sequence(state_history[:, :3], state_history[:, 3:7], anchor=0)
        relative_state = torch.cat((history_pose, state_history[:, 7:8]), dim=-1)
        target = window_targets(episode, torch.tensor([start]), self.horizon)
        return {
            "relative_state_history": relative_state.float(),
            "proprio_history": episode.proprio_history[start].float(),
            "history_timestamps": (episode.history_timestamps[start] - episode.timestamp[start]).float(),
            "height_map": episode.height_map[start].float(),
            "height_map_invalid": episode.height_map_invalid[start],
            **{name: value[0] for name, value in target.items()},
            "contains_collision": torch.tensor(item.collision),
            "low_motion": torch.tensor(item.low_motion),
        }

    @property
    def collision_flags(self) -> torch.Tensor:
        return torch.tensor([item.collision for item in self.indices], dtype=torch.bool)

    @property
    def low_motion_flags(self) -> torch.Tensor:
        return torch.tensor([item.low_motion for item in self.indices], dtype=torch.bool)


def make_collision_balanced_sampler(
    dataset: FDMWindowDataset,
    collision_fraction: float = 0.35,
    low_motion_fraction: float = 0.10,
    *,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Balance collision and low-motion marginals without changing disk data."""

    if not 0.0 <= collision_fraction <= 1.0 or not 0.0 <= low_motion_fraction <= 1.0:
        raise ValueError("Requested sampler fractions must be in [0, 1].")
    collision = dataset.collision_flags
    low_motion = dataset.low_motion_flags
    weights = torch.ones(len(dataset), dtype=torch.double)
    # Iterative proportional fitting preserves both requested marginals when
    # all necessary groups exist and degrades gracefully for empty groups.
    for _ in range(8):
        for flags, fraction in ((collision, collision_fraction), (low_motion, low_motion_fraction)):
            positive_sum = weights[flags].sum()
            negative_sum = weights[~flags].sum()
            if positive_sum > 0 and negative_sum > 0:
                weights[flags] *= fraction / positive_sum
                weights[~flags] *= (1.0 - fraction) / negative_sum
    return WeightedRandomSampler(weights, num_samples or len(dataset), replacement=True)
