"""Reset-safe FDM windows with optional precomputed CPU sample storage."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import groupby
from math import prod
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..utils.se2 import relative_pose_sequence
from ..utils.progress import ProgressLogger
from ..utils.memory import available_memory_bytes
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


def _episode_samples(
    episode: EpisodeData, items: list[WindowIndex], horizon: int, *, float_height: bool = True,
) -> dict[str, torch.Tensor]:
    """Materialize only requested frames, shared by memory and shard modes."""
    starts = torch.tensor([item.start for item in items], dtype=torch.long, device="cpu")
    state_history = episode.state_history_raw[starts]
    history_pose = relative_pose_sequence(state_history[..., :3], state_history[..., 3:7], anchor=0)
    height = episode.height_map[starts]
    return {
        "relative_state_history": torch.cat((history_pose, state_history[..., 7:8]), dim=-1).float(),
        "proprio_history": episode.proprio_history[starts].float(),
        "history_timestamps": (episode.history_timestamps[starts] - episode.timestamp[starts, None]).float(),
        "height_map": height.float() if float_height else height,
        "height_map_invalid": episode.height_map_invalid[starts],
        **window_targets(episode, starts, horizon),
        "contains_collision": torch.tensor([item.collision for item in items], dtype=torch.bool, device="cpu"),
        "low_motion": torch.tensor([item.low_motion for item in items], dtype=torch.bool, device="cpu"),
    }


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
        self._prepared_samples: dict[str, torch.Tensor] | None = None
        self._prepared_cache_mode = "memory"
        self._cache_directory: Path | None = None
        self.sample_pool_report: dict | None = None
        self._height_dtype = torch.float16
        self.shard_load_count = 0
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
            self.shard_load_count += 1
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
                # Vectorize the horizon checks within each episode. No per-start
                # torch reductions or scalar reads during indexing.
                starts = torch.arange(episode.num_frames, device="cpu")
                end = (starts + self.horizon).clamp_max(episode.num_frames - 1)
                collision_prefix = torch.cat((torch.zeros(1, dtype=torch.long, device="cpu"), episode.collision_now.long().cumsum(0)))
                collision = collision_prefix[end + 1] > collision_prefix[starts + 1]
                terminal = episode.terminated | episode.truncated
                terminal_prefix = torch.cat((torch.zeros(1, dtype=torch.long, device="cpu"), terminal.long().cumsum(0)))
                has_terminal = terminal_prefix[end + 1] > terminal_prefix[starts + 1]
                full = starts + self.horizon < episode.num_frames
                valid = episode.has_outgoing_command & ~episode.collision_now
                valid &= collision | ((full | self.include_incomplete_noncollision) & ~(full & has_terminal))
                low_motion = torch.linalg.vector_norm(current_xy[end] - current_xy, dim=-1) < self.low_motion_threshold
                selected = starts[valid].tolist()
                output.extend(WindowIndex(shard_index, episode_index, start, coll, low)
                              for start, coll, low in zip(selected, collision[valid].tolist(), low_motion[valid].tolist()))
                if self._statistics_accumulator is not None:
                    self._statistics_accumulator.update_episode(episode, output[first_index:])
                if len(output) > first_index and episode.height_map.dtype == torch.float32:
                    # v3 also permits float32 maps. Do not silently quantize them.
                    self._height_dtype = torch.float32
                progress.update(shard_index, detail=lambda: f"windows={len(output)}")
            progress.update(shard_index + 1, detail=lambda: f"windows={len(output)}")
        self._cache.clear()
        progress.update(len(self.shards), detail=lambda: f"windows={len(output)}", force=True)
        return output

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    @property
    def cache_mode(self) -> str:
        return self._prepared_cache_mode if self._prepared_samples is not None else "shard"

    def select_sample_pool(self, count: int, *, seed: int, collision_fraction=0.35, low_motion_fraction=0.10) -> dict:
        """Draw once before allocating samples; training then shuffles this pool.

        Repeats exist only in window sampling, never in stored raw episodes.
        Sorting requests by source enables one sequential pass to build caches.
        Natural dataset statistics remain unchanged for collection diagnostics.
        """
        if count < 1 or not len(self):
            raise ValueError("A sample pool needs positive count and nonempty source windows.")
        if self._prepared_samples is not None or self.sample_pool_report is not None:
            raise RuntimeError("Select the sample pool once, before preparing the cache.")
        sampler = make_collision_balanced_sampler(self, collision_fraction, low_motion_fraction, num_samples=count)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = torch.multinomial(sampler.weights, count, replacement=True, generator=generator).sort().values.tolist()
        source_count = len(self)
        self.indices = [self.indices[index] for index in selected]
        self._cache.clear()
        self.sample_pool_report = {
            "source_windows": source_count, "sampled_windows": count, "unique_windows": len(set(selected)),
            "repeated_windows": count - len(set(selected)), "seed": seed,
            "collision_fraction": float(self.collision_flags.float().mean()),
            "low_motion_fraction": float(self.low_motion_flags.float().mean()),
        }
        print(f"[FDM] Training sample pool: {self.sample_pool_report}", flush=True)
        return self.sample_pool_report

    def _sample_spec(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Shapes of the public sample fields for episode schema v3."""
        return {
            "relative_state_history": ((10, 5), torch.float32),
            "proprio_history": ((10, 96), torch.float32),
            "history_timestamps": ((10,), torch.float32),
            "height_map": ((1, 60, 46), self._height_dtype),
            "height_map_invalid": ((1, 60, 46), torch.bool),
            "future_commands": ((self.horizon, 3), torch.float32),
            "future_pose": ((self.horizon, 4), torch.float32),
            "future_collision": ((self.horizon,), torch.float32),
            "valid_mask": ((self.horizon,), torch.bool),
            "contains_collision": ((), torch.bool),
            "low_motion": ((), torch.bool),
        }

    @property
    def estimated_cache_bytes(self) -> int:
        return len(self) * sum(
            prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
            for shape, dtype in self._sample_spec().values()
        )

    @property
    def cached_bytes(self) -> int:
        if self._prepared_samples is None or self.cache_mode == "mmap":
            return 0
        return sum(value.numel() * value.element_size() for value in self._prepared_samples.values())

    def release_cache(self) -> None:
        """Drop sample storage and any raw shards before the next collection."""
        self._prepared_samples = None
        self._cache_directory = None
        self._cache.clear()

    @torch.no_grad()
    def prepare_cache(
        self, mode: str = "memory", *, max_cache_bytes: int | None = None,
        resident_cache_bytes: int = 0, log_interval_s: float = 10.0,
    ) -> None:
        """Prepare this dataset once; the budget includes an existing val cache.

        The explicit budget bounds resident sample tensors, not the whole process.
        Available-RAM checks additionally reserve room for raw shards and scratch
        tensors, but cannot guarantee against concurrent external allocations.
        """
        if mode not in ("memory", "shard", "auto", "mmap"):
            raise ValueError(f"Unknown dataset cache mode: {mode!r}")
        if resident_cache_bytes < 0 or (max_cache_bytes is not None and max_cache_bytes < 0):
            raise ValueError("Cache budgets must be nonnegative.")
        if mode == "shard":
            self.release_cache()
            print(f"[FDM] Dataset split={self.split} cache=shard (batch reads, {self.cache_size} shard LRU).", flush=True)
            return
        if mode == "mmap":
            self._prepare_mmap(log_interval_s)
            return

        required = self.estimated_cache_bytes
        combined = required + resident_cache_bytes
        budget = "unlimited" if max_cache_bytes is None else f"{max_cache_bytes / 2**30:.3f}GiB"
        print(
            f"[FDM] Cache plan split={self.split} windows={len(self)} "
            f"samples={required / 2**30:.3f}GiB resident={resident_cache_bytes / 2**30:.3f}GiB "
            f"combined={combined / 2**30:.3f}GiB budget={budget} height_dtype={self._height_dtype}",
            flush=True,
        )
        if max_cache_bytes is not None and combined > max_cache_bytes:
            if mode == "auto":
                print("[FDM] RAM cache budget exceeded; auto uses precomputed mmap samples.", flush=True)
                self._prepare_mmap(log_interval_s)
                return
            raise MemoryError(
                f"FDM sample caches need {combined / 2**30:.3f} GiB including resident validation data, "
                f"above --dataset-cache-gb={max_cache_bytes / 2**30:g}. "
                "Increase the budget or use --dataset-cache mmap. Existing shards are unchanged."
            )
        if self.cache_mode == "memory":
            return

        self._cache.clear()
        available = available_memory_bytes()
        scratch = max((path.stat().st_size for path in self.shards), default=0) + 16 * 2**20
        if available is not None and required + scratch > int(available * 0.8):
            if mode == "auto":
                print(
                    f"[FDM] Cache preflight: need={(required + scratch) / 2**30:.3f}GiB "
                    f"available={available / 2**30:.3f}GiB with 20% reserve; "
                    "auto uses precomputed mmap samples.", flush=True,
                )
                self._prepare_mmap(log_interval_s)
                return
            raise MemoryError(
                f"FDM needs {required / 2**30:.3f} GiB of new sample tensors plus preprocessing space; "
                f"only {available / 2**30:.3f} GiB RAM is currently available (20% kept in reserve). "
                "Free RAM or use --dataset-cache mmap."
            )

        started = time.monotonic()
        initial_loads = self.shard_load_count
        progress = ProgressLogger(f"cache split={self.split}", len(self), unit="windows", interval_s=log_interval_s)
        try:
            # Allocate final-size arrays once: no list of all samples followed by
            # a full-size concatenation/copy, and no full-dataset float32 map copy.
            storage = {
                name: torch.empty((len(self), *shape), dtype=dtype, device="cpu")
                for name, (shape, dtype) in self._sample_spec().items()
            }
            for shard, requests in groupby(enumerate(self.indices), key=lambda pair: pair[1].shard):
                self._prepare_shard(shard, requests, storage, progress)
                self._cache.clear()
        finally:
            self._cache.clear()
        self._prepared_samples = storage
        self._prepared_cache_mode = "memory"
        self._cache_directory = None
        progress.update(len(self), force=True)
        print(
            f"[FDM] Cache ready split={self.split} cache=memory windows={len(self)} "
            f"size={self.cached_bytes / 2**30:.3f}GiB elapsed={time.monotonic() - started:.2f}s "
            f"shard_loads={self.shard_load_count - initial_loads}; epoch reads use CPU tensors.",
            flush=True,
        )

    def _prepare_mmap(self, log_interval_s):
        from .prepared_cache import build_cache, cache_path

        if self.cache_mode == "mmap":
            return
        directory = cache_path(self)
        available = available_memory_bytes()
        scratch = max((path.stat().st_size for path in self.shards), default=0) + 16 * 2**20
        if not (directory / "complete.json").is_file() and available is not None and scratch > available * 0.8:
            raise MemoryError("Insufficient RAM even for one raw shard during mmap preprocessing. "
                              "Reduce collection buffers or prepare this round in a separate process.")
        self._cache.clear()
        progress = ProgressLogger(f"mmap split={self.split}", len(self), unit="windows", interval_s=log_interval_s)
        self._prepared_samples = build_cache(self, directory, progress)
        self._prepared_cache_mode = "mmap"
        self._cache_directory = directory
        progress.update(len(self), force=True)
        print(f"[FDM] Cache ready split={self.split} cache=mmap windows={len(self)} "
              f"files={self.estimated_cache_bytes / 2**30:.3f}GiB path={directory}; epochs do not load raw shards.", flush=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.cache_mode == "mmap":
            # Workers reopen mappings instead of serializing the entire sample pool.
            state["_prepared_samples"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._cache_directory is not None:
            from .prepared_cache import open_cache
            self._prepared_samples = open_cache(self._cache_directory, self._sample_spec(), len(self))

    def _prepare_shard(self, shard, requests, storage, progress) -> None:
        # This function's raw-episode references are released before the next
        # shard is loaded, keeping preprocessing memory bounded to one shard.
        episodes = self._load_shard(shard)
        for episode_index, episode_requests in groupby(requests, key=lambda pair: pair[1].episode):
            episode_requests = list(episode_requests)
            for offset in range(0, len(episode_requests), 256):
                chunk = episode_requests[offset : offset + 256]
                rows = torch.tensor([row for row, _ in chunk], dtype=torch.long, device="cpu")
                samples = _episode_samples(
                    episodes[episode_index], [item for _, item in chunk], self.horizon, float_height=False,
                )
                for name, value in samples.items():
                    storage[name].index_copy_(0, rows, value.to(storage[name].dtype))
                progress.update(chunk[-1][0] + 1)

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        """Fetch a DataLoader batch with at most one load per requested shard.

        A random batch typically spans more shards than the LRU cache can hold.
        Loading in sample order would repeatedly deserialize and validate whole
        shards. Group only the reads; preserve sampler order and duplicates in
        the returned samples, including with spawned DataLoader workers.
        """
        if not indices:
            return []
        if self._prepared_samples is not None:
            # Advanced indexing copies the batch so callers cannot mutate the
            # resident cache, and batches do not pin an old round's full storage.
            samples = {name: values[indices] for name, values in self._prepared_samples.items()}
            samples["height_map"] = samples["height_map"].float()
            return [{name: values[row] for name, values in samples.items()} for row in range(len(indices))]

        groups: dict[int, dict[int, list[tuple[int, WindowIndex]]]] = {}
        for position, index in enumerate(indices):
            item = self.indices[index]
            groups.setdefault(item.shard, {}).setdefault(item.episode, []).append((position, item))

        output: dict[int, dict[str, torch.Tensor]] = {}
        for shard, episode_groups in groups.items():
            episodes = self._load_shard(shard)
            for episode_index, requests in episode_groups.items():
                samples = _episode_samples(episodes[episode_index], [item for _, item in requests], self.horizon)
                for row, (position, _) in enumerate(requests):
                    output[position] = {name: values[row] for name, values in samples.items()}
        return [output[position] for position in range(len(indices))]

    @property
    def collision_flags(self) -> torch.Tensor:
        return torch.tensor([item.collision for item in self.indices], dtype=torch.bool, device="cpu")

    @property
    def low_motion_flags(self) -> torch.Tensor:
        return torch.tensor([item.low_motion for item in self.indices], dtype=torch.bool, device="cpu")


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
    weights = torch.ones(len(dataset), dtype=torch.double, device="cpu")
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
