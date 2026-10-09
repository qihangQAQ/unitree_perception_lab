"""Bounded CPU frame storage; episode boundaries never become synthetic transitions."""

from __future__ import annotations

from math import prod

import torch

from .schema import EpisodeData, TerminationReason


def frame_spec() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        "state_history_raw": ((10, 8), torch.float32),
        "proprio_history": ((10, 96), torch.float32),
        "history_timestamps": ((10,), torch.float64),
        "height_map": ((1, 60, 46), torch.float16),
        "height_map_invalid": ((1, 60, 46), torch.bool),
        "command": ((3,), torch.float32),
        "command_plan": ((10, 3), torch.float32),
        "has_outgoing_command": ((), torch.bool),
        "timestamp": ((), torch.float64),
        "delta_t": ((), torch.float32),
        "collision_now": ((), torch.bool),
        "contact_groups": ((3,), torch.bool),
        "terminated": ((), torch.bool),
        "truncated": ((), torch.bool),
        "termination_reason": ((), torch.int8),
        "episode_id": ((), torch.int64),
        "usd_origin_id": ((), torch.int32),
        "usd_region_split": ((), torch.int8),
    }


class FixedFrameBuffer:
    """One capacity per environment, shared across its episodes within a round.

    append() transfers one tensor per field for all due environments. Closed
    episodes are copied out, so neither a writer nor torch.save retains the
    full backing storage. The buffer can be reset/reused or released for training.
    """

    def __init__(self, num_envs: int, capacity: int):
        if num_envs < 1 or capacity < 1:
            raise ValueError("Buffer dimensions must be positive.")
        self.num_envs, self.capacity = num_envs, capacity
        self.storage = {
            name: torch.empty((num_envs, capacity, *shape), dtype=dtype, device="cpu")
            for name, (shape, dtype) in frame_spec().items()
        }
        self.fill = torch.zeros(num_envs, dtype=torch.long, device="cpu")
        self.starts = torch.zeros_like(self.fill)
        self.open = torch.zeros(num_envs, dtype=torch.bool, device="cpu")

    @staticmethod
    def estimate_bytes(num_envs: int, capacity: int) -> int:
        return num_envs * capacity * sum(
            prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
            for shape, dtype in frame_spec().values()
        )

    @property
    def frames(self) -> int:
        return int(self.fill.sum())

    @property
    def pending_frames(self) -> int:
        return int((self.fill - self.starts)[self.open].sum())

    def reset(self) -> None:
        # Uninitialized/old slots are never exposed: all reads use fill/starts.
        self.fill.zero_()
        self.starts.zero_()
        self.open.zero_()

    def begin(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(device="cpu", dtype=torch.long)
        if torch.any(self.open[ids]) or torch.any(self.fill[ids] >= self.capacity):
            raise RuntimeError("Cannot begin an episode in an open or full buffer slot.")
        self.starts[ids] = self.fill[ids]
        self.open[ids] = True

    def append(self, env_ids: torch.Tensor, **frame: torch.Tensor) -> None:
        ids = env_ids.to(device="cpu", dtype=torch.long)
        if len(ids) == 0:
            return
        if len(torch.unique(ids)) != len(ids):
            raise ValueError("A frame batch must contain distinct environments.")
        if set(frame) != set(self.storage):
            raise ValueError("Frame batch fields differ from the episode schema.")
        if not torch.all(self.open[ids]) or torch.any(self.fill[ids] >= self.capacity):
            raise RuntimeError("Attempted to append to a closed or full buffer slot.")
        slots = self.fill[ids]
        for name, target in self.storage.items():
            value = frame[name]
            if value.shape != (len(ids), *target.shape[2:]):
                raise ValueError(f"Unexpected batched shape for {name}: {value.shape}.")
            # Synchronous batch copy owns the values before the simulator advances.
            target[ids, slots] = value.detach().to(device="cpu", dtype=target.dtype)
        self.fill[ids] += 1

    def finish(self, env_id: int, reason: TerminationReason | None = None) -> EpisodeData | None:
        if not self.open[env_id]:
            return None
        start, end = int(self.starts[env_id]), int(self.fill[env_id])
        self.open[env_id] = False
        self.starts[env_id] = end
        if end == start:
            return None
        if reason is not None and not self.storage["collision_now"][env_id, end - 1]:
            self.storage["has_outgoing_command"][env_id, end - 1] = False
            self.storage["truncated"][env_id, end - 1] = True
            self.storage["termination_reason"][env_id, end - 1] = int(reason)
        episode = EpisodeData(**{name: values[env_id, start:end].clone() for name, values in self.storage.items()})
        episode.validate()
        return episode


class CollectionBudget:
    """Stop full rounds, or slow tails after enough *real* capacity is filled."""

    def __init__(self, target: int, min_fill: float = 0.95, tail_factor: float = 1.5):
        if target < 1 or not 0 < min_fill <= 1 or tail_factor <= 0:
            raise ValueError("Invalid fixed collection budget.")
        self.target, self.min_fill, self.tail_factor = target, min_fill, tail_factor
        self.next_progress = 0.1
        self.last_progress_s = 0.0
        self.intervals: list[float] = []

    def stop_reason(self, frames: int, elapsed: float) -> str | None:
        ratio = frames / self.target
        if ratio >= 1:
            return "capacity"
        if ratio >= self.next_progress and self.next_progress <= 0.9 + 1e-6:
            # Batched updates may cross more than one milestone at once.
            self.intervals.append(elapsed - self.last_progress_s)
            self.last_progress_s = elapsed
            while self.next_progress <= ratio:
                self.next_progress += 0.1
        if ratio >= self.min_fill and self.intervals:
            average = sum(self.intervals) / len(self.intervals)
            if elapsed - self.last_progress_s > self.tail_factor * max(average, 1e-6):
                return "slow_tail"
        return None
