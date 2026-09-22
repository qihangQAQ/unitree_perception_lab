"""Versioned, reset-safe rollout trajectory schema."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any

import torch

SCHEMA_VERSION = 3


class Split(IntEnum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    @classmethod
    def from_name(cls, name: str) -> "Split":
        return {"train": cls.TRAIN, "val": cls.VALIDATION, "test": cls.TEST}[name]


class TerminationReason(IntEnum):
    NONE = 0
    COLLISION = 1
    TIMEOUT = 2
    WATCHDOG = 3
    INITIALIZATION = 4


@dataclass
class EpisodeData:
    """One episode containing command-boundary frames and optional event frames.

    ``state_history_raw`` and ``proprio_history`` are newest-first along their
    history dimension. ``command_plan[t]`` is generated before executing the
    command at frame ``t`` and is never edited after a future collision.
    """

    state_history_raw: torch.Tensor
    proprio_history: torch.Tensor
    history_timestamps: torch.Tensor
    height_map: torch.Tensor
    height_map_invalid: torch.Tensor
    command: torch.Tensor
    command_plan: torch.Tensor
    has_outgoing_command: torch.Tensor
    timestamp: torch.Tensor
    delta_t: torch.Tensor
    collision_now: torch.Tensor
    contact_groups: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    termination_reason: torch.Tensor
    episode_id: torch.Tensor
    usd_origin_id: torch.Tensor
    usd_region_split: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.state_history_raw.shape[0])

    def validate(self, *, history_length: int = 10, horizon: int = 10) -> None:
        expected_tail_shapes = {
            "state_history_raw": (history_length, 8),
            "proprio_history": (history_length, 96),
            "history_timestamps": (history_length,),
            "height_map": (1, 60, 46),
            "height_map_invalid": (1, 60, 46),
            "command": (3,),
            "command_plan": (horizon, 3),
            "has_outgoing_command": (),
            "timestamp": (),
            "delta_t": (),
            "collision_now": (),
            "contact_groups": (3,),
            "terminated": (),
            "truncated": (),
            "termination_reason": (),
            "episode_id": (),
            "usd_origin_id": (),
            "usd_region_split": (),
        }
        if self.num_frames < 1:
            raise ValueError("An episode cannot be empty.")
        for name, tail_shape in expected_tail_shapes.items():
            tensor = getattr(self, name)
            expected = (self.num_frames, *tail_shape)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}.")
        if self.state_history_raw.dtype != torch.float32 or self.proprio_history.dtype != torch.float32:
            raise TypeError("State and proprioception histories must be float32.")
        if self.height_map.dtype not in (torch.float16, torch.float32):
            raise TypeError("height_map must be float16 or float32.")
        if self.command.dtype != torch.float32 or self.command_plan.dtype != torch.float32:
            raise TypeError("Commands must be float32.")
        for name in ("height_map_invalid", "has_outgoing_command", "collision_now", "terminated", "truncated"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be bool.")
        if torch.unique(self.episode_id).numel() != 1:
            raise ValueError("A shard episode contains more than one episode_id.")
        if torch.unique(self.usd_origin_id).numel() != 1 or torch.unique(self.usd_region_split).numel() != 1:
            raise ValueError("An episode cannot cross a USD origin or split.")
        if not torch.all(self.timestamp[1:] >= self.timestamp[:-1]):
            raise ValueError("Frame timestamps must be monotonic.")
        if not torch.all(self.history_timestamps[:, :-1] >= self.history_timestamps[:, 1:]):
            raise ValueError("History timestamps must be newest-first.")
        if torch.any(self.delta_t < 0.0):
            raise ValueError("delta_t cannot be negative.")
        event_frames = self.collision_now | self.terminated | self.truncated
        if torch.any(self.has_outgoing_command & event_frames):
            raise ValueError("Terminal/event frames cannot own an outgoing command.")

    def to_dict(self) -> dict[str, torch.Tensor]:
        self.validate()
        return {field.name: getattr(self, field.name).detach().cpu().contiguous() for field in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, validate: bool = True) -> "EpisodeData":
        missing = {field.name for field in fields(cls)} - data.keys()
        if missing:
            raise ValueError(f"Episode is missing schema fields: {sorted(missing)}")
        episode = cls(**{field.name: data[field.name] for field in fields(cls)})
        if validate:
            episode.validate()
        return episode


class EpisodeBuilder:
    """Accumulate per-frame tensors and finalize one validated episode."""

    def __init__(self) -> None:
        self._frames: list[dict[str, torch.Tensor]] = []

    def append(self, **frame: torch.Tensor) -> None:
        expected = {field.name for field in fields(EpisodeData)}
        if set(frame) != expected:
            raise ValueError(
                f"Frame keys differ from schema: missing={expected - set(frame)}, extra={set(frame) - expected}."
            )
        self._frames.append({name: value.detach().cpu() for name, value in frame.items()})

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def mark_last_truncated(self, reason: TerminationReason) -> None:
        """Make the latest valid observation a non-start terminal marker."""

        if not self._frames:
            return
        frame = self._frames[-1]
        frame["has_outgoing_command"] = torch.tensor(False)
        frame["truncated"] = torch.tensor(True)
        frame["termination_reason"] = torch.tensor(int(reason), dtype=torch.int8)

    def finalize(self) -> EpisodeData:
        if not self._frames:
            raise ValueError("Cannot finalize an empty episode.")
        episode = EpisodeData(
            **{name: torch.stack([frame[name] for frame in self._frames]) for name in self._frames[0]}
        )
        episode.validate()
        return episode
