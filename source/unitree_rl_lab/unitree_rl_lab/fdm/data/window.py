"""Shared, batched target construction for training and dataset diagnostics."""

from __future__ import annotations

import torch

from ..utils.se2 import relative_pose_sequence
from .schema import EpisodeData


def window_targets(episode: EpisodeData, starts: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
    """Freeze targets at the first collision and mask unavailable futures."""
    offsets = torch.arange(1, horizon + 1, device=starts.device)
    indices = starts[:, None] + offsets
    exists = indices < episode.num_frames
    indices = indices.clamp_max(episode.num_frames - 1)
    collisions = episode.collision_now[indices] & exists
    has_collision = collisions.any(dim=1)
    first_collision = collisions.long().argmax(dim=1)
    after_collision = has_collision[:, None] & (torch.arange(horizon, device=starts.device) >= first_collision[:, None])
    frozen_indices = indices.gather(1, first_collision[:, None])
    indices = torch.where(after_collision, frozen_indices, indices)
    # Missing non-collision futures use the anchor pose and are masked out.
    indices = torch.where(exists | after_collision, indices, starts[:, None])
    current = episode.state_history_raw[starts, 0]
    future = episode.state_history_raw[indices, 0]
    poses = torch.cat((current[:, None], future), dim=1)
    return {
        "future_commands": episode.command_plan[starts].float(),
        "future_pose": relative_pose_sequence(poses[..., :3], poses[..., 3:7], anchor=0)[:, 1:].float(),
        "future_collision": after_collision.float(),
        "valid_mask": exists | after_collision,
    }
