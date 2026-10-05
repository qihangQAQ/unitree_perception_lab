"""Contact decisions shared by FDM rollout labels and environment termination."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def recent_body_contacts(
    force_history: torch.Tensor, *, physics_steps: int, threshold: float
) -> torch.Tensor:
    """Return ``[env, body]`` contacts during the latest policy step.

    Isaac Lab stores the newest physics sample at history index zero. Only the
    ``physics_steps`` samples belonging to this policy step are considered; the
    remaining sensor history may contain contacts from the preceding step.
    """

    if force_history.ndim != 4 or force_history.shape[-1] != 3:
        raise ValueError("Contact force history must have shape [env, history, body, xyz].")
    if physics_steps < 1 or physics_steps > force_history.shape[1]:
        raise ValueError(
            f"Contact history has {force_history.shape[1]} samples, but the policy step needs {physics_steps}."
        )
    recent_forces = force_history[:, :physics_steps]
    return torch.any(torch.linalg.vector_norm(recent_forces, dim=-1) > threshold, dim=1)


def any_body_contact(contacts: torch.Tensor, body_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Reduce per-body contacts over a chosen navigation body group."""

    return torch.any(contacts[:, body_ids], dim=-1)


def clear_collision_delay(latch: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Clear the one-step termination latch for selected reset environments."""

    latch[env_ids] = False


def advance_collision_delay(latch: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
    """Return the preceding step's collisions and latch this step's result."""

    delayed = latch.clone()
    latch.copy_(now)
    return delayed
