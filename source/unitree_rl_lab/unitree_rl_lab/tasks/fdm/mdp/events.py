"""FDM reset helpers and collision-delay state reset."""

from __future__ import annotations

import torch

from unitree_rl_lab.fdm.utils.contact import clear_collision_delay


def clear_fdm_collision_delay(env, env_ids: torch.Tensor) -> None:
    if hasattr(env, "_fdm_previous_collision"):
        clear_collision_delay(env._fdm_previous_collision, env_ids)
