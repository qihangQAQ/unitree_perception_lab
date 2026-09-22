"""Stateful one-policy-step delayed navigation collision termination."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .observations import navigation_collision


def navigation_contact_delayed(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    now = navigation_collision(env, sensor_cfg=sensor_cfg, threshold=threshold)
    if not hasattr(env, "_fdm_previous_collision"):
        env._fdm_previous_collision = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    delayed = env._fdm_previous_collision.clone()
    env._fdm_previous_collision.copy_(now)
    return delayed
