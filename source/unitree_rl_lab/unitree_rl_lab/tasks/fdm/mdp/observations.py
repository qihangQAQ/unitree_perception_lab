"""FDM map extraction and collision observations."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.fdm.utils.height_map import door_aware_height_map


def fdm_height_map_with_invalid(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("fdm_height_scanner"),
    shape: tuple[int, int] = (60, 46),
    offset: float = 0.5,
    clip: tuple[float, float] = (-1.0, 1.5),
    invalid_sentinel: float = 1.5,
    door_probe_height: float = 0.5,
    door_height_threshold: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = env.scene.sensors[sensor_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device)
    return door_aware_height_map(
        sensor,
        env_ids,
        shape=shape,
        offset=offset,
        clip=clip,
        invalid_sentinel=invalid_sentinel,
        door_probe_height=door_probe_height,
        door_height_threshold=door_height_threshold,
    )


def fdm_height_map(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("fdm_height_scanner"),
    shape: tuple[int, int] = (60, 46),
    offset: float = 0.5,
    clip: tuple[float, float] = (-1.0, 1.5),
    invalid_sentinel: float = 1.5,
    door_probe_height: float = 0.5,
    door_height_threshold: float = 1.25,
) -> torch.Tensor:
    """Return the door-aware FDM map as ``[N, 1, 60, 46]``."""

    height, _ = fdm_height_map_with_invalid(
        env,
        sensor_cfg=sensor_cfg,
        shape=shape,
        offset=offset,
        clip=clip,
        invalid_sentinel=invalid_sentinel,
        door_probe_height=door_probe_height,
        door_height_threshold=door_height_threshold,
    )
    return height


def fdm_height_map_invalid(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("fdm_height_scanner"),
    shape: tuple[int, int] = (60, 46),
    offset: float = 0.5,
    clip: tuple[float, float] = (-1.0, 1.5),
    invalid_sentinel: float = 1.5,
    door_probe_height: float = 0.5,
    door_height_threshold: float = 1.25,
) -> torch.Tensor:
    _, invalid = fdm_height_map_with_invalid(
        env,
        sensor_cfg=sensor_cfg,
        shape=shape,
        offset=offset,
        clip=clip,
        invalid_sentinel=invalid_sentinel,
        door_probe_height=door_probe_height,
        door_height_threshold=door_height_threshold,
    )
    return invalid


def navigation_collision(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.linalg.vector_norm(forces, dim=-1) > threshold, dim=-1)
