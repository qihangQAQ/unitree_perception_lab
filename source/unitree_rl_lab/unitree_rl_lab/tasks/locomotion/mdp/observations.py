from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs import ManagerBasedEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return the independent contact state of every selected foot."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norms = torch.linalg.vector_norm(forces, dim=-1)
    return (torch.max(force_norms, dim=1).values > threshold).float()


def current_feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_to_vertical_ratio: float = 0.5,
) -> torch.Tensor:
    """Return whether each selected foot currently experiences a stumbling contact."""
    if horizontal_to_vertical_ratio < 0.0:
        raise ValueError(
            "horizontal_to_vertical_ratio must be non-negative, got "
            f"{horizontal_to_vertical_ratio}."
        )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    horizontal_force = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    vertical_force = torch.abs(forces[..., 2])
    return (horizontal_force > horizontal_to_vertical_ratio * vertical_force).float()


def _resolved_indices(indices: slice | list[int], count: int) -> tuple[int, ...]:
    """Convert resolved scene-entity indices to a hashable tuple."""
    if isinstance(indices, slice):
        return tuple(range(count))[indices]
    return tuple(int(index) for index in indices)


def _privileged_property_cache(env: ManagerBasedEnv) -> dict[tuple, torch.Tensor]:
    """Return the per-environment cache for startup-only physics properties."""
    cache = getattr(env, "_unitree_privileged_property_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_unitree_privileged_property_cache", cache)
    return cache


def payload(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the randomized mass delta of the selected payload bodies."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = _resolved_indices(asset_cfg.body_ids, asset.num_bodies)
    cache_key = ("payload", asset_cfg.name, body_ids)
    cache = _privileged_property_cache(env)
    if cache_key not in cache:
        # PhysX properties may live on CPU even when the simulation runs on CUDA.
        # Compute the delta on one device first, as Noetix does, and only then move
        # the privileged observation to the environment device.
        current_mass = asset.root_physx_view.get_masses()
        default_mass = asset.data.default_mass.to(
            device=current_mass.device,
            dtype=current_mass.dtype,
        )
        mass_delta = current_mass[:, body_ids] - default_mass[:, body_ids]
        cache[cache_key] = mass_delta.to(device=env.device)
    return cache[cache_key]


def material_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return mean static friction, dynamic friction, and restitution per selected body.

    Material randomization is a startup event in perception-pro, so the result is cached
    after its first observation. Averaging a body's collision shapes keeps three values
    per body even when a link owns more than one collision shape.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolved_indices(asset_cfg.body_ids, asset.num_bodies)
    cache_key = ("material_properties", asset_cfg.name, body_ids)
    cache = _privileged_property_cache(env)
    if cache_key not in cache:
        shape_counts = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_view = asset._physics_sim_view.create_rigid_body_view(link_path)
            shape_counts.append(link_view.max_shapes)

        if sum(shape_counts) != asset.root_physx_view.max_shapes:
            raise RuntimeError(
                "Failed to map articulation bodies to collision shapes for privileged "
                "material observations."
            )

        materials = asset.root_physx_view.get_material_properties()
        shape_offsets = [0]
        for shape_count in shape_counts:
            shape_offsets.append(shape_offsets[-1] + shape_count)

        body_materials = []
        for body_id in body_ids:
            shape_start = shape_offsets[body_id]
            shape_end = shape_offsets[body_id + 1]
            if shape_start == shape_end:
                raise RuntimeError(
                    f"Body {asset.body_names[body_id]!r} has no collision shapes."
                )
            body_materials.append(materials[:, shape_start:shape_end, :].mean(dim=1))

        cache[cache_key] = torch.cat(body_materials, dim=-1).to(env.device)
    return cache[cache_key]


def kp_params(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the current actuator stiffness gains for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness[:, asset_cfg.joint_ids]


def kd_params(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the current actuator damping gains for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping[:, asset_cfg.joint_ids]


def effort_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the current simulated effort limits for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_effort_limits[:, asset_cfg.joint_ids]


def height_scan_hpc(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.78,  # 默认值改为 G1 的合理站立高度
) -> torch.Tensor:
    """
    获取基于传感器坐标系的高度扫描图，并包含 NaN/Inf 工业级清洗。
    """
    # 提取传感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 1. 原始物理计算：地形相对高度 = 传感器Z - 击中点Z - 目标离地偏移量
    # 结果含义：0.0 代表完美平地，正数代表脚下有坑（击中点低），负数代表踩到台阶（击中点高）
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # 2.  核心防护：清洗 NaN 和 Inf (防御物理引擎异常或射线射穿地图)
    # nan=0.0: 如果没测到，保守假设脚下是平地
    # posinf/neginf: 限制在一个物理上不可能达到的极限值，后续会被 config 中的 clip 截断
    heights = torch.nan_to_num(heights, nan=0.0, posinf=10.0, neginf=-10.0)

    return heights
