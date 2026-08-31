"""Locomotion-specific event functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

from .commands.velocity_command import terrain_type_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_uniform_terrain_aware(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    terrain_specific_pos_range: dict[
        str, dict[str, tuple[float, float]]
    ]
    | None = None,
    terrain_specific_yaw: dict[str, tuple[float, ...]] | None = None,
    terrain_specific_velocity_range: dict[
        str, dict[str, tuple[float, float]]
    ]
    | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root state with pose and velocity overrides by assigned terrain."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    root_states = asset.data.default_root_state[env_ids].clone()

    pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
        device=asset.device,
    )
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0],
        pose_ranges[:, 1],
        (env_ids.numel(), len(pose_keys)),
        device=asset.device,
    )

    for terrain_type, position_ranges in (
        terrain_specific_pos_range or {}
    ).items():
        terrain_mask = terrain_type_mask(
            env,
            (terrain_type,),
            env_ids,
            use_assigned_terrain=True,
        )
        count = int(terrain_mask.sum().item())
        if count == 0:
            continue
        for component_index, component_name in enumerate(pose_keys[:3]):
            if component_name not in position_ranges:
                continue
            component_range = position_ranges[component_name]
            pose_samples[terrain_mask, component_index] = math_utils.sample_uniform(
                float(component_range[0]),
                float(component_range[1]),
                (count,),
                device=asset.device,
            )

    for terrain_type, yaw_choices in (terrain_specific_yaw or {}).items():
        if not yaw_choices:
            raise ValueError(
                f"Terrain-specific yaw choices for {terrain_type!r} cannot be empty."
            )
        terrain_mask = terrain_type_mask(
            env,
            (terrain_type,),
            env_ids,
            use_assigned_terrain=True,
        )
        count = int(terrain_mask.sum().item())
        if count == 0:
            continue
        yaw_values = torch.tensor(yaw_choices, device=asset.device)
        yaw_indices = torch.randint(
            0,
            len(yaw_choices),
            (count,),
            device=asset.device,
        )
        pose_samples[terrain_mask, 5] = yaw_values[yaw_indices]

    positions = (
        root_states[:, 0:3]
        + env.scene.env_origins[env_ids]
        + pose_samples[:, 0:3]
    )
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(
        root_states[:, 3:7], orientations_delta
    )

    velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys],
        device=asset.device,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0],
        velocity_ranges[:, 1],
        (env_ids.numel(), len(velocity_keys)),
        device=asset.device,
    )

    for terrain_type, component_ranges in (
        terrain_specific_velocity_range or {}
    ).items():
        terrain_mask = terrain_type_mask(
            env,
            (terrain_type,),
            env_ids,
            use_assigned_terrain=True,
        )
        count = int(terrain_mask.sum().item())
        if count == 0:
            continue
        for component_index, component_name in enumerate(velocity_keys):
            if component_name not in component_ranges:
                continue
            component_range = component_ranges[component_name]
            velocity_samples[terrain_mask, component_index] = (
                math_utils.sample_uniform(
                    float(component_range[0]),
                    float(component_range[1]),
                    (count,),
                    device=asset.device,
                )
            )

    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def push_by_setting_velocity_terrain_specific(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    terrain_specific_velocity_range: dict[
        str, dict[str, tuple[float, float]]
    ]
    | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply velocity pushes with optional overrides by assigned terrain."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys],
        device=asset.device,
    )
    velocity_deltas = math_utils.sample_uniform(
        velocity_ranges[:, 0],
        velocity_ranges[:, 1],
        (env_ids.numel(), len(velocity_keys)),
        device=asset.device,
    )

    for terrain_type, component_ranges in (
        terrain_specific_velocity_range or {}
    ).items():
        terrain_mask = terrain_type_mask(
            env,
            (terrain_type,),
            env_ids,
            use_assigned_terrain=True,
        )
        count = int(terrain_mask.sum().item())
        if count == 0:
            continue
        terrain_ranges = torch.tensor(
            [
                component_ranges.get(
                    key, velocity_range.get(key, (0.0, 0.0))
                )
                for key in velocity_keys
            ],
            device=asset.device,
        )
        velocity_deltas[terrain_mask] = math_utils.sample_uniform(
            terrain_ranges[:, 0],
            terrain_ranges[:, 1],
            (count, len(velocity_keys)),
            device=asset.device,
        )

    velocities = asset.data.root_vel_w[env_ids].clone() + velocity_deltas
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def register_virtual_obstacle_to_sensor(
    env,
    env_ids: torch.Tensor | None,
    sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg,
):
    """Register the terrain's analytical virtual obstacles with compatible sensors."""
    del env_ids
    if isinstance(sensor_cfgs, SceneEntityCfg):
        sensor_cfgs = [sensor_cfgs]

    virtual_obstacles = env.scene.terrain.virtual_obstacles
    for sensor_cfg in sensor_cfgs:
        sensor = env.scene[sensor_cfg.name]
        if not hasattr(sensor, "register_virtual_obstacles"):
            raise ValueError(
                f"Sensor '{sensor_cfg.name}' does not support virtual obstacles."
            )
        sensor.register_virtual_obstacles(virtual_obstacles)


def clear_privileged_property_cache(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
):
    """Discard physics-property values sampled before startup randomization."""
    del env_ids
    cache = getattr(env, "_unitree_privileged_property_cache", None)
    if cache is not None:
        cache.clear()


def randomize_camera_mount_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    orientation_range: dict[str, tuple[float, float]],
):
    """Randomize a ray-caster camera's mounting orientation per environment.

    The sampled roll, pitch, and yaw offsets are composed with the configured
    nominal camera mount. Sampling is always relative to that nominal mount, so
    repeated calls do not accumulate orientation drift.
    """
    camera = env.scene.sensors[sensor_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=camera.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=camera.device, dtype=torch.long)

    initial_offset_quat = getattr(
        camera, "_unitree_initial_offset_quat", None
    )
    if initial_offset_quat is None:
        initial_offset_quat = camera._offset_quat.clone()
        camera._unitree_initial_offset_quat = initial_offset_quat

    angle_ranges = torch.tensor(
        [
            orientation_range.get(axis, (0.0, 0.0))
            for axis in ("roll", "pitch", "yaw")
        ],
        device=camera.device,
        dtype=initial_offset_quat.dtype,
    )
    angle_offsets = math_utils.sample_uniform(
        angle_ranges[:, 0],
        angle_ranges[:, 1],
        (env_ids.numel(), 3),
        device=camera.device,
    )
    orientation_delta = math_utils.quat_from_euler_xyz(
        angle_offsets[:, 0], angle_offsets[:, 1], angle_offsets[:, 2]
    )
    camera._offset_quat[env_ids] = math_utils.quat_mul(
        initial_offset_quat[env_ids], orientation_delta
    )


def randomize_actuator_effort_limit(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_limit_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize simulated effort limits from their startup values.

    Isaac Lab 2.3 does not provide an effort-limit counterpart to
    ``randomize_actuator_gains``. This implementation keeps the actuator buffers,
    articulation data, and PhysX limits synchronized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    default_limits = getattr(asset, "_unitree_default_joint_effort_limits", None)
    if default_limits is None:
        default_limits = asset.data.joint_effort_limits.clone()
        setattr(asset, "_unitree_default_joint_effort_limits", default_limits)

    if isinstance(asset_cfg.joint_ids, slice):
        selected_joint_ids = None
    else:
        selected_joint_ids = torch.as_tensor(
            asset_cfg.joint_ids, device=asset.device, dtype=torch.long
        )

    for actuator in asset.actuators.values():
        if isinstance(actuator.joint_indices, slice):
            global_joint_ids: slice | torch.Tensor = slice(None)
            actuator_joint_ids = torch.arange(asset.num_joints, device=asset.device)
        else:
            global_joint_ids = actuator.joint_indices.to(asset.device)
            actuator_joint_ids = global_joint_ids

        if selected_joint_ids is None:
            local_joint_ids: slice | torch.Tensor = slice(None)
        else:
            local_joint_ids = torch.nonzero(
                torch.isin(actuator_joint_ids, selected_joint_ids), as_tuple=False
            ).squeeze(-1)
            if local_joint_ids.numel() == 0:
                continue

        limits = default_limits[env_ids][:, global_joint_ids].clone()
        limits = _randomize_prop_by_op(
            limits,
            effort_limit_distribution_params,
            dim_0_ids=None,
            dim_1_ids=local_joint_ids,
            operation=operation,
            distribution=distribution,
        )
        actuator.effort_limit_sim[env_ids] = limits
        if isinstance(actuator, ImplicitActuator):
            actuator.effort_limit[env_ids] = limits
        asset.write_joint_effort_limit_to_sim(
            limits,
            joint_ids=global_joint_ids,
            env_ids=env_ids,
        )
