"""SSR-style imagined foothold reward and touchdown-label generation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from unitree_rl_lab.utils.warp import raycast_mesh

from .commands.velocity_command import terrain_type_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _support_deficiency_from_heights(
    heights: torch.Tensor,
    sole_height: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Return the unsupported fraction without coupling the MDP to RSL-RL."""

    finite = torch.isfinite(heights)
    supported = finite & ((sole_height - heights) < tolerance)
    return 1.0 - supported.float().mean(dim=-1)


def _write_foothold_events(env: ManagerBasedRLEnv) -> None:
    """Detect post-action touchdowns while state is still available before reset."""

    env_u = env.unwrapped
    post_contact, post_forces = env_u._foothold_contact_state()
    decision_contact = env_u._foothold_decision_contact
    touchdown = post_contact & ~decision_contact
    stumble = torch.linalg.vector_norm(post_forces[..., :2], dim=-1) > (
        4.0 * torch.abs(post_forces[..., 2])
    )
    touchdown_valid = touchdown & (env_u._foothold_decision_air_time >= 0.04) & ~stumble

    robot: Articulation = env.scene["robot"]
    touchdown_xy_w = robot.data.body_pos_w[:, env_u._foothold_body_ids, :2].clone()

    terrain_id = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.int8)
    terrain_id[terrain_type_mask(env, ("stairs_up",))] = 0
    terrain_id[terrain_type_mask(env, ("stairs_down",))] = 1

    collection_terrain_types = getattr(env.cfg, "foothold_collection_terrain_types", None)
    if collection_terrain_types is None:
        collection_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    elif len(collection_terrain_types) == 0:
        collection_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        collection_mask = terrain_type_mask(env, tuple(collection_terrain_types))

    env.extras["foothold_events"] = {
        "decision_swing": (~decision_contact) & collection_mask.unsqueeze(-1),
        "decision_base_xy": env_u._foothold_decision_base_pos_w[:, :2].clone(),
        "decision_base_yaw": env_u._yaw_from_quat(
            env_u._foothold_decision_base_quat_w
        ).clone(),
        "touchdown": touchdown,
        "touchdown_valid": touchdown_valid & collection_mask.unsqueeze(-1),
        "touchdown_xy_w": touchdown_xy_w,
        "terrain_id": terrain_id,
    }


def imagined_foothold_guidance(
    env: ManagerBasedRLEnv,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    patch_size: tuple[float, float] = (0.225, 0.10),
    patch_resolution: float = 0.05,
    support_height_tolerance: float = 0.03,
    sole_offset: float = 0.06,
    reward_sigma: float = 0.25,
    curriculum_level_threshold: int = 3,
    terrain_types: tuple[str, ...] = ("stairs_up", "stairs_down"),
    downstairs_terrain_types: tuple[str, ...] = ("stairs_down",),
    downstairs_scale: float = 1.5,
    cast_height: float = 2.0,
) -> torch.Tensor:
    """Reward support under actual stance feet and imagined swing footholds."""

    env_u = env.unwrapped
    if not getattr(env_u, "_foothold_prediction_valid", False):
        return torch.zeros(env.num_envs, device=env.device)

    _write_foothold_events(env)

    active_mask = terrain_type_mask(env, terrain_types) & (
        env.scene.terrain.terrain_levels >= curriculum_level_threshold
    )
    log = env.extras.setdefault("log", {})
    log["Foothold/guidance_active_fraction"] = active_mask.float().mean().detach()
    if not active_mask.any():
        return torch.zeros(env.num_envs, device=env.device)

    active_ids = active_mask.nonzero(as_tuple=False).squeeze(-1)
    robot: Articulation = env.scene["robot"]
    base_pos_w = env_u._foothold_decision_base_pos_w[active_ids]
    base_quat_yaw = math_utils.yaw_quat(env_u._foothold_decision_base_quat_w[active_ids])
    mu = env_u._imagined_foothold_mu[active_ids]
    sigma = env_u._imagined_foothold_sigma[active_ids]
    contact = env_u._foothold_decision_contact[active_ids]

    unit_sigma_points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=env.device,
        dtype=mu.dtype,
    )
    sample_weights = torch.exp(-0.5 * torch.sum(torch.square(unit_sigma_points), dim=-1))
    sample_weights /= sample_weights.sum()
    candidate_b = mu.unsqueeze(2) + sigma[..., None, None] * unit_sigma_points

    num_active, num_feet, num_candidates, _ = candidate_b.shape
    candidate_vec_b = torch.nn.functional.pad(candidate_b, (0, 1))
    candidate_quat = base_quat_yaw[:, None, None, :].expand(
        -1,
        num_feet,
        num_candidates,
        -1,
    )
    candidate_w = math_utils.quat_apply(
        candidate_quat.reshape(-1, 4),
        candidate_vec_b.reshape(-1, 3),
    ).reshape(num_active, num_feet, num_candidates, 3)
    candidate_xy_w = candidate_w[..., :2] + base_pos_w[:, None, None, :2]

    foot_pos_w = robot.data.body_pos_w[active_ids][:, env_u._foothold_body_ids]
    candidate_xy_w = torch.where(
        contact[..., None, None],
        foot_pos_w[..., None, :2].expand_as(candidate_xy_w),
        candidate_xy_w,
    )

    nx = max(2, int(round(patch_size[0] / patch_resolution)) + 1)
    ny = max(2, int(round(patch_size[1] / patch_resolution)) + 1)
    patch_x = torch.linspace(-0.5 * patch_size[0], 0.5 * patch_size[0], nx, device=env.device)
    patch_y = torch.linspace(-0.5 * patch_size[1], 0.5 * patch_size[1], ny, device=env.device)
    grid_x, grid_y = torch.meshgrid(patch_x, patch_y, indexing="ij")
    patch_b = torch.stack(
        (grid_x.flatten(), grid_y.flatten(), torch.zeros(nx * ny, device=env.device)),
        dim=-1,
    )
    patch_quat = base_quat_yaw[:, None, :].expand(-1, nx * ny, -1)
    patch_w = math_utils.quat_apply(
        patch_quat.reshape(-1, 4),
        patch_b.unsqueeze(0).expand(num_active, -1, -1).reshape(-1, 3),
    ).reshape(num_active, nx * ny, 3)[..., :2]

    ray_xy = candidate_xy_w[..., None, :] + patch_w[:, None, None, :, :]
    ray_starts = torch.empty(*ray_xy.shape[:-1], 3, device=env.device)
    ray_starts[..., :2] = ray_xy
    ray_starts[..., 2] = base_pos_w[:, None, None, None, 2] + cast_height
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., 2] = -1.0

    scanner: RayCaster = env.scene.sensors[left_foot_height_scanner_cfg.name]
    mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    raycast_start = time.perf_counter()
    ray_hits, _, _, _ = raycast_mesh(
        ray_starts.reshape(-1, 3),
        ray_directions.reshape(-1, 3),
        mesh,
        max_dist=2.0 * cast_height,
    )
    raycast_time_ms = (time.perf_counter() - raycast_start) * 1_000.0
    heights = ray_hits[:, 2].reshape(
        num_active,
        num_feet,
        num_candidates,
        nx * ny,
    )
    finite = torch.isfinite(heights)
    imagined_sole_height = torch.where(
        finite,
        heights,
        torch.full_like(heights, -torch.inf),
    ).amax(dim=-1, keepdim=True)
    stance_sole_height = foot_pos_w[..., 2, None, None] - sole_offset
    sole_height = torch.where(
        contact[..., None, None],
        stance_sole_height,
        imagined_sole_height,
    )

    deficiency = _support_deficiency_from_heights(
        heights,
        sole_height,
        support_height_tolerance,
    )
    expected_deficiency = torch.sum(deficiency * sample_weights, dim=-1)
    stance_deficiency = deficiency[..., 0]
    per_foot_deficiency = torch.where(contact, stance_deficiency, expected_deficiency)
    reward = torch.exp(-torch.square(per_foot_deficiency.sum(dim=-1)) / reward_sigma)

    down_mask = terrain_type_mask(env, downstairs_terrain_types, env_ids=active_ids)
    reward *= torch.where(
        down_mask,
        torch.as_tensor(downstairs_scale, device=env.device),
        1.0,
    )
    full_reward = torch.zeros(env.num_envs, device=env.device)
    full_reward[active_ids] = reward

    log["Foothold/support_deficiency"] = per_foot_deficiency.mean().detach()
    log["Foothold/predicted_sigma"] = sigma.mean().detach()
    log["Foothold/raycast_time_ms"] = raycast_time_ms
    log["Foothold/rays_per_step"] = float(ray_starts.numel() // 3)
    if down_mask.any():
        log["Foothold/downstairs_deficiency"] = per_foot_deficiency[down_mask].mean().detach()
    return full_reward
