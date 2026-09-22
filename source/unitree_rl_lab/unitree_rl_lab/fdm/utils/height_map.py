"""Door-aware large-area height scans for the G1 forward dynamics model."""

from __future__ import annotations

from collections.abc import Callable

import torch


def door_aware_height_map(
    sensor,
    env_ids: torch.Tensor,
    *,
    shape: tuple[int, int] = (60, 46),
    offset: float = 0.5,
    clip: tuple[float, float] = (-1.0, 1.5),
    invalid_sentinel: float = 1.5,
    door_probe_height: float = 0.5,
    door_height_threshold: float = 1.25,
    raycast_fn: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FDM maps and invalid masks after the reference door recognition.

    The reference scans down from above the robot, then probes up and down
    from world height 0.5 m. A lower surface replaces the first downward hit
    when the intervening clearance is large enough. Only the FDM map is
    corrected; the sensor's hits and the frozen locomotion policy stay intact.
    """

    if raycast_fn is None:
        from isaaclab.utils.warp import raycast_mesh

        raycast_fn = raycast_mesh
    if len(sensor.cfg.mesh_prim_paths) != 1:
        raise ValueError("FDM door recognition requires exactly one terrain ray-caster mesh.")

    top_hits = sensor.data.ray_hits_w[env_ids]
    if top_hits.ndim != 3 or top_hits.shape[1:] != (shape[0] * shape[1], 3):
        raise ValueError(f"Expected ray hits [N, {shape[0] * shape[1]}, 3], got {tuple(top_hits.shape)}.")

    probe_origins = top_hits.clone()
    # RayCaster uses (inf, inf, inf) for a miss. Warp must never receive an
    # infinite ray origin; those cells remain invalid and cannot be corrected.
    valid_xy = torch.isfinite(probe_origins[..., :2]).all(dim=-1)
    probe_origins[..., :2] = torch.where(
        valid_xy.unsqueeze(-1), probe_origins[..., :2], torch.zeros_like(probe_origins[..., :2])
    )
    probe_origins[..., 2] = door_probe_height
    directions = torch.zeros_like(probe_origins)
    directions[..., 2] = -1.0
    mesh = sensor.meshes[sensor.cfg.mesh_prim_paths[0]]
    lower_hits = raycast_fn(
        probe_origins, directions, mesh=mesh, max_dist=sensor.cfg.max_distance
    )[0]
    directions[..., 2] = 1.0
    upper_hits = raycast_fn(
        probe_origins, directions, mesh=mesh, max_dist=sensor.cfg.max_distance
    )[0]

    top_z = top_hits[..., 2]
    lower_z = lower_hits[..., 2]
    upper_z = upper_hits[..., 2]
    use_lower_surface = (
        valid_xy
        & torch.isfinite(upper_z)
        & torch.isfinite(lower_z)
        & (upper_z < top_z - 1.0e-3)
        & (upper_z - lower_z > door_height_threshold)
    )
    corrected_z = torch.where(use_lower_surface, lower_z, top_z)
    invalid = ~torch.isfinite(corrected_z)
    height = corrected_z + offset - sensor.data.pos_w[env_ids, 2].unsqueeze(1)
    height = torch.where(invalid, torch.full_like(height, invalid_sentinel), height)
    height = height.clamp(*clip).unflatten(1, shape)
    invalid = invalid.unflatten(1, shape)
    return torch.flip(height, dims=[1]).unsqueeze(1), torch.flip(invalid, dims=[1]).unsqueeze(1)
