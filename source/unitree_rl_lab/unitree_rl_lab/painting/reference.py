"""Pure tensor timing, frame transforms and mutually exclusive ending conditions."""

from __future__ import annotations

import torch

from .trajectories import interpolate_path


def rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply a unit wxyz quaternion, with broadcast-compatible leading dimensions."""
    qxyz = quaternion[..., 1:].expand_as(vector)
    uv = torch.cross(qxyz, vector, dim=-1)
    return vector + 2 * (quaternion[..., :1] * uv + torch.cross(qxyz, uv, dim=-1))


def inverse_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return rotate(torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1), vector)


def reference_at_time(points, arc, lengths, speed, distance, elapsed, prepare_time, lookahead):
    progress = (speed * (elapsed - prepare_time).clamp_min(0)).minimum(lengths)
    moving = (elapsed >= prepare_time) & (progress < lengths)
    command_speed = torch.where(moving, speed, 0.0)
    queries = (progress[:, None] + command_speed[:, None] * lookahead[None]).minimum(lengths[:, None])
    surface, tangent = interpolate_path(points, arc, queries)
    target = surface.clone()
    target[..., 0] -= distance[:, None]
    velocity = tangent[:, 0] * command_speed[:, None]
    return target, velocity, progress, command_speed


def trajectory_observation(target_world, base_pos, base_quat, command_speed):
    relative = inverse_rotate(base_quat[:, None], target_world - base_pos[:, None])
    direction = torch.zeros_like(base_pos)
    direction[:, 0] = 1
    direction = inverse_rotate(base_quat, direction)
    return torch.cat((relative.flatten(1), direction, command_speed[:, None]), dim=-1)


def ending_masks(tilt, progress, length, endpoint_error, elapsed, deadline, tilt_limit=0.8, tolerance=0.03):
    bad = tilt > tilt_limit
    success = ~bad & (progress >= length) & (endpoint_error < tolerance)
    timeout = ~bad & ~success & (elapsed >= deadline)
    return bad, success, timeout
