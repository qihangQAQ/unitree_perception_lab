"""Batched SE(2) transforms used by the dataset and model."""

from __future__ import annotations

import torch


def yaw_from_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract yaw from xyzw quaternions without assuming unit sign."""

    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def relative_pose_sequence(position_w: torch.Tensor, quaternion_xyzw_w: torch.Tensor, anchor: int = 0) -> torch.Tensor:
    """Express a pose sequence relative to the anchor pose.

    Args:
        position_w: Tensor shaped ``[..., T, 3]``.
        quaternion_xyzw_w: Tensor shaped ``[..., T, 4]``.
        anchor: Sequence index used as the local origin.

    Returns:
        Tensor ``[..., T, 4]`` with ``x, y, sin(yaw), cos(yaw)``.
    """

    origin_xy = position_w[..., anchor, :2]
    yaw = yaw_from_quaternion_xyzw(quaternion_xyzw_w)
    origin_yaw = yaw[..., anchor]
    delta = position_w[..., :, :2] - origin_xy.unsqueeze(-2)
    cosine = torch.cos(origin_yaw).unsqueeze(-1)
    sine = torch.sin(origin_yaw).unsqueeze(-1)
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    relative_yaw = wrap_angle(yaw - origin_yaw.unsqueeze(-1))
    return torch.stack((local_x, local_y, torch.sin(relative_yaw), torch.cos(relative_yaw)), dim=-1)


def integrate_body_twists(step_twists: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate body-frame ``vx, vy, yaw_rate`` increments into SE(2) poses."""

    if step_twists.shape[-1] != 3:
        raise ValueError(f"Expected twist dimension 3, got {tuple(step_twists.shape)}.")
    yaw = torch.zeros(step_twists.shape[:-2], dtype=step_twists.dtype, device=step_twists.device)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    output = []
    for step in range(step_twists.shape[-2]):
        vx, vy, yaw_rate = step_twists[..., step, :].unbind(dim=-1)
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        x = x + dt * (cosine * vx - sine * vy)
        y = y + dt * (sine * vx + cosine * vy)
        yaw = wrap_angle(yaw + dt * yaw_rate)
        output.append(torch.stack((x, y, torch.sin(yaw), torch.cos(yaw)), dim=-1))
    return torch.stack(output, dim=-2)
