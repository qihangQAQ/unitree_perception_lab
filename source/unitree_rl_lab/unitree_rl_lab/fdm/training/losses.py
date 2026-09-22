"""Masked first-stage FDM objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from ..config import TrainCfg


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_value = mask.to(value.dtype)
    while mask_value.ndim < value.ndim:
        mask_value = mask_value.unsqueeze(-1)
    denominator = mask_value.expand_as(value).sum().clamp_min(1.0)
    return (value * mask_value).sum() / denominator


class FDMLoss(nn.Module):
    """Position, heading, cumulative collision, and post-collision stop loss."""

    def __init__(self, cfg: TrainCfg | None = None, *, collision_pos_weight: float | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TrainCfg()
        if collision_pos_weight is None:
            self.register_buffer("collision_pos_weight", None)
        else:
            self.register_buffer("collision_pos_weight", torch.tensor(float(collision_pos_weight)))

    def forward(
        self, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        predicted_pose = prediction["future_pose"]
        target_pose = target["future_pose"]
        valid = target["valid_mask"].bool()
        position = _masked_mean(
            functional.smooth_l1_loss(predicted_pose[..., :2], target_pose[..., :2], reduction="none"), valid
        )
        heading = _masked_mean(
            functional.smooth_l1_loss(predicted_pose[..., 2:4], target_pose[..., 2:4], reduction="none"), valid
        )
        collision_elementwise = functional.binary_cross_entropy_with_logits(
            prediction["collision_logits"],
            target["future_collision"].to(prediction["collision_logits"].dtype),
            pos_weight=self.collision_pos_weight,
            reduction="none",
        )
        collision = _masked_mean(collision_elementwise, valid)

        target_collision = target["future_collision"].bool()
        after_collision = torch.zeros_like(target_collision)
        after_collision[..., 1:] = target_collision[..., :-1]
        after_collision &= valid
        predicted_xy_step = predicted_pose[..., 1:, :2] - predicted_pose[..., :-1, :2]
        predicted_heading = torch.atan2(predicted_pose[..., 2], predicted_pose[..., 3])
        predicted_yaw_step = torch.atan2(
            torch.sin(predicted_heading[..., 1:] - predicted_heading[..., :-1]),
            torch.cos(predicted_heading[..., 1:] - predicted_heading[..., :-1]),
        )
        step_motion = torch.cat((predicted_xy_step, predicted_yaw_step.unsqueeze(-1)), dim=-1)
        stop = _masked_mean(step_motion.square(), after_collision[..., 1:])
        total = (
            self.cfg.position_weight * position
            + self.cfg.heading_weight * heading
            + self.cfg.collision_weight * collision
            + self.cfg.stop_weight * stop
        )
        return {
            "loss": total,
            "position_loss": position,
            "heading_loss": heading,
            "collision_loss": collision,
            "stop_loss": stop,
        }
