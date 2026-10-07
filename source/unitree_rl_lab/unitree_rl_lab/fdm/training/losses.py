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


def _masked_per_step_mean_sum(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum horizon losses after averaging valid samples and channels at each step."""
    mask_value = mask.to(value.dtype).unsqueeze(-1)
    step_sum = (value * mask_value).sum(dim=(0, 2))
    step_count = (mask_value.sum(dim=0).squeeze(-1) * value.shape[-1]).clamp_min(1.0)
    return (step_sum / step_count).sum()


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
        position = _masked_per_step_mean_sum(
            functional.mse_loss(predicted_pose[..., :2], target_pose[..., :2], reduction="none"), valid
        )
        heading = _masked_per_step_mean_sum(
            functional.mse_loss(predicted_pose[..., 2:4], target_pose[..., 2:4], reduction="none"), valid
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


class LossAccumulator:
    """Aggregate the objective's numerators/denominators across unequal batches."""

    def __init__(self, loss_fn: FDMLoss) -> None:
        self.loss_fn = loss_fn
        self.step_sums = None
        self.step_counts = None
        self.collision_sum = self.stop_sum = 0.0
        self.collision_count = self.stop_count = 0

    @torch.no_grad()
    def update(self, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> None:
        pose = prediction["future_pose"].detach().double()
        valid = target["valid_mask"].bool()
        error = (pose - target["future_pose"].double()).square()
        sums = torch.stack((error[..., :2].sum(-1), error[..., 2:4].sum(-1)), dim=-1)
        step_sums = (sums * valid[..., None]).sum(0).cpu()
        step_counts = (2 * valid.sum(0)).cpu()
        self.step_sums = step_sums if self.step_sums is None else self.step_sums + step_sums
        self.step_counts = step_counts if self.step_counts is None else self.step_counts + step_counts
        collision = functional.binary_cross_entropy_with_logits(
            prediction["collision_logits"].detach().double(), target["future_collision"].double(),
            pos_weight=self.loss_fn.collision_pos_weight, reduction="none",
        )
        self.collision_sum += float(collision[valid].sum())
        self.collision_count += int(valid.sum())
        stop_mask = target["future_collision"][:, :-1].bool() & valid[:, 1:]
        heading = torch.atan2(pose[..., 2], pose[..., 3])
        yaw_step = torch.atan2(torch.sin(heading[:, 1:] - heading[:, :-1]), torch.cos(heading[:, 1:] - heading[:, :-1]))
        motion = torch.cat((pose[:, 1:, :2] - pose[:, :-1, :2], yaw_step[..., None]), dim=-1)
        self.stop_sum += float(motion[stop_mask].square().sum())
        self.stop_count += int(stop_mask.sum()) * 3

    def result(self) -> dict[str, float]:
        position, heading = (self.step_sums / self.step_counts.clamp_min(1)[:, None]).sum(0).tolist()
        collision = self.collision_sum / max(self.collision_count, 1)
        stop = self.stop_sum / max(self.stop_count, 1)
        cfg = self.loss_fn.cfg
        return {
            "loss": cfg.position_weight * position + cfg.heading_weight * heading
            + cfg.collision_weight * collision + cfg.stop_weight * stop,
            "position_loss": position, "heading_loss": heading,
            "collision_loss": collision, "stop_loss": stop,
        }
