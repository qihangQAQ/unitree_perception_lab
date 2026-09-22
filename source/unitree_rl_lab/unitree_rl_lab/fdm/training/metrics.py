"""Validation metrics with stable empty-class handling."""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_metrics(
    prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], threshold: float = 0.5
) -> dict[str, float]:
    valid = target["valid_mask"].bool()
    position_error = torch.linalg.vector_norm(
        prediction["future_pose"][..., :2] - target["future_pose"][..., :2], dim=-1
    )
    heading_prediction = torch.atan2(prediction["future_pose"][..., 2], prediction["future_pose"][..., 3])
    heading_target = torch.atan2(target["future_pose"][..., 2], target["future_pose"][..., 3])
    heading_error = torch.abs(
        torch.atan2(torch.sin(heading_prediction - heading_target), torch.cos(heading_prediction - heading_target))
    )
    label = target["future_collision"].bool() & valid
    predicted = (torch.sigmoid(prediction["collision_logits"]) >= threshold) & valid
    true_positive = (predicted & label).sum().float()
    false_positive = (predicted & ~label & valid).sum().float()
    false_negative = (~predicted & label).sum().float()
    correct = ((predicted == label) & valid).sum().float()
    count = valid.sum().clamp_min(1).float()
    return {
        "position_mae_m": (position_error[valid].sum() / count).item(),
        "heading_mae_deg": torch.rad2deg(heading_error[valid].sum() / count).item(),
        "collision_precision": (true_positive / (true_positive + false_positive).clamp_min(1)).item(),
        "collision_recall": (true_positive / (true_positive + false_negative).clamp_min(1)).item(),
        "collision_accuracy": (correct / count).item(),
    }
