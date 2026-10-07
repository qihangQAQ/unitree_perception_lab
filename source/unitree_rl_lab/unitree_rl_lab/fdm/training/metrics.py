"""Global metric aggregation, independent of DataLoader batch boundaries."""

from __future__ import annotations

import torch

from ..utils.se2 import integrate_body_twists, wrap_angle


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5, command_timestep: float = 0.5) -> None:
        self.threshold = threshold
        self.command_timestep = command_timestep
        self.count = self.final_count = self.tp = self.fp = self.fn = self.correct = 0
        self.position = self.heading = self.final_position = self.model_squared = 0.0
        self.baseline_position = self.baseline_heading = self.baseline_final = self.baseline_squared = 0.0
        self.baseline_count = self.baseline_final_count = 0

    @torch.no_grad()
    def update(self, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> None:
        valid = target["valid_mask"].bool()
        predicted_pose = prediction["future_pose"].double()
        target_pose = target["future_pose"].double()
        error = torch.linalg.vector_norm(predicted_pose[..., :2] - target_pose[..., :2], dim=-1)
        predicted_yaw = torch.atan2(predicted_pose[..., 2], predicted_pose[..., 3])
        target_yaw = torch.atan2(target_pose[..., 2], target_pose[..., 3])
        yaw_error = wrap_angle(predicted_yaw - target_yaw).abs()
        self.count += int(valid.sum())
        self.final_count += int(valid[:, -1].sum())
        self.position += float(error[valid].sum())
        self.heading += float(yaw_error[valid].sum())
        self.final_position += float(error[:, -1][valid[:, -1]].sum())
        label = target["future_collision"].bool()
        predicted = torch.sigmoid(prediction["collision_logits"]) >= self.threshold
        self.tp += int((predicted & label & valid).sum())
        self.fp += int((predicted & ~label & valid).sum())
        self.fn += int((~predicted & label & valid).sum())
        self.correct += int(((predicted == label) & valid).sum())
        if "future_commands" in target:
            baseline = integrate_body_twists(target["future_commands"].double(), self.command_timestep)
            baseline_error = torch.linalg.vector_norm(baseline[..., :2] - target_pose[..., :2], dim=-1)
            baseline_yaw = torch.atan2(baseline[..., 2], baseline[..., 3])
            self.baseline_count += int(valid.sum())
            self.baseline_final_count += int(valid[:, -1].sum())
            self.baseline_position += float(baseline_error[valid].sum())
            self.baseline_heading += float(wrap_angle(baseline_yaw - target_yaw)[valid].abs().sum())
            self.baseline_final += float(baseline_error[:, -1][valid[:, -1]].sum())
            self.baseline_squared += float(baseline_error[valid].square().sum())
            self.model_squared += float(error[valid].square().sum())

    def result(self) -> dict[str, float | int | None]:
        result = {
            "position_mae_m": self.position / max(self.count, 1),
            "heading_mae_deg": self.heading / max(self.count, 1) * 180.0 / torch.pi,
            "final_position_mae_m": self.final_position / self.final_count if self.final_count else None,
            "collision_precision": self.tp / max(self.tp + self.fp, 1),
            "collision_recall": self.tp / max(self.tp + self.fn, 1),
            "collision_accuracy": self.correct / max(self.count, 1),
            "valid_targets": self.count, "valid_final_targets": self.final_count,
            "collision_tp": self.tp, "collision_fp": self.fp, "collision_fn": self.fn,
        }
        if self.baseline_count:
            result.update({
                "baseline_position_mae_m": self.baseline_position / self.baseline_count,
                "baseline_heading_mae_deg": self.baseline_heading / self.baseline_count * 180.0 / torch.pi,
                "baseline_final_position_mae_m": self.baseline_final / self.baseline_final_count
                if self.baseline_final_count else None,
                "position_mse_ratio_vs_baseline": self.model_squared / self.baseline_squared
                if self.baseline_squared > 0 else None,
            })
        return result


@torch.no_grad()
def compute_metrics(
    prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], threshold: float = 0.5
) -> dict[str, float | int | None]:
    accumulator = MetricAccumulator(threshold)
    accumulator.update(prediction, target)
    return accumulator.result()
