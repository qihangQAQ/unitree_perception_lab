"""Two-input ONNX export of the complete painting inference graph."""

import copy
from pathlib import Path

import torch
import torch.nn as nn


class PaintingOnnxModel(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.actor = copy.deepcopy(policy.actor)
        self.history_length = policy.history_length
        self.proprio_dim = policy.proprio_dim

    def forward(self, proprio_history, trajectory_command):
        history = proprio_history.reshape(-1, self.history_length, self.proprio_dim)
        velocity = self.estimator(history)
        return self.actor(torch.cat((history[:, -1], velocity, trajectory_command), dim=-1))


def export_painting_policy_as_onnx(policy, path, filename="policy.onnx"):
    Path(path).mkdir(parents=True, exist_ok=True)
    output = str(Path(path) / filename)
    model = PaintingOnnxModel(policy).cpu().eval()
    torch.onnx.export(
        model, (torch.zeros(1, 465), torch.zeros(1, 19)), output,
        input_names=["proprio_history", "trajectory_command"], output_names=["actions"],
        opset_version=18, export_params=True, dynamic_axes={}, dynamo=False,
    )
    return output
