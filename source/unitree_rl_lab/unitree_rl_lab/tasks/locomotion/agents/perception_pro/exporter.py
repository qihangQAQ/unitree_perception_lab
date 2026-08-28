"""ONNX exporter for the structured perception-pro policy."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn


class PerceptionProOnnxModel(nn.Module):
    """Inputs are frame-major policy history [1, 480] and the current height map [1, 187]."""

    def __init__(self, policy):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.height_encoder = copy.deepcopy(policy.height_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.history_length = policy.history_length
        self.proprio_dim = policy.proprio_dim
        self.height_map_dim = policy.height_map_dim

    @property
    def policy_input_dim(self) -> int:
        return self.history_length * self.proprio_dim

    def forward(self, policy: torch.Tensor, height_map: torch.Tensor) -> torch.Tensor:
        history = policy.reshape(-1, self.history_length, self.proprio_dim)
        current_proprio = history[:, -1]
        velocity_estimate, him_latent = self.estimator(history)
        terrain_latent = self.height_encoder(height_map, current_proprio, velocity_estimate)
        actor_input = torch.cat(
            (current_proprio, velocity_estimate, him_latent, terrain_latent),
            dim=-1,
        )
        return self.actor(actor_input)


def export_perception_pro_policy_as_onnx(policy, path: str, filename: str = "policy.onnx") -> str:
    """Export a perception-pro policy with fixed-size policy and height-map inputs."""

    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)
    model = PerceptionProOnnxModel(policy).cpu().eval()
    dummy_policy = torch.zeros(1, model.policy_input_dim)
    dummy_height_map = torch.zeros(1, model.height_map_dim)
    torch.onnx.export(
        model,
        (dummy_policy, dummy_height_map),
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["policy", "height_map"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    return output_path
