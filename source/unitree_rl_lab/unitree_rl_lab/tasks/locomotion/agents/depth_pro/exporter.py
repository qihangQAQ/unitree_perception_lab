"""ONNX exporter for the structured depth-pro policy."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn


class DepthProOnnxModel(nn.Module):
    """Flattened proprioceptive history and one channels-last depth image."""

    def __init__(self, policy):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.depth_encoder = copy.deepcopy(policy.depth_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.history_length = policy.history_length
        self.proprio_dim = policy.proprio_dim
        self.depth_image_shape = policy.depth_image_shape

    @property
    def policy_input_dim(self) -> int:
        return self.history_length * self.proprio_dim

    def forward(self, policy: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        history = policy.reshape(-1, self.history_length, self.proprio_dim)
        current_proprio = history[:, -1]
        velocity_estimate, him_latent = self.estimator(history)
        depth_latent = self.depth_encoder(depth, current_proprio, velocity_estimate)
        actor_input = torch.cat(
            (current_proprio, velocity_estimate, him_latent, depth_latent),
            dim=-1,
        )
        return self.actor(actor_input)


def export_depth_pro_policy_as_onnx(policy, path: str, filename: str = "policy.onnx") -> str:
    """Export a depth-pro policy with fixed-size proprioception and image inputs."""

    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)
    model = DepthProOnnxModel(policy).cpu().eval()
    dummy_policy = torch.zeros(1, model.policy_input_dim)
    dummy_depth = torch.zeros(1, *model.depth_image_shape, 1)
    torch.onnx.export(
        model,
        (dummy_policy, dummy_depth),
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["policy", "depth"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    return output_path
