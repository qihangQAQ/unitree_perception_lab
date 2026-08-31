"""ONNX exporter shared by height-map and depth HIM-MoE policies."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn


class HimMoeOnnxModel(nn.Module):
    """Inference-only actor with proprioceptive history and exteroception inputs."""

    def __init__(self, policy):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.exteroception_encoder = copy.deepcopy(policy.exteroception_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.history_length = policy.history_length
        self.proprio_dim = policy.proprio_dim

    @property
    def policy_input_dim(self) -> int:
        return self.history_length * self.proprio_dim

    def forward(self, policy: torch.Tensor, exteroception: torch.Tensor) -> torch.Tensor:
        history = policy.reshape(-1, self.history_length, self.proprio_dim)
        current_proprio = history[:, -1]
        velocity_estimate, him_latent = self.estimator(history)
        exteroception_latent = self.exteroception_encoder(
            exteroception,
            current_proprio,
            velocity_estimate,
        )
        actor_input = torch.cat(
            (current_proprio, velocity_estimate, him_latent, exteroception_latent),
            dim=-1,
        )
        return self.actor(actor_input)


def export_him_moe_policy_as_onnx(policy, path: str, filename: str = "policy.onnx") -> str:
    """Export a fixed-size HIM-MoE policy using the task's observation-group name."""

    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)
    model = HimMoeOnnxModel(policy).cpu().eval()
    dummy_policy = torch.zeros(1, model.policy_input_dim)
    dummy_exteroception = torch.zeros(1, *policy.exteroception_export_shape)
    torch.onnx.export(
        model,
        (dummy_policy, dummy_exteroception),
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["policy", policy.exteroception_group],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    return output_path
