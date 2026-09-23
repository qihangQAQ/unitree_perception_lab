"""Compact whole-body policy with six explicit history-based velocity estimates."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

from .networks import build_mlp


class PaintingActorCritic(ActorCritic):
    def __init__(
        self, obs, obs_groups, num_actions, history_length=5, proprio_dim=93,
        command_dim=19, proprio_group="policy", command_group="painting",
        velocity_group="velocity_targets", estimator_hidden_dims=(128, 64),
        actor_obs_normalization=False, activation="elu", **kwargs,
    ):
        if actor_obs_normalization:
            raise ValueError("Painting uses fixed sensor scales; actor normalization must be disabled.")
        if (history_length, proprio_dim, command_dim, num_actions) != (5, 93, 19, 29):
            raise ValueError("Painting deployment contract is 5x93 history, 19 commands and 29 actions.")
        expected = {proprio_group: (history_length, proprio_dim), command_group: (command_dim,), velocity_group: (6,)}
        for name, shape in expected.items():
            if name not in obs or tuple(obs[name].shape[1:]) != shape:
                raise ValueError(f"Expected observation {name!r} with shape [N, {shape}].")
        actor_dim = proprio_dim + 6 + command_dim
        synthetic = dict(obs)
        synthetic["_painting_actor"] = obs[command_group].new_zeros(len(obs[command_group]), actor_dim)
        super().__init__(
            synthetic, {"policy": ["_painting_actor"], "critic": obs_groups["critic"]}, num_actions,
            actor_obs_normalization=False, activation=activation, **kwargs,
        )
        self.obs_groups = obs_groups
        self.history_length, self.proprio_dim, self.command_dim = history_length, proprio_dim, command_dim
        self.proprio_group, self.command_group, self.velocity_group = proprio_group, command_group, velocity_group
        self.actor_input_dim = actor_dim
        self.estimator = nn.Sequential(
            nn.Flatten(start_dim=1),
            build_mlp(history_length * proprio_dim, list(estimator_hidden_dims), 6, activation),
        )

    def get_actor_obs(self, obs):
        history = obs[self.proprio_group]
        velocity = self.estimator(history).detach()
        return torch.cat((history[:, -1], velocity, obs[self.command_group]), dim=-1)

    def export_onnx(self, path, filename="policy.onnx"):
        from ..exporters.painting_exporter import export_painting_policy_as_onnx

        return export_painting_policy_as_onnx(self, path, filename)
