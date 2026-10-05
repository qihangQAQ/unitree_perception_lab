"""Compact painting policy with bounded Gaussian action statistics."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from torch.distributions import Normal

from .networks import build_mlp


class PaintingActorCritic(ActorCritic):
    interface_version = 2

    def __init__(
        self, obs, obs_groups, num_actions, history_length=5, proprio_dim=93,
        command_dim=27, proprio_group="policy", command_group="painting",
        velocity_group="velocity_targets", estimator_hidden_dims=(128, 64),
        actor_obs_normalization=False, activation="elu", min_action_std=0.05,
        max_action_std=0.4, mean_action_limit=2.0,
        action_mean_lower=None, action_mean_upper=None, action_std_max=None, **kwargs,
    ):
        if actor_obs_normalization:
            raise ValueError("Painting uses fixed sensor scales; actor normalization must be disabled.")
        if (history_length, proprio_dim, command_dim, num_actions) != (5, 93, 27, 29):
            raise ValueError("Painting v2 contract is 5x93 history, 27 commands and 29 actions.")
        if not 0 < min_action_std <= max_action_std:
            raise ValueError("Expected 0 < min_action_std <= max_action_std.")
        if mean_action_limit <= 0:
            raise ValueError("mean_action_limit must be positive.")
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
        self.num_actions = num_actions
        self.history_length, self.proprio_dim, self.command_dim = history_length, proprio_dim, command_dim
        self.proprio_group, self.command_group, self.velocity_group = proprio_group, command_group, velocity_group
        self.actor_input_dim = actor_dim
        self.estimator = nn.Sequential(
            nn.Flatten(start_dim=1),
            build_mlp(history_length * proprio_dim, list(estimator_hidden_dims), 6, activation),
        )

        def vector(value, default):
            value = [default] * num_actions if value is None else value
            result = torch.as_tensor(value, dtype=torch.float32)
            if result.shape != (num_actions,):
                raise ValueError(f"Expected {num_actions} action bounds, got {tuple(result.shape)}.")
            return result

        lower = vector(action_mean_lower, -mean_action_limit)
        upper = vector(action_mean_upper, mean_action_limit)
        std_min = torch.full((num_actions,), float(min_action_std))
        std_max = vector(action_std_max, max_action_std)
        if torch.any(lower >= 0) or torch.any(upper <= 0) or torch.any(lower >= upper):
            raise ValueError("Every bounded action-mean interval must strictly contain zero.")
        if torch.any(std_max < std_min):
            raise ValueError("Every action std maximum must be at least min_action_std.")
        self.register_buffer("action_mean_lower", lower)
        self.register_buffer("action_mean_upper", upper)
        self.register_buffer("action_std_min", std_min)
        self.register_buffer("action_std_max", std_max)
        with torch.no_grad():
            initial_std = torch.exp(self.log_std).clamp(min=std_min, max=std_max)
            self.log_std.copy_(initial_std.log())

    def bounded_action_mean(self, actor_obs):
        """Map the raw MLP output to asymmetric per-joint bounds with zero fixed."""
        raw = self.actor(actor_obs)
        positive = self.action_mean_upper
        negative = -self.action_mean_lower
        return torch.where(
            raw >= 0,
            positive * torch.tanh(raw / positive),
            negative * torch.tanh(raw / negative),
        )

    @property
    def effective_action_std(self):
        bounded_log_std = torch.clamp(self.log_std, self.action_std_min.log(), self.action_std_max.log())
        return torch.exp(bounded_log_std).clamp(min=self.action_std_min, max=self.action_std_max)

    @torch.no_grad()
    def project_distribution_parameters_(self):
        """Keep the optimized parameter itself inside the effective std interval."""
        self.log_std.clamp_(self.action_std_min.log(), self.action_std_max.log())

    def update_distribution(self, actor_obs):
        mean = self.bounded_action_mean(actor_obs)
        self.distribution = Normal(mean, self.effective_action_std.expand_as(mean))

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.bounded_action_mean(actor_obs)

    def get_actor_obs(self, obs):
        history = obs[self.proprio_group]
        velocity = self.estimator(history).detach()
        return torch.cat((history[:, -1], velocity, obs[self.command_group]), dim=-1)

    def export_onnx(self, path, filename="policy.onnx"):
        from ..exporters.painting_exporter import export_painting_policy_as_onnx

        return export_painting_policy_as_onnx(self, path, filename)
