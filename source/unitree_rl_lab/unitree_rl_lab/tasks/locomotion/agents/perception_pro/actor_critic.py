"""Actor-critic used by the G1 perception-pro task."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP

from .networks import DenseMoEActor, HeightMapCrossAttention, OldHIMEstimator


class PerceptionProActorCritic(nn.Module):
    """Policy combining Old-HIM state estimates with cross-attended terrain features."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 128, 64],
        critic_hidden_dims: list[int] = [256, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        proprio_group: str = "policy",
        height_map_group: str = "height_map",
        auxiliary_group: str = "auxiliary",
        history_length: int = 5,
        proprio_dim: int = 96,
        auxiliary_dim: int = 96,
        estimator_hidden_dims: list[int] = [256, 64, 16],
        target_hidden_dims: list[int] = [256, 64],
        num_prototypes: int = 32,
        him_temperature: float = 3.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        height_map_shape: tuple[int, int] = (11, 17),
        height_model_dim: int = 32,
        height_latent_dim: int = 32,
        attention_num_heads: int = 4,
        attention_batch_size: int = 4096,
        num_moe_experts: int = 4,
        moe_gate_hidden_dims: list[int] = [],
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected PerceptionProActorCritic arguments: {sorted(kwargs)}")
        if actor_obs_normalization:
            raise ValueError("Actor observation normalization is not supported for structured perception-pro inputs.")

        self.obs_groups = obs_groups
        self.proprio_group = proprio_group
        self.height_map_group = height_map_group
        self.auxiliary_group = auxiliary_group
        self.history_length = history_length
        self.proprio_dim = proprio_dim
        self.auxiliary_dim = auxiliary_dim
        self.height_map_dim = height_map_shape[0] * height_map_shape[1]
        self.height_latent_dim = height_latent_dim
        self.num_actions = num_actions

        self._validate_observation_shapes(obs)

        self.estimator = OldHIMEstimator(
            history_length=history_length,
            proprio_dim=proprio_dim,
            auxiliary_dim=auxiliary_dim,
            estimator_hidden_dims=estimator_hidden_dims,
            target_hidden_dims=target_hidden_dims,
            num_prototypes=num_prototypes,
            temperature=him_temperature,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            activation=activation,
        )
        self.height_encoder = HeightMapCrossAttention(
            height_map_shape=height_map_shape,
            proprio_dim=proprio_dim,
            model_dim=height_model_dim,
            output_dim=height_latent_dim,
            num_heads=attention_num_heads,
            attention_batch_size=attention_batch_size,
            activation=activation,
        )

        actor_input_dim = proprio_dim + 3 + self.estimator.latent_dim + height_latent_dim
        self.actor_input_dim = actor_input_dim
        self.actor = DenseMoEActor(
            input_dim=actor_input_dim,
            output_dim=num_actions,
            num_experts=num_moe_experts,
            expert_hidden_dims=actor_hidden_dims,
            gate_hidden_dims=moe_gate_hidden_dims,
            activation=activation,
        )

        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )
        self.actor_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type {noise_std_type!r}; expected 'scalar' or 'log'.")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _validate_observation_shapes(self, obs):
        proprio_shape = tuple(obs[self.proprio_group].shape[1:])
        expected_proprio_shape = (self.history_length, self.proprio_dim)
        if proprio_shape != expected_proprio_shape:
            raise ValueError(
                f"Observation group {self.proprio_group!r} must have shape [N, {self.history_length}, "
                f"{self.proprio_dim}], got {tuple(obs[self.proprio_group].shape)}."
            )
        if obs[self.height_map_group].shape[-1] != self.height_map_dim:
            raise ValueError(
                f"Observation group {self.height_map_group!r} must contain {self.height_map_dim} height samples, "
                f"got {obs[self.height_map_group].shape[-1]}."
            )
        if obs[self.auxiliary_group].shape[-1] != self.auxiliary_dim:
            raise ValueError(
                f"Observation group {self.auxiliary_group!r} must have {self.auxiliary_dim} values, "
                f"got {obs[self.auxiliary_group].shape[-1]}."
            )

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_forward(self, obs) -> torch.Tensor:
        history = obs[self.proprio_group]
        current_proprio = history[:, -1]
        velocity_estimate, him_latent = self.estimator(history)

        # Old-HIM owns its optimizer; policy gradients must not update the estimator.
        actor_velocity = velocity_estimate.detach()
        actor_him_latent = him_latent.detach()
        terrain_latent = self.height_encoder(
            obs[self.height_map_group],
            current_proprio,
            actor_velocity,
        )
        actor_input = torch.cat(
            (current_proprio, actor_velocity, actor_him_latent, terrain_latent),
            dim=-1,
        )
        return self.actor(actor_input)

    def update_distribution(self, obs):
        mean = self._actor_forward(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self._actor_forward(obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        return self.critic(self.critic_obs_normalizer(critic_obs))

    def get_critic_obs(self, obs):
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def export_onnx(self, path: str, filename: str = "policy.onnx") -> str:
        """Export the complete structured actor as a flattened-input ONNX model."""
        from .exporter import export_perception_pro_policy_as_onnx

        return export_perception_pro_policy_as_onnx(self, path, filename)
