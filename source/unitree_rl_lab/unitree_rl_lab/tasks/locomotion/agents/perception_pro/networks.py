"""Neural-network building blocks for the perception-pro policy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_activation(name: str) -> nn.Module:
    """Create an activation module from the names used by RSL-RL configs."""

    activations = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "lrelu": nn.LeakyReLU,
        "swish": nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}. Available: {sorted(activations)}") from exc


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    """Build an MLP whose final layer has no activation."""

    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), resolve_activation(activation)))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def sinkhorn_assignments(
    scores: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 3,
) -> torch.Tensor:
    """Return balanced prototype assignments using the Old-HIM Sinkhorn step."""

    assignments = torch.exp(scores / epsilon).transpose(0, 1)
    num_prototypes, batch_size = assignments.shape
    assignments /= assignments.sum().clamp_min(1.0e-12)

    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        assignments /= num_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        assignments /= batch_size
    return (assignments * batch_size).transpose(0, 1)


class OldHIMEstimator(nn.Module):
    """Estimate velocity and a future-aware latent from five proprioceptive frames."""

    def __init__(
        self,
        history_length: int,
        proprio_dim: int,
        auxiliary_dim: int,
        estimator_hidden_dims: list[int],
        target_hidden_dims: list[int],
        num_prototypes: int,
        temperature: float,
        sinkhorn_epsilon: float,
        sinkhorn_iterations: int,
        activation: str,
    ):
        super().__init__()
        if history_length < 2:
            raise ValueError(f"history_length must be at least 2, got {history_length}.")
        if len(estimator_hidden_dims) < 2:
            raise ValueError("estimator_hidden_dims must contain a hidden layer and the latent dimension.")
        if not target_hidden_dims:
            raise ValueError("target_hidden_dims must contain at least one dimension.")
        if num_prototypes < 2:
            raise ValueError(f"num_prototypes must be at least 2, got {num_prototypes}.")
        if temperature <= 0.0 or sinkhorn_epsilon <= 0.0:
            raise ValueError("HIM temperature and Sinkhorn epsilon must be positive.")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive.")

        self.history_length = history_length
        self.proprio_dim = proprio_dim
        self.auxiliary_dim = auxiliary_dim
        self.latent_dim = estimator_hidden_dims[-1]
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        self.encoder = build_mlp(
            history_length * proprio_dim,
            estimator_hidden_dims[:-1],
            3 + self.latent_dim,
            activation,
        )
        self.target = build_mlp(
            auxiliary_dim,
            target_hidden_dims,
            self.latent_dim,
            activation,
        )
        self.prototypes = nn.Embedding(num_prototypes, self.latent_dim)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (self.history_length, self.proprio_dim)
        if not torch.jit.is_tracing() and (
            history.ndim != 3 or tuple(history.shape[-2:]) != expected_shape
        ):
            raise ValueError(
                f"Expected proprio history [B, {self.history_length}, {self.proprio_dim}], "
                f"got {tuple(history.shape)}."
            )
        encoded = self.encoder(history.detach().flatten(start_dim=1))
        velocity = encoded[..., :3]
        latent = F.normalize(encoded[..., 3:], dim=-1, p=2.0)
        return velocity, latent

    def compute_loss(
        self,
        history: torch.Tensor,
        current_auxiliary: torch.Tensor,
        next_auxiliary: torch.Tensor,
        next_state_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity MSE and the symmetric Old-HIM prototype swap loss."""

        velocity, source_latent = self(history)
        velocity_target = current_auxiliary[..., -3:].detach()
        velocity_loss = F.mse_loss(velocity, velocity_target)

        valid = next_state_valid.squeeze(-1).bool()
        if valid.any():
            source_latent = source_latent[valid]
            target_latent = F.normalize(
                self.target(next_auxiliary[valid].detach()),
                dim=-1,
                p=2.0,
            )

            with torch.no_grad():
                self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=-1, p=2.0))

            source_scores = source_latent @ self.prototypes.weight.transpose(0, 1)
            target_scores = target_latent @ self.prototypes.weight.transpose(0, 1)
            source_assignments = sinkhorn_assignments(
                source_scores,
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
            )
            target_assignments = sinkhorn_assignments(
                target_scores,
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
            )
            source_log_prob = F.log_softmax(source_scores / self.temperature, dim=-1)
            target_log_prob = F.log_softmax(target_scores / self.temperature, dim=-1)
            swap_loss = -0.5 * (
                source_assignments * target_log_prob + target_assignments * source_log_prob
            ).mean()
        else:
            # Preserve a gradient entry for every estimator parameter in distributed runs.
            swap_loss = source_latent.sum() * 0.0
            swap_loss = swap_loss + sum(parameter.sum() * 0.0 for parameter in self.target.parameters())
            swap_loss = swap_loss + self.prototypes.weight.sum() * 0.0
        return velocity_loss, swap_loss


class HeightMapCrossAttention(nn.Module):
    """Encode an 11x17 height map into 187 spatial tokens and query it with robot state."""

    def __init__(
        self,
        height_map_shape: tuple[int, int],
        proprio_dim: int,
        model_dim: int,
        output_dim: int,
        num_heads: int,
        attention_batch_size: int,
        activation: str,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}.")
        if attention_batch_size <= 0:
            raise ValueError("attention_batch_size must be positive.")

        self.height_map_shape = height_map_shape
        self.height_map_dim = height_map_shape[0] * height_map_shape[1]
        self.attention_batch_size = attention_batch_size

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            resolve_activation(activation),
            nn.Conv2d(16, model_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, model_dim),
            resolve_activation(activation),
        )
        self.num_tokens = height_map_shape[0] * height_map_shape[1]
        self.position_embedding = nn.Parameter(torch.empty(1, self.num_tokens, model_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.query_projection = build_mlp(proprio_dim + 3, [64], model_dim, activation)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim),
            resolve_activation(activation),
            nn.Linear(2 * model_dim, model_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            resolve_activation(activation),
        )

    def _attend(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Apply attention in bounded chunks for large PPO mini-batches."""

        if torch.jit.is_tracing() or query.shape[0] <= self.attention_batch_size:
            return self.attention(query, tokens, tokens, need_weights=False)[0]

        outputs = []
        for start in range(0, query.shape[0], self.attention_batch_size):
            end = min(start + self.attention_batch_size, query.shape[0])
            output, _ = self.attention(
                query[start:end],
                tokens[start:end],
                tokens[start:end],
                need_weights=False,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        height_map: torch.Tensor,
        current_proprio: torch.Tensor,
        velocity_estimate: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            height_map.ndim != 2 or height_map.shape[-1] != self.height_map_dim
        ):
            raise ValueError(
                f"Expected flattened height map [B, {self.height_map_dim}], "
                f"got {tuple(height_map.shape)}."
            )

        image = height_map.reshape(height_map.shape[0], 1, *self.height_map_shape)
        tokens = self.cnn(image).flatten(start_dim=2).transpose(1, 2)
        tokens = tokens + self.position_embedding

        query_input = torch.cat((current_proprio, velocity_estimate), dim=-1)
        query = self.query_projection(query_input).unsqueeze(1)
        attention_output = self._attend(query, tokens)
        fused = self.attention_norm(query + attention_output)
        fused = self.feed_forward_norm(fused + self.feed_forward(fused))
        return self.output_projection(fused.squeeze(1))


class DenseMoEActor(nn.Module):
    """Reference-style dense softmax MoE actor without an auxiliary balance loss."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_hidden_dims: list[int],
        gate_hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"num_experts must be at least 2, got {num_experts}.")
        self.num_experts = num_experts
        self.gate = build_mlp(input_dim, gate_hidden_dims, num_experts, activation)
        self.experts = nn.ModuleList(
            [build_mlp(input_dim, expert_hidden_dims, output_dim, activation) for _ in range(num_experts)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_probabilities = F.softmax(self.gate(inputs), dim=-1)
        expert_outputs = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        return torch.einsum("be,bea->ba", gate_probabilities, expert_outputs)
