"""Neural-network modules shared by perceptive HIM-MoE policies."""

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
    """Estimate velocity and a future-aware latent from proprioceptive history."""

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


class SpatialCrossAttention(nn.Module):
    """Encode a height map or depth image and query its spatial tokens with robot state."""

    def __init__(
        self,
        observation_shape: tuple[int, int],
        input_mode: str,
        proprio_dim: int,
        model_dim: int,
        output_dim: int,
        first_conv_channels: int,
        first_conv_stride: int,
        num_heads: int,
        attention_batch_size: int,
        activation: str,
    ):
        super().__init__()
        if len(observation_shape) != 2 or any(dimension <= 0 for dimension in observation_shape):
            raise ValueError(f"observation_shape must contain two positive dimensions, got {observation_shape}.")
        if input_mode not in {"flat", "image"}:
            raise ValueError(f"input_mode must be 'flat' or 'image', got {input_mode!r}.")
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}.")
        if attention_batch_size <= 0:
            raise ValueError("attention_batch_size must be positive.")
        if first_conv_channels % 4 != 0 or model_dim % 8 != 0:
            raise ValueError("first_conv_channels and model_dim must be divisible by 4 and 8 respectively.")
        if first_conv_stride <= 0:
            raise ValueError("first_conv_stride must be positive.")

        self.observation_shape = observation_shape
        self.observation_dim = observation_shape[0] * observation_shape[1]
        self.input_mode = input_mode
        self.attention_batch_size = attention_batch_size

        self.cnn = nn.Sequential(
            nn.Conv2d(1, first_conv_channels, kernel_size=3, stride=first_conv_stride, padding=1),
            nn.GroupNorm(4, first_conv_channels),
            resolve_activation(activation),
            nn.Conv2d(first_conv_channels, model_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, model_dim),
            resolve_activation(activation),
        )
        token_rows = (observation_shape[0] + first_conv_stride - 1) // first_conv_stride
        token_cols = (observation_shape[1] + first_conv_stride - 1) // first_conv_stride
        self.num_tokens = token_rows * token_cols
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

    @property
    def export_input_shape(self) -> tuple[int, ...]:
        """Return the fixed ONNX shape, excluding its batch dimension."""

        if self.input_mode == "flat":
            return (self.observation_dim,)
        return (*self.observation_shape, 1)

    def validate_observation_shape(self, shape: tuple[int, ...]) -> None:
        """Validate an observation-manager shape without its batch dimension."""

        height, width = self.observation_shape
        if self.input_mode == "flat":
            if shape != (self.observation_dim,):
                raise ValueError(
                    f"Expected flattened spatial observation [{self.observation_dim}], got {shape}."
                )
            return

        supported_shapes = (
            (height, width),
            (height, width, 1),
            (1, height, width),
            (1, height, width, 1),
        )
        if shape not in supported_shapes:
            raise ValueError(
                f"Expected a {height}x{width} single-channel image, got {shape}."
            )

    def _as_channels_first(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize supported observation layouts to ``[B, 1, H, W]``."""

        height, width = self.observation_shape
        if self.input_mode == "flat":
            if not torch.jit.is_tracing() and (
                observation.ndim != 2 or observation.shape[-1] != self.observation_dim
            ):
                raise ValueError(
                    f"Expected flattened spatial observation [B, {self.observation_dim}], "
                    f"got {tuple(observation.shape)}."
                )
            return observation.reshape(observation.shape[0], 1, height, width)

        if torch.jit.is_tracing():
            # The image-mode ONNX contract is fixed to channels-last [B,H,W,1].
            return observation.permute(0, 3, 1, 2)
        if observation.ndim == 5 and observation.shape[1] == 1 and observation.shape[-1] == 1:
            observation = observation[:, 0]
        if observation.ndim == 4 and observation.shape[-1] == 1:
            observation = observation.permute(0, 3, 1, 2)
        elif observation.ndim == 4 and observation.shape[1] == 1:
            pass
        elif observation.ndim == 3:
            observation = observation.unsqueeze(1)
        else:
            raise ValueError(
                "Expected image [B,H,W], [B,H,W,1], [B,1,H,W], or [B,1,H,W,1], "
                f"got {tuple(observation.shape)}."
            )
        if tuple(observation.shape[-2:]) != (height, width):
            raise ValueError(
                f"Expected image {height}x{width}, got {tuple(observation.shape[-2:])}."
            )
        return observation

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
        observation: torch.Tensor,
        current_proprio: torch.Tensor,
        velocity_estimate: torch.Tensor,
    ) -> torch.Tensor:
        image = self._as_channels_first(observation)
        tokens = self.cnn(image).flatten(start_dim=2).transpose(1, 2)
        if not torch.jit.is_tracing() and tokens.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} spatial tokens, got {tokens.shape[1]}.")
        tokens = tokens + self.position_embedding

        query_input = torch.cat((current_proprio, velocity_estimate), dim=-1)
        query = self.query_projection(query_input).unsqueeze(1)
        attention_output = self._attend(query, tokens)
        fused = self.attention_norm(query + attention_output)
        fused = self.feed_forward_norm(fused + self.feed_forward(fused))
        return self.output_projection(fused.squeeze(1))


class DenseMoEActor(nn.Module):
    """Dense softmax MoE actor without an auxiliary balance loss."""

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
