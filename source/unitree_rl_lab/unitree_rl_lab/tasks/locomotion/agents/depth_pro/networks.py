"""Depth-specific neural-network blocks for the G1 depth-pro policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from unitree_rl_lab.tasks.locomotion.agents.perception_pro.networks import (
    build_mlp,
    resolve_activation,
)


class DepthCrossAttention(nn.Module):
    """Encode a depth image and query its spatial tokens with the robot state."""

    def __init__(
        self,
        depth_image_shape: tuple[int, int],
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

        self.depth_image_shape = depth_image_shape
        self.depth_image_dim = depth_image_shape[0] * depth_image_shape[1]
        self.attention_batch_size = attention_batch_size

        # Match the reference depth encoder: 16x24 becomes 8x12 spatial tokens.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 8),
            resolve_activation(activation),
            nn.Conv2d(8, model_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, model_dim),
            resolve_activation(activation),
        )
        token_rows = (depth_image_shape[0] + 1) // 2
        token_cols = (depth_image_shape[1] + 1) // 2
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

    def _as_channels_first(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize supported observation-manager layouts to ``[B, 1, H, W]``."""

        height, width = self.depth_image_shape
        if torch.jit.is_tracing():
            # The ONNX contract is fixed to channels-last [B,H,W,1].
            return depth.permute(0, 3, 1, 2)
        if depth.ndim == 5 and depth.shape[1] == 1 and depth.shape[-1] == 1:
            depth = depth[:, 0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        elif depth.ndim == 4 and depth.shape[1] == 1:
            pass
        elif depth.ndim == 3:
            depth = depth.unsqueeze(1)
        else:
            raise ValueError(
                "Expected depth [B,H,W], [B,H,W,1], [B,1,H,W], or [B,1,H,W,1], "
                f"got {tuple(depth.shape)}."
            )
        if not torch.jit.is_tracing() and tuple(depth.shape[-2:]) != (height, width):
            raise ValueError(
                f"Expected depth image {height}x{width}, got {tuple(depth.shape[-2:])}."
            )
        return depth

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
        depth: torch.Tensor,
        current_proprio: torch.Tensor,
        velocity_estimate: torch.Tensor,
    ) -> torch.Tensor:
        image = self._as_channels_first(depth)
        tokens = self.cnn(image).flatten(start_dim=2).transpose(1, 2)
        if not torch.jit.is_tracing() and tokens.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} depth tokens, got {tokens.shape[1]}.")
        tokens = tokens + self.position_embedding

        query_input = torch.cat((current_proprio, velocity_estimate), dim=-1)
        query = self.query_projection(query_input).unsqueeze(1)
        attention_output = self._attend(query, tokens)
        fused = self.attention_norm(query + attention_output)
        fused = self.feed_forward_norm(fused + self.feed_forward(fused))
        return self.output_projection(fused.squeeze(1))
