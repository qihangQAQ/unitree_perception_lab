"""Training-only model for SSR-style imagined foothold guidance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def world_xy_to_base_yaw(
    target_xy_w: torch.Tensor,
    base_xy_w: torch.Tensor,
    base_yaw: torch.Tensor,
) -> torch.Tensor:
    """Express world-frame XY targets in their matching base-yaw frames."""

    delta = target_xy_w - base_xy_w
    cos_yaw = torch.cos(base_yaw)
    sin_yaw = torch.sin(base_yaw)
    return torch.stack(
        (
            cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1],
            -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1],
        ),
        dim=-1,
    )


def support_deficiency_from_heights(
    heights: torch.Tensor,
    sole_height: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Return the unsupported fraction of a sole-sized terrain patch."""

    finite = torch.isfinite(heights)
    supported = finite & ((sole_height - heights) < tolerance)
    return 1.0 - supported.float().mean(dim=-1)


class FootholdImagination(nn.Module):
    """Predict each foot's future XY contact distribution from state and action."""

    def __init__(
        self,
        num_privileged_obs: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        min_sigma: float = 0.02,
        max_sigma: float = 0.50,
        max_forward_reach: float = 1.0,
        max_lateral_reach: float = 0.6,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = num_privileged_obs + num_actions
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 6))
        self.network = nn.Sequential(*layers)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.register_buffer(
            "_mu_scale",
            torch.tensor([max_forward_reach, max_lateral_reach], dtype=torch.float),
        )

    def forward(
        self,
        privileged_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``mu[N, 2, 2]`` and isotropic ``sigma[N, 2]``."""

        output = self.network(torch.cat((privileged_obs, actions), dim=-1)).view(-1, 2, 3)
        mu = torch.tanh(output[..., :2]) * self._mu_scale
        sigma = (F.softplus(output[..., 2]) + self.min_sigma).clamp_max(self.max_sigma)
        return mu, sigma

    @staticmethod
    def gaussian_nll(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute isotropic 2-D Gaussian NLL without its constant term."""

        squared_error = torch.sum(torch.square(targets - mu), dim=-1)
        loss = squared_error / (2.0 * torch.square(sigma)) + 2.0 * torch.log(sigma)
        if valid is None:
            return loss.mean()
        weights = valid.to(loss.dtype)
        return torch.sum(loss * weights) / weights.sum().clamp_min(1.0)
