"""CPU tests for the training-only foothold components."""

import math

import torch

from unitree_rl_lab.rsl_rl_ext.modules.foothold_imagination import (
    FootholdImagination,
    support_deficiency_from_heights,
    world_xy_to_base_yaw,
)
from unitree_rl_lab.rsl_rl_ext.storage.foothold_replay_buffer import FootholdReplayBuffer


def test_world_xy_to_base_yaw_rotates_and_translates():
    target = torch.tensor([[2.0, 3.0]])
    base = torch.tensor([[1.0, 1.0]])
    yaw = torch.tensor([math.pi / 2.0])

    actual = world_xy_to_base_yaw(target, base, yaw)

    torch.testing.assert_close(actual, torch.tensor([[2.0, -1.0]]), atol=1.0e-6, rtol=0.0)


def test_model_outputs_bounded_per_foot_distribution():
    model = FootholdImagination(8, 3, hidden_dims=(16, 8))

    mu, sigma = model(torch.randn(4, 8), torch.randn(4, 3))

    assert mu.shape == (4, 2, 2)
    assert sigma.shape == (4, 2)
    assert torch.all(torch.abs(mu[..., 0]) <= 1.0)
    assert torch.all(torch.abs(mu[..., 1]) <= 0.6)
    assert torch.all(sigma >= 0.02)
    assert torch.all(sigma <= 0.50)


def test_gaussian_nll_is_finite_and_differentiable():
    model = FootholdImagination(8, 3, hidden_dims=(16, 8))
    mu, sigma = model(torch.randn(4, 8), torch.randn(4, 3))

    loss = model.gaussian_nll(mu, sigma, torch.randn(4, 2, 2))
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_support_deficiency_counts_low_and_missing_samples():
    heights = torch.tensor([[1.0, 0.99, 0.8, float("inf")]])

    deficiency = support_deficiency_from_heights(
        heights,
        torch.tensor([[1.0]]),
        tolerance=0.03,
    )

    torch.testing.assert_close(deficiency, torch.tensor([0.5]))


def test_foothold_replay_buffer_wraps_and_samples():
    replay = FootholdReplayBuffer(4, 2, buffer_size=5, device="cpu")
    replay.insert(
        torch.arange(28, dtype=torch.float).view(7, 4),
        torch.arange(14, dtype=torch.float).view(7, 2),
        torch.arange(14, dtype=torch.float).view(7, 2),
        torch.arange(7) % 2,
        torch.arange(7) % 2,
    )

    sample = replay.sample(3)

    assert replay.num_samples == 5
    assert [item.shape for item in sample] == [(3, 4), (3, 2), (3, 2), (3,), (3,)]
