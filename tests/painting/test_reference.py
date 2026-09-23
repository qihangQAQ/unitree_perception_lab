import math

import numpy as np
import torch

from unitree_rl_lab.painting.inference import wall_targets_to_base
from unitree_rl_lab.painting.reference import ending_masks, reference_at_time, trajectory_observation


def _straight_reference(elapsed):
    arc = torch.linspace(0, 3.5, 351)[None]
    points = torch.zeros(1, 351, 3)
    points[..., 1] = -arc
    points[..., 2] = 1.2
    return reference_at_time(points, arc, torch.tensor([3.5]), torch.tensor([.2]), torch.tensor([.1]),
                             torch.tensor([elapsed]), 1.0, torch.tensor([0., .1, .2, .4, .8]))


def test_prepare_spray_and_terminal_hold_have_exact_time_budget():
    targets, velocity, progress, speed = _straight_reference(.5)
    assert speed.item() == 0 and progress.item() == 0
    torch.testing.assert_close(targets[:, 0], targets[:, -1])
    targets, velocity, progress, speed = _straight_reference(1.02)
    torch.testing.assert_close(progress, torch.tensor([.004]))
    torch.testing.assert_close(velocity, torch.tensor([[0., -.2, 0.]]))
    torch.testing.assert_close(targets[0, 0], torch.tensor([-.1, -.004, 1.2]))
    targets, velocity, progress, speed = _straight_reference(18.5)
    assert progress.item() == 3.5 and speed.item() == 0
    torch.testing.assert_close(targets[:, 0], targets[:, -1])
    assert velocity.abs().sum() == 0


def test_reference_does_not_quantize_velocity_at_one_centimeter():
    p0 = _straight_reference(3.0)[0][:, 0]
    p1 = _straight_reference(3.02)[0][:, 0]
    torch.testing.assert_close((p1 - p0) / .02, torch.tensor([[0., -.2, 0.]]), atol=2e-6, rtol=0)


def test_only_three_exclusive_endings_and_no_early_closed_loop_success():
    # early spatial endpoint; completed endpoint; ordinary limit; bad+success+limit; success+limit
    bad, success, timeout = ending_masks(
        torch.tensor([0., 0., 0., 1., 0.]), torch.tensor([0., 4., 4., 4., 4.]), torch.full((5,), 4.),
        torch.tensor([0., .02, .2, 0., 0.]), torch.tensor([0., 19., 20., 20., 20.]), torch.full((5,), 20.),
    )
    assert bad.tolist() == [False, False, False, True, False]
    assert success.tolist() == [False, True, False, False, True]
    assert timeout.tolist() == [False, False, True, False, False]
    assert ((bad.int() + success.int() + timeout.int()) <= 1).all()


def test_numpy_deployment_transform_matches_training_and_rotates_normal():
    base_pos = torch.tensor([[1., 2., 0.]])
    base_quat = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
    surface = torch.tensor([[[2., 3., 1.2]]]).expand(1, 5, 3).clone()
    target = surface.clone()
    target[..., 0] -= .1
    obs = trajectory_observation(target, base_pos, base_quat, torch.tensor([.2]))
    p, axis = wall_targets_to_base(surface[0].numpy(), base_pos[0].numpy(), base_quat[0].numpy(), .1)
    np.testing.assert_allclose(obs[0, :15].numpy(), p.ravel(), atol=3e-7)
    np.testing.assert_allclose(obs[0, 15:18].numpy(), axis, atol=3e-7)
    np.testing.assert_allclose(axis, [0, -1, 0], atol=3e-7)
    assert obs.shape == (1, 19)
