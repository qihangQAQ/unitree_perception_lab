import math

import pytest
import torch

from unitree_rl_lab.painting.trajectories import PaintingPathCfg, import_surface_path, interpolate_path, sample_paths


@pytest.mark.parametrize("kind", range(4))
def test_all_primitives_respect_plane_height_length_and_sampling(kind):
    cfg = PaintingPathCfg(probabilities=tuple(float(i == kind) for i in range(4)))
    lengths = torch.tensor([3.5, 6.75, 11.0])
    paths = sample_paths(lengths, torch.tensor([-1, 1, -1]), cfg, generator=torch.Generator().manual_seed(30 + kind))
    torch.testing.assert_close(paths.arc[:, -1], lengths)
    assert paths.points.shape[-1] == 6
    assert (paths.points[..., 0] == 0).all()
    assert (paths.points[..., 3] == -1).all()
    assert (paths.points[..., 4:] == 0).all()
    assert (paths.points[..., 2] >= 1.1).all() and (paths.points[..., 2] <= 1.4).all()
    assert (paths.primitive_counts[:, kind] > 0).all()
    for i in range(3):
        xyz = paths.points[i, :paths.counts[i], :3]
        delta = torch.linalg.vector_norm(torch.diff(xyz, dim=0), dim=-1)
        assert delta.max() < 0.01001
        assert delta[:-1].min() > 0.0095
        assert abs(float(delta.sum() - lengths[i])) < 0.015
        assert torch.sign(xyz[-1, 1]) == torch.tensor([-1, 1, -1])[i]


def test_direction_is_a_mirror_of_the_same_fixed_path():
    length = torch.tensor([4.0])
    left = sample_paths(length, torch.ones(1), PaintingPathCfg(), generator=torch.Generator().manual_seed(5))
    right = sample_paths(length, -torch.ones(1), PaintingPathCfg(), generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(left.points[..., 1], -right.points[..., 1])
    torch.testing.assert_close(left.points[..., 2], right.points[..., 2])


def test_closed_loop_queries_follow_order_and_endpoint_padding():
    points = torch.tensor([[[0., 0., 1.2], [0., .1, 1.2], [0., 0., 1.2], [0., 0., 1.2]]])
    arc = torch.tensor([[0., .1, .2, .2]])
    position, tangent = interpolate_path(points, arc, torch.tensor([[.05, .15, .2, 10.]]))
    torch.testing.assert_close(position[0, 0], position[0, 1])
    assert tangent[0, 0, 1] > 0 and tangent[0, 1, 1] < 0
    torch.testing.assert_close(position[0, 2:], points[0, 0].expand(2, 3))


def test_import_deduplicates_and_rejects_invalid_normals():
    points = torch.tensor([[0., 0., 1.2, -1., 0., 0.], [0., 0., 1.2, -1., 0., 0.],
                           [0., .035, 1.2, -1., 0., 0.]])
    path = import_surface_path(points, PaintingPathCfg(), 10)
    assert path.counts.item() == 5
    assert math.isclose(path.lengths.item(), .035, abs_tol=1e-6)
    points[:, 3] = 1
    with pytest.raises(ValueError, match="outward normal"):
        import_surface_path(points, PaintingPathCfg(), 10)


def test_shape_distribution_counts_primitives_in_the_consumed_prefix():
    paths = sample_paths(torch.full((128,), 3.5), -torch.ones(128), PaintingPathCfg(),
                         generator=torch.Generator().manual_seed(77))
    distribution = paths.primitive_counts.sum(0).float()
    distribution /= distribution.sum()
    torch.testing.assert_close(distribution, torch.tensor([.3, .2, .25, .25]), atol=.07, rtol=0)


def test_invalid_capacity_and_geometry_fail_before_training():
    with pytest.raises(ValueError, match="capacity"):
        sample_paths(torch.tensor([3.5]), torch.tensor([-1]), PaintingPathCfg(), capacity=5)
    with pytest.raises(ValueError, match="fit"):
        PaintingPathCfg(shape_height=(.2, .4)).validate()
