"""The FDM map must distinguish a passable door from its overhead beam."""

from types import SimpleNamespace

import torch

from unitree_rl_lab.fdm.utils.height_map import door_aware_height_map


def test_door_recognition_replaces_only_overhead_hits_and_preserves_sensor_data():
    top = torch.zeros(2, 4, 3)
    top[0, :, 2] = torch.tensor([2.4, 1.4, float("nan"), 2.4])
    top[1, :, 2] = 0.0
    top[1, 0] = float("inf")
    original = top.clone()
    sensor = SimpleNamespace(
        data=SimpleNamespace(ray_hits_w=top, pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])),
        cfg=SimpleNamespace(mesh_prim_paths=["/World/ground"], max_distance=10.0),
        meshes={"/World/ground": object()},
    )
    down_z = torch.tensor([[0.0, 0.0, float("nan"), 0.0], [0.0, 0.0, 0.0, 0.0]])
    up_z = torch.tensor([[2.0, 1.4, float("nan"), 0.9], [0.0, 0.0, 0.0, 0.0]])
    calls = []

    def fake_raycast(origins, directions, *, mesh, max_dist):
        assert torch.isfinite(origins).all()
        assert torch.all(origins[..., 2] == 0.5)
        assert mesh is sensor.meshes["/World/ground"] and max_dist == 10.0
        calls.append(float(directions[0, 0, 2]))
        hits = origins.clone()
        hits[..., 2] = down_z if calls[-1] < 0 else up_z
        return hits, None, None, None

    height, invalid = door_aware_height_map(
        sensor, torch.tensor([0, 1]), shape=(2, 2), raycast_fn=fake_raycast
    )

    assert calls == [-1.0, 1.0]
    assert height.shape == invalid.shape == (2, 1, 2, 2)
    # Grid-pattern orientation reverses the first spatial axis.
    assert torch.allclose(height[0, 0, 1], torch.tensor([-0.5, 0.9]))
    assert torch.allclose(height[0, 0, 0], torch.tensor([1.5, 1.5]))
    assert invalid[0, 0, 0].tolist() == [True, False]
    assert invalid[1, 0, 1, 0]
    assert invalid[1].sum() == 1
    torch.testing.assert_close(top, original, equal_nan=True)
