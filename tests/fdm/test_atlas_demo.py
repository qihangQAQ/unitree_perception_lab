"""CPU checks for atlas inference setup; no simulated locomotion claims."""

import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unitree_rl_lab.fdm.atlas_scene import AtlasScene
from unitree_rl_lab.fdm.terrain_atlas import SIZE, atlas_terrain, build_atlas


@pytest.fixture(scope="module")
def scene():
    # Emulate TerrainGenerator's net tile-local -> world-centred translation.
    meshes, origin = atlas_terrain(0.5, SimpleNamespace(size=SIZE, seed=0))
    meshes[0].apply_translation((-24.0, -18.0, 0.0))
    np.testing.assert_allclose(origin - np.array([24, 18, 0]), [-21.5, 14.5, 0])
    return AtlasScene(meshes[0], seed=0)


@pytest.mark.parametrize("zone", [f"{row}{col}" for row in "ABC" for col in range(1, 5)])
def test_all_zone_presets_have_supported_endpoints(scene, zone):
    start, goal = scene.choose_endpoints(zone=zone)
    assert scene.contains(start[:2]) and scene.contains(goal)
    assert scene.start_height == pytest.approx(0)
    assert scene.goal_height == pytest.approx(0)
    assert math.dist(start[:2], goal) >= 1


def test_original_command_goal_is_preserved_on_b3_ramp(scene):
    start, goal = scene.choose_endpoints(goal=(3.95, -1.85))
    assert scene.selected_zone == "B3"
    assert goal == (3.95, -1.85)
    np.testing.assert_allclose(start[:2], (1.4, 0))
    assert start[2] == pytest.approx(math.atan2(-1.85, 3.95 - 1.4))
    assert scene.start_height == pytest.approx(0)
    assert scene.goal_height == pytest.approx(0.4625)


def test_default_trial_keeps_stairs_between_endpoints(scene):
    start, goal = scene.choose_endpoints()
    assert scene.selected_zone == "B1"
    np.testing.assert_allclose(start, (-22.8, 0, 0))
    np.testing.assert_allclose(goal, (-13.2, 0))
    # The plateau is inside the live allowed bounds, not a 2-D forbidden polygon.
    assert scene.contains((-18, 0))
    assert scene.height_at((-18, 0))[0] == pytest.approx(0.75)


def test_explicit_platform_start_keeps_yaw_and_uses_actual_imported_mesh():
    atlas = build_atlas(0)
    for part in atlas.parts:
        if part.name == "B1_stair_bridge":
            vertices = part.mesh.vertices.copy()
            vertices[vertices[:, 2] > 0, 2] *= 1.2  # Actual platform now 0.90 m.
            part.mesh.vertices = vertices
    altered = AtlasScene(atlas.combined_mesh(), seed=0)
    start, goal = altered.choose_endpoints(start=(-18, 0, 1.25), goal=(-13.2, 0))
    assert start == (-18, 0, 1.25)
    assert goal == (-13.2, 0)
    assert altered.start_height == pytest.approx(0.90)


def test_zone_preset_can_be_overridden_without_moving_explicit_endpoints(scene):
    start, goal = scene.choose_endpoints(zone="B2", start=(1.4, 0, -0.7), goal=(3.95, -1.85))
    assert start == (1.4, 0, -0.7)
    assert goal == (3.95, -1.85)


def test_only_explicit_start_selects_its_zone_goal(scene):
    start, goal = scene.choose_endpoints(start=(-22.8, 0, 0.2))
    assert scene.selected_zone == "B1"
    assert start == (-22.8, 0, 0.2)
    np.testing.assert_allclose(goal, (-13.2, 0))


@pytest.mark.parametrize("args", [
    {"goal": (16.2, 8)},  # Wall in A4.
    {"goal": (-21, 0)},   # Across multiple stair risers in B1.
    {"start": (3.95, -1.85, 0)},  # Spawn on a ramp, not level ground.
    {"start": (-12, 12.8, 0)},    # Thin seam wall.
    {"goal": (23.9, 0)},
    {"goal": (25, 0)},
    {"goal": (float("nan"), 0)},
    {"start": (1.4, 0, float("inf"))},
    {"zone": "D1"},
    {"start": (-22.8, 0, 0), "goal": (-22.7, 0)},
])
def test_invalid_endpoints_raise_instead_of_silently_moving(scene, args):
    with pytest.raises(ValueError):
        scene.choose_endpoints(**args)


def test_world_boundary_is_large_and_does_not_exclude_raised_terrain(scene):
    assert scene.contains((-22, -16))
    assert scene.contains((21, 15))
    assert scene.contains((6, 0))  # Raised B3 platform.
    assert not scene.contains((24, 0))
    assert not scene.contains((0, -18))
    assert not scene.contains((float("nan"), 0))


def test_goal_marker_can_sample_slope_instead_of_being_buried_at_z_zero(scene):
    angles = np.linspace(0, math.tau, 49)
    xy = np.array((3.95, -1.85)) + 0.4 * np.column_stack((np.cos(angles), np.sin(angles)))
    z = scene.height_at(xy)
    assert np.isfinite(z).all() and (z > 0).all()
    np.testing.assert_allclose(z, (xy[:, 0] - 2.1) * 0.25, atol=1e-10)


def test_changed_seed_preserves_all_presets():
    scene = AtlasScene(build_atlas(7).combined_mesh(), seed=7)
    for zone in scene.zones:
        scene.choose_endpoints(zone=zone)


def test_cli_keeps_original_flags_and_switches_to_atlas(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[2] / "scripts" / "fdm"
    monkeypatch.syspath_prepend(str(scripts))

    def add_app_args(parser):
        parser.add_argument("--rendering_mode")

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(
        AppLauncher=SimpleNamespace(add_app_launcher_args=add_app_args)))
    cli = importlib.import_module("play_mixed_pillars")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    original = ["play_mixed_pillars.py", "--checkpoint", str(checkpoint),
                "--fdm-checkpoint", str(checkpoint), "--seed", "0",
                "--goal", "3.95", "-1.85", "--mppi-candidates", "256",
                "--rendering_mode", "balanced"]
    monkeypatch.setattr(sys, "argv", original)
    args = cli.parse_args()
    assert cli.TASK == "Unitree-G1-29dof-FDM-Atlas"
    assert args.goal == [3.95, -1.85] and args.start is None and args.zone is None
    monkeypatch.setattr(sys, "argv", original + ["--zone", "B3"])
    assert cli.parse_args().zone == "B3"
