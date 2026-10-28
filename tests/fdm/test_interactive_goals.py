"""Goal geometry, input semantics and argument validation without Isaac Sim."""

import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unitree_rl_lab.fdm.atlas_scene import AtlasScene
from unitree_rl_lab.fdm.interactive_goals import GoalSamplingCfg, LocalGoalSampler, SpaceRequest
from unitree_rl_lab.fdm.terrain_atlas import build_atlas


@pytest.fixture(scope="module")
def scene():
    return AtlasScene(build_atlas(0).combined_mesh())


def test_sampling_is_uniform_in_area_and_uses_current_center():
    sampler = LocalGoalSampler(None, GoalSamplingCfg(max_attempts=40_000), seed=10)
    center = np.array([17.0, -8.0])
    delta = sampler.candidates(center) - center
    r2 = (delta ** 2).sum(axis=1)
    assert r2.min() >= 4 and r2.max() <= 25
    assert abs(np.mean(r2) - 14.5) < 0.12
    assert abs(np.mean(r2 <= 14.5) - 0.5) < 0.015
    assert np.max(abs(np.mean(delta, axis=0))) < 0.05


@pytest.mark.parametrize("zone", [f"{row}{col}" for row in "ABC" for col in range(1, 5)])
def test_goals_respect_radius_surface_and_boundary(scene, zone):
    start = scene.choose_spawn(zone=zone)
    sampler = LocalGoalSampler(scene, seed=23)
    for _ in range(3):
        goal = sampler.sample(start[:2])
        assert goal is not None
        assert 2 <= math.dist(start[:2], goal[:2]) <= 5
        assert scene.contains(goal[:2])
        assert goal[2] == pytest.approx(scene.validate_goal(goal[:2], reject_obstacle_tops=True))


def test_rejection_is_bounded_without_unsafe_fallback():
    calls = []

    def reject(xy, **_):
        calls.append(xy)
        raise ValueError("unsupported")

    sampler = LocalGoalSampler(SimpleNamespace(validate_goal=reject), GoalSamplingCfg(max_attempts=7))
    assert sampler.sample((0, 0)) is None
    assert len(calls) == 7


def test_platforms_and_ramps_allowed_but_pillar_roofs_excluded(scene):
    assert scene.validate_goal((-18, 0), reject_obstacle_tops=True) == pytest.approx(0.75)
    assert scene.validate_goal((3.95, -1.85), reject_obstacle_tops=True) == pytest.approx(0.4625)
    assert scene.validate_goal((6, 12)) > 0
    with pytest.raises(ValueError, match="pillars or walls"):
        scene.validate_goal((6, 12), reject_obstacle_tops=True)


def test_goal_rng_repeatable_but_advances_between_rounds(scene):
    a, b = LocalGoalSampler(scene, seed=5), LocalGoalSampler(scene, seed=5)
    centers = [(-22.8, 0), (-17.9, 0), (-15, 3)]
    first = [a.sample(center) for center in centers]
    assert first == [b.sample(center) for center in centers]
    assert len(set(first)) == 3


def test_spawn_is_independent_of_old_preset_goal(scene):
    assert scene.choose_spawn(start=(-13.2, 0, 1.0)) == (-13.2, 0, 1.0)
    with pytest.raises(ValueError, match="flat z=0"):
        scene.choose_spawn(start=(-18, 0, 0))


def test_space_requires_release_and_fresh_press_while_waiting():
    gate = SpaceRequest()
    gate.key(True)  # During walking.
    gate.enable(True)
    gate.key(True)  # Still held after arrival.
    assert not gate.consume()
    gate.key(False)
    gate.key(True)
    assert gate.consume()
    assert not gate.consume()
    gate.key(True)
    assert not gate.consume()
    gate.key(False)
    gate.key(True)
    gate.enable(False)  # Pause discards queued requests.
    gate.enable(True)
    assert not gate.consume()


def test_input_callback_consumes_only_space():
    from unitree_rl_lab.fdm.runner.interactive_controls import InteractiveControls

    controls = object.__new__(InteractiveControls)
    controls.request = SpaceRequest()
    controls.request.enable(True)
    controls._keyboard = "viewport"
    controls._carb = SimpleNamespace(
        DeviceType=SimpleNamespace(KEYBOARD=1),
        KeyboardEventType=SimpleNamespace(KEY_PRESS=0, KEY_REPEAT=1, KEY_RELEASE=2, CHAR=3),
        KeyboardInput=SimpleNamespace(SPACE=32),
    )

    def event(kind, key=32, device=1):
        return SimpleNamespace(deviceType=device, event=SimpleNamespace(type=kind, input=key, keyboard="viewport"))

    assert controls._on_input(event(0)) is False
    assert controls.consume_next()
    assert controls._on_input(event(1)) is False
    assert not controls.consume_next()
    assert controls._on_input(event(2)) is False
    assert controls._on_input(event(3, " ")) is False
    assert controls._on_input(event(0, 65)) is True
    assert controls._on_input(event(0, device=2)) is True


@pytest.fixture
def parse(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts" / "fdm"))

    def add_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--rendering_mode", default="balanced")

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(
        AppLauncher=SimpleNamespace(add_app_launcher_args=add_args)))
    entry = importlib.import_module("play_interactive_goals")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")

    def parse_args(*extra):
        monkeypatch.setattr(sys, "argv", ["play_interactive_goals.py", "--checkpoint", str(checkpoint),
                                         "--fdm-checkpoint", str(checkpoint), *extra])
        return entry.parse_args(interactive=True)

    return parse_args


def test_interactive_cli_defaults(parse):
    args = parse()
    assert args.goal is None and args.goal_radius == 5 and args.goal_min_distance == 2
    assert args.num_envs == 1 and not args.headless


@pytest.mark.parametrize("args", [
    ("--headless",), ("--goal", "1", "2"), ("--goal-radius", "nan"),
    ("--goal-min-distance", "0"), ("--goal-min-distance", "6"),
    ("--goal-radius", "2"), ("--goal-min-distance", "inf"),
])
def test_interactive_cli_rejects_inconsistent_usage(parse, args):
    with pytest.raises(SystemExit) as exc:
        parse(*args)
    assert exc.value.code == 2
