"""Geometry, CLI isolation, and evaluation boundaries for generated arenas."""

from dataclasses import replace
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from unitree_rl_lab.fdm.benchmarks import BENCHMARK_TASK, BoxObstacle, get_scene, scene_meshes
from unitree_rl_lab.fdm.config import RolloutCfg
from unitree_rl_lab.fdm.planning.navigation import NavigationCfg
from unitree_rl_lab.fdm.runner.navigation_runner import NavigationRunner
from test_navigation import _NavEnv, _planner
from test_fixed_collection import _maps


@pytest.mark.parametrize('name', ['empty', 'box', 'wall'])
def test_scene_requires_detour_and_has_safe_reference_route(name):
    scene = get_scene(name)
    info = scene.metadata()
    assert info['direct_path_blocked'] == (name != 'empty')
    assert info['reference_used_for_control'] is False
    path = info['reference_path']
    assert path[0] == scene.start[:2] and path[-1] == scene.goal
    points = torch.tensor(path)
    assert scene.free(points).all()
    for a, b in zip(path, path[1:]):
        assert scene.segment_clear(a, b)
    if name == 'empty':
        assert info['reference_path_length_m'] == pytest.approx(5.)
    else:
        assert info['reference_path_length_m'] > 5.
    assert scene.metadata() == info


def test_reference_rejects_wall_that_blocks_entire_arena():
    scene = replace(get_scene('wall'), obstacles=(BoxObstacle((0., 0.), (.2, 10., 1.5)),))
    with pytest.raises(ValueError, match='no route'):
        scene.reference_path()
    with pytest.raises(ValueError, match='clearance'):
        get_scene('box').reference_path((0., 0.))


def test_thin_wall_segment_checks_interior_and_body_width():
    scene = get_scene('wall')
    # Both endpoints are free, but crossing the expanded wall is not.
    assert scene.free(torch.tensor([[-2., 1.2], [2., 1.2]])).all()
    assert not scene.segment_clear((-2., 1.2), (2., 1.2))
    assert scene.segment_clear((-2., 1.5), (2., 1.5))
    assert not scene.contains(torch.tensor([[4.7, 0.]])).item()
    assert not scene.contains(torch.tensor([[float('nan'), 0.]])).item()


@pytest.mark.parametrize('name', ['empty', 'box', 'wall'])
def test_mesh_geometry_matches_preflight_spec_and_stays_small(name):
    scene = get_scene(name)
    meshes = scene_meshes(scene)
    np.testing.assert_allclose(meshes[0].bounds, [[-5., -5., -.1], [5., 5., 0.]])
    assert sum(len(m.faces) for m in meshes) <= 24
    for mesh, obstacle in zip(meshes[1:], scene.obstacles):
        np.testing.assert_allclose(mesh.extents, obstacle.size)
        np.testing.assert_allclose(mesh.centroid, (*obstacle.center, obstacle.size[2]/2), atol=1e-12)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'/'fdm'))
    def add_args(parser):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cpu')
    monkeypatch.setitem(sys.modules, 'isaaclab.app', SimpleNamespace(
        AppLauncher=SimpleNamespace(add_app_launcher_args=add_args)))
    module = importlib.import_module('_navigation_common')
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'test checkpoint')
    import unitree_rl_lab.fdm.config as config
    monkeypatch.setattr(config, 'DEFAULT_TERRAIN_USD', tmp_path/'missing.usd')
    def parse(*extra, evaluation=False):
        flag = '--fdm-checkpoints' if evaluation else '--fdm-checkpoint'
        monkeypatch.setattr(sys, 'argv', ['navigate_fdm.py', '--checkpoint', str(checkpoint),
                                         flag, str(checkpoint), '--output', str(tmp_path/'output'), *extra])
        return module.parse_args(evaluation=evaluation)
    return parse


@pytest.mark.parametrize('evaluation', [False, True])
def test_generated_cli_does_not_require_usd_and_defaults_to_one_env(cli, evaluation):
    args = cli('--scene', 'box', evaluation=evaluation)
    assert args.terrain_usd is None and args.split == 'benchmark' and args.num_envs == 1


@pytest.mark.parametrize('extra', [('--split', 'val'), ('--num-envs', '2'), ('--start', '0', '0', '0'),
                                  ('--goal', '3', '0'), ('--goal-offset', '3', '0'),
                                  ('--scenarios', 'scene.json'), ('--terrain-usd', 'old.usd')])
def test_generated_cli_rejects_conflicting_scene_options(cli, extra):
    with pytest.raises(SystemExit) as exc:
        cli('--scene', 'box', *extra)
    assert exc.value.code == 2


def test_usd_cli_preserves_defaults_and_missing_asset_check(cli, tmp_path):
    with pytest.raises(SystemExit):
        cli()
    terrain = tmp_path/'terrain.usd'
    terrain.write_bytes(b'test terrain')
    args = cli('--terrain-usd', str(terrain), evaluation=True)
    assert args.num_envs == 8 and args.split == 'test' and args.scene is None


def test_benchmark_runner_uses_explicit_boundary_without_usd_metadata():
    env = _NavEnv()
    env.num_envs = 2  # Existing fake environment uses two rows; normal benchmark CLI enforces one.
    env.scene.terrain = SimpleNamespace()
    rollout = RolloutCfg(num_envs=2)
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2,29), reset=lambda: None)
    runner = NavigationRunner(env, policy, _planner(), rollout,
                              NavigationCfg(initialization_s=1., timeout_s=.04, log_interval_s=0),
                              region_contains=lambda xy: xy[:,0] < .5)
    with patch('unitree_rl_lab.fdm.runner.navigation_observation.door_aware_height_map', side_effect=_maps):
        result = runner.run(seed=2, goal_factory=lambda raw: (torch.tensor([[3.,0.],[3.,1.]]),
                                                              torch.ones(2,dtype=torch.bool)))
    assert result['trials'][1]['outcome'] == 'out_of_region'
    assert result['trials'][0]['outcome'] == 'timeout'


def test_benchmark_rollout_is_not_a_usd_training_split():
    RolloutCfg(task_name=BENCHMARK_TASK, terrain_usd_path='', split='benchmark', num_envs=1).validate()
    with pytest.raises(ValueError, match='Benchmark split'):
        RolloutCfg(split='benchmark', num_envs=1).validate()
