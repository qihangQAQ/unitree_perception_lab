from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from unitree_rl_lab.fdm.config import FDMModelCfg, RolloutCfg, TrainCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset
from unitree_rl_lab.fdm.data.shard_writer import iter_episodes
from unitree_rl_lab.fdm.data.source_sampling import select_mixed_pool
from unitree_rl_lab.fdm.data.window import window_targets
from unitree_rl_lab.fdm.models import G1HeightFDM
from unitree_rl_lab.fdm.planning.goals import GoalCfg, LocalGoalManager, NavigationGrid, SpatialSplit
from unitree_rl_lab.fdm.planning.mppi import MPPICfg, MPPIPlanner
from unitree_rl_lab.fdm.runner.stage2_collector import Stage2RolloutCollector
from unitree_rl_lab.fdm.runner.navigation_evaluation import compare_navigation, evaluate_navigation
from unitree_rl_lab.fdm.training import FDMTrainer
from unitree_rl_lab.fdm.training.stage2 import Stage2Cfg, load_initial_model, stage2_settings, validate_stage2_resume
from unitree_rl_lab.fdm.utils.se2 import integrate_body_twists
from test_fixed_collection import _BatchEnv, _maps
from test_schema_and_dataset import _episode


def _v4(collision=2, source=1, episode_id=4):
    episode = _episode(collision, episode_id)
    n = episode.num_frames
    episode.command_source = torch.full((n,), source, dtype=torch.int8)
    episode.goal_xy_w = torch.zeros(n, 2)
    episode.planner_model_version = torch.full((n,), 7, dtype=torch.int32)
    # Every command boundary changes the entire proposal; v3 and v4 targets
    # must therefore differ while sharing exactly the same motion labels.
    episode.command_plan[:, :, 0] += torch.arange(n)[:, None] * 10
    episode.command = episode.command_plan[:, 0].clone()
    return episode


def test_executed_commands_and_last_pre_collision_tail():
    episode = _v4()
    episode.command_plan[-1] = -777  # Event/reset frame must never supply commands.
    episode.command[-1] = -999
    targets = window_targets(episode, torch.tensor([0, 1]), 10)
    expected = torch.cat((episode.command[:2], episode.command_plan[1, 1:9]))
    torch.testing.assert_close(targets["future_commands"][0], expected)
    torch.testing.assert_close(targets["future_commands"][1], episode.command_plan[1])
    assert targets["future_collision"][0].tolist() == [0.] + [1.] * 9
    assert torch.equal(targets["future_pose"][0, 1:], targets["future_pose"][0, 1].expand(9, -1))


@pytest.mark.parametrize("cache", ["shard", "memory", "mmap"])
def test_v4_windows_use_executed_commands_and_never_cross_fragment(tmp_path, cache):
    with EpisodeShardWriter(tmp_path, "train", {}, schema_version=4) as writer:
        writer.append(_v4(None, episode_id=1))
        writer.append(_v4(episode_id=2))
    dataset = FDMWindowDataset(tmp_path, "train", collect_statistics=True, log_interval_s=0)
    assert len(dataset) == 3
    dataset.prepare_cache(cache, log_interval_s=0)
    torch.testing.assert_close(dataset[0]["future_commands"], _v4(None).command[:10])
    torch.testing.assert_close(dataset[1]["future_commands"], window_targets(_v4(), torch.tensor([0]), 10)["future_commands"][0])
    assert all(item.command_source == 1 for item in dataset.indices)


def test_schema_rejects_mixed_semantics_and_partial_provenance(tmp_path):
    episode = _v4()
    with EpisodeShardWriter(tmp_path / "v3", "train", {}) as writer:
        with pytest.raises(ValueError, match="semantics"):
            writer.append(episode)
    with EpisodeShardWriter(tmp_path / "v4", "train", {}, schema_version=4) as writer:
        with pytest.raises(ValueError, match="semantics"):
            writer.append(_episode())
    episode.goal_xy_w = None
    with pytest.raises(ValueError, match="all be present"):
        episode.validate()


def test_source_quotas_cap_rare_collision_repetition(tmp_path):
    with EpisodeShardWriter(tmp_path, "train", {}, schema_version=4) as writer:
        for index in range(20):
            writer.append(_v4(2 if index in (0, 10) else None, source=int(index >= 10), episode_id=index))
    dataset = FDMWindowDataset(tmp_path, "train", log_interval_s=0)
    report = select_mixed_pool(dataset, 100, mppi_fraction=.5, seed=4, max_repeats=2, collision_fraction=.8)
    assert max(Counter(dataset.indices).values()) <= 2
    assert all(source["collision"] <= 4 for source in report["sources"].values())
    assert report["sampled_windows"] == 44 < report["requested_windows"]
    assert report["sources"]["mppi"]["shortfall"] > 0


def test_source_pool_rounding_never_exceeds_tiny_quota(tmp_path):
    low = _v4(None, source=0, episode_id=2)
    low.state_history_raw[..., :3] = 0
    with EpisodeShardWriter(tmp_path, "train", {}, schema_version=4) as writer:
        writer.append(_v4(source=0, episode_id=1))
        writer.append(low)
        writer.append(_v4(None, source=0, episode_id=3))
    dataset = FDMWindowDataset(tmp_path, "train", log_interval_s=0)
    report = select_mixed_pool(dataset, 3, mppi_fraction=0, seed=1, collision_fraction=.5, low_motion_fraction=.5)
    assert report["sampled_windows"] == len(dataset) == 3


def test_cached_context_preserves_state_dict_and_full_forward():
    torch.set_num_threads(1)
    model = G1HeightFDM().eval()
    state, proprio = torch.randn(3, 10, 5), torch.randn(3, 10, 96)
    height, commands = torch.randn(3, 1, 60, 46), torch.rand(3, 10, 3)
    with torch.inference_mode():
        full = model(state, proprio, height, commands)
        context = model.encode_context(state, proprio, height)
        for start in range(3):
            cached = model.predict_from_context(context[start:start + 1], commands[start:start + 1])
            for key in full:
                torch.testing.assert_close(full[key][start:start + 1], cached[key], atol=1e-5, rtol=1e-5)
    assert not any("context" in name for name in model.state_dict())


class _IdealFDM(torch.nn.Module):
    def __init__(self, nan=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cfg = FDMModelCfg()
        self.nan = nan
        self.encodes = 0
        self.heights = []

    def encode_context(self, state, proprio, height):
        self.encodes += 1
        self.heights.extend(height[:, 0, 0, 0].tolist())
        return state[:, 0, :1]

    def predict_from_context(self, context, commands):
        pose = integrate_body_twists(commands, .5)
        if self.nan:
            pose[:] = torch.nan
        return {"future_pose": pose, "collision_logits": torch.full(commands.shape[:2], -8., device=commands.device)}


def _inputs(n=2):
    return (torch.arange(n), torch.zeros(n, 10, 5), torch.zeros(n, 10, 96), torch.zeros(n, 1, 60, 46),
            torch.zeros(n, 1, 60, 46, dtype=torch.bool), torch.tensor([[2., .2]]).repeat(n, 1))


def test_mppi_chunk_equivalence_bounds_and_one_observation_encode():
    a, b = _IdealFDM(), _IdealFDM()
    left = MPPIPlanner(a, 2, cfg=MPPICfg(candidates=8, inference_chunk=1), seed=23)
    right = MPPIPlanner(b, 2, cfg=MPPICfg(candidates=8, inference_chunk=64), seed=23)
    first = left.plan(*_inputs())
    second = right.plan(*_inputs())
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
    assert a.encodes == b.encodes == 1
    assert ((first >= left.lower) & (first <= left.upper)).all()
    unchanged = left.mean[1].clone()
    left.reset(torch.tensor([0]))
    assert not left.initialized[0] and left.initialized[1]
    torch.testing.assert_close(left.mean[1], unchanged)


@pytest.mark.parametrize("failure", ["nan", "missing_map"])
def test_mppi_all_invalid_stops(failure):
    planner = MPPIPlanner(_IdealFDM(nan=failure == "nan"), 2, cfg=MPPICfg(candidates=3))
    args = list(_inputs())
    if failure == "missing_map":
        args[4].fill_(True)
    result = planner.plan(*args)
    assert torch.isfinite(result).all() and not result.any()
    assert planner.all_invalid == 2


def test_weighted_mean_is_rescored_and_rejected_when_unsafe():
    planner = MPPIPlanner(_IdealFDM(), 2, cfg=MPPICfg(candidates=3))
    calls = []
    def score(context, plans, rows, goals, invalid, previous, map_transform=None):
        calls.append(plans.clone())
        if len(plans) == 6:
            return torch.zeros(6), torch.zeros(6)
        return torch.full((2,), 100.), torch.ones(2)
    with patch.object(planner, "_score", side_effect=score):
        result = planner.plan(*_inputs())
    assert len(calls) == 2 and planner.fallbacks == 2
    torch.testing.assert_close(result, calls[0].view(2, 3, 10, 3)[:, 0])


def test_map_validity_uses_torso_frame_transform():
    planner = MPPIPlanner(_IdealFDM(), 1, cfg=MPPICfg(candidates=3))
    commands = torch.zeros(1, 10, 3)
    commands[..., 0] = .4
    invalid = torch.zeros(1, 1, 60, 46, dtype=torch.bool)
    invalid[0, 0, 30, 7] = True  # First root-frame prediction: x=.2, y=0.
    args = (torch.zeros(1, 1), commands, torch.tensor([0]), torch.tensor([[2., 0.]]), invalid, torch.zeros(1, 3))
    assert not torch.isfinite(planner._score(*args)[0]).all()
    rotated = planner._score(*args, map_transform=torch.tensor([[0., 0., torch.pi / 2]]))[0]
    assert torch.isfinite(rotated).all()


def _terrain():
    free = torch.ones(150, 50, dtype=torch.bool)
    free[:, 25] = False  # Wall along y=0, separating each split into two components.
    return SimpleNamespace(all_usd_origins=torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]]),
                           origin_split_ids=torch.tensor([0, 1, 2]), cfg=SimpleNamespace(usd_uniform_env_spacing=10.),
                           fdm_navigation_grid={"origin": torch.tensor([-5., -5.]), "resolution": .2, "walkable": free})


def test_local_goals_share_component_and_actual_spatial_split():
    grid = NavigationGrid(_terrain())
    lookup = grid.spatial.lookup(torch.tensor([[4.99, 0.], [5., 0.], [25., 0.], [-5.01, 0.]]))
    assert lookup.tolist() == [0, 1, -1, -1]
    assert grid.lookup(torch.tensor([[0., -1.], [0., 1.]])).unique().numel() == 2
    goals = LocalGoalManager(grid, 2, seed=10)
    state = torch.tensor([[0., -2., 1., 0., 0., 0., 1., 0.], [10., 2., 1., 0., 0., 0., 1., 0.]])
    ids = torch.arange(2)
    goals.update(ids, state, 0.)
    assert goals.active.all()
    assert torch.equal(grid.lookup(goals.goal), grid.lookup(state[:, :2]))
    assert torch.equal(grid.spatial.lookup(goals.goal), grid.spatial.lookup(state[:, :2]))
    initial = goals.goal.clone()
    goals.update(ids, state, .5)
    torch.testing.assert_close(goals.goal, initial)


def _open_grid():
    return SimpleNamespace(device=torch.device("cpu"), lookup=lambda xy: torch.ones(xy.shape[:-1], dtype=torch.long),
                           spatial=SimpleNamespace(lookup=lambda xy: torch.zeros(xy.shape[:-1], dtype=torch.long)))


def test_stage2_subcollections_release_buffer_and_record_same_planner_maps(tmp_path):
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg)
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    model = _IdealFDM()
    counter = 0
    def maps(sensor, ids, **kwargs):
        nonlocal counter
        counter += 1
        height, invalid = _maps(sensor, ids)
        height.fill_(counter / 10.)
        return height, invalid
    with EpisodeShardWriter(tmp_path, "train", {}, schema_version=4) as writer, patch(
        "unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=maps
    ):
        collector = Stage2RolloutCollector(env, env.observations, policy, writer, cfg, model=model,
                                           navigation_grid=_open_grid(), mppi_cfg=MPPICfg(candidates=3))
        for batch in range(2):
            result = collector.collect(12, seed=4 + batch, mppi_fraction=.5, model_version=batch, min_fill=1, log_interval_s=0)
            assert collector.buffer is None and collector.pending_frames == 0
            assert result["frames_recorded"] == 24 and result["mppi_envs"] == 1
    episodes = list(iter_episodes(tmp_path, "train"))
    assert {int(ep.command_source[0]) for ep in episodes} == {0, 1}
    assert {int(ep.planner_model_version[0]) for ep in episodes} == {0, 1}
    for episode in episodes:
        if episode.command_source[0] == 1:
            for height in episode.height_map[episode.has_outgoing_command, 0, 0, 0]:
                assert float(height) in model.heights


def test_navigation_pair_uses_same_starts_goals_and_waits_for_outcomes():
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg)
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    with patch("unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=_maps):
        runs = [evaluate_navigation(env, env.observations, policy, cfg, _IdealFDM(), _open_grid(),
                                    mppi_cfg=MPPICfg(candidates=3), goal_cfg=GoalCfg(), seed=31, log_interval_s=0)
                for _ in range(2)]
    comparison = compare_navigation(*runs)
    assert comparison["matched_scenarios"]
    assert runs[0]["navigation"]["collision"] == 2
    assert comparison["delta"]["collision_rate"] == 0
    runs[1]["navigation"]["scenarios"][0]["goal"][0] += 1
    assert not compare_navigation(*runs)["matched_scenarios"]


def test_stage1_initialization_loads_weights_with_fresh_stage2_schedule(tmp_path):
    torch.set_num_threads(1)
    model = G1HeightFDM()
    initial = tmp_path / "initial.pt"
    torch.save({"model_cfg": asdict(model.cfg), "model_state_dict": model.state_dict(),
                "global_epoch": 144, "optimizer_state_dict": {"bad": "must not restore"}}, initial)
    loaded, cfg = load_initial_model(initial)
    trainer = FDMTrainer(loaded, TrainCfg(collection_rounds=10, epochs_per_round=4, learning_rate=1e-4, device="cpu"))
    assert trainer.global_epoch == 0 and trainer.scheduler.T_max == 40
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-4 and not trainer.optimizer.state
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name])
    settings = stage2_settings(Stage2Cfg(), MPPICfg(), GoalCfg(), RolloutCfg(), {})
    with pytest.raises(ValueError, match="stage-two"):
        validate_stage2_resume({"round": 0}, settings, tmp_path / "manifest.json")


def test_split_exit_seals_without_recording_outside_pose(tmp_path):
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg, collision_steps=(100000, 100000))
    grid = _open_grid()
    grid.spatial.lookup = lambda xy: (((xy[:, 0] % 100) > 1.5) & ((xy[:, 0] % 100) < 50)).long()
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    with EpisodeShardWriter(tmp_path, "train", {}, schema_version=4) as writer, patch(
        "unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=_maps
    ):
        collector = Stage2RolloutCollector(env, env.observations, policy, writer, cfg, model=_IdealFDM(),
                                           navigation_grid=grid, mppi_cfg=MPPICfg(candidates=3))
        collector.collect(12, seed=1, mppi_fraction=.5, min_fill=1, log_interval_s=0)
    episodes = list(iter_episodes(tmp_path, "train"))
    assert any(int(ep.termination_reason[-1]) == 6 for ep in episodes)
    assert all(not ep.collision_now.any() for ep in episodes)
    assert all(((ep.state_history_raw[:, 0, 0] % 100) <= 1.5).all() for ep in episodes)


def test_unavailable_goals_reset_instead_of_spending_whole_round_stopped(tmp_path):
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg, collision_steps=(100000, 100000))
    grid = _open_grid()
    grid.lookup = lambda xy: torch.zeros(xy.shape[:-1], dtype=torch.long)
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    with EpisodeShardWriter(tmp_path, "train", {}, schema_version=4) as writer, patch(
        "unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=_maps
    ):
        collector = Stage2RolloutCollector(env, env.observations, policy, writer, cfg, model=_IdealFDM(),
                                           navigation_grid=grid, goal_cfg=GoalCfg(stall_s=.5), mppi_cfg=MPPICfg(candidates=3))
        result = collector.collect(12, seed=1, mppi_fraction=1, min_fill=1, log_interval_s=0)
    episodes = list(iter_episodes(tmp_path, "train"))
    assert result["navigation"]["goal_unavailable_resets"] > 0
    assert any(int(ep.termination_reason[-1]) == 7 for ep in episodes)
    assert all(not ep.collision_now.any() for ep in episodes)
    assert all(not ep.command.any() for ep in episodes)


def test_two_round_entrypoint_and_current_stage2_resume(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    torch.set_num_threads(1)
    scripts = Path(__file__).resolve().parents[2] / "scripts" / "fdm"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("stage2_entry_test", scripts / "train_fdm_stage2.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    model = G1HeightFDM()
    initial = tmp_path / "initial.pt"
    torch.save({"model_cfg": asdict(model.cfg), "model_state_dict": model.state_dict()}, initial)
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg)
    env.close = lambda: None
    grid = _open_grid()
    active_split = [1]
    def activate(split):
        active_split[0] = {"train": 0, "val": 1, "test": 2}[split]
        env.scene.terrain.origin_split_ids.fill_(active_split[0])
    env.scene.terrain.activate_split = activate
    grid.spatial.lookup = lambda xy: torch.full(xy.shape[:-1], active_split[0], dtype=torch.long)
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    def environment(args, rollout):
        activate("val")
        return env, env.observations, policy, grid
    monkeypatch.setattr(entry, "create_environment", environment)
    monkeypatch.setattr(entry, "dataset_metadata", lambda env, rollout: {"rollout": rollout.as_dict() | {"split": "val"}})
    args = SimpleNamespace(dataset=str(tmp_path / "dataset"), output=str(tmp_path / "output"),
                           fdm_checkpoint=str(initial), resume=None, checkpoint="policy.pt", terrain_usd="terrain.usd",
                           task=cfg.task_name, num_envs=2, seed=3, disable_policy_corruption=False, device="cpu",
                           collection_rounds=2, collection_batches=2, epochs_per_round=1, batch_size=8, workers=0,
                           samples_per_round=16, learning_rate=1e-4, initial_mppi_fraction=.5, mppi_fraction=.5,
                           initial_mix_rounds=1, max_window_repeats=4, navigation_eval_interval=0,
                           mppi_candidates=3, mppi_iterations=1, inference_chunk=8, mppi_temperature=1., risk_threshold=.55,
                           goal_min_distance=1.5, goal_max_distance=3., goal_timeout=20., goal_stall_time=5.,
                           frames_per_env=12, validation_frames_per_env=12, collection_min_fill=1., collection_tail_factor=1.5,
                           collection_max_steps=0, dataset_cache="memory", dataset_cache_gb=1., log_interval=0)
    with patch("unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=_maps):
        entry._train(args)
        final = torch.load(Path(args.output) / "fdm_round_001.pt", weights_only=False)
        assert final["global_epoch"] == 2 and final["stage2"]["next_planner_model_version"] == 2
        assert set(final["metrics"]["validation"]) == {"random", "mppi"}
        args.resume = str(Path(args.output) / "fdm_round_000.pt")
        args.fdm_checkpoint = None
        args.output = str(tmp_path / "resumed")
        entry._train(args)
    resumed = torch.load(Path(args.output) / "fdm_round_001.pt", weights_only=False)
    assert resumed["global_epoch"] == 2 and resumed["stage2"]["initial_checkpoint_sha256"] == final["stage2"]["initial_checkpoint_sha256"]
    assert resumed["scheduler_state_dict"]["last_epoch"] == 2
