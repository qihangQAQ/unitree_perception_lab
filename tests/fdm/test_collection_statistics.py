import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from unitree_rl_lab.fdm.config import RolloutCfg, TrainCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset
from unitree_rl_lab.fdm.data.statistics import DatasetStatistics
from unitree_rl_lab.fdm.runner.collection_log import CollectionLog
from unitree_rl_lab.fdm.runner.collector import FDMRolloutCollector
from unitree_rl_lab.fdm.training.losses import FDMLoss, LossAccumulator
from unitree_rl_lab.fdm.training.metrics import MetricAccumulator
from unitree_rl_lab.fdm.training.trainer import FDMTrainer
from test_schema_and_dataset import _episode


def _dataset(path: Path, episodes, **kwargs):
    with EpisodeShardWriter(path, "train", {"fixture": "statistics"}, max_frames_per_shard=12) as writer:
        for episode in episodes:
            writer.append(episode)
    return FDMWindowDataset(path, "train", collect_statistics=True, allow_empty=True, log_interval_s=0, **kwargs)


def test_natural_window_and_episode_statistics_and_reused_report(tmp_path):
    collision = _episode()
    collision.contact_groups[-1, :2] = True
    dataset = _dataset(tmp_path / "data", [collision, _episode(None, episode_id=5)])
    summary = dataset.statistics
    assert summary["counts"] == {
        "episodes": 2, "frames": 15, "candidate_windows": 13, "valid_windows": 3,
        "excluded_windows": 10, "non_start_frames": 2, "collision_episodes": 1,
        "collision_windows": 2, "low_motion_windows": 0,
    }
    assert summary["collision_episode_fraction"] == 0.5
    assert summary["collision_window_fraction"] == pytest.approx(2 / 3)
    assert summary["statistics"]["endpoint_distance_m"]["mean"] == pytest.approx(13 / 3)
    assert summary["statistics"]["commands"]["mean"][0] == 5.5
    assert summary["statistics"]["velocity"]["max_abs"] == [2.0, 0.0, 0.0]
    assert summary["termination_reasons"] == {"collision": 1, "timeout": 1}
    assert summary["collision_groups"]["torso"]["episodes"] == 1
    assert summary["collision_groups"]["left_hand"]["episodes"] == 1
    assert summary["collision_groups"]["right_hand"]["episodes"] == 0
    assert summary["endpoint_histograms"]["distance_m"]["[1,2)"]["count"] == 1
    log = CollectionLog(tmp_path / "output")
    path = log.save_summary(dataset, reused=True)
    report = json.loads(path.read_text())
    assert report["reused"]
    assert report["collection"] is report["contact_force_peak_n"] is None
    assert len(report["shards"]) == 2
    assert report["dataset"] == summary


def test_real_timestamps_yaw_wrap_and_no_collision_padding_in_kinematics():
    episode = _episode()
    episode.timestamp[:] = torch.tensor([0.0, 0.5, 0.7], dtype=torch.float64)
    episode.state_history_raw[:, :, 0] = torch.tensor([0.0, 0.5, 0.7])[:, None]
    yaw = torch.deg2rad(torch.tensor([179.0, -179.0, -178.2]))
    episode.state_history_raw[:, :, 5] = torch.sin(yaw / 2)[:, None]
    episode.state_history_raw[:, :, 6] = torch.cos(yaw / 2)[:, None]
    stats = DatasetStatistics(10, 0.5, 0.05)
    stats.update_episode(episode, [])
    result = stats.result()["statistics"]
    assert result["velocity"]["count"] == [2, 2, 2]
    assert result["acceleration"]["count"] == [1, 1, 1]
    assert result["velocity"]["mean"][2] == pytest.approx(math.radians(4), abs=2e-6)
    assert result["acceleration"]["max_abs"] == pytest.approx([0, 0, 0], abs=3e-5)


def test_empty_short_stationary_and_nonfinite_data(tmp_path):
    empty = _dataset(tmp_path / "empty", [])
    assert empty.statistics["collision_window_fraction"] is None
    assert empty.statistics["statistics"]["endpoint_distance_m"]["mean"] is None
    short = _episode()
    short.collision_now[:] = False
    short.state_history_raw[..., 7] = 0
    short.termination_reason[-1] = 3
    dataset = _dataset(tmp_path / "short", [short])
    assert len(dataset) == 0
    assert dataset.statistics["counts"]["excluded_windows"] == 2
    stationary = _episode(None)
    stationary.state_history_raw[..., 0] = 0
    stationary.command_plan[:] = 0
    stationary.height_map[0, 0, 0, 0] = torch.nan
    stationary.height_map_invalid[0, 0, 0, 0] = True
    dataset = _dataset(tmp_path / "stationary", [stationary])
    stats = dataset.statistics
    assert stats["low_motion_window_fraction"] == 1
    assert stats["statistics"]["endpoint_distance_m"]["std"] == 0
    assert stats["statistics"]["baseline_position_error_m"]["max"] == 0
    assert stats["endpoint_histograms"]["distance_m"]["[0,1)"]["fraction"] == 1
    assert stats["quality"]["nonfinite_values"]["height_map"] == 1
    json.dumps(stats, allow_nan=False)


def test_collision_baseline_keeps_original_plan(tmp_path):
    episode = _episode()
    episode.state_history_raw[..., 0] *= 0.5
    episode.command_plan[:] = 0
    episode.command_plan[..., 0] = 1
    dataset = _dataset(tmp_path, [episode])
    # First target: x=.5, then collision x=1 is frozen while ideal x reaches 5.
    assert dataset.statistics["statistics"]["baseline_position_error_m"]["max"] == 4.5
    assert dataset.statistics["collision_window_fraction"] == 1


def test_losses_and_metrics_are_independent_of_batch_partitions():
    generator = torch.Generator().manual_seed(23)
    target = {
        "future_pose": torch.randn(5, 10, 4, generator=generator),
        "future_commands": torch.randn(5, 10, 3, generator=generator),
        "future_collision": torch.randint(0, 2, (5, 10), generator=generator).float(),
        "valid_mask": torch.rand(5, 10, generator=generator) > 0.2,
    }
    prediction = {
        "future_pose": torch.randn(5, 10, 4, generator=generator),
        "collision_logits": torch.randn(5, 10, generator=generator),
    }
    loss_fn = FDMLoss(TrainCfg(), collision_pos_weight=2.5)
    expected_loss = loss_fn(prediction, target)
    full_metrics = MetricAccumulator()
    full_metrics.update(prediction, target)
    for batch_size in (1, 2, 3, 5):
        losses = LossAccumulator(loss_fn)
        metrics = MetricAccumulator()
        for start in range(0, 5, batch_size):
            batch = {key: value[start : start + batch_size] for key, value in target.items()}
            predicted = {key: value[start : start + batch_size] for key, value in prediction.items()}
            losses.update(predicted, batch)
            metrics.update(predicted, batch)
        assert losses.result() == pytest.approx({key: float(value) for key, value in expected_loss.items()}, rel=1e-6)
        assert metrics.result() == pytest.approx(full_metrics.result(), rel=1e-10, abs=1e-10)


class _DummyPolicy:
    def act(self, observations):
        return torch.zeros(1, 29)

    def reset(self, done=None):
        pass


class _Scene(dict):
    def write_data_to_sim(self):
        pass


class _FakeEnv:
    """Deterministic reset, warmup rejection, then two-body navigation collision."""
    def __init__(self, cfg):
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 1
        self.cfg = SimpleNamespace(decimation=4)
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.local_step = self.cycle = 0
        self.observations = {"policy": torch.zeros(1, 283)}
        names = [*cfg.collision_body_names, "left_ankle_roll_link", "right_ankle_roll_link"]
        self.forces = torch.zeros(1, 6, len(names), 3)
        sensor = SimpleNamespace(body_names=names, data=SimpleNamespace(
            net_forces_w_history=self.forces, net_forces_w=self.forces[:, 0]))
        robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 1.0]]), root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]])))
        self.scene = _Scene(robot=robot)
        self.scene.sensors = {"contact_forces": sensor, "fdm_height_scanner": None}
        self.scene.terrain = SimpleNamespace(env_origin_ids=torch.tensor([0]), origin_split_ids=torch.tensor([0]))
        self.command_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(set_command=lambda *args: None))
        self.sim = SimpleNamespace(forward=lambda: None)
        self.observation_manager = SimpleNamespace(compute=lambda **kw: self.observations)

    def _reset_idx(self, ids):
        self.local_step = 0
        self.cycle += 1

    def step(self, actions):
        self.common_step_counter += 1
        self.local_step += 1
        # Force a settling rejection, then a warmup collision, then accepted episodes.
        collision_step = 15 if self.cycle == 1 else 80
        done = self.local_step == collision_step + 1
        self.forces.zero_()
        if self.cycle > 0:
            self.forces[:, :, -2:, 2] = 2
        if self.local_step == collision_step and self.cycle > 0:
            self.forces[:, 0, 0, 0] = 3
            self.forces[:, 2, 1, 0] = 7
        if done and self.cycle > 0:
            self._reset_idx(None)
        else:
            done = False
        self.scene["robot"].data.root_pos_w[0, 0] = self.local_step * 0.02
        # Isaac Lab returns freshly computed observations inside inference_mode.
        self.observations = {"policy": torch.zeros(1, 283)}
        return self.observations, None, torch.tensor([done]), torch.tensor([False]), {}


def _collect_fixture(root, interval):
    cfg = RolloutCfg(num_envs=1)
    env = _FakeEnv(cfg)
    log = CollectionLog(root / "logs")
    with EpisodeShardWriter(root / "data", "train", {"fixture": "collector"}) as writer:
        collector = FDMRolloutCollector(env, env.observations, _DummyPolicy(), writer, cfg)
        def prepare(ids):
            for index in ids.tolist():
                collector._frame_maps[index] = (torch.zeros(1, 60, 46), torch.zeros(1, 60, 46, dtype=torch.bool))
        collector._prepare_height_maps = prepare
        result = collector.collect(2, log_interval_s=interval, log=log, round_index=0)
    dataset = FDMWindowDataset(root / "data", "train", collect_statistics=True)
    path = log.save_summary(dataset, collection=result, round_index=0)
    return result, dataset, log, path


def test_collection_counters_forces_and_logging_do_not_change_data(tmp_path):
    result, dataset, log, path = _collect_fixture(tmp_path / "logged", 0.001)
    quiet, other, _, _ = _collect_fixture(tmp_path / "quiet", 0)
    assert result["episodes"] == 2
    assert result["diagnostics"]["spawn_rejections"] == 1
    assert result["diagnostics"]["warmup_collisions"] == 1
    assert result["contact_force_peak_n"] == {"torso": 3.0, "left_hand": 7.0, "right_hand": 0.0}
    assert result["completion_fraction"] == 1
    assert result["shards_written"] == 1
    assert result["timing_seconds"]["write"] > 0
    assert result["steps"] == quiet["steps"]
    assert dataset.statistics == other.statistics
    for index in range(len(dataset)):
        for key, value in dataset[index].items():
            torch.testing.assert_close(value, other[index][key])
    records = [json.loads(line) for line in (log.directory / "progress.jsonl").read_text().splitlines()]
    assert records[0]["status"] == "started"
    assert records[-1]["status"] == "completed"
    assert len(records) > 2
    assert json.loads(path.read_text())["collection"]["diagnostics"] == result["diagnostics"]


class _ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward_batch(self, batch):
        pose = torch.zeros_like(batch["future_pose"]) + self.offset
        pose[..., 3] = 1
        return {"future_pose": pose, "collision_logits": torch.zeros_like(batch["future_collision"]) + self.offset}


def test_training_reports_separate_fetch_and_optimization_times(tmp_path, monkeypatch, capsys):
    dataset = _dataset(tmp_path / "data", [_episode(), _episode(None, 5)])
    clock = [0.0]

    class TimedDataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            clock[0] += 5.0
            return dataset[index]

    class TimedModel(_ZeroModel):
        def forward_batch(self, batch):
            clock[0] += 2.0
            return super().forward_batch(batch)

    trainer = FDMTrainer(TimedModel(), TrainCfg(device="cpu"))
    monkeypatch.setattr("unitree_rl_lab.fdm.training.trainer.time.monotonic", lambda: clock[0])
    metrics = trainer.train_epoch(DataLoader(TimedDataset(), batch_size=2), log_interval_s=1000)
    assert math.isfinite(metrics["loss"])
    assert trainer.global_epoch == 1
    assert trainer.last_epoch_timing == {
        "data_seconds": 15.0, "step_seconds": 4.0, "mean_data_seconds": 7.5, "mean_step_seconds": 2.0,
    }
    output = capsys.readouterr().out
    assert "device=cpu batch_size=2 workers=0" in output
    assert "batches=1/2" in output  # First batch prints even before the log interval.
    assert "last_s(data=10.000,step=2.000)" in output
    assert "avg_s(data=7.500,step=2.000)" in output


def test_trainer_evaluation_uses_global_metrics(tmp_path):
    dataset = _dataset(tmp_path / "data", [_episode(), _episode(None, 5)])
    trainer = FDMTrainer(_ZeroModel(), TrainCfg(device="cpu"))
    first = trainer.evaluate(DataLoader(dataset, batch_size=1), log_interval_s=0)
    second = trainer.evaluate(DataLoader(dataset, batch_size=2), log_interval_s=0)
    assert first == pytest.approx(second)
    assert first["final_position_mae_m"] == pytest.approx(13 / 3)
    log = CollectionLog(tmp_path / "logs")
    path = log.save_summary(dataset, reused=True)
    log.save_evaluation(path, first, global_epoch=0)
    assert json.loads(path.read_text())["evaluation_before_training"]["global_epoch"] == 0


def test_zero_baseline_and_empty_masks_have_defined_results():
    target = {
        "future_pose": torch.zeros(2, 10, 4), "future_commands": torch.zeros(2, 10, 3),
        "future_collision": torch.zeros(2, 10), "valid_mask": torch.ones(2, 10, dtype=torch.bool),
    }
    target["future_pose"][..., 3] = 1
    prediction = {"future_pose": target["future_pose"].clone(), "collision_logits": torch.full((2, 10), -10.0)}
    metrics = MetricAccumulator()
    metrics.update(prediction, target)
    report = metrics.result()
    assert report["position_mse_ratio_vs_baseline"] is None
    assert report["collision_accuracy"] == 1
    assert report["collision_recall"] == report["collision_precision"] == 0
    target["valid_mask"][:] = False
    metrics = MetricAccumulator()
    metrics.update(prediction, target)
    report = metrics.result()
    assert report["valid_targets"] == 0
    assert report["final_position_mae_m"] is None
    json.dumps(report, allow_nan=False)


def test_nonuniform_kinematics_use_signed_velocity_changes():
    episode = _episode()
    episode.timestamp[:] = torch.tensor([0.0, 0.5, 0.7], dtype=torch.float64)
    episode.state_history_raw[:, :, 0] = torch.tensor([0.0, 0.5, 0.3])[:, None]
    stats = DatasetStatistics(10, 0.5, 0.05)
    stats.update_episode(episode, [])
    acceleration = stats.result()["statistics"]["acceleration"]
    assert acceleration["mean"][0] == pytest.approx(-2 / 0.35)
    assert acceleration["max_abs"][0] == pytest.approx(2 / 0.35)


def test_one_frame_episode_and_incomplete_future(tmp_path):
    episode = _episode(None)
    # Include masked short tails: the final frame is not a start, earlier tails have missing future.
    dataset = _dataset(tmp_path / "tails", [episode], include_incomplete_noncollision=True)
    tail = dataset[len(dataset) - 1]
    assert tail["valid_mask"].tolist() == [True] + [False] * 9
    assert tail["future_collision"].sum() == 0
    assert torch.equal(tail["future_pose"][1:, :2], torch.zeros(9, 2))
    from dataclasses import fields
    for field in fields(episode):
        setattr(episode, field.name, getattr(episode, field.name)[-1:].clone())
    dataset = _dataset(tmp_path / "one", [episode])
    assert dataset.statistics["statistics"]["velocity"]["mean"] == [None, None, None]
    assert dataset.statistics["statistics"]["episode_duration_s"]["mean"] == 0


def test_failed_collection_records_original_error_before_shutdown(tmp_path):
    cfg = RolloutCfg(num_envs=1)
    env = _FakeEnv(cfg)
    def fail_step(actions):
        raise RuntimeError("synthetic simulation failure")
    env.step = fail_step
    log = CollectionLog(tmp_path / "logs")
    with EpisodeShardWriter(tmp_path / "data", "train", {"fixture": "failure"}) as writer:
        collector = FDMRolloutCollector(env, env.observations, _DummyPolicy(), writer, cfg)
        with pytest.raises(RuntimeError, match="synthetic simulation failure"):
            collector.collect(1, log_interval_s=0, log=log)
    last = json.loads((log.directory / "progress.jsonl").read_text().splitlines()[-1])
    assert last["status"] == "failed"
    assert last["error"] == {"type": "RuntimeError", "message": "synthetic simulation failure"}
    assert last["episodes"] == 0


def test_split_and_policy_resets_support_inference_buffers(monkeypatch):
    import runpy
    from unitree_rl_lab.fdm.runner.frozen_policy import FrozenRecurrentPolicy
    script = Path(__file__).resolve().parents[2] / "scripts" / "fdm" / "train_fdm.py"
    monkeypatch.syspath_prepend(str(script.parent))
    namespace = runpy.run_path(str(script), run_name="__mp_main__")
    with torch.inference_mode():
        buffer = torch.ones(2)
    def reset(seed=None):
        buffer.zero_()
        return {"policy": buffer}, {}
    terrain = SimpleNamespace(activate_split=lambda split: None)
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(terrain=terrain)), reset=reset)
    observations = namespace["_activate_split"](env, "train", 43)
    assert observations["policy"].sum() == 0
    policy = FrozenRecurrentPolicy.__new__(FrozenRecurrentPolicy)
    policy.module = SimpleNamespace(reset=lambda dones: buffer.fill_(1))
    policy.reset()
    assert buffer.sum() == 2
