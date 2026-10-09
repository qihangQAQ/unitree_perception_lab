import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from unitree_rl_lab.fdm.config import FDMModelCfg, TrainCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, Split
from unitree_rl_lab.fdm.training import FDMTrainer
from unitree_rl_lab.fdm.training.resume import (
    check_checkpoint_destinations, load_round_shards, resume_dataset_metadata, validate_resume_checkpoint,
)
from test_batch_loading import _multi_shard_dataset
from test_collection_statistics import _ZeroModel
from test_schema_and_dataset import _episode


def _saved_round(tmp_path, dataset, cfg):
    trainer = FDMTrainer(_ZeroModel(), cfg)
    trainer.train_epoch(DataLoader(dataset, batch_size=7), log_interval_s=0)
    path = tmp_path / "fdm_round_000.pt"
    trainer.save_checkpoint(
        path, round_index=0, model_cfg=FDMModelCfg(), metrics={},
        dataset_manifest=str(dataset.root / "manifest.json"),
    )
    return trainer, path, torch.load(path, weights_only=False)


def test_resume_preserves_next_update_and_scheduler(tmp_path):
    dataset = _multi_shard_dataset(tmp_path / "data")
    cfg = TrainCfg(device="cpu", collection_rounds=3, epochs_per_round=1)
    original, _, checkpoint = _saved_round(tmp_path, dataset, cfg)
    loader = DataLoader(dataset, batch_size=7)
    original.train_epoch(loader, log_interval_s=0)
    restored = FDMTrainer(_ZeroModel(), cfg)
    assert restored.restore_checkpoint(checkpoint, model_cfg=FDMModelCfg()) == 1
    assert restored.global_epoch == 1
    restored.train_epoch(loader, log_interval_s=0)
    torch.testing.assert_close(restored.model.offset, original.model.offset, rtol=0, atol=0)
    assert restored.global_epoch == original.global_epoch == 2
    assert restored.scheduler.state_dict() == original.scheduler.state_dict()
    for key, value in original.optimizer.state[original.model.offset].items():
        torch.testing.assert_close(restored.optimizer.state[restored.model.offset][key], value, rtol=0, atol=0)


def test_stage_transition_can_load_weights_with_a_new_training_schedule(tmp_path):
    dataset = _multi_shard_dataset(tmp_path / "data")
    original, _, checkpoint = _saved_round(tmp_path, dataset, TrainCfg(device="cpu", collection_rounds=1, epochs_per_round=1))
    # A later MPPI stage can initialize the same FDM from stage-one weights;
    # it does not need to resume the completed stage-one scheduler or dataset.
    next_stage = FDMTrainer(_ZeroModel(), TrainCfg(device="cpu", collection_rounds=5, epochs_per_round=2))
    next_stage.model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["model_cfg"] == FDMModelCfg().__dict__
    torch.testing.assert_close(next_stage.model.offset, original.model.offset, rtol=0, atol=0)
    assert next_stage.global_epoch == 0 and not next_stage.optimizer.state
    assert next_stage.scheduler.T_max == 10


def test_resume_rejects_wrong_schedule_or_incomplete_checkpoint(tmp_path):
    dataset = _multi_shard_dataset(tmp_path / "data")
    cfg = TrainCfg(device="cpu", collection_rounds=3, epochs_per_round=1)
    _, _, checkpoint = _saved_round(tmp_path, dataset, cfg)
    for key, value in (("global_epoch", 7), ("model_cfg", {}), ("round", -1)):
        invalid = {**checkpoint, key: value}
        with pytest.raises(ValueError):
            validate_resume_checkpoint(invalid, cfg, FDMModelCfg())
    with pytest.raises(ValueError, match="TOTAL"):
        validate_resume_checkpoint(checkpoint, TrainCfg(device="cpu", collection_rounds=2), FDMModelCfg())
    with pytest.raises(ValueError, match="full FDM"):
        validate_resume_checkpoint({"model_state_dict": {}}, cfg, FDMModelCfg())
    for key in ("checkpoint_version", "torch_rng_state", "cuda_rng_states"):
        old = {name: value for name, value in checkpoint.items() if name != key}
        with pytest.raises(ValueError, match="current training format"):
            validate_resume_checkpoint(old, cfg, FDMModelCfg())
    old_cfg = {key: value for key, value in checkpoint["train_cfg"].items() if key != "samples_per_round"}
    with pytest.raises(ValueError, match="Unsupported"):
        validate_resume_checkpoint({**checkpoint, "train_cfg": old_cfg}, cfg, FDMModelCfg())
    # Changing runtime resource settings does not reset the learning schedule.
    runtime_cfg = copy.copy(cfg)
    runtime_cfg.batch_size, runtime_cfg.num_workers, runtime_cfg.checkpoint_dir = 2, 0, "another-output"
    assert validate_resume_checkpoint(checkpoint, runtime_cfg, FDMModelCfg()) == 1


def _summary(dataset, path, round_index=1):
    manifest = json.loads((dataset.root / "manifest.json").read_text())
    entries = [entry for entry in manifest["shards"] if entry["split"] == "train"]
    report = {
        "split": "train", "round": round_index,
        "collection": {"status": "completed", "collection_mode": "fixed",
                       "frames_recorded": sum(entry["frames"] for entry in entries)},
        "manifest": str((dataset.root / "manifest.json").resolve()),
        "shards": [str(p.resolve()) for p in dataset.shards],
        "dataset": {"counts": {key: sum(entry[key] for entry in entries) for key in ("frames", "episodes")}},
    }
    path.write_text(json.dumps(report))
    return report


def test_reuse_checks_round_and_exact_shards(tmp_path):
    dataset = _multi_shard_dataset(tmp_path / "data")
    manifest_path = dataset.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["collection"] = {"mode": "fixed", "version": 1}
    manifest_path.write_text(json.dumps(manifest))
    path = tmp_path / "pending.json"
    report = _summary(dataset, path)
    assert load_round_shards(path, dataset.root, round_index=1) == dataset.shards
    variants = [
        {**report, "round": 0}, {**report, "split": "val"},
        {**report, "shards": report["shards"] * 2},
        {**report, "collection": {"status": "interrupted"}},
        {**report, "manifest": str(tmp_path / "another" / "manifest.json")},
        {**report, "shards": report["shards"][:-1]},
        {**report, "collection": {**report["collection"], "frames_recorded": 1}},
        {**report, "collection": {"status": "completed", "collection_mode": "episodes"}},
    ]
    for invalid in variants:
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            load_round_shards(path, dataset.root, round_index=1)


def test_resume_preserves_manifest_and_existing_checkpoints(tmp_path):
    recorded = {"git_commit": "old", "rollout": {"seed": 42}, "policy_checkpoint_sha256": "abc"}
    current = {**recorded, "git_commit": "new"}
    assert resume_dataset_metadata(current, recorded) == recorded
    assert current["git_commit"] == "new"
    with pytest.raises(ValueError, match="policy_checkpoint_sha256"):
        resume_dataset_metadata({**current, "policy_checkpoint_sha256": "different"}, recorded)
    (tmp_path / "fdm_round_000.pt").write_bytes(b"existing")
    check_checkpoint_destinations(tmp_path, 1, 3)
    with pytest.raises(FileExistsError):
        check_checkpoint_destinations(tmp_path, 0, 3)


@pytest.mark.parametrize("resume", [False, True])
def test_online_entrypoint_starts_fresh_or_resumes_fixed_rounds(tmp_path, monkeypatch, resume):
    """Exercise real datasets/trainer/checkpoints; replace only simulator/policy."""
    import unitree_rl_lab.fdm.models as models
    import unitree_rl_lab.fdm.runner as runner

    dataset = _multi_shard_dataset(tmp_path / "data")
    metadata = {"fixture": "batch-reads"}
    with EpisodeShardWriter(dataset.root, "val", metadata) as writer:
        episode = _episode()
        episode.usd_region_split.fill_(int(Split.from_name("val")))
        writer.append(episode)
    cfg = TrainCfg(device="cpu", collection_rounds=3, epochs_per_round=1, num_workers=0)
    _, checkpoint_path, _ = _saved_round(tmp_path, dataset, cfg)
    checkpoint_bytes = checkpoint_path.read_bytes()
    summary_path = tmp_path / "pending.json"
    _summary(dataset, summary_path)

    script = Path(__file__).resolve().parents[2] / "scripts/fdm/train_fdm.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("fdm_resume_test_entrypoint", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from _common import resolve_collection_settings
    metadata["collection"] = resolve_collection_settings(SimpleNamespace())
    manifest_path = dataset.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"] = metadata
    manifest_path.write_text(json.dumps(manifest))
    env = SimpleNamespace(device="cpu", action_manager=SimpleNamespace(total_action_dim=29))
    env.unwrapped = env
    env.reset = lambda **_: ({}, {})
    env.close = lambda: None
    config = SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace()),
                             observations=SimpleNamespace(policy=SimpleNamespace()))
    for name, attrs in {
        "gymnasium": {"make": lambda *a, **k: env},
        "isaaclab_tasks": {},
        "isaaclab_tasks.utils": {},
        "isaaclab_tasks.utils.parse_cfg": {"load_cfg_from_registry": lambda *a: {}},
        "unitree_rl_lab.tasks": {},
        "unitree_rl_lab.utils.parser_cfg": {"parse_env_cfg": lambda *a, **k: config},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(module, "dataset_metadata", lambda *a: metadata)
    monkeypatch.setattr(module, "configure_safe_spawn", lambda *a: None)
    monkeypatch.setattr(module, "_activate_split", lambda *a: {})
    monkeypatch.setattr(models, "G1HeightFDM", lambda cfg: _ZeroModel())
    monkeypatch.setattr(runner, "FrozenRecurrentPolicy", lambda *a: SimpleNamespace(reset=lambda: None))
    collected = []

    class Collector:
        builders = []

        def __init__(self, env, observations, policy, writer, cfg):
            self.writer = writer

        def collect(self, frames_per_env, *, round_index=None, **kwargs):
            collected.append(round_index)
            assert frames_per_env == 150 and kwargs["seed"] == 43 + round_index
            for i in range(8):
                self.writer.append(_episode(episode_id=100 + i))
            self.writer.flush()
            return {"status": "completed", "collection_mode": "fixed", "frames_recorded": 24,
                    "diagnostics": {}, "timing_seconds": {}, "contact_force_peak_n": None}

    monkeypatch.setattr(runner, "FixedRolloutCollector", Collector)
    args = SimpleNamespace(
        task="fixture", checkpoint="policy.pt", terrain_usd="fixture.usd", dataset=str(dataset.root),
        output=str(tmp_path / "continued"), resume=str(checkpoint_path) if resume else None,
        resume_round_summary=str(summary_path) if resume else None,
        num_envs=1, seed=42, disable_policy_corruption=False, collection_rounds=3, epochs_per_round=1,
        batch_size=7, workers=0, device="cpu",
        dataset_cache="mmap", dataset_cache_gb=None, log_interval=0,
        samples_per_round=20, eval_after_collection=False,
    )
    module._train(args)
    assert collected == ([2] if resume else [0, 1, 2])
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    for index in range(1 if resume else 0, 3):
        result = torch.load(Path(args.output) / f"fdm_round_{index:03d}.pt", weights_only=False)
        assert result["round"] == index and result["global_epoch"] == index + 1
    reused = list((Path(args.output) / "collection").glob("*/train_round_001_summary.json"))
    assert json.loads(reused[0].read_text())["reused"] is resume
    report = json.loads(reused[0].read_text())
    assert report["collection"]["collection_mode"] == "fixed"
    assert report["training_sample_pool"]["sampled_windows"] == 20
    assert report["training_cache"]["mode"] == "mmap"
