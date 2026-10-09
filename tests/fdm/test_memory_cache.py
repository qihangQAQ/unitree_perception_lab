import argparse
import gc
import importlib.util
from pathlib import Path
from unittest.mock import patch
import weakref

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from unitree_rl_lab.fdm.config import TrainCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset, make_collision_balanced_sampler
from unitree_rl_lab.fdm.training import FDMTrainer
from test_batch_loading import _multi_shard_dataset
from test_collection_statistics import _ZeroModel
from test_schema_and_dataset import _episode


@pytest.fixture(autouse=True)
def sufficient_ram(monkeypatch):
    monkeypatch.setattr("unitree_rl_lab.fdm.data.trajectory_dataset.available_memory_bytes", lambda: 64 * 2**30)


def test_cache_preserves_samples_and_multiple_epochs_never_read_or_transform(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    order = list(reversed(range(len(dataset)))) + [0, 0, 4, 1]
    expected = default_collate(dataset.__getitems__(order))
    flags = dataset.collision_flags.clone(), dataset.low_motion_flags.clone()
    with patch("torch.load", wraps=torch.load) as load:
        dataset.prepare_cache(max_cache_bytes=dataset.estimated_cache_bytes, log_interval_s=0)
    assert load.call_count == len(dataset.shards)
    assert dataset.cache_mode == "memory"
    assert dataset.cached_bytes == dataset.estimated_cache_bytes
    assert dataset._prepared_samples["height_map"].dtype == torch.float16
    assert not dataset._cache
    torch.testing.assert_close(dataset.collision_flags, flags[0])
    torch.testing.assert_close(dataset.low_motion_flags, flags[1])
    loader = DataLoader(dataset, batch_size=17, sampler=order, num_workers=0)
    with patch("torch.load", side_effect=AssertionError("cache reread a shard")), patch(
        "unitree_rl_lab.fdm.data.trajectory_dataset._episode_samples",
        side_effect=AssertionError("cache recomputed a window"),
    ):
        dataset.prepare_cache(log_interval_s=0)  # Idempotent; validation can be reused.
        for _ in range(3):
            for offset, actual in zip(range(0, len(order), 17), loader):
                for name, value in actual.items():
                    torch.testing.assert_close(value, expected[name][offset : offset + 17])
    assert dataset.__getitems__([]) == []


def test_combined_cache_budget_rejects_before_loading_and_keeps_validation(tmp_path):
    validation = _multi_shard_dataset(tmp_path / "val")
    train = _multi_shard_dataset(tmp_path / "train")
    validation.prepare_cache(log_interval_s=0)
    expected = validation[0]
    combined = validation.cached_bytes + train.estimated_cache_bytes
    with patch("torch.load", side_effect=AssertionError("budget must be checked before loading")):
        with pytest.raises(MemoryError, match="including resident validation data"):
            train.prepare_cache(max_cache_bytes=combined - 1, resident_cache_bytes=validation.cached_bytes)
    assert train.cached_bytes == 0
    assert train.cache_mode == "shard"
    for name, value in validation[0].items():
        torch.testing.assert_close(value, expected[name])
    train.prepare_cache(max_cache_bytes=combined, resident_cache_bytes=validation.cached_bytes, log_interval_s=0)
    assert validation.cached_bytes + train.cached_bytes == combined


def test_low_available_ram_rejects_and_explicit_shard_mode_still_works(tmp_path, monkeypatch):
    dataset = _multi_shard_dataset(tmp_path)
    monkeypatch.setattr("unitree_rl_lab.fdm.data.trajectory_dataset.available_memory_bytes", lambda: 1)
    with pytest.raises(MemoryError, match="RAM is currently available"):
        dataset.prepare_cache(log_interval_s=0)
    dataset.prepare_cache("shard", max_cache_bytes=1, log_interval_s=0)
    assert dataset.cache_mode == "shard"
    assert dataset.cached_bytes == 0
    assert len(list(DataLoader(dataset, batch_size=7))) > 0


def test_float32_height_maps_are_not_quantized_in_memory(tmp_path):
    with EpisodeShardWriter(tmp_path, "train", {"fixture": "mixed-precision"}, max_frames_per_shard=3) as writer:
        writer.append(_episode())
        precise = _episode(episode_id=5)
        precise.height_map = torch.full_like(precise.height_map, 1.000123, dtype=torch.float32)
        writer.append(precise)
    dataset = FDMWindowDataset(tmp_path, "train", log_interval_s=0)
    expected = default_collate(dataset.__getitems__(list(range(len(dataset)))))
    dataset.prepare_cache(log_interval_s=0)
    assert dataset._prepared_samples["height_map"].dtype == torch.float32
    assert dataset.cached_bytes == dataset.estimated_cache_bytes
    actual = default_collate(dataset.__getitems__(list(range(len(dataset)))))
    torch.testing.assert_close(actual["height_map"], expected["height_map"], rtol=0, atol=0)


def test_releasing_round_drops_storage_without_invalidating_copied_batch(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    dataset.prepare_cache(log_interval_s=0)
    sample = dataset[0]
    original = sample["proprio_history"].clone()
    sample["proprio_history"].add_(10)
    torch.testing.assert_close(dataset[0]["proprio_history"], original)
    refs = [weakref.ref(value) for value in dataset._prepared_samples.values()]
    dataset.release_cache()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert dataset.cached_bytes == 0
    assert not dataset._cache
    assert dataset.cache_mode == "shard"
    torch.testing.assert_close(sample["proprio_history"], original + 10)
    with patch("torch.load", wraps=torch.load) as load:
        torch.testing.assert_close(dataset[0]["proprio_history"], original)
    assert load.call_count == 1


def test_partial_cache_failure_does_not_publish_storage_or_leave_raw_shards(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    load = torch.load
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture failed shard read")
        return load(*args, **kwargs)

    with patch("torch.load", side_effect=fail_on_second), pytest.raises(OSError, match="fixture failed"):
        dataset.prepare_cache(log_interval_s=0)
    assert dataset.cached_bytes == 0
    assert not dataset._cache
    dataset.prepare_cache(log_interval_s=0)
    assert dataset.cached_bytes == dataset.estimated_cache_bytes


def test_cached_training_and_validation_and_weighted_sampling(tmp_path, capsys):
    dataset = _multi_shard_dataset(tmp_path)
    sampler = make_collision_balanced_sampler(dataset, num_samples=13)
    torch.manual_seed(77)
    expected_order = list(sampler)
    dataset.prepare_cache(log_interval_s=0)
    cached_sampler = make_collision_balanced_sampler(dataset, num_samples=13)
    torch.manual_seed(77)
    assert list(cached_sampler) == expected_order
    train = DataLoader(dataset, batch_size=7, sampler=cached_sampler)
    val = DataLoader(dataset, batch_size=11, shuffle=False)
    trainer = FDMTrainer(_ZeroModel(), TrainCfg(device="cpu"))
    with patch("torch.load", side_effect=AssertionError("epoch loaded disk")):
        for _ in range(2):
            losses = trainer.train_epoch(train, log_interval_s=1e-9)
            metrics = trainer.evaluate(val, log_interval_s=0)
            assert torch.isfinite(torch.tensor(losses["loss"]))
            assert torch.isfinite(torch.tensor(metrics["position_mae_m"]))
    output = capsys.readouterr().out
    assert "cache=memory" in output
    assert "shard_loads=0" in output
    assert trainer.global_epoch == 2


def test_cache_cli_defaults_and_budget_validation():
    path = Path(__file__).resolve().parents[2] / "scripts/fdm/_common.py"
    spec = importlib.util.spec_from_file_location("fdm_test_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.add_dataset_cache_args(parser)
    args = parser.parse_args([])
    assert args.dataset_cache == "auto" and args.dataset_cache_gb is None
    args = parser.parse_args(["--dataset-cache", "shard", "--dataset-cache-gb", "2.5"])
    assert args.dataset_cache == "shard" and args.dataset_cache_gb == 2.5
    for invalid in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset-cache-gb", invalid])


def test_available_ram_check_honors_container_limit_and_missing_files(monkeypatch):
    from unitree_rl_lab.fdm.utils.memory import available_memory_bytes

    files = {
        "/proc/meminfo": "MemTotal: 8192 kB\nMemAvailable: 4096 kB\n",
        "/sys/fs/cgroup/memory.max": "2097152",
        "/sys/fs/cgroup/memory.current": "1048576",
    }

    def read(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(path)
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    assert available_memory_bytes() == 2**20
    files["/sys/fs/cgroup/memory.max"] = "max"
    assert available_memory_bytes() == 4 * 2**20
    files.clear()
    assert available_memory_bytes() is None


def test_auto_cache_uses_precomputed_mmap_and_retains_validation(tmp_path, monkeypatch):
    validation = _multi_shard_dataset(tmp_path / "val")
    train = _multi_shard_dataset(tmp_path / "train")
    validation.prepare_cache(log_interval_s=0)
    retained = validation.cached_bytes
    monkeypatch.setattr("unitree_rl_lab.fdm.data.trajectory_dataset.available_memory_bytes", lambda: 1)
    with patch("torch.load", side_effect=AssertionError("preflight must not load shards")):
        with pytest.raises(MemoryError, match="one raw shard"):
            train.prepare_cache("auto", resident_cache_bytes=retained, log_interval_s=0)
    assert train.cache_mode == "shard" and validation.cached_bytes == retained
    assert len(list(DataLoader(train, batch_size=7))) > 0
    monkeypatch.setattr("unitree_rl_lab.fdm.data.trajectory_dataset.available_memory_bytes", lambda: 64 * 2**30)
    train.prepare_cache("auto", max_cache_bytes=1, log_interval_s=0)
    assert train.cache_mode == "mmap" and train.cached_bytes == 0
    with patch("torch.load", side_effect=AssertionError("epochs must not load raw shards")):
        assert len(list(DataLoader(train, batch_size=7))) > 0
    train.prepare_cache("auto", log_interval_s=0)
    assert train.cache_mode == "memory"


@pytest.mark.parametrize("version", [1, 2])
def test_ram_estimate_reclaims_clean_file_pages_but_honors_container_limit(monkeypatch, version):
    from unitree_rl_lab.fdm.utils.memory import memory_snapshot

    gib = 2**30
    base = "/sys/fs/cgroup" if version == 2 else "/sys/fs/cgroup/memory"
    limit, current = ("memory.max", "memory.current") if version == 2 else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    stats = {"file": 8, "shmem": 1, "inactive_file": 6, "file_dirty": 1, "file_writeback": 1}
    if version == 1:
        stats = {"total_cache": 8, "total_shmem": 1, "total_inactive_file": 6, "total_dirty": 1, "total_writeback": 1}
    files = {
        "/proc/meminfo": f"MemAvailable: {100 * gib // 1024} kB\n",
        f"{base}/{limit}": str(16 * gib), f"{base}/{current}": str(14 * gib),
        f"{base}/memory.stat": "\n".join(f"{key} {value * gib}" for key, value in stats.items()),
    }

    def read(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(path)
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    # Raw headroom is 2 GiB; only 4 of the 8 GiB of file pages count as reclaimable.
    assert memory_snapshot()["available"] == 6 * gib
    files["/proc/meminfo"] = f"MemAvailable: {3 * gib // 1024} kB\n"
    assert memory_snapshot()["available"] == 3 * gib
    files[f"{base}/memory.stat"] = ""
    assert memory_snapshot()["available"] == 2 * gib
