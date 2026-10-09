import importlib.util
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from unitree_rl_lab.fdm.data import FDMWindowDataset
from unitree_rl_lab.fdm.data.prepared_cache import cache_path
from test_batch_loading import _multi_shard_dataset


def test_pool_precedes_materialization_and_reports_real_repeats(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    original = list(dataset.indices)
    natural_count = len(dataset)
    report = dataset.select_sample_pool(120, seed=7)
    assert report["source_windows"] == natural_count
    assert report["sampled_windows"] == len(dataset) == 120
    assert report["repeated_windows"] == 120 - report["unique_windows"] > 0
    assert dataset.indices == sorted(dataset.indices, key=lambda row: (row.shard, row.episode, row.start))
    assert all(item in original for item in dataset.indices)
    other = FDMWindowDataset(tmp_path, "train", include_incomplete_noncollision=True, log_interval_s=0)
    other.select_sample_pool(120, seed=7)
    assert other.indices == dataset.indices
    expected = default_collate(dataset.__getitems__(list(range(len(dataset)))))
    dataset.prepare_cache("memory", log_interval_s=0)
    with patch("torch.load", side_effect=AssertionError("epoch reread raw shard")):
        actual = next(iter(DataLoader(dataset, batch_size=120)))
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])
    with pytest.raises(RuntimeError, match="before preparing"):
        dataset.select_sample_pool(80, seed=8)


def test_pool_size_reduces_cache_instead_of_only_epoch_length(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    original_bytes, count = dataset.estimated_cache_bytes, len(dataset)
    dataset.select_sample_pool(7, seed=3)
    assert dataset.estimated_cache_bytes == original_bytes // count * 7


def test_mmap_cache_reuses_samples_and_reopens_instead_of_pickling_storage(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    dataset.select_sample_pool(31, seed=1)
    order = [30, 0, 4, 0, 18]
    expected = default_collate(dataset.__getitems__(order))
    dataset.prepare_cache("mmap", log_interval_s=0)
    directory = dataset._cache_directory
    assert (directory / "complete.json").is_file()
    assert dataset.cached_bytes == 0
    serialized = pickle.dumps(dataset)
    assert len(serialized) < dataset.estimated_cache_bytes / 2
    reopened = pickle.loads(serialized)
    with patch("torch.load", side_effect=AssertionError("mmap epoch read raw shard")):
        for _ in range(3):
            actual = default_collate(reopened.__getitems__(order))
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])
        dataset.release_cache()
        dataset.prepare_cache("mmap", log_interval_s=0)
    assert dataset._cache_directory == directory
    assert dataset.shard_load_count == reopened.shard_load_count
    sample = dataset[0]
    sample["height_map"].fill_(123)
    assert not torch.any(dataset[0]["height_map"] == 123)


def test_mmap_key_changes_with_pool_seed_or_source_revision(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    first = cache_path(dataset)
    dataset.select_sample_pool(10, seed=1)
    second = cache_path(dataset)
    assert first != second
    path = dataset.shards[0]
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert cache_path(dataset) != second


def test_failed_mmap_build_is_not_published_and_can_retry(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    directory = cache_path(dataset)
    with patch.object(dataset, "_prepare_shard", side_effect=OSError("interrupted preprocessing")):
        with pytest.raises(OSError, match="interrupted"):
            dataset.prepare_cache("mmap", log_interval_s=0)
    assert not directory.exists()
    assert not list(directory.parent.glob("*.tmp"))
    assert dataset.cache_mode == "shard" and not dataset._cache
    dataset.prepare_cache("mmap", log_interval_s=0)
    assert dataset.cache_mode == "mmap"


def test_collection_mode_defaults_and_resume_do_not_mix_metadata():
    path = Path(__file__).resolve().parents[2] / "scripts/fdm/_common.py"
    spec = importlib.util.spec_from_file_location("fdm_settings_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace()
    settings = module.resolve_collection_settings(args)
    assert settings["mode"] == "fixed" and args.frames_per_env == 150
    assert module.resolve_collection_settings(SimpleNamespace(), {"collection": settings}) == settings
    with pytest.raises(ValueError, match="new --dataset"):
        module.resolve_collection_settings(SimpleNamespace(), {"rollout": {}})
    with pytest.raises(ValueError, match="settings differ"):
        module.resolve_collection_settings(SimpleNamespace(frames_per_env=200), {"collection": settings})
    with pytest.raises(ValueError, match="at least"):
        module.resolve_collection_settings(SimpleNamespace(frames_per_env=10))
