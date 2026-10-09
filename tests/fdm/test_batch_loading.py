from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset
from unitree_rl_lab.fdm.data.window import window_targets
from unitree_rl_lab.fdm.utils.se2 import relative_pose_sequence
from test_schema_and_dataset import _episode


def _multi_shard_dataset(root: Path):
    with EpisodeShardWriter(root, "train", {"fixture": "batch-reads"}, max_frames_per_shard=15) as writer:
        for shard in range(4):
            for collision in (2, None):
                episode = _episode(collision, episode_id=shard * 2 + int(collision is None))
                episode.height_map.fill_(shard)
                if shard == 1:
                    episode.state_history_raw[..., 0] = 0
                writer.append(episode)
    return FDMWindowDataset(root, "train", include_incomplete_noncollision=True, log_interval_s=0)


def _scalar_reference(dataset, index):
    """The pre-batch-loading sample path, independent of __getitems__."""
    item = dataset.indices[index]
    episode = dataset._load_shard(item.shard)[item.episode]
    start = item.start
    state = episode.state_history_raw[start]
    pose = relative_pose_sequence(state[:, :3], state[:, 3:7], anchor=0)
    targets = window_targets(episode, torch.tensor([start]), dataset.horizon)
    return {
        "relative_state_history": torch.cat((pose, state[:, 7:8]), dim=-1).float(),
        "proprio_history": episode.proprio_history[start].float(),
        "history_timestamps": (episode.history_timestamps[start] - episode.timestamp[start]).float(),
        "height_map": episode.height_map[start].float(),
        "height_map_invalid": episode.height_map_invalid[start],
        **{name: value[0] for name, value in targets.items()},
        "contains_collision": torch.tensor(item.collision),
        "low_motion": torch.tensor(item.low_motion),
    }


def test_random_batches_load_each_shard_once_and_preserve_samples(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    assert len(dataset.shards) == 4 > dataset.cache_size
    # Interleave all shards repeatedly, include duplicates and a partial batch.
    by_shard = [[i for i, item in enumerate(dataset.indices) if item.shard == shard] for shard in range(4)]
    order = [indices[row] for row in range(len(by_shard[0])) for indices in by_shard]
    order += [order[2], order[0], order[2]]
    expected = default_collate([_scalar_reference(dataset, index) for index in order])
    dataset._cache.clear()
    loader = DataLoader(dataset, batch_size=17, sampler=order, num_workers=0)
    iterator = iter(loader)
    for offset in range(0, len(order), 17):
        requested = order[offset : offset + 17]
        with patch("torch.load", wraps=torch.load) as load:
            actual = next(iterator)
        paths = [call.args[0] for call in load.call_args_list]
        assert len(paths) == len(set(paths))  # Never reread a shard within a batch.
        assert len(paths) <= len({dataset.indices[index].shard for index in requested})
        assert len(dataset._cache) <= dataset.cache_size
        for name, value in actual.items():
            torch.testing.assert_close(value, expected[name][offset : offset + 17])
    assert dataset.__getitems__([]) == []


def test_batch_samples_do_not_retain_full_episode_storage(tmp_path):
    dataset = _multi_shard_dataset(tmp_path)
    indices = [i for i, item in enumerate(dataset.indices) if item.shard == 0 and item.episode == 1][:2]
    samples = dataset.__getitems__(indices)
    episode = dataset._cache[0][1]
    for name in ("proprio_history", "height_map_invalid"):
        selected = samples[0][name]
        original = getattr(episode, name)
        assert selected.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        assert selected.untyped_storage().nbytes() == 2 * selected.numel() * selected.element_size()
