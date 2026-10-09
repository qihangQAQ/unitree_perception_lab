from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from unitree_rl_lab.fdm.config import RolloutCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset, TerminationReason
from unitree_rl_lab.fdm.data.rollout_buffer import CollectionBudget, FixedFrameBuffer
from unitree_rl_lab.fdm.data.shard_writer import iter_episodes
from unitree_rl_lab.fdm.runner.fixed_collector import FixedRolloutCollector
from test_schema_and_dataset import _episode


def test_fixed_buffer_batches_own_values_and_keep_episode_boundaries(tmp_path):
    buffer = FixedFrameBuffer(2, 6)
    episodes = [_episode(episode_id=10), _episode(episode_id=11)]
    ids = torch.tensor([0, 1])
    buffer.begin(ids)
    for index in range(3):
        frame = {name: torch.stack([getattr(ep, name)[index] for ep in episodes]) for name in buffer.storage}
        buffer.append(ids, **frame)
        frame["state_history_raw"].fill_(10000)  # No alias to a simulator-owned batch.
    first = buffer.finish(0)
    second = buffer.finish(1, TerminationReason.ROUND_CUT)
    assert buffer.pending_frames == 0
    assert first.episode_id.unique().tolist() == [10]
    assert second.termination_reason[-1] == int(TerminationReason.COLLISION)
    torch.testing.assert_close(first.state_history_raw, episodes[0].state_history_raw)
    assert first.height_map.untyped_storage().nbytes() == first.height_map.numel() * 2
    buffer.reset()
    buffer.storage["state_history_raw"].fill_(-10000)
    torch.testing.assert_close(first.state_history_raw, episodes[0].state_history_raw)
    with EpisodeShardWriter(tmp_path, "train", {}) as writer:
        writer.append(first)
        writer.append(second)
    assert len(list(iter_episodes(tmp_path, "train"))) == 2


def test_buffer_capacity_and_cut_does_not_create_future_targets(tmp_path):
    episode = _episode(None)
    buffer = FixedFrameBuffer(1, 12)
    ids = torch.tensor([0])
    buffer.begin(ids)
    for frame in range(12):
        buffer.append(ids, **{name: getattr(episode, name)[frame:frame + 1] for name in buffer.storage})
    with pytest.raises(RuntimeError, match="full"):
        buffer.append(ids, **{name: getattr(episode, name)[:1] for name in buffer.storage})
    result = buffer.finish(0, TerminationReason.ROUND_CUT)
    assert result.truncated[-1] and not result.has_outgoing_command[-1]
    with EpisodeShardWriter(tmp_path, "train", {}) as writer:
        writer.append(result)
    dataset = FDMWindowDataset(tmp_path, "train", log_interval_s=0)
    assert len(dataset) == 1
    assert dataset[0]["valid_mask"].all()
    assert not dataset[0]["future_collision"].any()


def test_tail_budget_uses_average_capacity_and_stops_only_after_threshold():
    budget = CollectionBudget(1000)
    for index in range(1, 10):
        assert budget.stop_reason(index * 100, float(index)) is None
    assert budget.stop_reason(949, 100) is None
    assert budget.stop_reason(950, 10) is None
    assert budget.stop_reason(950, 11) == "slow_tail"
    assert budget.stop_reason(1000, 11) == "capacity"


class _Scene(dict):
    def write_data_to_sim(self):
        pass


class _BatchEnv:
    def __init__(self, cfg, collision_steps=(80, 55)):
        self.unwrapped = self
        self.device, self.num_envs, self.step_dt = "cpu", 2, 0.02
        self.cfg = SimpleNamespace(decimation=4)
        self.common_step_counter = 0
        self.reset_calls = []
        self.local_steps = torch.zeros(2, dtype=torch.long)
        self.collision_steps = torch.tensor(collision_steps)
        self.origins = torch.tensor([0., 100.])
        names = [*cfg.collision_body_names, "left_ankle_roll_link", "right_ankle_roll_link"]
        self.forces = torch.zeros(2, 6, len(names), 3)
        sensor = SimpleNamespace(body_names=names, data=SimpleNamespace(
            net_forces_w_history=self.forces, net_forces_w=self.forces[:, 0]))
        robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[0., 0., 1.], [100., 0., 1.]]),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1)))
        self.scene = _Scene(robot=robot)
        self.scene.sensors = {"contact_forces": sensor, "fdm_height_scanner": SimpleNamespace(reset=lambda ids: None)}
        self.scene.terrain = SimpleNamespace(env_origin_ids=torch.tensor([0, 1]), origin_split_ids=torch.tensor([0, 0]))
        self.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(set_command=lambda *a: None))
        self.sim = SimpleNamespace(forward=lambda: None)
        self.observations = {"policy": torch.zeros(2, 283)}
        self.observation_manager = SimpleNamespace(compute=lambda **kw: self.observations)

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        self._reset_idx(torch.arange(2))
        return self.observations, {}

    def _reset_idx(self, ids):
        self.local_steps[ids] = 0
        self.scene["robot"].data.root_pos_w[ids, 0] = self.origins[ids]

    def step(self, actions):
        self.common_step_counter += 1
        self.local_steps += 1
        self.forces.zero_()
        self.forces[:, :, -2:, 2] = 2
        collision = self.local_steps == self.collision_steps
        self.forces[collision, 2, 0, 0] = 7
        done = self.local_steps == self.collision_steps + 1
        self._reset_idx(torch.nonzero(done).flatten())
        self.scene["robot"].data.root_pos_w[:, 0] = self.origins + self.local_steps * 0.02
        self.observations = {"policy": torch.zeros(2, 283)}
        return self.observations, None, done, torch.zeros_like(done), {}


def _maps(sensor, ids, **kwargs):
    return torch.zeros(len(ids), 1, 60, 46), torch.zeros(len(ids), 1, 60, 46, dtype=torch.bool)


def test_multiple_rounds_release_pending_data_and_preserve_plans(tmp_path):
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg)
    policy_resets = []
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: policy_resets.append(done))
    with EpisodeShardWriter(tmp_path, "train", {}) as writer, patch(
        "unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=_maps
    ):
        collector = FixedRolloutCollector(env, env.observations, policy, writer, cfg)
        for index in range(6):
            result = collector.collect(12, seed=42 + index, min_fill=1, log_interval_s=0, round_index=index)
            assert result["frames_recorded"] == 24
            assert result["completion_fraction"] == 1
            assert result["stop_reason"] == "capacity"
            assert collector.buffer is None
            assert collector.pending_frames == collector.pending_episodes == 0
            assert collector.frame_counts.tolist() == [12, 12]
    assert env.reset_calls == list(range(42, 48))
    assert sum(item is None for item in policy_resets) == 6
    episodes = list(iter_episodes(tmp_path, "train"))
    ids = [int(ep.episode_id[0]) for ep in episodes]
    assert len(set(ids)) == len(ids)
    for episode in episodes:
        episode.validate()
        assert episode.delta_t[0] == 0
        assert episode.usd_origin_id.unique().numel() == 1
        assert episode.state_history_raw[-1, 0, 0] - episode.state_history_raw[0, 0, 0] < 5
        for frame in range(episode.num_frames - 1):
            if episode.has_outgoing_command[frame + 1]:
                torch.testing.assert_close(episode.command_plan[frame, 1:], episode.command_plan[frame + 1, :-1])


def test_step_limit_records_real_partial_round_without_fabrication(tmp_path):
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg, collision_steps=(100000, 100000))
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    with EpisodeShardWriter(tmp_path, "train", {}) as writer, patch(
        "unitree_rl_lab.fdm.runner.fixed_collector.door_aware_height_map", side_effect=_maps
    ):
        collector = FixedRolloutCollector(env, env.observations, policy, writer, cfg)
        result = collector.collect(20, min_fill=.5, max_steps=310, log_interval_s=0)
    assert result["stop_reason"] == "step_limit"
    assert result["frames_recorded"] == 24 < result["target_frames"]
    episodes = list(iter_episodes(tmp_path, "train"))
    assert sum(ep.num_frames for ep in episodes) == result["frames_recorded"]
    assert all(int(ep.termination_reason[-1]) == int(TerminationReason.ROUND_CUT) for ep in episodes)


def test_unfilled_round_fails_at_step_limit_and_releases_buffer(tmp_path):
    cfg = RolloutCfg(num_envs=2)
    env = _BatchEnv(cfg)
    policy = SimpleNamespace(act=lambda obs: torch.zeros(2, 29), reset=lambda done=None: None)
    with EpisodeShardWriter(tmp_path, "train", {}) as writer:
        collector = FixedRolloutCollector(env, env.observations, policy, writer, cfg)
        with pytest.raises(RuntimeError, match="below required"):
            collector.collect(12, max_steps=1, log_interval_s=0)
        assert collector.buffer is None
