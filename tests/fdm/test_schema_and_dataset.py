from pathlib import Path

import torch

from unitree_rl_lab.fdm.data import EpisodeData, EpisodeShardWriter, FDMWindowDataset, Split, TerminationReason


def _episode(collision_frame: int | None = 2, episode_id: int = 4) -> EpisodeData:
    frames = 3 if collision_frame is not None else 12
    state = torch.zeros(frames, 10, 8, dtype=torch.float32)
    state[..., 6] = 1.0  # xyzw identity quaternion
    for frame in range(frames):
        state[frame, :, 0] = float(frame)
    collision = torch.zeros(frames, dtype=torch.bool)
    if collision_frame is not None:
        collision[collision_frame] = True
        state[collision_frame, 0, 7] = 1.0
    outgoing = torch.ones(frames, dtype=torch.bool)
    outgoing[-1] = False
    terminated = torch.zeros(frames, dtype=torch.bool)
    truncated = torch.zeros(frames, dtype=torch.bool)
    reason = torch.zeros(frames, dtype=torch.int8)
    if collision_frame is not None:
        reason[collision_frame] = int(TerminationReason.COLLISION)
    else:
        truncated[-1] = True
        reason[-1] = int(TerminationReason.TIMEOUT)
    timestamp = torch.arange(frames, dtype=torch.float64) * 0.5
    history_timestamps = timestamp[:, None] - torch.arange(10, dtype=torch.float64)[None] * 0.05
    command_plan = torch.zeros(frames, 10, 3)
    command_plan[..., 0] = torch.arange(1, 11)
    command_plan[..., 1] = torch.linspace(-0.2, 0.2, 10)
    command_plan[..., 2] = 0.2
    return EpisodeData(
        state_history_raw=state,
        proprio_history=torch.randn(frames, 10, 96),
        history_timestamps=history_timestamps,
        height_map=torch.zeros(frames, 1, 60, 46, dtype=torch.float16),
        height_map_invalid=torch.zeros(frames, 1, 60, 46, dtype=torch.bool),
        command=command_plan[:, 0].clone(),
        command_plan=command_plan,
        has_outgoing_command=outgoing,
        timestamp=timestamp,
        delta_t=torch.cat((torch.zeros(1), torch.full((frames - 1,), 0.5))),
        collision_now=collision,
        contact_groups=torch.zeros(frames, 3, dtype=torch.bool),
        terminated=terminated,
        truncated=truncated,
        termination_reason=reason,
        episode_id=torch.full((frames,), episode_id, dtype=torch.long),
        usd_origin_id=torch.full((frames,), 8, dtype=torch.int32),
        usd_region_split=torch.full((frames,), int(Split.TRAIN), dtype=torch.int8),
    )


def test_collision_window_freezes_pose_and_keeps_preplanned_commands(tmp_path: Path):
    episode = _episode()
    episode.validate()
    with EpisodeShardWriter(tmp_path, "train", {"fixture": 1}, max_frames_per_shard=100) as writer:
        writer.append(episode)
    dataset = FDMWindowDataset(tmp_path, "train")
    sample = dataset[0]
    assert torch.equal(sample["future_commands"], episode.command_plan[0])
    assert sample["future_collision"].tolist() == [0.0, 1.0] + [1.0] * 8
    assert torch.all(sample["valid_mask"])
    assert torch.allclose(sample["future_pose"][1:], sample["future_pose"][1].expand(9, -1))
    assert sample["history_timestamps"][0] == 0.0


def test_normal_window_never_uses_terminal_frame_as_start(tmp_path: Path):
    with EpisodeShardWriter(tmp_path, "train", {"fixture": 2}) as writer:
        writer.append(_episode(collision_frame=None))
    dataset = FDMWindowDataset(tmp_path, "train")
    assert len(dataset) == 1
    assert all(not index.collision for index in dataset.indices)


def test_manifest_metadata_is_stable_across_json_round_trip(tmp_path: Path):
    metadata = {"shape": (1, 60, 46), "nested": {"ratios": (0.8, 0.1, 0.1)}}
    with EpisodeShardWriter(tmp_path, "train", metadata) as writer:
        writer.append(_episode())
    # Tuple-valued process metadata must compare equal to its JSON list form.
    with EpisodeShardWriter(tmp_path, "train", metadata) as writer:
        assert writer.next_episode_id == 1
