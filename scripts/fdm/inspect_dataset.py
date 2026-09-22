"""Validate a dataset manifest and summarize its episodes/windows."""

from __future__ import annotations

import argparse
from collections import Counter

from unitree_rl_lab.fdm.data import FDMWindowDataset
from unitree_rl_lab.fdm.data.shard_writer import iter_episodes, read_manifest

parser = argparse.ArgumentParser(description="Inspect a G1 FDM dataset.")
parser.add_argument("--dataset", required=True)
parser.add_argument("--split", choices=("train", "val", "test"), default="train")
args = parser.parse_args()


def main() -> None:
    _, manifest = read_manifest(args.dataset)
    shards = [item for item in manifest["shards"] if item["split"] == args.split]
    reasons: Counter[int] = Counter()
    episode_count = frame_count = collision_episodes = 0
    for episode in iter_episodes(args.dataset, args.split):
        episode.validate()
        episode_count += 1
        frame_count += episode.num_frames
        collision_episodes += int(episode.collision_now.any())
        reasons.update(int(value) for value in episode.termination_reason.tolist())
    dataset = FDMWindowDataset(args.dataset, args.split)
    print(f"schema_version: {manifest['schema_version']}")
    print(f"shards: {len(shards)}")
    print(f"episodes: {episode_count}")
    print(f"frames: {frame_count}")
    print(f"windows: {len(dataset)}")
    print(f"collision_episodes: {collision_episodes}")
    print(f"termination_reason_frame_counts: {dict(reasons)}")


if __name__ == "__main__":
    main()
