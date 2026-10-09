"""Collect frozen-policy G1 trajectories into atomic FDM shards."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher
from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD

from _common import (add_collection_args, collect_fixed, configure_safe_spawn, dataset_metadata,
                     require_input_file, resolve_collection_settings)

parser = argparse.ArgumentParser(description="Collect G1 FDM rollout trajectories.")
parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--terrain-usd", default=str(DEFAULT_TERRAIN_USD), help="USD terrain (default: bundled FDM terrain).")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", default="logs/fdm_g1/collection", help="Root for collection reports.")
parser.add_argument("--log-interval", type=float, default=10.0)
parser.add_argument("--split", choices=("train", "val", "test"), default="train")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable-policy-corruption", action="store_true")
add_collection_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.log_interval < 0:
    parser.error("Environment count must be positive and --log-interval must be nonnegative.")

try:
    args.checkpoint = require_input_file(args.checkpoint, "--checkpoint")
    args.terrain_usd = require_input_file(args.terrain_usd, "--terrain-usd")
except ValueError as exc:
    parser.error(str(exc))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.fdm.config import RolloutCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset
from unitree_rl_lab.fdm.runner import FixedRolloutCollector, FrozenRecurrentPolicy
from unitree_rl_lab.fdm.runner.collection_log import CollectionLog
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main() -> None:
    from pathlib import Path
    from unitree_rl_lab.fdm.data.shard_writer import read_manifest
    from unitree_rl_lab.fdm.training.resume import resume_dataset_metadata

    existing_metadata = None
    if (Path(args.dataset) / "manifest.json").exists():
        existing_metadata = read_manifest(args.dataset)[1]["metadata"]
    settings = resolve_collection_settings(args, existing_metadata)
    collection_log = CollectionLog(args.output)
    rollout_cfg = RolloutCfg(
        task_name=args.task,
        policy_checkpoint=args.checkpoint,
        terrain_usd_path=args.terrain_usd,
        dataset_root=args.dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        split=args.split,
        policy_observation_corruption=not args.disable_policy_corruption,
    )
    rollout_cfg.validate()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.scene.terrain.usd_path = args.terrain_usd
    env_cfg.scene.terrain.active_split = args.split
    env_cfg.observations.policy.enable_corruption = rollout_cfg.policy_observation_corruption
    configure_safe_spawn(env_cfg, rollout_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    observations, _ = env.reset(seed=args.seed)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    policy = FrozenRecurrentPolicy(
        observations,
        env.unwrapped.action_manager.total_action_dim,
        agent_cfg,
        args.checkpoint,
        env.unwrapped.device,
    )
    metadata = dataset_metadata(env, rollout_cfg)
    metadata["collection"] = settings
    if existing_metadata is not None:
        metadata = resume_dataset_metadata(metadata, existing_metadata)
    with EpisodeShardWriter(
        args.dataset,
        args.split,
        metadata,
        max_frames_per_shard=rollout_cfg.max_frames_per_shard,
    ) as writer:
        collector = FixedRolloutCollector(env, observations, policy, writer, rollout_cfg)
        result = collect_fixed(collector, args, split=args.split, seed=args.seed, log=collection_log)
        new_shards = list(writer.written_paths)
    dataset = FDMWindowDataset(
        args.dataset, args.split, horizon=rollout_cfg.prediction_horizon, shard_paths=new_shards,
        log_interval_s=args.log_interval, collect_statistics=True, allow_empty=True,
    )
    collection_log.save_summary(dataset, collection=result)
    print(f"[FDM] Collected {result['episodes']} {args.split} episodes into {args.dataset}.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False)
