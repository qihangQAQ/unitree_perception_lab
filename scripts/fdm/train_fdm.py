"""Alternate frozen-policy rollout collection and first-stage FDM training."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Online collect/train loop for the G1 height-map FDM.")
parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
parser.add_argument("--checkpoint", required=True, help="Frozen G1 locomotion policy checkpoint.")
parser.add_argument("--terrain-usd", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", default="logs/fdm_g1")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--validation-episodes", type=int, default=128)
parser.add_argument("--episodes-per-round", type=int, default=256)
parser.add_argument("--collection-rounds", type=int, default=20)
parser.add_argument("--epochs-per-round", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable-policy-corruption", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch.utils.data import DataLoader

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.fdm.config import FDMModelCfg, RolloutCfg, TrainCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset, make_collision_balanced_sampler
from unitree_rl_lab.fdm.data.shard_writer import read_manifest
from unitree_rl_lab.fdm.models import G1HeightFDM
from unitree_rl_lab.fdm.runner import FDMRolloutCollector, FrozenRecurrentPolicy
from unitree_rl_lab.fdm.training import FDMTrainer
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from _common import configure_safe_spawn, dataset_metadata


def _split_shards(dataset_root: str, split: str) -> list[Path]:
    root, manifest = read_manifest(dataset_root)
    return [root / item["path"] for item in manifest["shards"] if item["split"] == split]


def _activate_split(env, split: str, seed: int):
    env.unwrapped.scene.terrain.activate_split(split)
    observations, _ = env.reset(seed=seed)
    return observations


def main() -> None:
    rollout_cfg = RolloutCfg(
        task_name=args.task,
        policy_checkpoint=args.checkpoint,
        terrain_usd_path=args.terrain_usd,
        dataset_root=args.dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        policy_observation_corruption=not args.disable_policy_corruption,
    )
    train_cfg = TrainCfg(
        collection_rounds=args.collection_rounds,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        num_workers=args.workers,
        checkpoint_dir=args.output,
        device=args.device,
    )
    rollout_cfg.validate()
    train_cfg.validate()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.scene.terrain.usd_path = args.terrain_usd
    env_cfg.scene.terrain.active_split = "val"
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

    existing_validation = _split_shards(args.dataset, "val") if (Path(args.dataset) / "manifest.json").exists() else []
    if not existing_validation:
        rollout_cfg.split = "val"
        with EpisodeShardWriter(
            args.dataset,
            "val",
            metadata,
            max_frames_per_shard=rollout_cfg.max_frames_per_shard,
        ) as writer:
            collector = FDMRolloutCollector(env, observations, policy, writer, rollout_cfg)
            collector.collect(args.validation_episodes)
    else:
        print(f"[FDM] Reusing {len(existing_validation)} fixed validation shards.")

    validation_dataset = FDMWindowDataset(args.dataset, "val", horizon=rollout_cfg.prediction_horizon)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model_cfg = FDMModelCfg()
    trainer = FDMTrainer(G1HeightFDM(model_cfg), train_cfg)

    observations = _activate_split(env, "train", args.seed + 1)
    policy.reset()
    rollout_cfg.split = "train"
    with EpisodeShardWriter(
        args.dataset,
        "train",
        metadata,
        max_frames_per_shard=rollout_cfg.max_frames_per_shard,
    ) as writer:
        collector = FDMRolloutCollector(env, observations, policy, writer, rollout_cfg)
        for round_index in range(train_cfg.collection_rounds):
            before = {path.resolve() for path in _split_shards(args.dataset, "train")}
            collector.collect(args.episodes_per_round)
            after = _split_shards(args.dataset, "train")
            new_shards = [path for path in after if path.resolve() not in before]
            if not new_shards:
                raise RuntimeError(f"Collection round {round_index} produced no new train shard.")
            round_dataset = FDMWindowDataset(
                args.dataset,
                "train",
                horizon=rollout_cfg.prediction_horizon,
                shard_paths=new_shards,
            )
            sampler = make_collision_balanced_sampler(
                round_dataset,
                train_cfg.collision_window_fraction,
                train_cfg.low_motion_fraction,
            )
            train_loader = DataLoader(
                round_dataset,
                batch_size=train_cfg.batch_size,
                sampler=sampler,
                num_workers=train_cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            train_metrics = {}
            for _ in range(train_cfg.epochs_per_round):
                train_metrics = trainer.train_epoch(train_loader)
            validation_metrics = trainer.evaluate(validation_loader)
            metrics = {"train": train_metrics, "validation": validation_metrics}
            checkpoint_path = Path(args.output) / f"fdm_round_{round_index:03d}.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                round_index=round_index,
                model_cfg=model_cfg,
                metrics=metrics,
                dataset_manifest=str(Path(args.dataset) / "manifest.json"),
            )
            print(
                f"[FDM] round={round_index:03d} windows={len(round_dataset)} "
                f"val_position={validation_metrics['position_mae_m']:.4f}m "
                f"val_collision_recall={validation_metrics['collision_recall']:.3f}"
            )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
