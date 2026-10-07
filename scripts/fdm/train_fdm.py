"""Alternate frozen-policy rollout collection and first-stage FDM training."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from _common import configure_safe_spawn, dataset_metadata, require_input_file


def _parse_args() -> argparse.Namespace:
    # Spawned DataLoader workers import this script. They must not parse CLI
    # arguments, import simulator tasks, or create a second SimulationApp.
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD

    parser = argparse.ArgumentParser(description="Online collect/train loop for the G1 height-map FDM.")
    parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
    parser.add_argument("--checkpoint", required=True, help="Frozen G1 locomotion policy checkpoint.")
    parser.add_argument("--terrain-usd", default=str(DEFAULT_TERRAIN_USD), help="USD terrain (default: bundled FDM terrain).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="logs/fdm_g1")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--validation-episodes", type=int, default=128)
    parser.add_argument("--episodes-per-round", type=int, default=256)
    parser.add_argument("--collection-rounds", type=int, default=20)
    parser.add_argument("--epochs-per-round", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for low RAM; >0 uses spawn).")
    parser.add_argument("--log-interval", type=float, default=10.0, help="Progress interval in seconds (0 disables it).")
    parser.add_argument("--eval-after-collection", action="store_true", help="Evaluate the current FDM on each new dataset before training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-policy-corruption", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for option in (
        "num_envs", "validation_episodes", "episodes_per_round", "collection_rounds", "epochs_per_round", "batch_size"
    ):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive.")
    if args.workers < 0 or args.log_interval < 0:
        parser.error("--workers and --log-interval must be nonnegative.")
    try:
        args.checkpoint = require_input_file(args.checkpoint, "--checkpoint")
        args.terrain_usd = require_input_file(args.terrain_usd, "--terrain-usd")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _split_shards(dataset_root: str, split: str) -> list[Path]:
    from unitree_rl_lab.fdm.data.shard_writer import read_manifest

    root, manifest = read_manifest(dataset_root)
    return [root / item["path"] for item in manifest["shards"] if item["split"] == split]


def _activate_split(env, split: str, seed: int):
    import torch

    # Sensor/observation buffers may have been allocated during inference rollouts.
    with torch.inference_mode():
        env.unwrapped.scene.terrain.activate_split(split)
        observations, _ = env.reset(seed=seed)
    return observations


def _train(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from torch.utils.data import DataLoader

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.fdm.config import FDMModelCfg, RolloutCfg, TrainCfg
    from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset, make_collision_balanced_sampler
    from unitree_rl_lab.fdm.models import G1HeightFDM
    from unitree_rl_lab.fdm.runner import FDMRolloutCollector, FrozenRecurrentPolicy
    from unitree_rl_lab.fdm.runner.collection_log import CollectionLog
    from unitree_rl_lab.fdm.training import FDMTrainer
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

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
    collection_log = CollectionLog(args.output)
    # Forking the live CUDA/Kit process can inherit locked runtime threads.
    loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if train_cfg.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    print(
        f"[FDM] Schedule: envs={args.num_envs}, validation_episodes={args.validation_episodes}, "
        f"episodes_per_round={args.episodes_per_round}, rounds={args.collection_rounds}, "
        f"epochs_per_round={args.epochs_per_round}, batch_size={args.batch_size}, workers={args.workers}",
        flush=True,
    )
    if max(args.validation_episodes, args.episodes_per_round) > 100 * args.num_envs:
        print(
            "[FDM] Collection requires many sequential episodes per environment. Reducing --num-envs "
            "does not reduce episode counts; use smaller --validation-episodes and --episodes-per-round "
            "for a smoke test. --batch-size only affects the later FDM training stage.",
            flush=True,
        )

    print("[FDM] Creating rollout environment...", flush=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.scene.terrain.usd_path = args.terrain_usd
    env_cfg.scene.terrain.active_split = "val"
    env_cfg.observations.policy.enable_corruption = rollout_cfg.policy_observation_corruption
    configure_safe_spawn(env_cfg, rollout_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    print("[FDM] Resetting validation environment...", flush=True)
    observations, _ = env.reset(seed=args.seed)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    print(f"[FDM] Loading frozen policy: {args.checkpoint}", flush=True)
    policy = FrozenRecurrentPolicy(
        observations,
        env.unwrapped.action_manager.total_action_dim,
        agent_cfg,
        args.checkpoint,
        env.unwrapped.device,
    )
    print("[FDM] Frozen policy loaded. Preparing dataset metadata...", flush=True)
    metadata = dataset_metadata(env, rollout_cfg)

    existing_validation = _split_shards(args.dataset, "val") if (Path(args.dataset) / "manifest.json").exists() else []
    validation_collection = None
    if not existing_validation:
        rollout_cfg.split = "val"
        with EpisodeShardWriter(
            args.dataset,
            "val",
            metadata,
            max_frames_per_shard=rollout_cfg.max_frames_per_shard,
        ) as writer:
            collector = FDMRolloutCollector(env, observations, policy, writer, rollout_cfg)
            validation_collection = collector.collect(
                args.validation_episodes, log_interval_s=args.log_interval, log=collection_log
            )
    else:
        print(f"[FDM] Reusing {len(existing_validation)} fixed validation shards.", flush=True)

    validation_dataset = FDMWindowDataset(
        args.dataset, "val", horizon=rollout_cfg.prediction_horizon, log_interval_s=args.log_interval,
        collect_statistics=True, allow_empty=True,
    )
    validation_report = collection_log.save_summary(
        validation_dataset, collection=validation_collection, reused=bool(existing_validation)
    )
    if not len(validation_dataset):
        raise RuntimeError(f"No valid validation windows. See collection diagnostics: {validation_report}")
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    model_cfg = FDMModelCfg()
    trainer = FDMTrainer(G1HeightFDM(model_cfg), train_cfg)
    if args.eval_after_collection:
        collection_log.save_evaluation(
            validation_report, trainer.evaluate(validation_loader, log_interval_s=args.log_interval),
            global_epoch=trainer.global_epoch,
        )

    print("[FDM] Switching to train terrain split...", flush=True)
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
            print(f"[FDM] Round {round_index + 1}/{train_cfg.collection_rounds}: collecting train episodes...", flush=True)
            before = {path.resolve() for path in _split_shards(args.dataset, "train")}
            collection = collector.collect(
                args.episodes_per_round, log_interval_s=args.log_interval, log=collection_log, round_index=round_index
            )
            after = _split_shards(args.dataset, "train")
            new_shards = [path for path in after if path.resolve() not in before]
            if not new_shards:
                raise RuntimeError(f"Collection round {round_index} produced no new train shard.")
            round_dataset = FDMWindowDataset(
                args.dataset,
                "train",
                horizon=rollout_cfg.prediction_horizon,
                shard_paths=new_shards,
                log_interval_s=args.log_interval,
                collect_statistics=True,
                allow_empty=True,
            )
            report = collection_log.save_summary(
                round_dataset, round_index=round_index, collection=collection,
                sampler_targets={
                    "collision_window_fraction": train_cfg.collision_window_fraction,
                    "low_motion_window_fraction": train_cfg.low_motion_fraction,
                },
            )
            if not len(round_dataset):
                raise RuntimeError(f"No valid training windows in round {round_index}. See: {report}")
            if args.eval_after_collection:
                collection_log.save_evaluation(
                    report,
                    trainer.evaluate(DataLoader(round_dataset, shuffle=False, **loader_kwargs), log_interval_s=args.log_interval),
                    global_epoch=trainer.global_epoch,
                )
            sampler = make_collision_balanced_sampler(
                round_dataset,
                train_cfg.collision_window_fraction,
                train_cfg.low_motion_fraction,
            )
            train_loader = DataLoader(
                round_dataset,
                sampler=sampler,
                **loader_kwargs,
            )
            train_metrics = {}
            for _ in range(train_cfg.epochs_per_round):
                train_metrics = trainer.train_epoch(train_loader, log_interval_s=args.log_interval)
            validation_metrics = trainer.evaluate(validation_loader, log_interval_s=args.log_interval)
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
                f"val_collision_recall={validation_metrics['collision_recall']:.3f} checkpoint={checkpoint_path}",
                flush=True,
            )
    env.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    args = _parse_args()
    print("[FDM] Starting Isaac Sim...", flush=True)
    app_launcher = AppLauncher(args)
    try:
        _train(args)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app_launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
