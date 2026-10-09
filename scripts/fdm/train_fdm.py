"""Alternate frozen-policy rollout collection and first-stage FDM training."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from _common import (
    add_collection_args, add_dataset_cache_args, collect_fixed, configure_safe_spawn, dataset_metadata,
    prepare_dataset_cache, require_input_file, resolve_collection_settings,
)


def _parse_args() -> argparse.Namespace:
    # Spawned DataLoader workers import this script. They must not parse CLI
    # arguments, import simulator tasks, or create a second SimulationApp.
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD

    parser = argparse.ArgumentParser(description="Online collect/train loop for the G1 height-map FDM.")
    parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
    parser.add_argument("--checkpoint", required=True, help="Frozen G1 locomotion policy checkpoint.")
    parser.add_argument("--resume", help="Resume a full checkpoint from the current fixed-capacity training format.")
    parser.add_argument(
        "--resume-round-summary",
        help="Completed collection summary for the next untrained round; reuse its shards instead of collecting again.",
    )
    parser.add_argument("--terrain-usd", default=str(DEFAULT_TERRAIN_USD), help="USD terrain (default: bundled FDM terrain).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="logs/fdm_g1")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collection-rounds", type=int, default=20)
    parser.add_argument("--epochs-per-round", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for low RAM; >0 uses spawn).")
    parser.add_argument("--samples-per-round", type=int,
                        help="Preselect this many train windows before caching (new runs: 80000; 0: all windows).")
    add_collection_args(parser)
    add_dataset_cache_args(parser)
    parser.add_argument("--log-interval", type=float, default=10.0, help="Progress interval in seconds (0 disables it).")
    parser.add_argument("--eval-after-collection", action="store_true", help="Evaluate the current FDM on each new dataset before training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-policy-corruption", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for option in (
        "num_envs", "collection_rounds", "epochs_per_round", "batch_size"
    ):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive.")
    if args.workers < 0 or args.log_interval < 0:
        parser.error("--workers and --log-interval must be nonnegative.")
    if args.samples_per_round is not None and args.samples_per_round < 0:
        parser.error("--samples-per-round must be nonnegative.")
    if args.resume_round_summary and not args.resume:
        parser.error("--resume-round-summary requires --resume.")
    try:
        args.checkpoint = require_input_file(args.checkpoint, "--checkpoint")
        args.terrain_usd = require_input_file(args.terrain_usd, "--terrain-usd")
        if args.resume:
            args.resume = require_input_file(args.resume, "--resume")
        if args.resume_round_summary:
            args.resume_round_summary = require_input_file(args.resume_round_summary, "--resume-round-summary")
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
    from unitree_rl_lab.fdm.runner import FixedRolloutCollector, FrozenRecurrentPolicy
    from unitree_rl_lab.fdm.runner.collection_log import CollectionLog
    from unitree_rl_lab.fdm.training import FDMTrainer
    from unitree_rl_lab.fdm.training.resume import (
        check_checkpoint_destinations, load_round_shards, resume_dataset_metadata, validate_resume_checkpoint,
    )
    from unitree_rl_lab.fdm.data.shard_writer import read_manifest
    from unitree_rl_lab.fdm.utils.memory import log_memory
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    existing_metadata = None
    if (Path(args.dataset) / "manifest.json").exists():
        _, existing_manifest = read_manifest(args.dataset)
        existing_metadata = existing_manifest["metadata"]
    collection_settings = resolve_collection_settings(args, existing_metadata)
    resume_state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    samples_per_round = args.samples_per_round if args.samples_per_round is not None else 80_000
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
        samples_per_round=samples_per_round,
        checkpoint_dir=args.output,
        device=args.device,
    )
    rollout_cfg.validate()
    train_cfg.validate()
    model_cfg = FDMModelCfg()
    start_round = 0
    reused_round_shards = None
    recorded_metadata = None
    if args.resume:
        start_round = validate_resume_checkpoint(resume_state, train_cfg, model_cfg)
        if args.samples_per_round is None:
            samples_per_round = train_cfg.samples_per_round = resume_state["train_cfg"]["samples_per_round"]
            train_cfg.validate()
        if Path(resume_state["dataset_manifest"]).resolve() != (Path(args.dataset) / "manifest.json").resolve():
            raise ValueError("--dataset must be the dataset used by the resume checkpoint.")
        _, manifest = read_manifest(args.dataset)
        recorded_metadata = manifest["metadata"]
        if not any(entry["split"] == "val" for entry in manifest["shards"]):
            raise ValueError("Resume requires the original fixed validation shards.")
        if args.resume_round_summary:
            reused_round_shards = load_round_shards(
                args.resume_round_summary, args.dataset, round_index=start_round,
            )
        print(f"[FDM] Resume checkpoint: {args.resume}; next round={start_round + 1}/{args.collection_rounds}.", flush=True)
    check_checkpoint_destinations(args.output, start_round, train_cfg.collection_rounds)
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
        f"[FDM] Schedule: envs={args.num_envs}, rounds={args.collection_rounds}, "
        f"epochs_per_round={args.epochs_per_round}, batch_size={args.batch_size}, workers={args.workers}, "
        f"dataset_cache={args.dataset_cache} collection_mode=fixed "
        f"samples_per_round={samples_per_round} collection={collection_settings}",
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
    metadata["collection"] = collection_settings
    if recorded_metadata is not None:
        (collection_log.directory / "resume_context.json").write_text(json.dumps({
            "checkpoint": args.resume, "start_round": start_round,
            "round_summary": args.resume_round_summary, "current_metadata": metadata,
        }, indent=2))
        metadata = resume_dataset_metadata(metadata, recorded_metadata)
    elif existing_metadata is not None:
        metadata = resume_dataset_metadata(metadata, existing_metadata)
    log_memory("environment_ready", log=collection_log)

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
            collector = FixedRolloutCollector(env, observations, policy, writer, rollout_cfg)
            validation_collection = collect_fixed(collector, args, split="val", seed=args.seed, log=collection_log)
            del collector
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
    log_memory("before_validation_cache", log=collection_log)
    prepare_dataset_cache(validation_dataset, args)
    log_memory("after_validation_cache", log=collection_log)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    trainer = FDMTrainer(G1HeightFDM(model_cfg), train_cfg)
    if resume_state is not None:
        trainer.restore_checkpoint(resume_state, model_cfg=model_cfg)
        del resume_state
    if args.eval_after_collection:
        collection_log.save_evaluation(
            validation_report, trainer.evaluate(validation_loader, log_interval_s=args.log_interval),
            global_epoch=trainer.global_epoch,
        )

    rollout_cfg.split = "train"
    with EpisodeShardWriter(
        args.dataset,
        "train",
        metadata,
        max_frames_per_shard=rollout_cfg.max_frames_per_shard,
    ) as writer:
        collector = None
        for round_index in range(start_round, train_cfg.collection_rounds):
            reused = round_index == start_round and reused_round_shards is not None
            if reused:
                new_shards = reused_round_shards
                collection = json.loads(Path(args.resume_round_summary).read_text()).get("collection")
                print(
                    f"[FDM] Round {round_index + 1}/{train_cfg.collection_rounds}: "
                    f"reusing {len(new_shards)} completed train shards from {args.resume_round_summary}.", flush=True,
                )
            else:
                if collector is None:
                    print("[FDM] Switching to train terrain split...", flush=True)
                    observations = _activate_split(env, "train", args.seed + 1)
                    policy.reset()
                    collector = FixedRolloutCollector(env, observations, policy, writer, rollout_cfg)
                print(f"[FDM] Round {round_index + 1}/{train_cfg.collection_rounds}: collecting fixed-capacity train data...", flush=True)
                log_memory(f"round_{round_index:03d}_before_collection", collector=collector, log=collection_log)
                before = {path.resolve() for path in _split_shards(args.dataset, "train")}
                collection = collect_fixed(collector, args, split="train", seed=args.seed + 1 + round_index,
                                           log=collection_log, round_index=round_index)
                after = _split_shards(args.dataset, "train")
                new_shards = [path for path in after if path.resolve() not in before]
            if not new_shards:
                raise RuntimeError(f"Collection round {round_index} produced no new train shard.")
            log_memory(f"round_{round_index:03d}_before_index", collector=collector, log=collection_log)
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
                round_dataset, round_index=round_index, collection=collection, reused=reused,
                sampler_targets={
                    "collision_window_fraction": train_cfg.collision_window_fraction,
                    "low_motion_window_fraction": train_cfg.low_motion_fraction,
                },
            )
            if not len(round_dataset):
                raise RuntimeError(f"No valid training windows in round {round_index}. See: {report}")
            # Evaluate the natural distribution before drawing the weighted pool.
            if args.eval_after_collection:
                collection_log.save_evaluation(
                    report,
                    trainer.evaluate(DataLoader(round_dataset, shuffle=False, **loader_kwargs), log_interval_s=args.log_interval),
                    global_epoch=trainer.global_epoch,
                )
            if samples_per_round:
                round_dataset.select_sample_pool(
                    samples_per_round, seed=args.seed + round_index,
                    collision_fraction=train_cfg.collision_window_fraction,
                    low_motion_fraction=train_cfg.low_motion_fraction,
                )
            log_memory(f"round_{round_index:03d}_before_cache", collector=collector, log=collection_log)
            prepare_dataset_cache(round_dataset, args, resident_cache_bytes=validation_dataset.cached_bytes)
            log_memory(f"round_{round_index:03d}_after_cache", collector=collector, log=collection_log)
            collection_log.save_sample_pool(report, round_dataset)
            sampler = None if samples_per_round else make_collision_balanced_sampler(
                round_dataset,
                train_cfg.collision_window_fraction,
                train_cfg.low_motion_fraction,
            )
            train_loader = DataLoader(
                round_dataset,
                sampler=sampler,
                shuffle=bool(samples_per_round),
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
            # Keep validation resident, but release this round before collecting
            # another one. DataLoader also holds a reference to its dataset.
            released = round_dataset.cached_bytes
            round_dataset.release_cache()
            del train_loader, sampler, round_dataset
            log_memory(f"round_{round_index:03d}_released", collector=collector, log=collection_log)
            print(
                f"[FDM] Released train sample cache: {released / 2**30:.3f}GiB; "
                f"validation_cache={validation_dataset.cache_mode}.", flush=True,
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
