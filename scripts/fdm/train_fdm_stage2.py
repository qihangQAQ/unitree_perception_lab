"""Stage two: bounded mixed random/MPPI rollouts, followed by FDM-only updates."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path

from _common import (add_collection_args, add_dataset_cache_args, dataset_metadata,
                     prepare_dataset_cache, require_input_file, resolve_collection_settings, sha256_file)
from _stage2_common import add_planner_args, create_environment, planner_configs


def _parse_args():
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
    parser.add_argument("--checkpoint", required=True, help="Frozen locomotion policy; never optimized.")
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--fdm-checkpoint", help="Initial FDM weights; stage-one optimizer/schedule are discarded.")
    initial.add_argument("--resume", help="Continue a completed round of this stage-two format only.")
    parser.add_argument("--terrain-usd", default=str(DEFAULT_TERRAIN_USD))
    parser.add_argument("--dataset", required=True, help="Dedicated stage-two root; do not use the stage-one dataset.")
    parser.add_argument("--output", default="logs/fdm_g1/stage2_mppi_v1")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collection-batches", type=int, default=4)
    parser.add_argument("--collection-rounds", type=int, default=10)
    parser.add_argument("--epochs-per-round", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples-per-round", type=int, default=80_000)
    parser.add_argument("--mppi-fraction", type=float, default=.40)
    parser.add_argument("--initial-mppi-fraction", type=float, default=.25)
    parser.add_argument("--initial-mix-rounds", type=int, default=2)
    parser.add_argument("--max-window-repeats", type=int, default=4)
    parser.add_argument("--navigation-eval-interval", type=int, default=5,
                        help="Initial/final and every N rounds paired navigation trials; 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=float, default=10)
    parser.add_argument("--disable-policy-corruption", action="store_true")
    add_planner_args(parser)
    add_collection_args(parser)
    add_dataset_cache_args(parser)
    parser.set_defaults(dataset_cache_gb=8.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        for name in ("checkpoint", "terrain_usd", "fdm_checkpoint", "resume"):
            if getattr(args, name):
                setattr(args, name, require_input_file(getattr(args, name), "--" + name.replace("_", "-")))
        if min(args.num_envs, args.collection_rounds, args.epochs_per_round, args.batch_size, args.samples_per_round) < 1:
            raise ValueError("Environment, training and sample counts must be positive.")
        if args.workers < 0 or args.log_interval < 0 or not 0 < args.learning_rate < 1:
            raise ValueError("Invalid worker count, log interval or learning rate.")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _train(args):
    import torch
    from torch.utils.data import DataLoader
    from unitree_rl_lab.fdm.config import RolloutCfg, TrainCfg
    from unitree_rl_lab.fdm.data import EpisodeShardWriter, FDMWindowDataset
    from unitree_rl_lab.fdm.data.shard_writer import _atomic_json_dump, read_manifest
    from unitree_rl_lab.fdm.data.source_sampling import select_mixed_pool
    from unitree_rl_lab.fdm.runner.collection_log import CollectionLog
    from unitree_rl_lab.fdm.runner.stage2_collector import Stage2RolloutCollector
    from unitree_rl_lab.fdm.runner.navigation_evaluation import evaluate_navigation, compare_navigation
    from unitree_rl_lab.fdm.training import FDMTrainer
    from unitree_rl_lab.fdm.training.resume import check_checkpoint_destinations, resume_dataset_metadata, validate_resume_checkpoint
    from unitree_rl_lab.fdm.training.stage2 import (Stage2Cfg, load_initial_model, stage2_settings, validate_stage2_resume)
    from unitree_rl_lab.fdm.utils.memory import log_memory

    root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    train_root = root / "train"
    collection = resolve_collection_settings(args)
    mppi_cfg, goal_cfg = planner_configs(args)
    stage_cfg = Stage2Cfg(args.collection_batches, args.initial_mppi_fraction, args.mppi_fraction,
                          args.initial_mix_rounds, args.max_window_repeats, args.navigation_eval_interval)
    stage_cfg.validate()
    rollout = RolloutCfg(task_name=args.task, policy_checkpoint=args.checkpoint, terrain_usd_path=args.terrain_usd,
                         dataset_root=str(root), num_envs=args.num_envs, seed=args.seed,
                         policy_observation_corruption=not args.disable_policy_corruption, split="val")
    rollout.validate()
    train_cfg = TrainCfg(collection_rounds=args.collection_rounds, epochs_per_round=args.epochs_per_round,
                          batch_size=args.batch_size, learning_rate=args.learning_rate, num_workers=args.workers,
                          samples_per_round=args.samples_per_round, checkpoint_dir=str(output), device=args.device)
    settings = stage2_settings(stage_cfg, mppi_cfg, goal_cfg, rollout, collection)
    state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    stage_state = None
    model, model_cfg = load_initial_model(args.resume or args.fdm_checkpoint)
    start_round = 0
    if state is not None:
        stage_state = validate_stage2_resume(state, settings, train_root / "manifest.json")
        start_round = validate_resume_checkpoint(state, train_cfg, model_cfg)
        if not (root / "validation.json").is_file():
            raise ValueError("Resume requires the original fixed validation datasets and baseline report.")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"New stage-two training requires an empty --dataset: {root}")
        stage_state = {"version": 1, "settings": settings, "initial_checkpoint": args.fdm_checkpoint,
                       "initial_checkpoint_sha256": sha256_file(args.fdm_checkpoint)}
    check_checkpoint_destinations(output, start_round, train_cfg.collection_rounds)
    torch.manual_seed(args.seed)
    trainer = FDMTrainer(model, train_cfg)
    if state is not None:
        trainer.restore_checkpoint(state, model_cfg=model_cfg)
        del state
    root.mkdir(parents=True, exist_ok=True)
    log = CollectionLog(output)
    print(f"[FDM] Stage2 settings: {json.dumps(settings)}; lr={train_cfg.learning_rate:g}, "
          f"samples={train_cfg.samples_per_round}, batch={train_cfg.batch_size}.", flush=True)
    env, observations, policy, grid = create_environment(args, rollout)
    try:
        metadata = dataset_metadata(env, rollout)
        metadata.update(schema_semantics="v4 executed commands; post-collision tail from last pre-event decision",
                        collection=collection, stage2_settings=settings,
                        initial_checkpoint_sha256=stage_state["initial_checkpoint_sha256"])
        if args.resume:
            _, manifest = read_manifest(train_root)
            metadata = resume_dataset_metadata(metadata, manifest["metadata"])
        log_memory("stage2_environment_ready", log=log)
        loader_kwargs = {"batch_size": train_cfg.batch_size, "num_workers": args.workers,
                         "pin_memory": trainer.device.type == "cuda"}
        if args.workers:
            loader_kwargs["multiprocessing_context"] = "spawn"
        snapshot = copy.deepcopy(trainer.model).eval().requires_grad_(False)
        validation = {}
        baseline = json.loads((root / "validation.json").read_text()) if args.resume else {}
        for name, fraction in (("random", 0.0), ("mppi", 1.0)):
            destination = root / f"validation_{name}"
            if not args.resume:
                with EpisodeShardWriter(destination, "val", metadata, schema_version=4) as writer:
                    collector = Stage2RolloutCollector(env, observations, policy, writer, rollout, model=snapshot,
                                                       navigation_grid=grid, mppi_cfg=mppi_cfg, goal_cfg=goal_cfg)
                    collector.collect(args.validation_frames_per_env, seed=args.seed + 10_000 + int(fraction),
                                      min_fill=args.collection_min_fill, tail_factor=args.collection_tail_factor,
                                      max_steps=args.collection_max_steps, log_interval_s=args.log_interval,
                                      log=log, mppi_fraction=fraction, model_version=0)
                    del collector
            dataset = FDMWindowDataset(destination, "val", collect_statistics=True, log_interval_s=args.log_interval)
            prepare_dataset_cache(dataset, args, resident_cache_bytes=sum(d.cached_bytes for d in validation.values()))
            validation[name] = dataset
            if not args.resume:
                baseline[name] = trainer.evaluate(DataLoader(dataset, **loader_kwargs),
                                                   log_interval_s=args.log_interval, detailed=True)
        if not args.resume:
            _atomic_json_dump(root / "validation.json", baseline)
        baseline_navigation = None
        if stage_cfg.navigation_eval_interval:
            baseline_path = root / "navigation_baseline.json"
            if args.resume:
                baseline_navigation = json.loads(baseline_path.read_text())
            else:
                baseline_navigation = evaluate_navigation(env, observations, policy, rollout, snapshot, grid,
                                                          mppi_cfg=mppi_cfg, goal_cfg=goal_cfg, seed=args.seed + 90_000,
                                                          log_interval_s=args.log_interval)
                _atomic_json_dump(baseline_path, baseline_navigation)
        del snapshot
        rollout.split = "train"
        with EpisodeShardWriter(train_root, "train", metadata, schema_version=4) as writer:
            for round_index in range(start_round, train_cfg.collection_rounds):
                env.unwrapped.scene.terrain.activate_split("train")
                rollout.split = "train"
                snapshot = copy.deepcopy(trainer.model).eval().requires_grad_(False)
                first_shard = len(writer.written_paths)
                collections = []
                fraction = stage_cfg.fraction(round_index)
                collector = Stage2RolloutCollector(env, observations, policy, writer, rollout, model=snapshot,
                                                   navigation_grid=grid, mppi_cfg=mppi_cfg, goal_cfg=goal_cfg)
                for batch in range(stage_cfg.batches_per_round):
                    print(f"[FDM] Stage2 round={round_index + 1}/{train_cfg.collection_rounds} "
                          f"collect_batch={batch + 1}/{stage_cfg.batches_per_round} MPPI={fraction:.0%}", flush=True)
                    collections.append(collector.collect(
                        args.frames_per_env, seed=args.seed + 1_000_000 + round_index * stage_cfg.batches_per_round + batch,
                        min_fill=args.collection_min_fill, tail_factor=args.collection_tail_factor,
                        max_steps=args.collection_max_steps, log_interval_s=args.log_interval, log=log,
                        round_index=round_index, mppi_fraction=fraction, model_version=round_index,
                    ))
                    log_memory(f"stage2_round_{round_index:03d}_batch_{batch}_released", collector=collector, log=log)
                del collector, snapshot
                shards = writer.written_paths[first_shard:]
                dataset = FDMWindowDataset(train_root, "train", shard_paths=shards, collect_statistics=True,
                                            log_interval_s=args.log_interval)
                report = log.save_summary(dataset, round_index=round_index, collection={"status": "completed",
                                           "batches": collections, "frames_recorded": sum(c["frames_recorded"] for c in collections)})
                select_mixed_pool(dataset, train_cfg.samples_per_round, mppi_fraction=fraction, seed=args.seed + round_index,
                                  collision_fraction=train_cfg.collision_window_fraction,
                                  low_motion_fraction=train_cfg.low_motion_fraction, max_repeats=stage_cfg.max_window_repeats)
                prepare_dataset_cache(dataset, args, resident_cache_bytes=sum(d.cached_bytes for d in validation.values()))
                log.save_sample_pool(report, dataset)
                log_memory(f"stage2_round_{round_index:03d}_cache_ready", log=log)
                loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
                train_metrics = {}
                for _ in range(train_cfg.epochs_per_round):
                    train_metrics = trainer.train_epoch(loader, log_interval_s=args.log_interval)
                metrics = {"train": train_metrics, "validation": {
                    name: trainer.evaluate(DataLoader(data, **loader_kwargs), log_interval_s=args.log_interval, detailed=True)
                    for name, data in validation.items()
                }, "sample_pool": dataset.sample_pool_report}
                dataset.release_cache()
                del loader, dataset
                if stage_cfg.navigation_eval_interval and ((round_index + 1) % stage_cfg.navigation_eval_interval == 0
                                                          or round_index + 1 == train_cfg.collection_rounds):
                    env.unwrapped.scene.terrain.activate_split("val")
                    rollout.split = "val"
                    snapshot = copy.deepcopy(trainer.model).eval().requires_grad_(False)
                    current = evaluate_navigation(env, observations, policy, rollout, snapshot, grid, mppi_cfg=mppi_cfg,
                                                   goal_cfg=goal_cfg, seed=args.seed + 90_000, log_interval_s=args.log_interval)
                    comparison = compare_navigation(baseline_navigation, current)
                    _atomic_json_dump(output / f"navigation_round_{round_index:03d}.json", comparison)
                    metrics["navigation"] = {key: value for key, value in comparison.items() if key not in ("baseline", "current")}
                    del snapshot
                path = trainer.save_checkpoint(output / f"fdm_round_{round_index:03d}.pt", round_index=round_index,
                                                model_cfg=model_cfg, metrics=metrics, dataset_manifest=str(train_root / "manifest.json"),
                                                stage2_state={**stage_state, "next_planner_model_version": round_index + 1})
                log_memory(f"stage2_round_{round_index:03d}_released", log=log)
                print(f"[FDM] Stage2 round={round_index:03d} "
                      f"random_position={metrics['validation']['random']['position_mae_m']:.4f}m "
                      f"mppi_position={metrics['validation']['mppi']['position_mae_m']:.4f}m checkpoint={path}", flush=True)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher
    args = _parse_args()
    launcher = AppLauncher(args)
    try:
        _train(args)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
