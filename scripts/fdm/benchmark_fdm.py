"""Time data loading and FDM optimization on existing shards without Isaac Sim."""

from __future__ import annotations

import argparse
import json
import time

from _common import add_dataset_cache_args, prepare_dataset_cache


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from unitree_rl_lab.fdm.config import FDMModelCfg, TrainCfg
    from unitree_rl_lab.fdm.data import FDMWindowDataset, make_collision_balanced_sampler
    from unitree_rl_lab.fdm.models import G1HeightFDM
    from unitree_rl_lab.fdm.training import FDMTrainer
    from unitree_rl_lab.fdm.training.resume import load_round_shards

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--round-summary", help="Benchmark only this completed round, matching online training.")
    parser.add_argument("--samples-per-round", type=int, default=80000, help="Preselect windows before caching; 0 uses all.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2, help="Repeat the sampled epoch to check cache reuse.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    add_dataset_cache_args(parser)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1 or args.epochs < 1:
        parser.error("--batch-size, --batches and --epochs must be positive.")
    if args.samples_per_round < 0:
        parser.error("--samples-per-round must be nonnegative.")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Fix the CUDA environment or explicitly pass --device cpu.")

    torch.manual_seed(args.seed)
    print(
        f"[FDM] Benchmark: existing train shards, device={args.device}, cpu_threads={torch.get_num_threads()}, "
        f"epochs={args.epochs}, batches_per_epoch={args.batches}, batch_size={args.batch_size}, "
        f"dataset_cache={args.dataset_cache}. No simulator or checkpoint writes.",
        flush=True,
    )
    started = time.monotonic()
    shards = None
    if args.round_summary:
        from pathlib import Path
        report = json.loads(Path(args.round_summary).read_text())
        shards = load_round_shards(args.round_summary, args.dataset, round_index=report["round"])
    dataset = FDMWindowDataset(args.dataset, "train", shard_paths=shards)
    print(
        f"[FDM] Index ready in {time.monotonic() - started:.2f}s; "
        f"shards={len(dataset.shards)} total_shard_gib={sum(p.stat().st_size for p in dataset.shards) / 2**30:.2f}",
        flush=True,
    )
    cfg = TrainCfg(device=args.device, batch_size=args.batch_size, num_workers=0)
    if args.samples_per_round:
        dataset.select_sample_pool(args.samples_per_round, seed=args.seed,
                                   collision_fraction=cfg.collision_window_fraction, low_motion_fraction=cfg.low_motion_fraction)
    prepare_dataset_cache(dataset, args)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=args.batches * args.batch_size) \
        if args.samples_per_round else make_collision_balanced_sampler(
        dataset, cfg.collision_window_fraction, cfg.low_motion_fraction,
        num_samples=args.batches * args.batch_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    trainer = FDMTrainer(G1HeightFDM(FDMModelCfg()), cfg)
    # This disposable model measures the same optimizer path as online training.
    # Small nonzero interval prints every completed batch without disabling logs.
    for epoch in range(args.epochs):
        trainer.train_epoch(loader, log_interval_s=1e-9)
        print(
            f"[FDM] Benchmark epoch={epoch + 1} timing: {json.dumps(trainer.last_epoch_timing, sort_keys=True)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
