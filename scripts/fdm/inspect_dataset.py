"""Summarize stored episodes with the same statistics as online collection."""

from __future__ import annotations

import argparse

from unitree_rl_lab.fdm.data import FDMWindowDataset
from unitree_rl_lab.fdm.runner.collection_log import CollectionLog, print_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a G1 FDM dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output", help="Optionally save reports under this directory's collection/<run_id>/.")
    parser.add_argument("--log-interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.log_interval < 0:
        parser.error("--log-interval must be nonnegative.")
    dataset = FDMWindowDataset(
        args.dataset, args.split, collect_statistics=True, allow_empty=True, log_interval_s=args.log_interval
    )
    if args.output:
        CollectionLog(args.output).save_summary(dataset, reused=True)
    else:
        print_summary({"split": args.split, "round": None, "reused": True, "dataset": dataset.statistics})
    print("[FDM] Live timings, spawn retries and force peaks are unavailable from stored episode tensors.")


if __name__ == "__main__":
    main()
