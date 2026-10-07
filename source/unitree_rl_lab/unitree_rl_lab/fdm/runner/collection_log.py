"""Local collection progress and atomic, versioned summary reports."""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..data.shard_writer import _atomic_json_dump
from ..data.statistics import CONTACT_GROUPS


def print_summary(report: dict) -> None:
    """Show data quality in the terminal; the JSON contains the complete report."""
    data = report["dataset"]
    rows = dict(data["counts"])
    for key in ("collision_episode_fraction", "collision_window_fraction", "low_motion_window_fraction"):
        rows[key] = data[key]
    rows["termination_reasons"] = data["termination_reasons"]
    rows["collision_groups"] = data["collision_groups"]
    rows["invalid_height_fraction"] = data["quality"]["invalid_height_fraction"]
    rows["nonfinite_values"] = data["quality"]["nonfinite_values"]
    rows["invalid_time_intervals"] = data["quality"]["invalid_time_intervals"]
    rows.update(data["statistics"])
    rows.update(data["endpoint_histograms"])
    if report.get("collection") is not None:
        for name in ("diagnostics", "timing_seconds", "contact_force_peak_n"):
            rows[name] = report["collection"][name]
    print(f"[FDM] Dataset summary split={report['split']} round={report['round']} reused={report['reused']}", flush=True)
    for name, value in rows.items():
        print(f"  {name:32s} {json.dumps(value, allow_nan=False)}", flush=True)


class CollectionLog:
    """Keep each invocation separate, including runs that reuse fixed validation."""

    def __init__(self, output: str | Path) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self.directory = Path(output) / "collection" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        print(f"[FDM] Collection reports: {self.directory}", flush=True)

    def event(self, value: dict) -> None:
        record = {"report_version": 1, "run_id": self.run_id, "time_utc": datetime.now(timezone.utc).isoformat(), **value}
        with (self.directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def save_summary(
        self, dataset, *, round_index: int | None = None, collection: dict | None = None,
        reused: bool = False, sampler_targets: dict | None = None, evaluation: dict | None = None,
    ) -> Path:
        report = {
            "report_version": 1, "run_id": self.run_id, "split": dataset.split, "round": round_index,
            "reused": reused, "manifest": str((dataset.root / "manifest.json").resolve()),
            "shards": [str(path.resolve()) for path in dataset.shards],
            "dataset": dataset.statistics, "collection": collection,
            "sampler_targets": sampler_targets, "evaluation_before_training": evaluation,
            "contact_force_peak_n": collection["contact_force_peak_n"] if collection else None,
        }
        name = dataset.split if round_index is None else f"{dataset.split}_round_{round_index:03d}"
        path = self.directory / f"{name}_summary.json"
        _atomic_json_dump(path, report)
        print_summary(report)
        print(f"[FDM] Summary saved: {path}", flush=True)
        return path

    def save_evaluation(self, path: Path, metrics: dict, *, global_epoch: int) -> None:
        report = json.loads(path.read_text())
        report["evaluation_before_training"] = {"global_epoch": global_epoch, "metrics": metrics}
        _atomic_json_dump(path, report)
        print(f"[FDM] Evaluation before training (epoch={global_epoch}): {json.dumps(metrics)}", flush=True)


class CollectionTracker:
    """Cheap per-step counters; synchronize small diagnostic tensors only when logging."""

    def __init__(self, collector, target: int, interval_s: float, log: CollectionLog | None, round_index: int | None):
        self.collector = collector
        self.target = target
        self.interval_s = interval_s
        self.log = log
        self.round_index = round_index
        self.start = self.last_log = time.monotonic()
        self.initial_step = collector.env.common_step_counter
        self.initial_completed = collector.completed_episodes
        self.initial_shards = len(collector.writer.written_paths)
        self.initial_write_time = collector.writer.write_seconds
        self.initial_command_time = collector._command_seconds
        self.frames = 0
        self.policy_seconds = self.sim_seconds = self.data_seconds = 0.0
        self.counters = torch.zeros(4, dtype=torch.long, device=collector.device)
        self.force_peaks = torch.zeros(3, device=collector.device)
        self.accepted_force_episodes = 0
        self.last_record = None
        self.update(force=True, status="started")

    def update(self, *, force: bool = False, status: str = "collecting", error: dict | None = None) -> dict | None:
        now = time.monotonic()
        if not force and (self.interval_s <= 0 or now - self.last_log < self.interval_s):
            return None
        collector = self.collector
        elapsed = now - self.start
        completed = collector.completed_episodes - self.initial_completed
        steps = collector.env.common_step_counter - self.initial_step
        phases = torch.bincount(collector.phase.long(), minlength=4).cpu().tolist()
        counters = self.counters.cpu().tolist()
        timings = {
            "policy": self.policy_seconds, "simulation": self.sim_seconds,
            "command": collector._command_seconds - self.initial_command_time,
            "data": self.data_seconds,
            "write": collector.writer.write_seconds - self.initial_write_time,
        }
        record = {
            "event": "collection", "status": status, "error": error, "split": collector.cfg.split, "round": self.round_index,
            "episodes": completed, "target_episodes": self.target, "completion_fraction": completed / self.target,
            "frames_recorded": self.frames, "shards_written": len(collector.writer.written_paths) - self.initial_shards,
            "steps": steps, "sim_seconds_per_env": steps * collector.env.step_dt, "elapsed_seconds": elapsed,
            "episodes_per_second": completed / max(elapsed, 1.0e-9),
            "steps_per_second": steps / max(elapsed, 1.0e-9),
            "eta_seconds": elapsed * (self.target - completed) / completed if completed else None,
            "phases": dict(zip(("settling", "warmup", "active", "waiting_for_reset"), phases)),
            "diagnostics": dict(zip(("spawn_rejections", "warmup_collisions", "warmup_resets", "resets"), counters)),
            "timing_seconds": timings,
            "timing_definition": "cumulative host wall time; asynchronous GPU work may be charged at a later sync",
            "contact_force_peak_n": {
                name: value if math.isfinite(value) else None
                for name, value in zip(CONTACT_GROUPS, self.force_peaks.cpu().tolist())
            }
            if self.accepted_force_episodes else None,
            "force_scope": "max per body over physics substeps during active trajectories accepted in this call",
        }
        if self.log is not None:
            self.log.event(record)
        if self.interval_s > 0 or status != "collecting":
            eta = f"{record['eta_seconds']:.0f}s" if record["eta_seconds"] is not None else "pending"
            print(
                f"[FDM] collect split={collector.cfg.split} round={self.round_index} status={status} "
                f"episodes={completed}/{self.target} ({100 * completed / self.target:.1f}%) "
                f"frames={self.frames} shards={record['shards_written']} steps={steps} "
                f"steps/s={record['steps_per_second']:.1f} episodes/s={record['episodes_per_second']:.3f} "
                f"eta={eta} phases={record['phases']} diagnostics={record['diagnostics']} "
                f"times_s={ {name: round(value, 2) for name, value in timings.items()} }",
                flush=True,
            )
        self.last_log = now
        self.last_record = record
        return record
