"""Validate round-boundary resumes before collecting or overwriting data."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import FDMModelCfg, TrainCfg
from ..data.shard_writer import read_manifest


CHECKPOINT_VERSION = 1


def validate_resume_checkpoint(checkpoint: dict, cfg: TrainCfg, model_cfg: FDMModelCfg) -> int:
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "model_cfg", "train_cfg", "round", "global_epoch", "dataset_manifest",
        "checkpoint_version", "torch_rng_state", "cuda_rng_states",
    }
    if not isinstance(checkpoint, dict) or required - checkpoint.keys():
        raise ValueError("--resume requires a full FDM checkpoint from the current training format.")
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION or "samples_per_round" not in checkpoint["train_cfg"]:
        raise ValueError("Unsupported FDM checkpoint format; start a new training run.")
    if checkpoint["model_cfg"] != model_cfg.__dict__:
        raise ValueError("FDM model configuration differs from the resume checkpoint.")
    runtime_options = {"batch_size", "num_workers", "checkpoint_dir", "device", "samples_per_round"}
    mismatches = [
        key for key, value in cfg.__dict__.items()
        if key not in runtime_options and checkpoint["train_cfg"].get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Resume training configuration differs: {', '.join(mismatches)}. "
            "--collection-rounds is the original TOTAL, not the number of remaining rounds."
        )
    round_index, epoch = checkpoint["round"], checkpoint["global_epoch"]
    if type(round_index) is not int or round_index < 0 or type(epoch) is not int:
        raise ValueError("Invalid round/epoch in FDM checkpoint.")
    if epoch != (round_index + 1) * cfg.epochs_per_round:
        raise ValueError("Only checkpoints saved at a completed round boundary can be resumed.")
    scheduler = checkpoint["scheduler_state_dict"]
    if scheduler.get("last_epoch") != epoch or scheduler.get("T_max") != cfg.collection_rounds * cfg.epochs_per_round:
        raise ValueError("Checkpoint scheduler does not match its epoch/schedule.")
    if round_index + 1 >= cfg.collection_rounds:
        raise ValueError("This checkpoint has already completed the configured training schedule.")
    return round_index + 1


def check_checkpoint_destinations(output: str | Path, start_round: int, total_rounds: int) -> None:
    for index in range(start_round, total_rounds):
        for suffix in (".pt", ".json"):
            path = Path(output) / f"fdm_round_{index:03d}{suffix}"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}. Choose a new --output directory.")


def resume_dataset_metadata(current: dict, recorded: dict) -> dict:
    """Accept a code revision update, but keep all collection semantics strict."""
    current = json.loads(json.dumps(current))
    recorded = json.loads(json.dumps(recorded))
    keys = (current.keys() | recorded.keys()) - {"git_commit"}
    differences = sorted(key for key in keys if current.get(key) != recorded.get(key))
    if differences:
        raise ValueError(f"Resume dataset metadata differs in: {', '.join(differences)}.")
    if current.get("git_commit") != recorded.get("git_commit"):
        print(
            f"[FDM] Resume code revision: {recorded.get('git_commit')} -> {current.get('git_commit')}; "
            "collection metadata matches. Original manifest provenance is retained.", flush=True,
        )
    return recorded


def load_round_shards(
    summary_path: str | Path, dataset_root: str | Path, *, round_index: int,
) -> list[Path]:
    """Reuse only the explicitly selected, completed next-round collection."""
    report = json.loads(Path(summary_path).read_text())
    root, manifest = read_manifest(dataset_root)
    if report.get("split") != "train" or report.get("round") != round_index:
        raise ValueError("--resume-round-summary must describe the next untrained train round.")
    if Path(report["manifest"]).resolve() != (root / "manifest.json").resolve():
        raise ValueError("The round summary belongs to a different dataset manifest.")
    collection = report.get("collection") or {}
    if collection.get("status") != "completed":
        raise ValueError("The selected round collection was not completed.")
    entries = {(root / item["path"]).resolve(): item for item in manifest["shards"] if item["split"] == "train"}
    paths = [Path(value).resolve() for value in report.get("shards", [])]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Round summary has empty or duplicate shard paths.")
    for path in paths:
        if path not in entries or not path.is_file():
            raise ValueError(f"Round shard is missing or not registered as train data: {path}")
    counts = report["dataset"]["counts"]
    for key in ("episodes", "frames"):
        if sum(entries[path][key] for path in paths) != counts[key]:
            raise ValueError(f"Round summary {key} differs from the dataset manifest.")
    settings = manifest["metadata"].get("collection", {})
    if collection.get("collection_mode") != "fixed" or settings.get("mode") != "fixed" or settings.get("version") != 1:
        raise ValueError("Fixed collection requires a fixed-capacity round summary and manifest.")
    if collection.get("frames_recorded") != counts["frames"]:
        raise ValueError("Fixed round frame count differs from its archived records.")
    return paths
