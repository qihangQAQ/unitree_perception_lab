"""Reusable trainer for offline epochs and online collection rounds."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..config import FDMModelCfg, TrainCfg
from ..utils.progress import ProgressLogger
from .losses import FDMLoss, LossAccumulator
from .metrics import MetricAccumulator
from .resume import CHECKPOINT_VERSION, validate_resume_checkpoint


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


class FDMTrainer:
    """Optimize an FDM while keeping validation data fixed across rounds."""

    def __init__(self, model: torch.nn.Module, cfg: TrainCfg | None = None) -> None:
        self.cfg = cfg or TrainCfg()
        self.cfg.validate()
        requested = torch.device(self.cfg.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print(f"[FDM] WARNING: requested {requested}, but CUDA is unavailable; training on CPU.", flush=True)
            requested = torch.device("cpu")
        self.device = requested
        self.model = model.to(self.device)
        self.loss_fn = FDMLoss(self.cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.collection_rounds * self.cfg.epochs_per_round
        )
        self.global_epoch = 0
        self.last_epoch_timing: dict[str, float] = {}

    def _loader_description(self, loader: DataLoader) -> str:
        shards = getattr(loader.dataset, "shards", None)
        return (
            f"device={self.device} batch_size={loader.batch_size} workers={loader.num_workers} "
            f"windows={len(loader.dataset)} shards={len(shards) if shards is not None else 'n/a'} "
            f"cache={getattr(loader.dataset, 'cache_mode', 'n/a')}"
        )

    def train_epoch(self, loader: DataLoader, *, log_interval_s: float = 10.0) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        print(f"[FDM] Training {self._loader_description(loader)}", flush=True)
        data_seconds = step_seconds = 0.0
        initial_loads = getattr(loader.dataset, "shard_load_count", 0)
        last_data = last_step = 0.0

        def detail() -> str:
            loads = (
                str(loader.dataset.shard_load_count - initial_loads)
                if loader.num_workers == 0 and hasattr(loader.dataset, "shard_load_count") else "n/a"
            )
            return (
                f"loss={totals['loss'] / batches:.4f} "
                f"last_s(data={last_data:.3f},step={last_step:.3f}) "
                f"avg_s(data={data_seconds / batches:.3f},step={step_seconds / batches:.3f}) "
                f"shard_loads={loads}"
            )

        progress = ProgressLogger(
            f"train epoch={self.global_epoch + 1}", len(loader), unit="batches", interval_s=log_interval_s
        )
        ready = time.monotonic()
        for batch in loader:
            step_started = time.monotonic()
            last_data = step_started - ready
            batch = _move_batch(batch, self.device)
            prediction = self.model.forward_batch(batch)
            losses = self.loss_fn(prediction, batch)
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm)
            self.optimizer.step()
            batches += 1
            for name, value in losses.items():
                totals[name] += float(value.detach())
            # Reading the CUDA loss scalars synchronizes the preceding transfer,
            # forward, backward and optimizer work before timing the next fetch.
            last_step = time.monotonic() - step_started
            data_seconds += last_data
            step_seconds += last_step
            progress.update(batches, detail=detail, force=batches == 1)
            ready = time.monotonic()
        if batches == 0:
            raise ValueError("Training loader produced no batches.")
        self.scheduler.step()
        self.global_epoch += 1
        self.last_epoch_timing = {
            "data_seconds": data_seconds,
            "step_seconds": step_seconds,
            "mean_data_seconds": data_seconds / batches,
            "mean_step_seconds": step_seconds / batches,
        }
        progress.update(batches, detail=detail, force=True)
        return {name: value / batches for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, *, log_interval_s: float = 10.0) -> dict[str, Any]:
        self.model.eval()
        loss_totals = LossAccumulator(self.loss_fn)
        metrics = MetricAccumulator(command_timestep=getattr(loader.dataset, "command_timestep", 0.5))
        batches = 0
        progress = ProgressLogger("validate", len(loader), unit="batches", interval_s=log_interval_s)
        for batch in loader:
            batch = _move_batch(batch, self.device)
            prediction = self.model.forward_batch(batch)
            loss_totals.update(prediction, batch)
            metrics.update(prediction, batch)
            batches += 1
            progress.update(batches)
        if batches == 0:
            raise ValueError("Validation loader produced no batches.")
        progress.update(batches, force=True)
        return {**loss_totals.result(), **metrics.result()}

    def restore_checkpoint(self, checkpoint: dict, *, model_cfg: FDMModelCfg) -> int:
        """Restore optimization state; return the next zero-based collection round."""
        next_round = validate_resume_checkpoint(checkpoint, self.cfg, model_cfg)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_epoch = checkpoint["global_epoch"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng = checkpoint["cuda_rng_states"]
        if cuda_rng and self.device.type == "cuda" and len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng)
        print(
            f"[FDM] Resumed FDM: completed_rounds={next_round}/{self.cfg.collection_rounds} "
            f"global_epoch={self.global_epoch} next_epoch={self.global_epoch + 1} "
            f"lr={self.optimizer.param_groups[0]['lr']:.9g}; optimizer and scheduler restored.", flush=True,
        )
        return next_round

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        round_index: int,
        model_cfg: FDMModelCfg,
        metrics: dict[str, Any],
        dataset_manifest: str,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "train_cfg": self.cfg.__dict__,
                "round": round_index,
                "global_epoch": self.global_epoch,
                "metrics": metrics,
                "dataset_manifest": dataset_manifest,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            },
            temporary,
        )
        os.replace(temporary, destination)
        metrics_path = destination.with_suffix(".json")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return destination
