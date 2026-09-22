"""Reusable trainer for offline epochs and online collection rounds."""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..config import FDMModelCfg, TrainCfg
from .losses import FDMLoss
from .metrics import compute_metrics


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


class FDMTrainer:
    """Optimize an FDM while keeping validation data fixed across rounds."""

    def __init__(self, model: torch.nn.Module, cfg: TrainCfg | None = None) -> None:
        self.cfg = cfg or TrainCfg()
        self.cfg.validate()
        requested = torch.device(self.cfg.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
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

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
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
        if batches == 0:
            raise ValueError("Training loader produced no batches.")
        self.scheduler.step()
        self.global_epoch += 1
        return {name: value / batches for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            batch = _move_batch(batch, self.device)
            prediction = self.model.forward_batch(batch)
            for name, value in self.loss_fn(prediction, batch).items():
                totals[name] += float(value)
            for name, value in compute_metrics(prediction, batch).items():
                totals[name] += value
            batches += 1
        if batches == 0:
            raise ValueError("Validation loader produced no batches.")
        return {name: value / batches for name, value in totals.items()}

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
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "train_cfg": self.cfg.__dict__,
                "round": round_index,
                "global_epoch": self.global_epoch,
                "metrics": metrics,
                "dataset_manifest": dataset_manifest,
            },
            temporary,
        )
        os.replace(temporary, destination)
        metrics_path = destination.with_suffix(".json")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return destination
