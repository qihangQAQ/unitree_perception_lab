"""Evaluate an FDM checkpoint on a fixed validation or test split."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from unitree_rl_lab.fdm.config import FDMModelCfg, TrainCfg
from unitree_rl_lab.fdm.data import FDMWindowDataset
from unitree_rl_lab.fdm.models import G1HeightFDM
from unitree_rl_lab.fdm.training import FDMTrainer

parser = argparse.ArgumentParser(description="Evaluate a trained G1 FDM.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--split", choices=("val", "test"), default="val")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()


def main() -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = FDMModelCfg(**payload["model_cfg"])
    model = G1HeightFDM(model_cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    train_cfg = TrainCfg(**payload["train_cfg"])
    train_cfg.device = args.device
    trainer = FDMTrainer(model, train_cfg)
    dataset = FDMWindowDataset(args.dataset, args.split, horizon=model_cfg.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    for name, value in sorted(trainer.evaluate(loader).items()):
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
