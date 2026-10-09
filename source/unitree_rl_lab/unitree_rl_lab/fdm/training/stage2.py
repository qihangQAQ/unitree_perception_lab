"""Stage-two initialization and strict, current-format round resume helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..config import FDMModelCfg
from ..models import G1HeightFDM


@dataclass
class Stage2Cfg:
    batches_per_round: int = 4
    initial_mppi_fraction: float = .25
    mppi_fraction: float = .40
    initial_mix_rounds: int = 2
    max_window_repeats: int = 4
    navigation_eval_interval: int = 5  # 0 disables; otherwise initial/final + interval.

    def validate(self):
        if self.batches_per_round < 1 or self.max_window_repeats < 1 or self.initial_mix_rounds < 0:
            raise ValueError("Invalid stage-two collection schedule.")
        if not 0 <= self.initial_mppi_fraction <= 1 or not 0 <= self.mppi_fraction <= 1:
            raise ValueError("MPPI fractions must be in [0, 1].")
        if self.navigation_eval_interval < 0:
            raise ValueError("Navigation evaluation interval must be nonnegative.")

    def fraction(self, round_index):
        return self.initial_mppi_fraction if round_index < self.initial_mix_rounds else self.mppi_fraction


def load_initial_model(path):
    """Load architecture and weights only; never import a stage-one optimizer."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FDMModelCfg(**state["model_cfg"])
    cfg.validate()
    model = G1HeightFDM(cfg)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Initialization checkpoint contains nonfinite model parameters.")
    return model, cfg


def validate_stage2_resume(state, settings, dataset_manifest):
    from pathlib import Path
    stage = state.get("stage2", {})
    if stage.get("version") != 1 or stage.get("settings") != settings:
        raise ValueError("--resume requires this stage-two format and the same collection/planner settings.")
    if Path(state["dataset_manifest"]).resolve() != Path(dataset_manifest).resolve():
        raise ValueError("Stage-two resume must use its original dataset.")
    if stage.get("next_planner_model_version") != state["round"] + 1:
        raise ValueError("Invalid stage-two planner snapshot version.")
    return stage


def stage2_settings(stage_cfg, mppi_cfg, goal_cfg, rollout_cfg, collection):
    return {"stage2": asdict(stage_cfg), "mppi": asdict(mppi_cfg), "goals": asdict(goal_cfg),
            "num_envs": rollout_cfg.num_envs, "seed": rollout_cfg.seed, "collection": collection}
