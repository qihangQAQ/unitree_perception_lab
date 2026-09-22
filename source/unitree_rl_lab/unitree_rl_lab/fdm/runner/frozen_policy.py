"""Strict loader for the frozen recurrent RSL-RL locomotion actor."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


class FrozenRecurrentPolicy:
    """Construct only the actor-critic module, not the training runner/optimizer."""

    def __init__(
        self,
        observations: dict[str, torch.Tensor],
        num_actions: int,
        agent_cfg: Any,
        checkpoint: str | Path,
        device: str | torch.device,
    ) -> None:
        from rsl_rl.modules import ActorCriticRecurrent
        from rsl_rl.utils import resolve_obs_groups

        self.device = torch.device(device)
        cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else deepcopy(agent_cfg)
        self.clip_actions = cfg.get("clip_actions")
        policy_cfg = deepcopy(cfg["policy"])
        class_name = policy_cfg.pop("class_name")
        if class_name != "ActorCriticRecurrent":
            raise ValueError(f"Expected ActorCriticRecurrent checkpoint, got {class_name!r}.")
        obs_groups = resolve_obs_groups(observations, deepcopy(cfg["obs_groups"]), ["critic"])
        self.module = ActorCriticRecurrent(observations, obs_groups, num_actions, **policy_cfg).to(self.device)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen locomotion checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in payload:
            raise KeyError(f"Checkpoint {checkpoint_path} has no model_state_dict.")
        self.module.load_state_dict(payload["model_state_dict"], strict=True)
        self.module.eval()
        self.module.requires_grad_(False)
        self.checkpoint_path = checkpoint_path

    @torch.inference_mode()
    def act(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = self.module.act_inference(observations)
        if self.clip_actions is not None:
            actions = actions.clamp(-float(self.clip_actions), float(self.clip_actions))
        return actions

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.module.reset(dones)
