"""Painting-specific joint action with an explicit executable target guard."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class SafeJointPositionAction(JointPositionAction):
    """Clamp position targets to the asset's soft limits and expose diagnostics.

    RSL-RL's scalar action clip is still kept as a final policy-output guard.  This
    term protects the physically meaningful target after applying the configured
    per-joint scale and default-position offset.
    """

    cfg: "SafeJointPositionActionCfg"

    def __init__(self, cfg: "SafeJointPositionActionCfg", env):
        super().__init__(cfg, env)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids].clone()
        margin = float(cfg.soft_limit_margin)
        if margin < 0:
            raise ValueError("soft_limit_margin must be nonnegative.")
        limits[..., 0] += margin
        limits[..., 1] -= margin
        if torch.any(limits[..., 0] >= limits[..., 1]):
            raise ValueError("soft_limit_margin leaves an empty joint target interval.")
        self._safe_joint_pos_limits = limits
        self._attempted_actions = torch.zeros_like(self._processed_actions)
        self._target_limit_hits = torch.zeros_like(self._processed_actions, dtype=torch.bool)

        scale = self._scale
        if not isinstance(scale, torch.Tensor):
            scale = torch.full_like(self._processed_actions, float(scale))
        offset = self._offset
        if not isinstance(offset, torch.Tensor):
            offset = torch.full_like(self._processed_actions, float(offset))
        if torch.any(scale == 0):
            raise ValueError("Painting joint action scales must be nonzero.")
        raw_a = (limits[..., 0] - offset) / scale
        raw_b = (limits[..., 1] - offset) / scale
        self._safe_raw_action_limits = torch.stack((torch.minimum(raw_a, raw_b), torch.maximum(raw_a, raw_b)), -1)
        if torch.any(self._safe_raw_action_limits[..., 0] > 0) or torch.any(
            self._safe_raw_action_limits[..., 1] < 0
        ):
            raise ValueError("The configured default joint pose lies outside the guarded target limits.")

    @property
    def safe_joint_pos_limits(self) -> torch.Tensor:
        return self._safe_joint_pos_limits

    @property
    def safe_raw_action_limits(self) -> torch.Tensor:
        return self._safe_raw_action_limits

    @property
    def attempted_actions(self) -> torch.Tensor:
        return self._attempted_actions

    @property
    def target_limit_hits(self) -> torch.Tensor:
        return self._target_limit_hits

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._attempted_actions.copy_(self._processed_actions)
        lower, upper = self._safe_joint_pos_limits.unbind(-1)
        self._target_limit_hits.copy_((self._processed_actions < lower) | (self._processed_actions > upper))
        self._processed_actions.clamp_(min=lower, max=upper)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._attempted_actions[env_ids] = 0.0
        self._target_limit_hits[env_ids] = False


@configclass
class SafeJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SafeJointPositionAction
    soft_limit_margin: float = 0.0
