"""Externally controlled velocity command term for FDM rollouts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class ExternalVelocityCommand(CommandTerm):
    """A command buffer written by the collector and never internally sampled."""

    def __init__(self, cfg: "ExternalVelocityCommandCfg", env) -> None:
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
        super().__init__(cfg, env)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_command(self, command: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command = torch.as_tensor(command, device=self.device, dtype=self._command.dtype)
        if command.shape != (len(env_ids), 3):
            raise ValueError(f"Expected command shape {(len(env_ids), 3)}, got {tuple(command.shape)}.")
        self._command[env_ids] = command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self._command[env_ids] = 0.0

    def _update_command(self) -> None:
        pass


@configclass
class ExternalVelocityCommandCfg(CommandTermCfg):
    class_type: type = ExternalVelocityCommand
    # Isaac Lab samples this interval with ``uniform_`` during every reset, for
    # which infinity is invalid.  A finite ~31 year interval effectively turns
    # internal resampling off while the collector remains the sole command owner.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
