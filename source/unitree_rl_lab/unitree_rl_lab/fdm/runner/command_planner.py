"""Persistent preplanned velocity commands for collision-safe supervision."""

from __future__ import annotations

import torch

from ..config import CommandSamplingCfg


class CorrelatedCommandPlanner:
    """Maintain a rolling ``[H, 3]`` plan for every simulator environment.

    Advancing shifts the existing plan and samples only one new tail command.
    Thus a plan stored before collision remains the exact counterfactual plan and
    future entries never become reset commands or artificial zeros.
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        cfg: CommandSamplingCfg,
        device: str | torch.device,
        seed: int,
    ) -> None:
        cfg.validate()
        self.num_envs = num_envs
        self.horizon = horizon
        self.cfg = cfg
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.plan = torch.zeros(num_envs, horizon, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.plan[:, 0]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        previous = torch.zeros(len(env_ids), 3, device=self.device)
        previous[:, 0] = self._uniform(len(env_ids), self.cfg.min_forward_speed, self.cfg.max_forward_speed)
        previous[:, 1] = self._uniform(len(env_ids), self.cfg.min_lateral_speed, self.cfg.max_lateral_speed)
        for step in range(self.horizon):
            previous = self._sample_next(previous)
            self.plan[env_ids, step] = previous

    def advance(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        previous_tail = self.plan[env_ids, -1].clone()
        self.plan[env_ids, :-1] = self.plan[env_ids, 1:].clone()
        self.plan[env_ids, -1] = self._sample_next(previous_tail)

    def _uniform(self, count: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(count, device=self.device, generator=self.generator)

    def _sample_next(self, previous: torch.Tensor) -> torch.Tensor:
        count = previous.shape[0]
        selector = torch.rand(count, device=self.device, generator=self.generator)
        correlated_limit = self.cfg.correlated_probability
        straight_limit = correlated_limit + self.cfg.straight_probability
        turn_limit = straight_limit + self.cfg.turn_probability

        target_vx = self._uniform(count, self.cfg.min_forward_speed, self.cfg.max_forward_speed)
        target_vy = self._uniform(count, self.cfg.min_lateral_speed, self.cfg.max_lateral_speed)
        target_wz = self._uniform(count, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        correlated_vx = self.cfg.correlation * previous[:, 0] + (1.0 - self.cfg.correlation) * target_vx
        correlated_vy = self.cfg.correlation * previous[:, 1] + (1.0 - self.cfg.correlation) * target_vy
        correlated_wz = self.cfg.correlation * previous[:, 2] + (1.0 - self.cfg.correlation) * target_wz

        command = torch.zeros(count, 3, device=self.device)
        correlated = selector < correlated_limit
        command[correlated, 0] = correlated_vx[correlated]
        command[correlated, 1] = correlated_vy[correlated]
        command[correlated, 2] = correlated_wz[correlated]

        straight = (selector >= correlated_limit) & (selector < straight_limit)
        command[straight, 0] = target_vx[straight]
        command[straight, 1] = target_vy[straight]
        if torch.any(straight):
            command[straight, 2] = torch.randn(
                int(straight.sum()), device=self.device, generator=self.generator
            ) * self.cfg.straight_yaw_std

        turn = (selector >= straight_limit) & (selector < turn_limit)
        command[turn, 0] = self._uniform(int(turn.sum()), 0.0, self.cfg.turn_forward_max)
        command[turn, 1] = target_vy[turn]
        command[turn, 2] = target_wz[turn]
        # The remaining stop samples stay exactly zero.
        command[:, 0].clamp_(self.cfg.min_forward_speed, self.cfg.max_forward_speed)
        command[:, 1].clamp_(self.cfg.min_lateral_speed, self.cfg.max_lateral_speed)
        command[:, 2].clamp_(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        return command
