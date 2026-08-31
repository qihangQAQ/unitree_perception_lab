"""Replay storage for delayed touchdown labels."""

from __future__ import annotations

import torch


class FootholdReplayBuffer:
    """GPU ring buffer for state/action samples labeled at a future touchdown."""

    def __init__(
        self,
        privileged_obs_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str,
        storage_dtype: torch.dtype = torch.float16,
    ):
        self.privileged_obs = torch.empty(
            buffer_size,
            privileged_obs_dim,
            device=device,
            dtype=storage_dtype,
        )
        self.actions = torch.empty(
            buffer_size,
            action_dim,
            device=device,
            dtype=storage_dtype,
        )
        self.targets = torch.empty(buffer_size, 2, device=device, dtype=storage_dtype)
        self.foot_ids = torch.empty(buffer_size, device=device, dtype=torch.long)
        self.terrain_ids = torch.empty(buffer_size, device=device, dtype=torch.int8)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(
        self,
        privileged_obs: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        foot_ids: torch.Tensor,
        terrain_ids: torch.Tensor,
    ) -> None:
        """Insert a batch, retaining only its newest samples if it exceeds capacity."""

        num_samples = privileged_obs.shape[0]
        if num_samples == 0:
            return
        if num_samples > self.buffer_size:
            privileged_obs = privileged_obs[-self.buffer_size :]
            actions = actions[-self.buffer_size :]
            targets = targets[-self.buffer_size :]
            foot_ids = foot_ids[-self.buffer_size :]
            terrain_ids = terrain_ids[-self.buffer_size :]
            num_samples = self.buffer_size

        indices = (torch.arange(num_samples, device=self.device) + self.step) % self.buffer_size
        self.privileged_obs[indices] = privileged_obs.to(self.privileged_obs.dtype)
        self.actions[indices] = actions.to(self.actions.dtype)
        self.targets[indices] = targets.to(self.targets.dtype)
        self.foot_ids[indices] = foot_ids
        self.terrain_ids[indices] = terrain_ids.to(self.terrain_ids.dtype)
        self.step = (self.step + num_samples) % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + num_samples)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample stored labels and restore floating tensors to fp32."""

        if self.num_samples == 0:
            raise RuntimeError("Cannot sample an empty foothold replay buffer.")
        indices = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        return (
            self.privileged_obs[indices].float(),
            self.actions[indices].float(),
            self.targets[indices].float(),
            self.foot_ids[indices],
            self.terrain_ids[indices].long(),
        )
