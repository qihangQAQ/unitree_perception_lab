"""PPO augmented with SSR-style delayed foothold supervision."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO

from ..modules.foothold_imagination import FootholdImagination, world_xy_to_base_yaw
from ..storage.foothold_replay_buffer import FootholdReplayBuffer


class FootholdPPO(PPO):
    """Keep PPO unchanged while independently learning future foot contacts."""

    def __init__(
        self,
        policy,
        foothold_learning_rate: float = 1.0e-3,
        foothold_hidden_dims: tuple[int, ...] = (512, 256, 128),
        foothold_min_sigma: float = 0.02,
        foothold_max_sigma: float = 0.50,
        foothold_replay_buffer_size: int = 50_000,
        foothold_pending_steps: int = 16,
        foothold_batch_size: int = 2_048,
        foothold_updates_per_iteration: int = 4,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if foothold_pending_steps <= 0:
            raise ValueError("foothold_pending_steps must be positive.")
        if foothold_replay_buffer_size <= 0:
            raise ValueError("foothold_replay_buffer_size must be positive.")
        if foothold_batch_size <= 0:
            raise ValueError("foothold_batch_size must be positive.")
        if foothold_updates_per_iteration < 0:
            raise ValueError("foothold_updates_per_iteration must be non-negative.")

        self.foothold_learning_rate = foothold_learning_rate
        self.foothold_hidden_dims = tuple(foothold_hidden_dims)
        self.foothold_min_sigma = foothold_min_sigma
        self.foothold_max_sigma = foothold_max_sigma
        self.foothold_replay_buffer_size = foothold_replay_buffer_size
        self.foothold_pending_steps = foothold_pending_steps
        self.foothold_batch_size = foothold_batch_size
        self.foothold_updates_per_iteration = foothold_updates_per_iteration

        self.foothold_imagination: FootholdImagination | None = None
        self.foothold_optimizer: optim.Optimizer | None = None
        self.foothold_replay: FootholdReplayBuffer | None = None
        self._foothold_prediction_sink: Callable[[torch.Tensor, torch.Tensor], None] | None = None

    def set_foothold_prediction_sink(
        self,
        sink: Callable[[torch.Tensor, torch.Tensor], None],
    ) -> None:
        """Install the environment callback receiving current-step predictions."""

        self._foothold_prediction_sink = sink

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        super().init_storage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
        )
        critic_dim = self.policy.get_critic_obs(obs).shape[-1]
        action_dim = actions_shape[0]
        self.foothold_imagination = FootholdImagination(
            critic_dim,
            action_dim,
            hidden_dims=self.foothold_hidden_dims,
            min_sigma=self.foothold_min_sigma,
            max_sigma=self.foothold_max_sigma,
        ).to(self.device)
        self.foothold_optimizer = optim.Adam(
            self.foothold_imagination.parameters(),
            lr=self.foothold_learning_rate,
        )
        self.foothold_replay = FootholdReplayBuffer(
            critic_dim,
            action_dim,
            self.foothold_replay_buffer_size,
            self.device,
        )

        pending_shape = (num_envs, 2, self.foothold_pending_steps)
        self._foothold_pending_obs = torch.empty(
            *pending_shape,
            critic_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._foothold_pending_actions = torch.empty(
            *pending_shape,
            action_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._foothold_pending_base_xy = torch.empty(
            *pending_shape,
            2,
            device=self.device,
        )
        self._foothold_pending_yaw = torch.empty(*pending_shape, device=self.device)
        self._foothold_pending_terrain = torch.empty(
            *pending_shape,
            device=self.device,
            dtype=torch.int8,
        )
        self._foothold_pending_count = torch.zeros(
            num_envs,
            2,
            device=self.device,
            dtype=torch.long,
        )
        self._foothold_labeled_total = 0
        self._foothold_discarded_total = 0

    def act(self, obs):
        actions = super().act(obs)
        if self.foothold_imagination is not None and self._foothold_prediction_sink is not None:
            with torch.no_grad():
                critic_obs = self.policy.get_critic_obs(self.transition.observations)
                mu, sigma = self.foothold_imagination(critic_obs, self.transition.actions)
            self._foothold_prediction_sink(mu, sigma)
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        # Consume touchdown labels before PPO clears the transition cached by act().
        self._process_foothold_events(extras, dones)
        super().process_env_step(obs, rewards, dones, extras)

    def _process_foothold_events(self, extras: dict, dones: torch.Tensor) -> None:
        if self.foothold_replay is None or "foothold_events" not in extras:
            return
        events = extras["foothold_events"]
        swing = events["decision_swing"].to(self.device)
        base_xy = events["decision_base_xy"].to(self.device)
        base_yaw = events["decision_base_yaw"].to(self.device)
        terrain_ids = events["terrain_id"].to(self.device)
        critic_obs = self.policy.get_critic_obs(self.transition.observations)
        actions = self.transition.actions

        for foot_id in range(2):
            env_ids = swing[:, foot_id].nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            counts = self._foothold_pending_count[env_ids, foot_id]
            overflow = counts >= self.foothold_pending_steps
            if overflow.any():
                overflow_ids = env_ids[overflow]
                self._foothold_discarded_total += int(
                    self._foothold_pending_count[overflow_ids, foot_id].sum().item()
                )
                self._foothold_pending_count[overflow_ids, foot_id] = 0
                counts = self._foothold_pending_count[env_ids, foot_id]

            self._foothold_pending_obs[env_ids, foot_id, counts] = critic_obs[env_ids].to(torch.float16)
            self._foothold_pending_actions[env_ids, foot_id, counts] = actions[env_ids].to(torch.float16)
            self._foothold_pending_base_xy[env_ids, foot_id, counts] = base_xy[env_ids]
            self._foothold_pending_yaw[env_ids, foot_id, counts] = base_yaw[env_ids]
            self._foothold_pending_terrain[env_ids, foot_id, counts] = terrain_ids[env_ids]
            self._foothold_pending_count[env_ids, foot_id] = counts + 1

        touchdown = events["touchdown_valid"].to(self.device)
        touchdown_xy = events["touchdown_xy_w"].to(self.device)
        step_ids = torch.arange(self.foothold_pending_steps, device=self.device)
        for foot_id in range(2):
            env_ids = touchdown[:, foot_id].nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            counts = self._foothold_pending_count[env_ids, foot_id]
            valid_steps = step_ids.unsqueeze(0) < counts.unsqueeze(1)
            if not valid_steps.any():
                continue

            queued_obs = self._foothold_pending_obs[env_ids, foot_id][valid_steps]
            queued_actions = self._foothold_pending_actions[env_ids, foot_id][valid_steps]
            queued_base_xy = self._foothold_pending_base_xy[env_ids, foot_id][valid_steps]
            queued_yaw = self._foothold_pending_yaw[env_ids, foot_id][valid_steps]
            queued_terrain = self._foothold_pending_terrain[env_ids, foot_id][valid_steps]
            targets_w = touchdown_xy[env_ids, foot_id].unsqueeze(1).expand(
                -1,
                self.foothold_pending_steps,
                -1,
            )[valid_steps]
            targets_b = world_xy_to_base_yaw(targets_w, queued_base_xy, queued_yaw)
            foot_ids = torch.full(
                (queued_obs.shape[0],),
                foot_id,
                device=self.device,
                dtype=torch.long,
            )
            self.foothold_replay.insert(
                queued_obs,
                queued_actions,
                targets_b,
                foot_ids,
                queued_terrain,
            )
            self._foothold_labeled_total += int(queued_obs.shape[0])
            self._foothold_pending_count[env_ids, foot_id] = 0

        invalid_touchdown = events["touchdown"].to(self.device) & ~touchdown
        invalid_env_ids, invalid_foot_ids = invalid_touchdown.nonzero(as_tuple=True)
        if invalid_env_ids.numel() > 0:
            self._foothold_discarded_total += int(
                self._foothold_pending_count[invalid_env_ids, invalid_foot_ids].sum().item()
            )
            self._foothold_pending_count[invalid_env_ids, invalid_foot_ids] = 0

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._foothold_discarded_total += int(
                self._foothold_pending_count[done_ids].sum().item()
            )
            self._foothold_pending_count[done_ids] = 0

    def update(self):
        loss_dict = super().update()
        loss_dict.update(self._update_foothold_imagination())
        return loss_dict

    def _update_foothold_imagination(self) -> dict[str, float]:
        mean_nll = 0.0
        mean_rmse = 0.0
        mean_upstairs_rmse = 0.0
        mean_downstairs_rmse = 0.0
        num_updates = 0
        upstairs_updates = 0
        downstairs_updates = 0

        if (
            self.foothold_imagination is not None
            and self.foothold_optimizer is not None
            and self.foothold_replay is not None
            and self.foothold_replay.num_samples > 0
        ):
            batch_size = min(self.foothold_batch_size, self.foothold_replay.num_samples)
            for _ in range(self.foothold_updates_per_iteration):
                privileged_obs, actions, targets, foot_ids, terrain_ids = self.foothold_replay.sample(batch_size)
                mu, sigma = self.foothold_imagination(privileged_obs, actions)
                batch_indices = torch.arange(batch_size, device=self.device)
                selected_mu = mu[batch_indices, foot_ids]
                selected_sigma = sigma[batch_indices, foot_ids]
                nll = FootholdImagination.gaussian_nll(selected_mu, selected_sigma, targets)

                self.foothold_optimizer.zero_grad()
                nll.backward()
                if self.is_multi_gpu:
                    for parameter in self.foothold_imagination.parameters():
                        if parameter.grad is not None:
                            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                            parameter.grad /= self.gpu_world_size
                nn.utils.clip_grad_norm_(self.foothold_imagination.parameters(), self.max_grad_norm)
                self.foothold_optimizer.step()

                squared_distance = torch.sum(torch.square(selected_mu.detach() - targets), dim=-1)
                mean_nll += nll.item()
                mean_rmse += torch.sqrt(torch.mean(squared_distance)).item()
                up_mask = terrain_ids == 0
                if up_mask.any():
                    mean_upstairs_rmse += torch.sqrt(torch.mean(squared_distance[up_mask])).item()
                    upstairs_updates += 1
                down_mask = terrain_ids == 1
                if down_mask.any():
                    mean_downstairs_rmse += torch.sqrt(torch.mean(squared_distance[down_mask])).item()
                    downstairs_updates += 1
                num_updates += 1

        if num_updates:
            mean_nll /= num_updates
            mean_rmse /= num_updates
        if upstairs_updates:
            mean_upstairs_rmse /= upstairs_updates
        if downstairs_updates:
            mean_downstairs_rmse /= downstairs_updates

        replay_samples = 0.0 if self.foothold_replay is None else float(self.foothold_replay.num_samples)
        return {
            "foothold_nll": mean_nll,
            "foothold_rmse": mean_rmse,
            "foothold_upstairs_rmse": mean_upstairs_rmse,
            "foothold_downstairs_rmse": mean_downstairs_rmse,
            "foothold_replay_samples": replay_samples,
            "foothold_labeled_total": float(self._foothold_labeled_total),
            "foothold_pending_samples": float(self._foothold_pending_count.sum().item()),
            "foothold_discarded_total": float(self._foothold_discarded_total),
        }

    def broadcast_parameters(self):
        super().broadcast_parameters()
        if self.foothold_imagination is not None:
            state = [self.foothold_imagination.state_dict()]
            torch.distributed.broadcast_object_list(state, src=0)
            self.foothold_imagination.load_state_dict(state[0])
