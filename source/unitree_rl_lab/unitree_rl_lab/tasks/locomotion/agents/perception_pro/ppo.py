"""PPO with a separately optimized Old-HIM estimator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO

from .storage import PerceptionProRolloutStorage


class PerceptionProPPO(PPO):
    """PPO plus velocity regression and Old-HIM swapped prototype prediction."""

    def __init__(
        self,
        policy,
        velocity_loss_coef: float = 1.0,
        him_swap_loss_coef: float = 1.0,
        estimator_learning_rate: float = 1.0e-3,
        estimator_max_grad_norm: float = 10.0,
        auxiliary_group: str = "auxiliary",
        auxiliary_dim: int = 96,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("PerceptionProPPO currently expects rnd_cfg=None and symmetry_cfg=None.")

        self.velocity_loss_coef = velocity_loss_coef
        self.him_swap_loss_coef = him_swap_loss_coef
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.auxiliary_group = auxiliary_group
        self.auxiliary_dim = auxiliary_dim

        estimator_parameter_ids = {id(parameter) for parameter in self.policy.estimator.parameters()}
        self.policy_parameters = [
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad and id(parameter) not in estimator_parameter_ids
        ]
        self.estimator_parameters = [
            parameter for parameter in self.policy.estimator.parameters() if parameter.requires_grad
        ]
        # Replace PPO's all-parameter optimizer: Old-HIM must only see its own losses.
        self.optimizer = optim.Adam(self.policy_parameters, lr=self.learning_rate)
        self.estimator_optimizer = optim.Adam(
            self.estimator_parameters,
            lr=estimator_learning_rate,
        )
        self.transition = PerceptionProRolloutStorage.Transition()

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = PerceptionProRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            auxiliary_dim=self.auxiliary_dim,
        )

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        next_auxiliary = obs[self.auxiliary_group]
        if next_auxiliary.shape[-1] != self.auxiliary_dim:
            raise ValueError(
                f"Expected {self.auxiliary_dim} Old-HIM target values, got {next_auxiliary.shape[-1]}."
            )
        self.transition.next_auxiliary = next_auxiliary.detach()
        # Isaac Lab returns reset observations for done environments, so they cannot be future targets.
        self.transition.next_state_valid = (1.0 - dones.float()).reshape(-1, 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _reduce_gradients(self, parameters: list[torch.nn.Parameter]) -> None:
        """Average one optimizer's gradients without mixing PPO and estimator parameters."""

        flattened_gradients = []
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            flattened_gradients.append(parameter.grad.reshape(-1))
        all_gradients = torch.cat(flattened_gradients)
        torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
        all_gradients /= self.gpu_world_size

        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            parameter.grad.copy_(all_gradients[offset : offset + numel].view_as(parameter))
            offset += numel

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_velocity_loss = 0.0
        mean_him_swap_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for sample in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                _,
                _,
                next_auxiliary_batch,
                next_state_valid_batch,
            ) = sample

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = torch.square(value_batch - returns_batch)
                value_losses_clipped = torch.square(value_clipped - returns_batch)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = torch.square(returns_batch - value_batch).mean()

            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            self.optimizer.zero_grad()
            ppo_loss.backward()
            if self.is_multi_gpu:
                self._reduce_gradients(self.policy_parameters)
            nn.utils.clip_grad_norm_(self.policy_parameters, self.max_grad_norm)
            self.optimizer.step()

            velocity_loss, him_swap_loss = self.policy.estimator.compute_loss(
                obs_batch[self.policy.proprio_group],
                obs_batch[self.auxiliary_group],
                next_auxiliary_batch,
                next_state_valid_batch,
            )
            estimator_loss = (
                self.velocity_loss_coef * velocity_loss
                + self.him_swap_loss_coef * him_swap_loss
            )
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            if self.is_multi_gpu:
                self._reduce_gradients(self.estimator_parameters)
            nn.utils.clip_grad_norm_(self.estimator_parameters, self.estimator_max_grad_norm)
            self.estimator_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_velocity_loss += velocity_loss.item()
            mean_him_swap_loss += him_swap_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "velocity_estimation": mean_velocity_loss / num_updates,
            "him_swap": mean_him_swap_loss / num_updates,
        }
