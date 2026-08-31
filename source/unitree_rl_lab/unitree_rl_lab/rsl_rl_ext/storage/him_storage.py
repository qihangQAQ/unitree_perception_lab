"""RSL-RL rollout storage carrying the next physical observation for Old-HIM."""

from __future__ import annotations

import torch

from rsl_rl.storage import RolloutStorage


class HimRolloutStorage(RolloutStorage):
    """Standard feed-forward PPO storage plus the next Old-HIM target."""

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.next_auxiliary = None
            self.next_state_valid = None

    def __init__(self, *args, auxiliary_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_auxiliary = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            auxiliary_dim,
            device=self.device,
        )
        self.next_state_valid = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            1,
            device=self.device,
        )

    def add_transitions(self, transition: Transition):
        if transition.next_auxiliary is None or transition.next_state_valid is None:
            raise ValueError("HIM transition is missing its next-state supervision.")
        self.next_auxiliary[self.step].copy_(transition.next_auxiliary)
        self.next_state_valid[self.step].copy_(transition.next_state_valid)
        super().add_transitions(transition)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This generator is only available for reinforcement-learning storage.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        next_auxiliary = self.next_auxiliary.flatten(0, 1)
        next_state_valid = self.next_state_valid.flatten(0, 1)

        for _ in range(num_epochs):
            for mini_batch_index in range(num_mini_batches):
                start = mini_batch_index * mini_batch_size
                end = (mini_batch_index + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield (
                    observations[batch_indices],
                    actions[batch_indices],
                    values[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                    old_actions_log_prob[batch_indices],
                    old_mu[batch_indices],
                    old_sigma[batch_indices],
                    (None, None),
                    None,
                    next_auxiliary[batch_indices],
                    next_state_valid[batch_indices],
                )
