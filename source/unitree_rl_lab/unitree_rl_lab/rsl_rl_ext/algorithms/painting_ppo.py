"""Upstream PPO plus an independently supervised, deployable velocity estimator."""

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO


class PaintingPPO(PPO):
    def __init__(self, policy, estimator_learning_rate=1e-3, estimator_max_grad_norm=1.0,
                 velocity_loss_coef=1.0, action_clip=3.0, action_groups=None, **kwargs):
        super().__init__(policy, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("PaintingPPO requires rnd_cfg=None and symmetry_cfg=None.")
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.velocity_loss_coef = velocity_loss_coef
        estimator_ids = {id(p) for p in policy.estimator.parameters()}
        self.optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if id(p) not in estimator_ids], lr=self.learning_rate
        )
        self._std_projection_hook = self.optimizer.register_step_post_hook(
            lambda optimizer, args, hook_kwargs: self.policy.project_distribution_parameters_()
        )
        self.estimator_optimizer = torch.optim.Adam(policy.estimator.parameters(), lr=estimator_learning_rate)
        if action_clip <= 0:
            raise ValueError("action_clip must be positive.")
        self.action_clip = float(action_clip)
        self.action_groups = dict(action_groups or {"all": list(range(policy.num_actions))})
        self._diagnostic_samples = 0
        self._raw_clip_counts = torch.zeros(policy.num_actions, device=self.device)

    def act(self, obs):
        actions = super().act(obs)
        with torch.no_grad():
            self._raw_clip_counts += (actions.abs() > self.action_clip).sum(0)
            self._diagnostic_samples += actions.shape[0]
        return actions

    def update(self):
        # Keep the encoder fixed during PPO's epochs so actor inputs do not change
        # under a separate optimizer while likelihood ratios are being fitted.
        losses = super().update()
        # RolloutStorage.clear resets its cursor, retaining the completed rollout tensors.
        sums = torch.zeros(2, device=self.device)
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs = batch[0]
            prediction = self.policy.estimator(obs[self.policy.proprio_group])
            target = obs[self.policy.velocity_group].detach()
            base_loss = F.mse_loss(prediction[:, :3], target[:, :3])
            tcp_loss = F.mse_loss(prediction[:, 3:], target[:, 3:])
            self.estimator_optimizer.zero_grad(set_to_none=True)
            (self.velocity_loss_coef * (base_loss + tcp_loss)).backward()
            if self.is_multi_gpu:
                for parameter in self.policy.estimator.parameters():
                    torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                    parameter.grad.div_(self.gpu_world_size)
            torch.nn.utils.clip_grad_norm_(self.policy.estimator.parameters(), self.estimator_max_grad_norm)
            self.estimator_optimizer.step()
            sums += torch.stack((base_loss.detach(), tcp_loss.detach()))
            updates += 1
        # Do not leave encoder gradients in PPO's global gradient clipping/reduction.
        self.estimator_optimizer.zero_grad(set_to_none=True)
        losses["base_velocity_estimation"] = (sums[0] / max(updates, 1)).item()
        losses["tcp_velocity_estimation"] = (sums[1] / max(updates, 1)).item()
        denominator = max(self._diagnostic_samples, 1)
        clip_rate = self._raw_clip_counts / denominator
        std = self.policy.effective_action_std.detach()
        losses["action/clip_rate"] = clip_rate.mean().item()
        for name, indices in self.action_groups.items():
            index = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            if index.numel():
                losses[f"action/std_{name}"] = std[index].mean().item()
        self._raw_clip_counts.zero_()
        self._diagnostic_samples = 0
        return losses
