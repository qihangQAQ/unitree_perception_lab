"""RSL-RL runner for the shared HIM, cross-attention, and MoE policy."""

from __future__ import annotations

import warnings

import torch

from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from ..algorithms.him_moe_ppo import HimMoePPO
from ..modules.actor_critic import HimMoeActorCritic


class HimMoeRunner(OnPolicyRunner):
    """Construct and checkpoint the project-wide perceptive policy extension."""

    def _construct_algorithm(self, obs) -> HimMoePPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "empirical_normalization is deprecated; use policy normalization fields instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name != "HimMoeActorCritic":
            raise ValueError(f"Unsupported HIM-MoE policy class: {policy_class_name!r}.")
        policy = HimMoeActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        if algorithm_class_name != "HimMoePPO":
            raise ValueError(f"Unsupported HIM-MoE algorithm class: {algorithm_class_name!r}.")
        algorithm = HimMoePPO(
            policy,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def save(self, path: str, infos=None):
        """Save both PPO and independent Old-HIM optimizer states."""

        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        """Restore new or legacy task-specific policy and optimizer states."""

        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            estimator_optimizer_state = loaded_dict.get("estimator_optimizer_state_dict")
            if estimator_optimizer_state is not None:
                self.alg.estimator_optimizer.load_state_dict(estimator_optimizer_state)
            else:
                warnings.warn(
                    "Checkpoint has no Old-HIM optimizer state; the estimator optimizer starts fresh.",
                    stacklevel=2,
                )
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
