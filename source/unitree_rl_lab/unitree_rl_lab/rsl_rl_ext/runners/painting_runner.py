"""Painting runner with estimator checkpoints and deterministic episode clock initialization."""

import torch
from rsl_rl.runners import OnPolicyRunner

from ..algorithms.painting_ppo import PaintingPPO
from ..modules.painting_actor_critic import PaintingActorCritic


class PaintingRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs):
        if self.policy_cfg.pop("class_name") != "PaintingActorCritic":
            raise ValueError("PaintingRunner requires PaintingActorCritic.")
        if self.alg_cfg.pop("class_name") != "PaintingPPO":
            raise ValueError("PaintingRunner requires PaintingPPO.")
        policy = PaintingActorCritic(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        algorithm = PaintingPPO(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **self.alg_cfg)
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # The shared train entry point requests random episode lengths. Painting
        # has a sampled deadline and timed path starting at reset instead.
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path, infos=None):
        torch.save({
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)
        if self.logger_type in ("neptune", "wandb") and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True, map_location=None):
        checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
        self.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")
