"""Painting runner with estimator checkpoints and deterministic episode clock initialization."""

import random

import numpy as np
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
        action = self.env.unwrapped.action_manager.get_term("JointPositionAction")
        raw_limits = action.safe_raw_action_limits[0].detach()
        action_clip = float(self.env.clip_actions or 3.0)
        lower = raw_limits[:, 0].clamp_min(-action_clip)
        upper = raw_limits[:, 1].clamp_max(action_clip)
        radius = torch.minimum(-lower, upper)
        min_std = float(self.policy_cfg.get("min_action_std", 0.05))
        max_std = float(self.policy_cfg.get("max_action_std", 0.4))
        mean_limit = float(self.policy_cfg.get("mean_action_limit", 2.0))
        # Three maximum-standard-deviations remain between a saturated mean
        # and the closest executable action boundary.
        std_max = (0.2 * radius).clamp(min=min_std, max=max_std)
        mean_lower = torch.maximum(lower + 3.0 * std_max, torch.full_like(lower, -mean_limit))
        mean_upper = torch.minimum(upper - 3.0 * std_max, torch.full_like(upper, mean_limit))
        if torch.any(mean_lower >= 0) or torch.any(mean_upper <= 0):
            raise ValueError("Joint limits leave no zero-containing guarded action-mean interval.")
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.update(
            action_mean_lower=mean_lower.cpu().tolist(),
            action_mean_upper=mean_upper.cpu().tolist(),
            action_std_max=std_max.cpu().tolist(),
        )
        policy = PaintingActorCritic(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        names = list(action._joint_names)
        groups = {
            "legs": [i for i, name in enumerate(names) if any(x in name for x in ("hip", "knee", "ankle"))],
            "waist": [i for i, name in enumerate(names) if name.startswith("waist_")],
            "left_arm": [i for i, name in enumerate(names) if name.startswith("left_") and any(x in name for x in ("shoulder", "elbow", "wrist"))],
            "right_arm": [i for i, name in enumerate(names) if name.startswith("right_") and any(x in name for x in ("shoulder", "elbow", "wrist"))],
        }
        algorithm = PaintingPPO(
            policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg,
            action_clip=action_clip, action_groups=groups, **self.alg_cfg
        )
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # The shared train entry point requests random episode lengths. Painting
        # has a sampled deadline and timed path starting at reset instead.
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path, infos=None):
        torch.save({
            "painting_interface_version": self.alg.policy.interface_version,
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)
        if self.logger_type in ("neptune", "wandb") and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True, map_location=None):
        checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
        version = checkpoint.get("painting_interface_version")
        if version != self.alg.policy.interface_version:
            raise ValueError(
                f"Painting checkpoint interface {version!r} is incompatible with v{self.alg.policy.interface_version}. "
                "V1/19-command checkpoints must not be resumed for Painting V2."
            )
        self.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
        self.alg.policy.project_distribution_parameters_()
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")
