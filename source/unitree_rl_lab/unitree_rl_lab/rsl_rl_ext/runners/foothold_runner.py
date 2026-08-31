"""RSL-RL runner that wires foothold predictions into the training environment."""

from __future__ import annotations

import warnings

import torch
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from ..algorithms.foothold_ppo import FootholdPPO


class FootholdRunner(OnPolicyRunner):
    """Construct and checkpoint the perception policy plus its training-only model."""

    def _construct_algorithm(self, obs) -> FootholdPPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "empirical_normalization is deprecated; use policy normalization fields instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_class_name = self.policy_cfg.pop("class_name")
        policy_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
        }
        if policy_class_name not in policy_classes:
            raise ValueError(f"Unsupported foothold policy class: {policy_class_name!r}.")
        policy = policy_classes[policy_class_name](
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        if algorithm_class_name != "FootholdPPO":
            raise ValueError(f"Unsupported foothold algorithm class: {algorithm_class_name!r}.")
        algorithm = FootholdPPO(
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

        prediction_sink = getattr(self.env.unwrapped, "set_imagined_footholds", None)
        if prediction_sink is None:
            raise AttributeError("FootholdRunner requires an environment with set_imagined_footholds().")
        algorithm.set_foothold_prediction_sink(prediction_sink)
        return algorithm

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "foothold_imagination_state_dict": self.alg.foothold_imagination.state_dict(),
            "foothold_optimizer_state_dict": self.alg.foothold_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        foothold_state = loaded_dict.get("foothold_imagination_state_dict")
        if foothold_state is not None:
            self.alg.foothold_imagination.load_state_dict(foothold_state)
        else:
            warnings.warn(
                "Checkpoint has no foothold model; it will be trained from scratch.",
                stacklevel=2,
            )
        if self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            foothold_optimizer_state = loaded_dict.get("foothold_optimizer_state_dict")
            if foothold_optimizer_state is not None:
                self.alg.foothold_optimizer.load_state_dict(foothold_optimizer_state)
            if self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def train_mode(self):
        super().train_mode()
        self.alg.foothold_imagination.train()

    def eval_mode(self):
        super().eval_mode()
        self.alg.foothold_imagination.eval()
