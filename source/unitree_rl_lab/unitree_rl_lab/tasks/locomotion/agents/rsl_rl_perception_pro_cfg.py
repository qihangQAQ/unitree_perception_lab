"""RSL-RL configuration for Unitree G1 perception-pro."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPerceptionProActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "PerceptionProActorCritic"

    proprio_group: str = "policy"
    height_map_group: str = "height_map"
    auxiliary_group: str = "auxiliary"

    history_length: int = 5
    proprio_dim: int = 96
    auxiliary_dim: int = 96
    estimator_hidden_dims: list[int] = [256, 64, 16]
    target_hidden_dims: list[int] = [256, 64]
    num_prototypes: int = 32
    him_temperature: float = 3.0
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 3

    # GridPatternCfg(ordering="xy") flattens the 17 x-points inside the 11 y-rows.
    height_map_shape: tuple[int, int] = (11, 17)
    height_model_dim: int = 32
    height_latent_dim: int = 32
    attention_num_heads: int = 4
    attention_batch_size: int = 4096

    num_moe_experts: int = 4
    # Match the reference dense MoE: a direct linear softmax gate and no balancing loss.
    moe_gate_hidden_dims: list[int] = []


@configclass
class RslRlPerceptionProAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PerceptionProPPO"
    velocity_loss_coef: float = 1.0
    him_swap_loss_coef: float = 1.0
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 10.0
    auxiliary_group: str = "auxiliary"
    auxiliary_dim: int = 96


@configclass
class UnitreePerceptionProRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = (
        "unitree_rl_lab.tasks.locomotion.agents.perception_pro.runner:PerceptionProRunner"
    )
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    clip_actions = 18.0
    experiment_name = "Unitree-perception-pro"
    empirical_normalization = False
    obs_groups = {"policy": ["policy", "height_map"], "critic": ["critic"]}

    policy = RslRlPerceptionProActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPerceptionProAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
