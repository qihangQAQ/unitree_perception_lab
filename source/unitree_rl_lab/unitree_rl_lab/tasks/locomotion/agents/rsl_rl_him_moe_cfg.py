"""Shared RSL-RL configuration for cross-attention + Old-HIM + MoE policies."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlHimMoeActorCriticCfg(RslRlPpoActorCriticCfg):
    """Common policy fields; task configs only select the exteroceptive input adapter."""

    class_name: str = "HimMoeActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 256, 128]
    activation: str = "elu"

    proprio_group: str = "policy"
    exteroception_group: str = ""
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

    exteroception_shape: tuple[int, int] = (0, 0)
    exteroception_input_mode: str = "flat"
    exteroception_model_dim: int = 32
    exteroception_latent_dim: int = 32
    exteroception_first_conv_channels: int = 16
    exteroception_first_conv_stride: int = 1
    attention_num_heads: int = 4
    attention_batch_size: int = 4096

    num_moe_experts: int = 4
    moe_gate_hidden_dims: list[int] = []


@configclass
class RslRlHimMoeAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO fields used by the independently optimized Old-HIM estimator."""

    class_name: str = "HimMoePPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg = None
    symmetry_cfg = None

    velocity_loss_coef: float = 1.0
    him_swap_loss_coef: float = 1.0
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 10.0
    auxiliary_group: str = "auxiliary"
    auxiliary_dim: int = 96


@configclass
class RslRlHimMoeRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Training defaults shared by all HIM-MoE exteroception modalities."""

    class_name: str = "unitree_rl_lab.rsl_rl_ext.runners.him_moe_runner:HimMoeRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    clip_actions = 18.0
    experiment_name = ""
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlHimMoeActorCriticCfg()
    algorithm = RslRlHimMoeAlgorithmCfg()
