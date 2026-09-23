"""Small actor with explicit base/TCP velocity supervision."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PaintingActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "PaintingActorCritic"
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 256, 128]
    activation: str = "elu"
    init_noise_std: float = 0.5
    noise_std_type: str = "log"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    history_length: int = 5
    proprio_dim: int = 93
    command_dim: int = 19
    proprio_group: str = "policy"
    command_group: str = "painting"
    velocity_group: str = "velocity_targets"
    estimator_hidden_dims: list[int] = [128, 64]


@configclass
class PaintingAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PaintingPPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    rnd_cfg = None
    symmetry_cfg = None
    estimator_learning_rate: float = 1e-3
    estimator_max_grad_norm: float = 1.0
    velocity_loss_coef: float = 1.0


@configclass
class PaintingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "unitree_rl_lab.rsl_rl_ext.runners.painting_runner:PaintingRunner"
    seed: int = 42
    num_steps_per_env: int = 32
    max_iterations: int = 30000
    save_interval: int = 500
    experiment_name: str = "G1-Painting"
    clip_actions: float = 3.0
    empirical_normalization: bool = False
    obs_groups: dict = {"policy": ["policy", "painting"], "critic": ["critic"]}
    policy: PaintingActorCriticCfg = PaintingActorCriticCfg()
    algorithm: PaintingAlgorithmCfg = PaintingAlgorithmCfg()
