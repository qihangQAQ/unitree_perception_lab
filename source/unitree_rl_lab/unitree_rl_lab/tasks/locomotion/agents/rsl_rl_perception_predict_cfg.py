"""RSL-RL configuration for perception with imagined foothold guidance."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_perception_cfg import UnitreePerceptionRunnerCfg


@configclass
class RslRlFootholdAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Standard PPO plus an independently optimized foothold imagination model."""

    class_name: str = "FootholdPPO"
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

    foothold_learning_rate: float = 1.0e-3
    foothold_hidden_dims: tuple[int, ...] = (512, 256, 128)
    foothold_min_sigma: float = 0.02
    foothold_max_sigma: float = 0.50
    foothold_replay_buffer_size: int = 50_000
    foothold_pending_steps: int = 16
    foothold_batch_size: int = 2_048
    foothold_updates_per_iteration: int = 4


@configclass
class UnitreePerceptionPredictRunnerCfg(UnitreePerceptionRunnerCfg):
    """Keep the perception LSTM policy and select the foothold training stack."""

    class_name: str = "unitree_rl_lab.rsl_rl_ext.runners.foothold_runner:FootholdRunner"
    # experiment_name = "Unitree-Velocity_perception-predict"
    algorithm = RslRlFootholdAlgorithmCfg()
