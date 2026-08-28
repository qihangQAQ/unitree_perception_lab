from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# ==============================================================================
# 1) Perception-specific configuration classes
# ==============================================================================

@configclass
class RslRlPerceptionActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    Recurrent Actor-Critic configuration (aligned with LeggedLab G1Rough).

    Uses standard ActorCriticRecurrent from rsl_rl with LSTM.
    No terrain encoder — height scan goes directly to LSTM.
    """
    # Network class name (standard rsl_rl recurrent)
    class_name: str = "ActorCriticRecurrent"

    # LSTM parameters
    lstm_hidden_size: int = 256


@configclass
class RslRlPerceptionAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    Perception algorithm configuration.

    Currently uses the same algorithm as PPO, but kept separate for future extensions.
    """
    # Algorithm class name (uses base PPO algorithm)
    # class_name: str = "PPO"


# ==============================================================================
# 2) Runner configuration for training
# ==============================================================================

@configclass
class UnitreePerceptionRunnerCfg(RslRlOnPolicyRunnerCfg):
    # # ============== WandB Configuration ===========
    # logger = "wandb"
    # wandb_project = "Unitree_g1_Velocity_Perception"
    # run_name = "Perception_Run"
    # experiment_name = "unitree_perception"
    # # ==============================================

    # Runner class name (uses base OnPolicyRunner)
    class_name: str = "OnPolicyRunner"

    # Basic training parameters (consistent with original PPO)
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    # Policy configuration with LSTM (aligned with LeggedLab G1Rough)
    policy = RslRlPerceptionActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        noise_std_type="scalar",
        lstm_hidden_size=256,
    )

    # Algorithm configuration (same as PPO)
    algorithm = RslRlPpoAlgorithmCfg(
        # Original PPO parameters
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
    )
