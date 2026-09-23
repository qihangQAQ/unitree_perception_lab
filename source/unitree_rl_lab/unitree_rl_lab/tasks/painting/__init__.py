"""G1 wall painting task registration."""

import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Painting",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.painting_env_cfg:PaintingEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.painting_env_cfg:PaintingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PaintingPPORunnerCfg",
    },
)
