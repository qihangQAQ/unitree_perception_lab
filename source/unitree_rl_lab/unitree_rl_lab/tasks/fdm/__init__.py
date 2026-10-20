"""Register the G1 FDM rollout environment."""

import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-FDM-Mixed-Pillars",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_pillars_env_cfg:MixedPillarsEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_predict_cfg:UnitreePerceptionPredictRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-G1-29dof-FDM-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rollout_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rollout_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_predict_cfg:"
            "UnitreePerceptionPredictRunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-FDM-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_predict_cfg:UnitreePerceptionPredictRunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-FDM-Benchmark",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:BenchmarkEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_predict_cfg:UnitreePerceptionPredictRunnerCfg"
        ),
    },
)
