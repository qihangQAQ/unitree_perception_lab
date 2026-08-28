import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# 速度命令控制，训练和播放时均使用源项目的额外 G1 模型
gym.register(
    id="Unitree-G1-29dof-Velocity-Extra",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_extra_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_extra_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# 速度命令感知控制（带高程图和 LSTM）
gym.register(
    id="Unitree-G1-29dof-Velocity-perception",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg"
        ),
    },
)


# 高程图感知独立消融：三个实验分别直接继承 Perception 基线
gym.register(
    id="Unitree-G1-29dof-Velocity-perception-Exp1",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_ablation_env_cfg:Exp1RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_ablation_env_cfg:Exp1RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-perception-Exp2",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_ablation_env_cfg:Exp2RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_ablation_env_cfg:Exp2RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-perception-Exp3",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_ablation_env_cfg:Exp3RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_ablation_env_cfg:Exp3RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg"
        ),
    },
)


# 高程图感知升级控制，使用源项目的额外 G1 模型
gym.register(
    id="Unitree-G1-29dof-Velocity-perception-upgrade",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_upgrade_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_upgrade_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg"
        ),
    },
)


# 高程图感知 Pro：历史状态估计、交叉注意力和 Actor MoE。
gym.register(
    id="Unitree-G1-29dof-Velocity-perception-pro",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_pro_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_pro_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_pro_cfg:UnitreePerceptionProRunnerCfg"
        ),
    },
)


# 深度相机 Pro：Old-HIM、深度交叉注意力和 Actor MoE。
gym.register(
    id="Unitree-G1-29dof-Velocity-depth",
    entry_point="unitree_rl_lab.envs:TerrainLoggingManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_depth_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_depth_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_depth_cfg:UnitreeDepthRunnerCfg"
        ),
    },
)
