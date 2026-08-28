from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import G1_CFG

from .velocity_perception_env_cfg import RobotEnvCfg as PerceptionRobotEnvCfg
from .velocity_perception_env_cfg import RobotPlayEnvCfg as PerceptionRobotPlayEnvCfg


@configclass
class RobotEnvCfg(PerceptionRobotEnvCfg):
    """Perception environment using the extra G1 model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotPlayEnvCfg(PerceptionRobotPlayEnvCfg):
    """Play configuration using the extra G1 model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
