"""Local FDM navigation: the runner owns terminal detection before any reset."""

from isaaclab.utils import configclass

from .rollout_env_cfg import RobotEnvCfg


@configclass
class NavigationEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # No ManagerBasedRLEnv automatic reset at 80 seconds or after contact.
        # NavigationRunner detects contacts/falls at every policy step and
        # freezes the outcome before the next explicit batch reset.
        self.terminations.time_out = None
        self.terminations.navigation_collision = None
        self.episode_length_s = 1e6
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
