"""Depth-pro specialization of the separately optimized Old-HIM PPO."""

from unitree_rl_lab.tasks.locomotion.agents.perception_pro.ppo import PerceptionProPPO


class DepthProPPO(PerceptionProPPO):
    """PPO plus velocity regression and Old-HIM swapped prototype prediction."""

