"""Collect frozen-policy G1 trajectories into atomic FDM shards."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect G1 FDM rollout trajectories.")
parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--terrain-usd", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--split", choices=("train", "val", "test"), default="train")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--num-episodes", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable-policy-corruption", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.fdm.config import RolloutCfg
from unitree_rl_lab.fdm.data import EpisodeShardWriter
from unitree_rl_lab.fdm.runner import FDMRolloutCollector, FrozenRecurrentPolicy
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from _common import configure_safe_spawn, dataset_metadata


def main() -> None:
    rollout_cfg = RolloutCfg(
        task_name=args.task,
        policy_checkpoint=args.checkpoint,
        terrain_usd_path=args.terrain_usd,
        dataset_root=args.dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        split=args.split,
        policy_observation_corruption=not args.disable_policy_corruption,
    )
    rollout_cfg.validate()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.scene.terrain.usd_path = args.terrain_usd
    env_cfg.scene.terrain.active_split = args.split
    env_cfg.observations.policy.enable_corruption = rollout_cfg.policy_observation_corruption
    configure_safe_spawn(env_cfg, rollout_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    observations, _ = env.reset(seed=args.seed)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    policy = FrozenRecurrentPolicy(
        observations,
        env.unwrapped.action_manager.total_action_dim,
        agent_cfg,
        args.checkpoint,
        env.unwrapped.device,
    )
    metadata = dataset_metadata(env, rollout_cfg)
    with EpisodeShardWriter(
        args.dataset,
        args.split,
        metadata,
        max_frames_per_shard=rollout_cfg.max_frames_per_shard,
    ) as writer:
        collector = FDMRolloutCollector(env, observations, policy, writer, rollout_cfg)
        result = collector.collect(args.num_episodes)
    print(f"[FDM] Collected {result['episodes']} {args.split} episodes into {args.dataset}.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
