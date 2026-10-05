"""Exercise real Isaac Lab managers, asynchronous reset and a short PPO rollout.

Run with the same Isaac Sim environment as scripts/rsl_rl/train.py.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import faulthandler
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.painting.painting_env_cfg import PaintingEnvCfg
from unitree_rl_lab.tasks.painting.agents.rsl_rl_ppo_cfg import PaintingPPORunnerCfg
from unitree_rl_lab.rsl_rl_ext.runners.painting_runner import PaintingRunner
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

try:
    faulthandler.dump_traceback_later(30, repeat=True)
    cfg = PaintingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    print("Creating painting environment", flush=True)
    raw = gym.make("Unitree-G1-29dof-Painting", cfg=cfg)
    print("Wrapping and resetting painting environment", flush=True)
    env = RslRlVecEnvWrapper(raw, clip_actions=3.0)
    print("Painting environment reset complete", flush=True)
    obs = env.get_observations()
    assert obs["policy"].shape == (args.num_envs, 5, 93)
    assert obs["painting"].shape == (args.num_envs, 27)
    assert obs["critic"].shape == (args.num_envs, 128)
    assert obs["velocity_targets"].shape == (args.num_envs, 6)
    command = raw.unwrapped.command_manager.get_term("painting_command")
    original_paths = command.points.clone()
    for _ in range(args.steps):
        obs, reward, done, info = env.step(torch.zeros(args.num_envs, 29, device=args.device))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
    command.refresh()
    assert not command.end_effector_active.any()
    assert torch.allclose(command.base_velocity_command[:, 0], torch.zeros(args.num_envs, device=args.device))
    print("TCP", command.tcp_position, "target", command.target_world[:, 0], "axis", command.spray_axis)
    # Changing the shared episode counter must not change this command's clock.
    before = command.elapsed.clone()
    raw.unwrapped.episode_length_buf += 100
    command.refresh()
    torch.testing.assert_close(before, command.elapsed)
    if args.num_envs > 1:
        path_other = command.points[1].clone()
        raw.unwrapped._reset_idx(torch.tensor([0], device=args.device))
        torch.testing.assert_close(command.points[1], path_other)
    runner_cfg = PaintingPPORunnerCfg()
    runner_cfg.num_steps_per_env = 4
    runner_cfg.algorithm.num_learning_epochs = 1
    runner_cfg.algorithm.num_mini_batches = 1
    runner = PaintingRunner(env, runner_cfg.to_dict(), log_dir="/tmp/painting-smoke", device=args.device)
    runner.learn(1, init_at_random_ep_len=True)
    runner.alg.policy.export_onnx("/tmp/painting-smoke/exported")
    export_deploy_cfg(raw.unwrapped, "/tmp/painting-smoke", observation_group_names=["policy", "painting"])
    assert set(raw.unwrapped.termination_manager.active_terms) == {"time_out", "bad_posture", "success"}
    print("PAINTING SMOKE PASSED: managers, observations, clock, reset, PPO and ONNX")
    env.close()
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
