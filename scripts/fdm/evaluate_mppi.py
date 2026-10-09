"""Compare two FDMs under identical MPPI settings and matched navigation starts/goals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from _common import require_input_file, sha256_file
from _stage2_common import add_planner_args, create_environment, planner_configs


def _parse_args():
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-29dof-FDM-Rollout")
    parser.add_argument("--checkpoint", required=True, help="Frozen locomotion policy.")
    parser.add_argument("--baseline", required=True, help="Baseline FDM, usually stage-one round 017.")
    parser.add_argument("--fdm-checkpoint", required=True, help="FDM to compare against the baseline.")
    parser.add_argument("--terrain-usd", default=str(DEFAULT_TERRAIN_USD))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90_042)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True, help="New JSON report path.")
    parser.add_argument("--log-interval", type=float, default=10)
    parser.add_argument("--disable-policy-corruption", action="store_true")
    add_planner_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        for name in ("checkpoint", "baseline", "fdm_checkpoint", "terrain_usd"):
            setattr(args, name, require_input_file(getattr(args, name), "--" + name.replace("_", "-")))
        if args.num_envs < 1 or args.log_interval < 0:
            raise ValueError("Invalid environment count or logging interval.")
        if Path(args.output).exists():
            raise ValueError(f"Refusing to overwrite {args.output}.")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _evaluate(args):
    from unitree_rl_lab.fdm.config import RolloutCfg
    from unitree_rl_lab.fdm.data.shard_writer import _atomic_json_dump
    from unitree_rl_lab.fdm.runner.navigation_evaluation import compare_navigation, evaluate_navigation
    from unitree_rl_lab.fdm.training.stage2 import load_initial_model

    mppi_cfg, goal_cfg = planner_configs(args)
    rollout = RolloutCfg(task_name=args.task, policy_checkpoint=args.checkpoint, terrain_usd_path=args.terrain_usd,
                         num_envs=args.num_envs, seed=args.seed, split=args.split,
                         policy_observation_corruption=not args.disable_policy_corruption)
    env, observations, policy, grid = create_environment(args, rollout)
    try:
        env.unwrapped.scene.terrain.activate_split(args.split)
        results = []
        for path in (args.baseline, args.fdm_checkpoint):
            model, _ = load_initial_model(path)
            model = model.to(env.unwrapped.device).eval().requires_grad_(False)
            print(f"[FDM] Navigation evaluation: {path}", flush=True)
            results.append(evaluate_navigation(env, observations, policy, rollout, model, grid,
                                               mppi_cfg=mppi_cfg, goal_cfg=goal_cfg, seed=args.seed,
                                               log_interval_s=args.log_interval))
            del model
        comparison = compare_navigation(*results)
        comparison.update(mppi=asdict(mppi_cfg), goals=asdict(goal_cfg), split=args.split,
                          checkpoints={name: {"path": path, "sha256": sha256_file(path)}
                                       for name, path in (("baseline", args.baseline), ("current", args.fdm_checkpoint))})
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(destination, comparison)
        print(json.dumps({key: value for key, value in comparison.items() if key not in ("baseline", "current")}, indent=2))
        print(f"[FDM] Navigation report saved: {destination}", flush=True)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher
    args = _parse_args()
    launcher = AppLauncher(args)
    try:
        _evaluate(args)
    finally:
        launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
