"""Stage-two CLI/simulator helpers; safe to import in spawned loader workers."""

from __future__ import annotations


def add_planner_args(parser):
    parser.add_argument("--mppi-candidates", type=int, default=128)
    parser.add_argument("--mppi-iterations", type=int, default=1)
    parser.add_argument("--inference-chunk", type=int, default=2048)
    parser.add_argument("--mppi-temperature", type=float, default=1.0)
    parser.add_argument("--risk-threshold", type=float, default=.55)
    parser.add_argument("--goal-min-distance", type=float, default=1.5)
    parser.add_argument("--goal-max-distance", type=float, default=3.0)
    parser.add_argument("--goal-timeout", type=float, default=20.0)
    parser.add_argument("--goal-stall-time", type=float, default=5.0)


def planner_configs(args):
    from unitree_rl_lab.fdm.planning.mppi import MPPICfg
    from unitree_rl_lab.fdm.planning.goals import GoalCfg

    mppi = MPPICfg(candidates=args.mppi_candidates, iterations=args.mppi_iterations,
                   inference_chunk=args.inference_chunk, temperature=args.mppi_temperature,
                   high_risk_threshold=args.risk_threshold)
    goal = GoalCfg(minimum_distance=args.goal_min_distance, maximum_distance=args.goal_max_distance,
                   timeout_s=args.goal_timeout, stall_s=args.goal_stall_time)
    mppi.validate()
    goal.validate()
    return mppi, goal


def create_environment(args, rollout_cfg):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.fdm.planning.goals import NavigationGrid
    from unitree_rl_lab.fdm.runner import FrozenRecurrentPolicy
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from _common import configure_safe_spawn

    print("[FDM] Creating stage-two rollout environment...", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.scene.terrain.usd_path = args.terrain_usd
    cfg.scene.terrain.active_split = "val"
    cfg.observations.policy.enable_corruption = rollout_cfg.policy_observation_corruption
    configure_safe_spawn(cfg, rollout_cfg)
    cfg.events.reset_base.func.cfg.retain_navigation_grid = True
    cfg.events.reset_base.func.cfg.navigation_clearance = planner_configs(args)[1].terrain_clearance
    env = gym.make(args.task, cfg=cfg)
    try:
        observations, _ = env.reset(seed=args.seed)
        policy = FrozenRecurrentPolicy(observations, env.unwrapped.action_manager.total_action_dim,
                                       load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                       args.checkpoint, env.unwrapped.device)
        grid = NavigationGrid(env.unwrapped.scene.terrain)
        return env, observations, policy, grid
    except BaseException:
        env.close()
        raise
