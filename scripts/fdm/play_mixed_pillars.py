"""Visual-only MPPI + FDM demo on the atlas (legacy script name retained)."""

from __future__ import annotations

import argparse
import math

from _common import require_input_file


TASK = "Unitree-G1-29dof-FDM-Atlas"


def parse_args(*, interactive=False):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=(
        "Interactive FDM navigation: reach a random nearby goal, then press SPACE for the next." if interactive else __doc__
    ))
    parser.add_argument("--checkpoint", default=(
        "logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt"
    ))
    parser.add_argument("--fdm-checkpoint", default="logs/fdm_g1/stage2_mppi_v1/fdm_round_004.pt")
    parser.add_argument("--seed", type=int, default=0, help="Terrain and planner seed; matches the terrain preview.")
    parser.add_argument("--num-envs", type=int, choices=(1,), default=1)
    parser.add_argument("--zone", choices=[f"{row}{col}" for row in "ABC" for col in range(1, 5)],
                        help=("Initial flat-spawn zone; defaults to B1. Subsequent goals may cross zones." if interactive
                              else "Atlas endpoint preset. Default B1, or the zone containing an explicit goal/start."))
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"),
                        help="World start; yaw in radians. Needs a level standing patch; ground Z is sampled.")
    if interactive:
        parser.add_argument("--goal-radius", type=float, default=5.0, help="Maximum XY sampling distance in metres.")
        parser.add_argument("--goal-min-distance", type=float, default=2.0, help="Minimum XY sampling distance in metres.")
        parser.set_defaults(goal=None)
    else:
        parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"),
                            help="World goal on flat ground, a platform or a gentle ramp. Defaults to the zone preset.")
    parser.add_argument("--mppi-candidates", type=int, default=256)
    parser.add_argument("--mppi-iterations", type=int, default=1)
    parser.add_argument("--inference-chunk", type=int, default=2048)
    parser.add_argument("--max-forward-speed", type=float, default=0.8)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="Navigation timeout in simulated seconds.")
    parser.add_argument("--stall-time", type=float, default=15.0, help="Seconds without progress before recovery.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        args.checkpoint = require_input_file(args.checkpoint, "--checkpoint")
        args.fdm_checkpoint = require_input_file(args.fdm_checkpoint, "--fdm-checkpoint")
        if not 0 <= args.seed < 2**31:
            raise ValueError("--seed must be a nonnegative 31-bit integer.")
        for coordinates in (args.start, args.goal):
            if coordinates is not None and not all(math.isfinite(value) for value in coordinates):
                raise ValueError("Start and goal coordinates must be finite.")
        if args.mppi_candidates < 8 or min(args.mppi_iterations, args.inference_chunk) < 1:
            raise ValueError("Require at least 8 MPPI candidates and positive iteration/chunk counts.")
        for name in ("max_forward_speed", "max_yaw_rate", "timeout", "stall_time"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
        if args.max_forward_speed > 1.2 or args.max_yaw_rate > 1.0:
            raise ValueError("Model command bounds require vx <= 1.2 m/s and |wz| <= 1 rad/s.")
        if interactive:
            if args.headless:
                raise ValueError("Interactive goals require a GUI window; omit --headless.")
            if not (math.isfinite(args.goal_radius) and math.isfinite(args.goal_min_distance)
                    and 0.4 < args.goal_min_distance < args.goal_radius):
                raise ValueError("Require 0.4 < --goal-min-distance < --goal-radius, both finite.")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run_demo(env, app, args, policy, planner, rollout, navigation, visualizer, observations):
    """An inference-only loop: bounded histories, live display, no evaluation recorder."""
    import torch

    from unitree_rl_lab.fdm.planning.navigation import GoalProgress, local_goal
    from unitree_rl_lab.fdm.runner.navigation_observation import NavigationObservation

    adapter = NavigationObservation(env, rollout)
    adapter.reset()
    policy.reset()
    planner.reset(seed=args.seed)
    ids = torch.zeros(1, dtype=torch.long, device=env.device)
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    commands = torch.zeros(1, 3, device=env.device)
    goals = torch.tensor([env.scene.terrain.demo_goal], dtype=torch.float32, device=env.device)
    dt = rollout.policy_dt
    warmup_steps = math.ceil(navigation.initialization_s / dt)
    scan_steps = max(1, round(env.cfg.scene.height_scanner.update_period / dt))
    step, navigation_step, stable = 0, 0, 0
    initialized = False
    phase, progress = "settling", None
    print(f"[Demo] FDM atlas 48 x 36 m; zone {env.scene.terrain.demo_scene.selected_zone}; "
          f"start ground Z={env.scene.terrain.demo_start_height:.3f} m, "
          f"goal ground Z={env.scene.terrain.demo_goal_height:.3f} m.", flush=True)
    print(f"[Demo] Start {env.scene.terrain.demo_start}; goal {env.scene.terrain.demo_goal}.", flush=True)
    print("[Demo] Standing for 2 s to initialize FDM history. Blue: FDM scan; red: policy scan; "
          "green: executed command; yellow: goal. Camera is free.", flush=True)

    if visualizer is not None:
        visualizer.update_fdm_scan()
        visualizer.update_policy_scan()
        visualizer.update_command(commands)

    def stop(reason):
        commands.zero_()
        adapter.set_commands(commands, observations)
        if visualizer is not None:
            visualizer.update_command(commands)
        print(f"[Demo] {reason}; paused for inspection.", flush=True)
        return reason

    while app.is_running():
        if env.sim.is_stopped():
            return None
        if not env.sim.is_playing():
            env.sim.render()
            continue
        collision, _, feet = adapter.contacts()
        raw = adapter.raw(collision)
        if not torch.isfinite(raw).all() or not torch.isfinite(observations["policy"]).all():
            return stop("Non-finite robot state or observation")
        if bool(collision[0]):
            return stop("Body/hand collision")
        if float(adapter.robot.data.projected_gravity_b[0, 2]) > -0.5:
            return stop("Robot fell")
        if not bool(env.scene.terrain.demo_scene.contains(raw[0, :2].cpu().numpy())):
            return stop("Robot reached the floor boundary")

        if phase == "settling" and step >= warmup_steps:
            upright = float(adapter.robot.data.projected_gravity_b[0, 2]) < -0.7
            if not initialized or not bool(feet[0]) or not upright or int(adapter.count[0]) < rollout.history_length:
                return stop("Could not initialize a stable stance and full FDM history")
            progress = GoalProgress(goals, raw, navigation)
            phase = "navigating"
            print("[Demo] Navigating with MPPI + FDM.", flush=True)

        if phase == "navigating":
            now = navigation_step * dt
            status = progress.update(
                raw, adapter.robot.data.root_lin_vel_w[:, :2].norm(dim=-1), active, now,
                adapter.robot.data.root_ang_vel_w[:, 2],
            )
            if bool(status["reached"][0]):
                phase = "reached"
                commands.zero_()
                adapter.set_commands(commands, observations)
                print("[Demo] Goal reached. Holding zero velocity; close the window to exit.", flush=True)
                if args.headless:
                    return None
            elif bool(status["exhausted"][0]):
                return stop("No progress after recovery attempts")
            elif now >= navigation.timeout_s:
                return stop("Navigation timeout")
            else:
                if bool(status["recover"][0]):
                    planner.reset(ids)
                    commands.zero_()
                    print("[Demo] No recent progress; restarting the MPPI candidate sequence.", flush=True)
                if bool(status["near"][0]):
                    commands.zero_()
                elif navigation_step % rollout.policy_steps_per_command == 0:
                    state, proprio, height, invalid, transform = adapter.inputs(ids, observations, step * dt)
                    planner.previous[ids] = commands
                    plans = planner.plan(
                        ids, state, proprio, height, invalid, local_goal(goals, raw),
                        map_transform=transform, align_mask=progress.align,
                    )
                    commands.copy_(plans[:, 0])
                    if visualizer is not None:
                        visualizer.update_fdm_scan(invalid)
            navigation_step += 1

        if visualizer is not None:
            if phase != "navigating" and step % rollout.policy_steps_per_command == 0:
                visualizer.update_fdm_scan()
            if step % scan_steps == 0:
                visualizer.update_policy_scan()
            visualizer.update_command(commands)

        if (not torch.isfinite(commands).all() or (commands[:, 0] < 0).any()
                or (commands[:, 0] > navigation.forward_speed + 1e-6).any()
                or (commands[:, 1] != 0).any()
                or (commands[:, 2].abs() > navigation.yaw_rate + 1e-6).any()):
            return stop("Planner returned an invalid velocity command")
        adapter.set_commands(commands, observations)
        actions = policy.act(observations)
        if not torch.isfinite(actions).all():
            return stop("Policy returned non-finite actions")
        observations, _, terminated, truncated, _ = env.step(actions)
        if bool((terminated | truncated).any()):
            raise RuntimeError("Unexpected automatic reset in the atlas demo.")
        step += 1
        collision, _, feet = adapter.contacts()
        raw = adapter.raw(collision)
        safe = bool(feet[0] & ~collision[0]) and float(adapter.robot.data.projected_gravity_b[0, 2]) < -0.7
        stable = stable + 1 if safe else 0
        initialized |= stable >= rollout.initial_settle_steps
        if phase != "reached":
            adapter.tick(ids if initialized else ids[:0], raw, observations, step * dt)
    return None


def main(args, app, *, interactive=False):
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from unitree_rl_lab.fdm.config import CommandSamplingCfg, RolloutCfg
    from unitree_rl_lab.fdm.planning.mppi import MPPICfg, MPPIPlanner
    from unitree_rl_lab.fdm.planning.navigation import NavigationCfg
    from unitree_rl_lab.fdm.runner.frozen_policy import FrozenRecurrentPolicy
    from unitree_rl_lab.fdm.training.stage2 import load_initial_model
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    command_cfg = CommandSamplingCfg(
        max_forward_speed=args.max_forward_speed, min_lateral_speed=0.0,
        max_lateral_speed=0.0, max_yaw_rate=args.max_yaw_rate,
    )
    rollout = RolloutCfg(
        task_name=TASK, num_envs=1, seed=args.seed, policy_checkpoint=args.checkpoint,
        terrain_usd_path="", policy_observation_corruption=False, command=command_cfg,
    )
    rollout.validate()
    navigation = NavigationCfg(
        timeout_s=args.timeout, stall_s=args.stall_time, forward_speed=args.max_forward_speed,
        yaw_rate=args.max_yaw_rate, log_interval_s=0.0,
    )
    navigation.validate()
    model, model_cfg = load_initial_model(args.fdm_checkpoint)
    if (model_cfg.history_length != rollout.history_length or model_cfg.horizon != rollout.prediction_horizon
            or abs(model_cfg.command_timestep - rollout.command_timestep) > 1e-8):
        raise ValueError("FDM checkpoint history/horizon/timing does not match the frozen policy interface.")
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.scene.terrain.terrain_generator.seed = args.seed
    cfg.scene.terrain.demo_zone = args.zone
    cfg.scene.terrain.demo_spawn_only = interactive
    cfg.scene.terrain.demo_start = None if args.start is None else tuple(args.start)
    cfg.scene.terrain.demo_goal = None if args.goal is None else tuple(args.goal)
    wrapped, visualizer, controls = None, None, None
    try:
        wrapped = gym.make(TASK, cfg=cfg)
        env = wrapped.unwrapped
        if interactive and abs(env.scene.terrain.demo_start_height) > 0.015:
            raise ValueError("Interactive mode must initially spawn on the flat z=0 floor; choose another --start.")
        if env.termination_manager.active_terms:
            raise ValueError("The demo must not automatically reset after a collision or timeout.")
        if abs(env.step_dt - rollout.policy_dt) > 1e-8 or abs(cfg.sim.dt - rollout.physics_dt) > 1e-8:
            raise ValueError("Simulation timing differs from FDM training.")
        observations, _ = env.reset(seed=args.seed)
        if not args.headless:
            # Frame the selected experiment once; leave manual camera control free.
            sx, sy, _ = env.scene.terrain.demo_start
            gx, gy = (sx, sy) if interactive else env.scene.terrain.demo_goal
            cx, cy = (sx + gx) / 2, (sy + gy) / 2
            span = max(8.0, math.hypot(gx - sx, gy - sy))
            ground_z = env.scene.terrain.demo_start_height
            if not interactive:
                ground_z = max(ground_z, env.scene.terrain.demo_goal_height)
            env.sim.set_camera_view(
                eye=(cx + 0.7 * span, cy - 0.9 * span, ground_z + 0.85 * span),
                target=(cx, cy, ground_z + 0.4),
            )
        policy = FrozenRecurrentPolicy(
            observations, env.action_manager.total_action_dim, load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"),
            args.checkpoint, env.device,
        )
        planner = MPPIPlanner(
            model.to(env.device), 1,
            cfg=MPPICfg(candidates=args.mppi_candidates, iterations=args.mppi_iterations,
                        inference_chunk=args.inference_chunk, noise_std=(0.45, 0.0, 0.65)),
            command_cfg=command_cfg, seed=args.seed, forward_only=True, turn_candidates=True,
            record_predictions=False, track_statistics=False,
        )
        if not args.headless:
            from unitree_rl_lab.fdm.runner.pillar_visualizer import PillarDemoVisualizer
            visualizer = PillarDemoVisualizer(
                env, env.scene.terrain.demo_goal, navigation.arrival_radius,
                goal_height_at=env.scene.terrain.demo_scene.height_at,
            )
        if interactive:
            from unitree_rl_lab.fdm.runner.interactive_controls import InteractiveControls
            from unitree_rl_lab.fdm.runner.interactive_demo import run_interactive
            controls = InteractiveControls()
        with torch.inference_mode():
            if interactive:
                stopped = run_interactive(
                    env, app, args, policy, planner, rollout, navigation, visualizer, observations, controls
                )
            else:
                stopped = run_demo(env, app, args, policy, planner, rollout, navigation, visualizer, observations)
        if stopped and app.is_running() and not args.headless:
            env.sim.pause()
            while app.is_running():
                env.sim.render()
    finally:
        if controls is not None:
            controls.close()
        if visualizer is not None:
            visualizer.close()
        if wrapped is not None:
            wrapped.close()


if __name__ == "__main__":
    args = parse_args()
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(args).app
    try:
        main(args, simulation_app)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
