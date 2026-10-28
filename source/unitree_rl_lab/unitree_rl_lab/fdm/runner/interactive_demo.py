"""Continuous FDM inference with stationary waits and SPACE-triggered local goals."""

import math

import torch

from ..interactive_goals import GoalSamplingCfg, LocalGoalSampler
from ..planning.navigation import GoalProgress, local_goal
from .navigation_observation import NavigationObservation


def run_interactive(env, app, args, policy, planner, rollout, navigation, visualizer, observations, controls):
    adapter = NavigationObservation(env, rollout)
    adapter.reset()
    policy.reset()
    planner.reset(seed=args.seed)
    scene = env.scene.terrain.demo_scene
    sampler = LocalGoalSampler(
        scene, GoalSamplingCfg(radius=args.goal_radius, min_distance=args.goal_min_distance), seed=args.seed
    )
    ids = torch.zeros(1, dtype=torch.long, device=env.device)
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    commands = torch.zeros(1, 3, device=env.device)
    dt = rollout.policy_dt
    warmup_steps = math.ceil(navigation.initialization_s / dt)
    scan_steps = max(1, round(env.cfg.scene.height_scanner.update_period / dt))
    hud_steps = max(1, round(0.2 / dt))
    step = stable = navigation_step = round_number = 0
    initialized = False
    phase, reason = "settling", "Standing for 2 seconds to initialize observations."
    goal = goals = progress = None
    distance = None
    controls.set_waiting(False)
    controls.update(phase, round_number, goal, distance, reason)
    if visualizer is not None:
        visualizer.set_goal(None)
    print(f"[Interactive] Flat spawn {env.scene.terrain.demo_start}. "
          f"Targets are sampled {args.goal_min_distance:g}-{args.goal_radius:g} m from the current robot.", flush=True)
    print("[Interactive] The first goal starts automatically after settling. "
          "Focus the viewport; press SPACE after stopping for the next goal.", flush=True)

    def wait_for_space(message):
        nonlocal phase, reason
        phase, reason = "waiting", message
        commands.zero_()
        adapter.set_commands(commands, observations)
        controls.set_waiting(True)
        controls.update(phase, round_number, goal, distance, reason)
        print(f"[Interactive] Round {round_number}: {message}", flush=True)

    def stop(message):
        commands.zero_()
        adapter.set_commands(commands, observations)
        controls.set_waiting(False)
        controls.update("stopped", round_number, goal, distance, message)
        if visualizer is not None:
            visualizer.update_command(commands)
        print(f"[Interactive] {message}; paused for inspection.", flush=True)
        return message

    def start_round(raw):
        nonlocal goal, goals, progress, phase, reason, navigation_step, round_number, distance
        controls.set_waiting(False)
        commands.zero_()
        # Snapshot the actual current pose, never the previous target or spawn.
        center = raw[0, :2].cpu().numpy().copy()
        goal = sampler.sample(center)
        if visualizer is not None:
            visualizer.set_goal(None if goal is None else goal[:2])
        if goal is None:
            goals = progress = None
            distance = None
            wait_for_space("No valid target found in the requested radius. Press SPACE to retry.")
            return
        round_number += 1
        navigation_step = 0
        phase, reason = "navigating", "Walking to the yellow target. SPACE is ignored while walking."
        goals = torch.tensor([goal[:2]], device=env.device, dtype=raw.dtype)
        # Preserve policy memory, sensor history and the goal RNG across rounds.
        planner.reset(ids)
        progress = GoalProgress(goals, raw, navigation)
        env.scene.terrain.demo_goal = goal[:2]
        env.scene.terrain.demo_goal_height = goal[2]
        distance = math.dist(center, goal[:2])
        controls.update(phase, round_number, goal, distance, reason)
        print(f"[Interactive] Round {round_number}: start XY={tuple(center)}, "
              f"goal XYZ={goal}, distance={distance:.2f} m.", flush=True)
        if visualizer is not None:
            cx, cy = (center + goals[0].cpu().numpy()) / 2
            z = max(float(raw[0, 2]), goal[2])
            env.sim.set_camera_view(eye=(cx + 6, cy - 8, z + 7), target=(cx, cy, z))

    while app.is_running():
        if env.sim.is_stopped():
            return None
        if not env.sim.is_playing():
            controls.set_waiting(False)
            controls.update("paused", round_number, goal, distance, "Resume using the simulation Play button.")
            env.sim.render()
            continue
        controls.set_waiting(phase == "waiting")
        collision, _, feet = adapter.contacts()
        raw = adapter.raw(collision)
        if not torch.isfinite(raw).all() or not torch.isfinite(observations["policy"]).all():
            return stop("Non-finite robot state or observation")
        if bool(collision[0]):
            return stop("Body/hand collision")
        upright = float(adapter.robot.data.projected_gravity_b[0, 2]) < -0.7
        if float(adapter.robot.data.projected_gravity_b[0, 2]) > -0.5:
            return stop("Robot fell")
        if not bool(scene.contains(raw[0, :2].cpu().numpy())):
            return stop("Robot reached the map boundary")

        if phase == "settling" and step >= warmup_steps:
            if not initialized or not bool(feet[0]) or not upright or int(adapter.count[0]) < rollout.history_length:
                return stop("Could not initialize a stable stance and full FDM history")
            start_round(raw)
        elif phase == "waiting" and controls.consume_next():
            speed = float(adapter.robot.data.root_lin_vel_w[0, :2].norm())
            yaw_rate = abs(float(adapter.robot.data.root_ang_vel_w[0, 2]))
            if bool(feet[0]) and upright and speed <= navigation.arrival_speed and yaw_rate <= navigation.arrival_yaw_rate:
                start_round(raw)
            else:
                reason = "Robot is still settling. Press SPACE again once it is standing steadily."
                print(f"[Interactive] {reason}", flush=True)

        if phase == "navigating":
            now = navigation_step * dt
            status = progress.update(
                raw, adapter.robot.data.root_lin_vel_w[:, :2].norm(dim=-1), active, now,
                adapter.robot.data.root_ang_vel_w[:, 2],
            )
            distance = float(status["distance"][0])
            if bool(status["reached"][0]):
                wait_for_space("Goal reached. Holding zero velocity. Press SPACE for the next goal.")
            elif bool(status["exhausted"][0]):
                wait_for_space("No progress after recovery attempts. Press SPACE to choose another goal.")
            elif now >= navigation.timeout_s:
                wait_for_space("Navigation timeout. Press SPACE to choose another goal.")
            else:
                if bool(status["recover"][0]):
                    planner.reset(ids)
                    commands.zero_()
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
        else:
            commands.zero_()

        if visualizer is not None:
            if phase != "navigating" and step % rollout.policy_steps_per_command == 0:
                visualizer.update_fdm_scan()
            if step % scan_steps == 0:
                visualizer.update_policy_scan()
            visualizer.update_command(commands)
        if step % hud_steps == 0:
            if goal is not None:
                distance = math.dist(raw[0, :2].cpu().numpy(), goal[:2])
            controls.update(phase, round_number, goal, distance, reason)
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
            raise RuntimeError("Unexpected automatic reset in the interactive demo.")
        step += 1
        collision, _, feet = adapter.contacts()
        raw = adapter.raw(collision)
        safe = bool(feet[0] & ~collision[0]) and float(adapter.robot.data.projected_gravity_b[0, 2]) < -0.7
        stable = stable + 1 if safe else 0
        initialized |= stable >= rollout.initial_settle_steps
        # Keep rolling time and history during stationary waits as well as walking.
        adapter.tick(ids if initialized else ids[:0], raw, observations, step * dt)
    return None
