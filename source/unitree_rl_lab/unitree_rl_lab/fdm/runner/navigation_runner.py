"""Frozen FDM/MPPI navigation to fixed goals, independent of dataset collection."""

from collections import Counter
import math
import time

import torch

from ..planning.navigation import GoalProgress, NavigationCfg, local_goal
from ..utils.memory import memory_snapshot
from .navigation_observation import NavigationObservation


class NavigationRunner:
    def __init__(self, env, policy, planner, rollout_cfg, cfg=None, *, record=None, visualize=None, running=None,
                 region_contains=None):
        self.env = env.unwrapped
        self.policy, self.planner = policy, planner
        self.rollout, self.cfg = rollout_cfg, cfg or NavigationCfg()
        self.cfg.validate()
        self.rollout.validate()
        if self.env.termination_manager.active_terms:
            raise ValueError("Navigation needs a task without automatic resets; use FDM-Navigation.")
        if not planner.forward_only:
            raise ValueError("G1 navigation requires the forward-only planner.")
        if abs(self.env.step_dt - rollout_cfg.policy_dt) > 1e-8:
            raise ValueError("Navigation policy timestep differs from the FDM training timestep.")
        self.adapter = NavigationObservation(self.env, rollout_cfg)
        self.device = self.adapter.device
        self.n = self.env.num_envs
        self.ids = torch.arange(self.n, device=self.device)
        self.record = record or (lambda record: None)
        self.visualize = visualize
        self.running = running or (lambda: True)
        self.region_contains = region_contains
        if self.rollout.split == 'benchmark' and region_contains is None:
            raise ValueError("Generated benchmarks require an explicit arena boundary.")

    def _step(self, obs, commands):
        if not torch.isfinite(commands).all() or (commands[:, 0] < 0).any() or (commands[:, 1] != 0).any():
            raise RuntimeError("Navigation attempted a non-finite, backward or lateral command.")
        if (commands[:, 0] > self.cfg.forward_speed + 1e-6).any() \
                or (commands[:, 2].abs() > self.cfg.yaw_rate + 1e-6).any():
            raise RuntimeError("Navigation command exceeds configured bounds.")
        self.adapter.set_commands(commands, obs)
        obs, _, terminated, truncated, _ = self.env.step(self.policy.act(obs))
        if torch.any(terminated | truncated):
            raise RuntimeError("Unexpected automatic environment reset during navigation.")
        return obs

    @torch.inference_mode()
    def run(self, *, seed, goal_factory, start_setter=None, trial_ids=None):
        """Run one vectorized batch. goal_factory is used ONLY during setup.

        It returns (world goals [N,2], valid [N]); map connectivity may be used
        there for scenario generation, but is never supplied to the planner.
        """
        trial_ids = list(range(self.n)) if trial_ids is None else list(trial_ids)
        if len(trial_ids) != self.n:
            raise ValueError("One trial id per environment is required.")
        torch.manual_seed(seed)
        resetter = self.env.event_manager.get_term_cfg('reset_base').func
        if hasattr(resetter, '_generator'):
            resetter._generator.manual_seed(seed)
        obs, _ = self.env.reset(seed=seed)
        if start_setter is not None:
            start_setter(self.env)
            self.env.scene.write_data_to_sim()
            self.env.sim.forward()
            self.env.scene.update(0.)
            obs = self.env.observation_manager.compute()
        self.policy.reset()
        self.adapter.reset()
        self.planner.reset(seed=seed)
        self.planner.reset_statistics()
        if self.planner.generators is not None:
            self.planner.seed_streams([seed + 100003 * (i + 1) for i in trial_ids])
        commands = torch.zeros(self.n, 3, device=self.device)
        stable = torch.zeros(self.n, dtype=torch.long, device=self.device)
        initialized = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        warm_failure = torch.zeros_like(initialized)
        dt = self.rollout.policy_dt
        # Fixed, model-independent settling duration gives matched starts.
        for step in range(math.ceil(self.cfg.initialization_s / dt)):
            if not self.running():
                raise InterruptedError("Simulation closed during navigation initialization.")
            obs = self._step(obs, commands)
            collision, _, feet = self.adapter.contacts()
            upright = self.adapter.robot.data.projected_gravity_b[:, 2] < -.7
            safe = feet & upright & ~collision
            stable = torch.where(safe, stable + 1, 0)
            warm_failure |= initialized & collision
            initialized |= stable >= self.rollout.initial_settle_steps
            raw = self.adapter.raw(collision)
            self.adapter.tick(self.ids[initialized & ~warm_failure], raw, obs, (step + 1) * dt)
        raw = self.adapter.raw()
        prepared = raw.clone()
        goals, goal_valid = goal_factory(prepared)
        goals = torch.as_tensor(goals, device=self.device, dtype=torch.float32)
        goal_valid = torch.as_tensor(goal_valid, device=self.device, dtype=torch.bool)
        if goals.shape != (self.n, 2) or goal_valid.shape != (self.n,):
            raise ValueError("Goal factory must return [N,2] goals and [N] validity.")
        init_valid = (initialized & ~warm_failure & safe & (self.adapter.count >= self.rollout.history_length)
                      & torch.isfinite(raw).all(dim=1))
        goal_valid &= torch.isfinite(goals).all(dim=1)
        progress = GoalProgress(goals, raw, self.cfg)
        active = init_valid & goal_valid
        outcomes = ['running' if bool(active[i]) else 'initialization_failed' if not bool(init_valid[i])
                    else 'invalid_goal' for i in range(self.n)]
        path = torch.zeros(self.n, device=self.device)
        elapsed = torch.zeros(self.n, device=self.device)
        final = raw.clone()
        contacts = torch.zeros(self.n, 3, dtype=torch.bool, device=self.device)
        start_wall = time.monotonic()
        last_log = start_wall
        start_time = self.env.common_step_counter * dt
        # Anchor timestamps in simulation time, as in stage-two collection.
        self.adapter.timestamps += start_time - math.ceil(self.cfg.initialization_s / dt) * dt
        snapshots = [memory_snapshot()]
        self.record({'event':'prepared', 'seed':seed, 'trials':trial_ids, 'starts':prepared.cpu().tolist(),
                     'goals':goals.cpu().tolist(), 'valid':active.cpu().tolist()})
        self.record({'event':'memory', 'time_s':0., 'bytes':snapshots[0]})
        print(f"[FDM] Navigation prepared: seed={seed} active={int(active.sum())}/{self.n} "
              f"timeout={self.cfg.timeout_s:g}s; first start={prepared[0,:2].cpu().tolist()} "
              f"goal={goals[0].cpu().tolist()}.", flush=True)

        def finish(mask, name, now):
            ids = torch.nonzero(mask & active).flatten()
            for i in ids.tolist():
                outcomes[i] = name
                self.record({'event':'terminal', 'trial':trial_ids[i], 'outcome':name, 'time_s':now,
                             'state':raw[i].cpu().tolist(), 'distance_m':float((goals[i]-raw[i,:2]).norm())})
            final[ids] = raw[ids]
            elapsed[ids] = now
            active[ids] = False
            commands[ids] = 0

        step = 0
        while bool(active.any()):
            now = step * dt
            if not self.running():
                finish(active.clone(), 'interrupted', now)
                break
            speed = self.adapter.robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
            status = progress.update(raw, speed, active, now, self.adapter.robot.data.root_ang_vel_w[:,2])
            finish(status['reached'], 'reached', now)
            finish(status['exhausted'], 'stalled', now)
            if now + 1e-7 >= self.cfg.timeout_s:
                finish(active.clone(), 'timeout', now)
            recovery = torch.nonzero(status['recover'] & active).flatten()
            if len(recovery):
                self.planner.reset(recovery)  # Advance random streams rather than replay the same failed samples.
                commands[recovery] = 0
                for i in recovery.tolist():
                    self.record({'event':'recover', 'trial':trial_ids[i], 'time_s':now,
                                 'attempt':int(progress.recoveries[i])})
            commands[status['near'] | ~active] = 0
            if not bool(active.any()):
                break
            # All live trials use the same 0.5 s command boundaries.
            if step % self.rollout.policy_steps_per_command == 0:
                ids = torch.nonzero(active & ~status['near']).flatten()
                if len(ids):
                    state, proprio, height, invalid, transform = self.adapter.inputs(ids, obs, start_time + now)
                    self.planner.previous[ids] = commands[ids]
                    plans = self.planner.plan(ids, state, proprio, height, invalid,
                                              local_goal(goals[ids], raw[ids]), map_transform=transform,
                                              align_mask=progress.align[ids])
                    commands[ids] = plans[:, 0]
                self.record({'event':'decision', 'time_s':now, 'trials':trial_ids,
                             'active':active.cpu().tolist(), 'state':raw.cpu().tolist(),
                             'goals':goals.cpu().tolist(), 'commands':commands.cpu().tolist(),
                             'align':progress.align.cpu().tolist(),
                             'risk':self.planner.last_risk.cpu().tolist(),
                             'invalid':self.planner.last_invalid.cpu().tolist(),
                             'prediction_local':self.planner.last_pose.cpu().tolist()})
                if self.visualize:
                    self.visualize(raw, goals, commands, self.planner.last_pose, active, progress.align)
            previous_xy = raw[:, :2].clone()
            obs = self._step(obs, commands)
            step += 1
            collision, groups, _ = self.adapter.contacts()
            raw = self.adapter.raw(collision)
            finite = torch.isfinite(raw).all(dim=1) & torch.isfinite(obs['policy']).all(dim=1)
            contacts[active] |= groups[active]
            path[active & finite] += (raw[active & finite, :2] - previous_xy[active & finite]).norm(dim=-1)
            # Events outrank goal success at the same policy step; no reset has occurred.
            finish(~finite, 'nonfinite', step * dt)
            finish(collision, 'collision', step * dt)
            fallen = self.adapter.robot.data.projected_gravity_b[:, 2] > -.5
            finish(fallen, 'fall', step * dt)
            # This boundary is evaluation bookkeeping, not a navigation map input.
            if self.region_contains is not None:
                inside = self.region_contains(raw[:, :2])
            else:
                from ..planning.goals import SpatialSplit
                if not hasattr(self, '_spatial'):
                    self._spatial = SpatialSplit(self.env.scene.terrain)
                split_id = {'train':0, 'val':1, 'test':2}[self.rollout.split]
                inside = self._spatial.lookup(raw[:, :2]) == split_id
            finish(~inside, 'out_of_region', step * dt)
            self.adapter.tick(self.ids[active], raw, obs, start_time + step * dt)
            wall = time.monotonic()
            if self.cfg.log_interval_s and wall - last_log >= self.cfg.log_interval_s:
                snapshots.append(memory_snapshot())
                self.record({'event':'memory', 'time_s':step*dt, 'bytes':snapshots[-1]})
                print(f"[FDM] navigate sim={step*dt:.1f}s active={int(active.sum())}/{self.n} "
                      f"outcomes={dict(Counter(outcomes))} distance="
                      f"{(goals[active]-raw[active,:2]).norm(dim=-1).mean().item() if active.any() else 0:.2f}m", flush=True)
                last_log = wall
        commands.zero_()
        self.adapter.set_commands(commands, obs)
        wall_seconds = time.monotonic() - start_wall
        snapshots.append(memory_snapshot())
        self.record({'event':'memory', 'time_s':step*dt, 'bytes':snapshots[-1]})
        trials = [dict(trial=trial_ids[i], seed=seed, outcome=outcomes[i], start=prepared[i].cpu().tolist(),
                       goal=goals[i].cpu().tolist(), final_state=final[i].cpu().tolist(),
                       elapsed_s=float(elapsed[i]), path_length_m=float(path[i]),
                       final_distance_m=float((goals[i]-final[i,:2]).norm()),
                       recoveries=int(progress.recoveries[i]), contact_groups=contacts[i].cpu().tolist())
                  for i in range(self.n)]
        return dict(trials=trials, planning=self.planner.statistics(), wall_seconds=wall_seconds,
                    simulated_seconds=step*dt, realtime_factor=step*dt/max(wall_seconds,1e-9),
                    sampled_peak_rss_bytes=max(s.get('process_rss') or 0 for s in snapshots),
                    command_constraint_violations=0)
