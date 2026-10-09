"""Fixed-capacity mixed random/MPPI collection with immutable round snapshots."""

from __future__ import annotations

import time

import torch

from ..data.schema import TerminationReason, Split
from ..planning.goals import LocalGoalManager
from ..planning.mppi import MPPIPlanner
from ..utils.se2 import relative_pose_sequence, yaw_from_quaternion_xyzw
from .collector import _Phase
from .fixed_collector import FixedRolloutCollector


class Stage2RolloutCollector(FixedRolloutCollector):
    extra_frame_spec = {"command_source": ((), torch.int8), "goal_xy_w": ((2,), torch.float32),
                        "planner_model_version": ((), torch.int32)}

    def __init__(self, *args, model, navigation_grid, mppi_cfg=None, goal_cfg=None, single_goal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mppi = MPPIPlanner(model, self.num_envs, cfg=mppi_cfg, command_cfg=self.cfg.command, seed=self.cfg.seed)
        self.goals = LocalGoalManager(navigation_grid, self.num_envs, cfg=goal_cfg,
                                      seed=self.cfg.seed, single_goal=single_goal)
        self.source = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        self.model_version = 0
        self._boundary_snapshot = None
        self._evaluation_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluation_started = False

    @torch.inference_mode()
    def collect(self, frames_per_env=150, *, mppi_fraction=.4, model_version=0, seed=None, **kwargs):
        if not 0 <= mppi_fraction <= 1:
            raise ValueError("MPPI source fraction must be in [0, 1].")
        seed = self.cfg.seed if seed is None else seed
        self.source.zero_()
        # Assignment is fixed within this batch, including across episode resets.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        selected = torch.randperm(self.num_envs, device=self.device, generator=generator)[:round(self.num_envs * mppi_fraction)]
        self.source[selected] = 1
        self.model_version = model_version
        self._boundary_snapshot = None
        self._evaluation_ready.zero_()
        self._evaluation_started = False
        self.mppi.reset(seed=seed + 100_003)
        self.mppi.reset_statistics()
        self.goals.reset(seed=seed + 200_003)
        self.planner.generator.manual_seed(seed + 300_007)
        # The safe-spawn sampler owns an independent generator; env.reset(seed)
        # alone does not reseed it. This also makes paired evaluation reproducible.
        if hasattr(self.env, "event_manager"):
            reset = self.env.event_manager.get_term_cfg("reset_base").func
            if hasattr(reset, "_generator"):
                reset._generator.manual_seed(seed)
        result = super().collect(frames_per_env, seed=seed, **kwargs)
        self.goals.finish(torch.nonzero(self.goals.active).flatten(), "censored", self._simulation_time())
        result.update(mppi_fraction=mppi_fraction, mppi_envs=len(selected), planner_model_version=model_version,
                      planning=self.mppi.statistics(), navigation=self.goals.statistics())
        print(f"[FDM] Stage2 collection: planning={result['planning']} navigation="
              f"{ {k: v for k, v in result['navigation'].items() if k not in ('scenarios', 'outcomes')} }", flush=True)
        if kwargs.get("log") is not None:
            kwargs["log"].event({"event": "stage2_collection", **result})
        return result

    def _prepare_boundary(self, ids, timestamp):
        self._boundary_snapshot = None
        if not len(ids):
            return
        if self.goals.single_goal and not self._evaluation_started:
            self._evaluation_ready[ids] = True
        # Force the boundary anchor if the 20 Hz history sampler is out of phase.
        stale = ids[self.history_timestamps[ids, 0] != timestamp]
        raw = self._raw_state(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self._push_history(stale, raw, timestamp)
        height, invalid = super()._frame_maps_batch(ids)
        # Quantize once: the planner and stored training frame see the SAME map.
        self._boundary_snapshot = (timestamp, ids, height.half(), invalid)

    def _frame_maps_batch(self, ids):
        snapshot = self._boundary_snapshot
        if snapshot is not None and snapshot[0] == self._simulation_time() and torch.equal(snapshot[1], ids):
            return snapshot[2:]
        return super()._frame_maps_batch(ids)

    def _plan(self, method, *args):
        if not hasattr(self, "mppi"):
            return super()._plan(method, *args)
        ids = args[0] if args else torch.arange(self.num_envs, device=self.device)
        if method == "reset":
            self.mppi.reset(ids)
            self.goals.reset(ids)
            if not self._evaluation_started:
                self._evaluation_ready[ids] = False
            super()._plan(method, *args)
            # MPPI needs a stationary, full history before its first decision.
            # Random warmup can leave narrow free space before a goal is chosen.
            self.planner.plan[ids[self.source[ids] == 1]] = 0
            return
        if method != "advance":
            raise ValueError(f"Unexpected planning method {method}.")
        release_barrier = False
        if self.goals.single_goal and not self._evaluation_started:
            # Both compared models run identical zero-command settling/history
            # preparation until EVERY environment is ready. No model-dependent
            # reset or RNG consumption can change the initial scenarios.
            if not bool(self._evaluation_ready.all()):
                self.planner.plan[ids] = 0
                return
            self._evaluation_started = release_barrier = True
            ids = torch.arange(self.num_envs, device=self.device)
            self._prepare_boundary(ids, self._simulation_time())
            args = (ids,)
        mppi_ids = ids[self.source[ids] == 1]
        self.mppi.previous[mppi_ids] = self.planner.command[mppi_ids]
        super()._plan(method, *args)
        if not len(mppi_ids):
            return
        started = time.monotonic()
        raw = self.state_history[mppi_ids, 0]
        self.goals.update(mppi_ids, raw, self._simulation_time())
        usable = self.goals.active[mppi_ids]
        self.planner.plan[mppi_ids[~usable]] = 0
        mppi_ids, raw = mppi_ids[usable], raw[usable]
        if len(mppi_ids):
            snapshot = self._boundary_snapshot
            if snapshot is None:
                raise RuntimeError("MPPI requires a coherent command-boundary observation.")
            rows = torch.searchsorted(snapshot[1], mppi_ids)
            history = self.state_history[mppi_ids]
            relative = relative_pose_sequence(history[..., :3], history[..., 3:7])
            state = torch.cat((relative, history[..., 7:8]), dim=-1)
            transform = None
            sensor = getattr(self.height_sensor, "data", None)
            if sensor is not None:
                sensor_yaw = yaw_from_quaternion_xyzw(sensor.quat_w[mppi_ids][:, (1, 2, 3, 0)])
                root_yaw = yaw_from_quaternion_xyzw(raw[:, 3:7])
                delta = raw[:, :2] - sensor.pos_w[mppi_ids, :2]
                transform = torch.stack((sensor_yaw.cos() * delta[:, 0] + sensor_yaw.sin() * delta[:, 1],
                                         -sensor_yaw.sin() * delta[:, 0] + sensor_yaw.cos() * delta[:, 1],
                                         root_yaw - sensor_yaw), dim=-1)
            plans = self.mppi.plan(mppi_ids, state, self.proprio_history[mppi_ids], snapshot[2][rows],
                                   snapshot[3][rows], self.goals.local(mppi_ids, raw), map_transform=transform)
            self.planner.plan[mppi_ids] = plans
        self._command_seconds += time.monotonic() - started
        if release_barrier:
            self.steps_remaining[ids] = self.cfg.policy_steps_per_command
            self._set_commands(ids, self.planner.command[ids])

    def _extra_frame_fields(self, ids):
        goal = torch.where((self.source[ids] == 1)[:, None], self.goals.goal[ids], self.state_history[ids, 0, :2])
        return {"command_source": self.source[ids], "goal_xy_w": goal,
                "planner_model_version": torch.full((len(ids),), self.model_version, device=self.device, dtype=torch.int32)}

    def _finish_many(self, ids, reason=None):
        if len(ids) and hasattr(self, "goals"):
            outcome = {None: "collision", TerminationReason.TIMEOUT: "timeout", TerminationReason.WATCHDOG: "watchdog",
                       TerminationReason.SPLIT_EXIT: "split_exit", TerminationReason.ROUND_CUT: "censored",
                       TerminationReason.GOAL_UNAVAILABLE: "censored"}[reason]
            # Base collector passes None for both collision events and its time
            # limit. Active command count distinguishes those two physics cases.
            if reason is None:
                timeout = self.active_commands[ids] >= self.cfg.max_episode_commands
                self.goals.finish(ids[timeout], "timeout", self._simulation_time())
                ids_collision = ids[~timeout]
                self.goals.finish(ids_collision, "collision", self._simulation_time())
            else:
                self.goals.finish(ids, outcome, self._simulation_time())
        return super()._finish_many(ids, reason)

    def _handle_resets(self, done):
        active = torch.nonzero(done & (self.phase == int(_Phase.ACTIVE))).flatten()
        self.goals.finish(active, "watchdog", self._simulation_time())
        super()._handle_resets(done)
        ids = torch.nonzero(done).flatten()
        self.mppi.reset(ids)
        self.goals.reset(ids)
        if not self._evaluation_started:
            self._evaluation_ready[ids] = False

    def _step_exclusions(self, done):
        current = self.goals.grid.spatial.lookup(self.robot.data.root_pos_w[:, :2])
        outside = (current != int(Split.from_name(self.cfg.split))) & ~done & (self.phase != self.FULL)
        ids = torch.nonzero(outside).flatten()
        if len(ids):
            active = ids[self.phase[ids] == int(_Phase.ACTIVE)]
            # Do not record an out-of-split observation or label it as collision.
            self._finish_many(active, TerminationReason.SPLIT_EXIT)
            self._selective_reset(ids, rejected_spawn=False)
        return outside

    def _after_step(self):
        if not self.goals.single_goal:
            waiting = self.goals.unavailable_since
            stalled = ((self.phase == int(_Phase.ACTIVE)) & (self.source == 1) & ~self.goals.active
                       & (waiting >= 0) & (self._simulation_time() - waiting >= self.goals.cfg.stall_s))
            ids = torch.nonzero(stalled).flatten()
            if len(ids):
                self.goals.counts["goal_unavailable_resets"] += len(ids)
                self._finish_many(ids, TerminationReason.GOAL_UNAVAILABLE)
                self._selective_reset(ids, rejected_spawn=False)
            return None
        ids = torch.nonzero(self.goals.finished & (self.phase != self.FULL)).flatten()
        self._finish_many(ids, TerminationReason.ROUND_CUT)
        self.phase[ids] = self.FULL
        # Mark inactive slots full so a later simulator autoreset cannot reopen
        # them. Evaluation stops on goals, never on the number of stored frames.
        self.frame_counts[ids] = self.buffer.capacity
        self._set_commands(ids, torch.zeros(len(ids), 3, device=self.device))
        return "evaluation_complete" if bool(self.goals.finished.all()) else None
