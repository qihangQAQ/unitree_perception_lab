"""Round-local, fixed-capacity collection with batched command and frame updates."""

from __future__ import annotations

import math
import time
import traceback

import torch

from ..data.rollout_buffer import CollectionBudget, FixedFrameBuffer
from ..data.schema import TerminationReason
from ..utils.height_map import door_aware_height_map
from ..utils.memory import available_memory_bytes
from .collection_log import CollectionTracker
from .collector import FDMRolloutCollector, _Phase


class FixedRolloutCollector(FDMRolloutCollector):
    """Reuse the G1 sensor/history semantics, with no pending episodes across rounds.

    Full environments stop contributing frames. Slow-tail stopping retains real
    records only; the dataset sampler, not the raw trajectories, handles repeats.
    Storage is bounded during collection and released before sample preparation.
    """

    collection_mode = "fixed"
    FULL = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer: FixedFrameBuffer | None = None
        self.frame_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stop_reason: str | None = None

    @property
    def pending_frames(self) -> int:
        return self.buffer.pending_frames if self.buffer is not None else 0

    @property
    def pending_episodes(self) -> int:
        return int(self.buffer.open.sum()) if self.buffer is not None else 0

    def _record_batch(self, ids, groups, timestamp, *, collision=False, timeout=None):
        if not len(ids):
            return
        if torch.any(self.history_count[ids] < self.cfg.history_length):
            raise RuntimeError("FDM frame history is not full.")
        self.height_sensor.reset(ids)
        height, invalid = door_aware_height_map(
            self.height_sensor, ids, shape=(self.cfg.map_height, self.cfg.map_width),
            clip=(self.cfg.map_clip_min, self.cfg.map_clip_max),
            invalid_sentinel=self.cfg.invalid_height_sentinel,
            door_probe_height=self.cfg.map_door_probe_height,
            door_height_threshold=self.cfg.map_door_height_threshold,
        )
        timeout = torch.zeros(len(ids), dtype=torch.bool, device=self.device) if timeout is None else timeout
        collisions = torch.full_like(timeout, collision)
        reason = torch.where(timeout, int(TerminationReason.TIMEOUT), int(TerminationReason.NONE))
        reason = torch.where(collisions, int(TerminationReason.COLLISION), reason)
        importer = self.env.scene.terrain
        origin = importer.env_origin_ids[ids]
        self.buffer.append(
            ids,
            state_history_raw=self.state_history[ids], proprio_history=self.proprio_history[ids],
            history_timestamps=self.history_timestamps[ids], height_map=height.half(), height_map_invalid=invalid,
            command=self.planner.command[ids], command_plan=self.planner.plan[ids],
            has_outgoing_command=~(collisions | timeout),
            timestamp=torch.full((len(ids),), timestamp, device=self.device, dtype=torch.float64),
            delta_t=(timestamp - self.last_frame_timestamp[ids]).float(),
            collision_now=collisions, contact_groups=groups[ids], terminated=torch.zeros_like(timeout),
            truncated=timeout, termination_reason=reason,
            episode_id=self.episode_ids[ids], usd_origin_id=origin,
            usd_region_split=importer.origin_split_ids[origin],
        )
        self.last_frame_timestamp[ids] = timestamp
        self.frame_counts[ids] += 1
        self._tracker.frames += len(ids)

    def _finish_many(self, ids, reason=None):
        if not len(ids):
            return
        accepted = 0
        for env_id in ids.cpu().tolist():
            episode = self.buffer.finish(env_id, reason)
            if episode is not None:
                self.writer.append(episode)
                accepted += 1
        self.completed_episodes += accepted
        self._tracker.accepted_force_episodes += accepted
        self._tracker.force_peaks = torch.maximum(
            self._tracker.force_peaks, self._episode_force_peaks[ids].amax(dim=0)
        )

    def _handle_resets(self, done):
        ids = torch.nonzero(done).flatten()
        if not len(ids):
            return
        active = ids[self.phase[ids] == int(_Phase.ACTIVE)]
        self._finish_many(active, TerminationReason.WATCHDOG)
        self.policy.reset(done)
        self._tracker.counters[0] += (self.phase[ids] == int(_Phase.SETTLING)).sum()
        self._tracker.counters[2] += (self.phase[ids] == int(_Phase.WARMUP)).sum()
        self._tracker.counters[3] += len(ids)
        unfinished = ids[self.frame_counts[ids] < self.buffer.capacity]
        self.phase[ids] = self.FULL
        self.phase[unfinished] = int(_Phase.SETTLING)
        self.settle_steps[unfinished] = 0
        self.spawn_attempts[unfinished] += 1
        if torch.any(self.spawn_attempts[unfinished] > self.cfg.spawn_attempts):
            raise RuntimeError("Safe-spawn rejection exceeded the configured attempts.")
        self.history_count[ids] = 0
        self.steps_remaining[ids] = 0
        self._set_commands(ids, torch.zeros(len(ids), 3, device=self.device))

    @torch.inference_mode()
    def collect(
        self, frames_per_env=150, *, seed=None, min_fill=0.95, tail_factor=1.5, max_steps=0,
        log_interval_s=10.0, log=None, round_index=None,
    ):
        if frames_per_env < self.cfg.prediction_horizon + 2:
            raise ValueError("frames_per_env must be at least prediction_horizon + 2.")
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative.")
        budget = CollectionBudget(self.num_envs * frames_per_env, min_fill, tail_factor)
        needed = FixedFrameBuffer.estimate_bytes(self.num_envs, frames_per_env)
        available = available_memory_bytes()
        if available is not None and needed + 256 * 2**20 > available * 0.8:
            raise MemoryError(
                f"Fixed rollout buffer needs {needed / 2**30:.2f} GiB plus writer space; "
                f"available={available / 2**30:.2f} GiB. Reduce --num-envs or --frames-per-env."
            )
        print(f"[FDM] Fixed collection: envs={self.num_envs} frames_per_env={frames_per_env} "
              f"target_frames={budget.target} buffer={needed / 2**30:.3f}GiB min_fill={min_fill}.", flush=True)
        self.buffer = FixedFrameBuffer(self.num_envs, frames_per_env)
        self.frame_counts.zero_()
        self.stop_reason = None
        tracker = CollectionTracker(self, budget.target, log_interval_s, log, round_index)
        self._tracker = tracker
        try:
            # Every call starts a new round, including the recurrent actor state.
            self.observations, _ = self.env.reset(seed=self.cfg.seed if seed is None else seed)
            self.policy.reset()
            self.phase.fill_(int(_Phase.SETTLING))
            self.settle_steps.zero_()
            self.spawn_attempts.fill_(1)
            self.steps_remaining.zero_()
            self.active_commands.zero_()
            self.history_elapsed.zero_()
            self.history_count.zero_()
            self._episode_force_peaks.zero_()
            self._set_commands(torch.arange(self.num_envs, device=self.device),
                               torch.zeros(self.num_envs, 3, device=self.device))
            # Bound even rounds whose bad environments prevent reaching min_fill.
            step_limit = max_steps or math.ceil(4 * (frames_per_env + 1) * self.cfg.policy_steps_per_command
                                               + 250 * self.cfg.spawn_attempts)
            self._fixed_steps(tracker, budget, step_limit)
            pending = torch.nonzero(self.buffer.open).flatten().to(self.device)
            self._finish_many(pending, TerminationReason.ROUND_CUT)
            self.writer.flush()
            self.phase.fill_(self.FULL)
            record = tracker.update(force=True, status="completed")
            return {**record, "total_completed": self.completed_episodes}
        except BaseException as exc:
            traceback.print_exc()
            tracker.update(force=True, status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                           error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            # Raw data are archived, so keep neither pending episodes nor a large
            # raw buffer resident while constructing/using the training cache.
            self.buffer = None
            self._tracker = None

    def _fixed_steps(self, tracker, budget, step_limit):
        for step in range(step_limit):
            started = time.monotonic()
            actions = self.policy.act(self.observations)
            after_policy = time.monotonic()
            self.observations, _, terminated, truncated, _ = self.env.step(actions)
            after_sim = time.monotonic()
            tracker.policy_seconds += after_policy - started
            tracker.sim_seconds += after_sim - after_policy
            command_before, write_before = self._command_seconds, self.writer.write_seconds
            done = terminated | truncated
            collision, groups = self._collision_and_groups()
            active = (self.phase == int(_Phase.ACTIVE)) & ~done
            self._episode_force_peaks = torch.maximum(
                self._episode_force_peaks, torch.where(active[:, None], self._step_force_peaks, 0.0)
            )
            timestamp = self._simulation_time()
            raw_state = self._raw_state(collision)
            self._sample_due_history(raw_state, timestamp)
            collision_ids = torch.nonzero(active & collision).flatten()
            event_history_ids = collision_ids[self.history_timestamps[collision_ids, 0] != timestamp]
            self._push_history(event_history_ids, raw_state, timestamp)
            self._record_batch(collision_ids, groups, timestamp, collision=True)
            self._finish_many(collision_ids)
            self.phase[collision_ids] = int(_Phase.WAITING_FOR_RESET)
            self._set_commands(collision_ids, torch.zeros(len(collision_ids), 3, device=self.device))

            warmup_collision = (self.phase == int(_Phase.WARMUP)) & collision & ~done
            tracker.counters[1] += warmup_collision.sum()
            self.phase[warmup_collision] = int(_Phase.WAITING_FOR_RESET)
            moving = ((self.phase == int(_Phase.WARMUP)) | (self.phase == int(_Phase.ACTIVE))) & ~collision & ~done
            self.steps_remaining[moving] -= 1
            boundaries = moving & (self.steps_remaining == 0)
            born = boundaries & (self.phase == int(_Phase.WARMUP))
            continuing = boundaries & (self.phase == int(_Phase.ACTIVE))
            self.active_commands[continuing] += 1
            timeout = continuing & (self.active_commands >= self.cfg.max_episode_commands)
            born_ids = torch.nonzero(born).flatten()
            planned_ids = torch.nonzero(born | (continuing & ~timeout)).flatten()
            self._plan("advance", planned_ids)
            self.buffer.begin(born_ids)
            self.episode_ids[born_ids] = torch.arange(
                self._next_episode_id, self._next_episode_id + len(born_ids), device=self.device
            )
            self._next_episode_id += len(born_ids)
            self._episode_force_peaks[born_ids] = 0
            self.last_frame_timestamp[born_ids] = timestamp
            self.active_commands[born_ids] = 0
            self.phase[born_ids] = int(_Phase.ACTIVE)
            boundary_ids = torch.nonzero(boundaries).flatten()
            self._record_batch(boundary_ids, groups, timestamp, timeout=timeout[boundary_ids])
            self.steps_remaining[planned_ids] = self.cfg.policy_steps_per_command
            self._set_commands(planned_ids, self.planner.command[planned_ids])
            timeout_ids = torch.nonzero(timeout).flatten()
            self._finish_many(timeout_ids)
            self.phase[timeout_ids] = int(_Phase.WAITING_FOR_RESET)

            newly_full = (self.frame_counts >= self.buffer.capacity) & (self.phase != self.FULL)
            full_ids = torch.nonzero(newly_full).flatten()
            self._finish_many(full_ids, TerminationReason.ROUND_CUT)
            self.phase[full_ids] = self.FULL
            self._set_commands(full_ids, torch.zeros(len(full_ids), 3, device=self.device))

            settling = (self.phase == int(_Phase.SETTLING)) & ~done
            self.settle_steps[settling] += 1
            forces = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w, dim=-1)
            both_feet = torch.all(forces[:, self.foot_body_ids] > self.cfg.collision_force_threshold, dim=-1)
            safe = (settling & ~collision & both_feet & (self.robot.data.projected_gravity_b[:, 2] < -0.7)
                    & (self.robot.data.root_pos_w[:, 2] > 0.3))
            ready = safe & (self.settle_steps >= self.cfg.initial_settle_steps)
            self.spawn_attempts[ready] = 0
            self._begin_warmup(torch.nonzero(ready).flatten())
            stuck = settling & (self.settle_steps >= max(250, self.cfg.initial_settle_steps))
            self._selective_reset(torch.nonzero(stuck).flatten(), rejected_spawn=True)
            self._selective_reset(torch.nonzero(timeout & ~newly_full).flatten(), rejected_spawn=False)
            self._handle_resets(done)
            tracker.data_seconds += max(0.0, time.monotonic() - after_sim
                                        - (self._command_seconds - command_before)
                                        - (self.writer.write_seconds - write_before))
            self.stop_reason = budget.stop_reason(tracker.frames, time.monotonic() - tracker.start)
            tracker.update()
            if self.stop_reason is not None:
                return
        if tracker.frames / budget.target >= budget.min_fill:
            self.stop_reason = "step_limit"
            return
        raise RuntimeError(f"Collection reached {step_limit} policy steps with fill={tracker.frames / budget.target:.3f}, "
                           f"below required {budget.min_fill:.3f}; inspect spawn/termination diagnostics.")
