"""Parallel closed-loop rollout collector for the frozen G1 policy."""

from __future__ import annotations

from enum import IntEnum

import torch

from ..config import RolloutCfg
from ..data.schema import EpisodeBuilder, Split, TerminationReason
from ..data.shard_writer import EpisodeShardWriter
from ..utils.height_map import door_aware_height_map
from .command_planner import CorrelatedCommandPlanner
from .frozen_policy import FrozenRecurrentPolicy


class _Phase(IntEnum):
    SETTLING = 0
    WARMUP = 1
    ACTIVE = 2
    WAITING_FOR_RESET = 3


class FDMRolloutCollector:
    """Collect command-level frames while the recurrent locomotion actor runs at 50 Hz."""

    def __init__(
        self,
        env,
        observations: dict[str, torch.Tensor],
        policy: FrozenRecurrentPolicy,
        writer: EpisodeShardWriter,
        cfg: RolloutCfg,
    ) -> None:
        cfg.validate()
        self.env = env.unwrapped
        self.observations = observations
        self.policy = policy
        self.writer = writer
        self.cfg = cfg
        self.device = torch.device(self.env.device)
        self.num_envs = self.env.num_envs
        if tuple(observations["policy"].shape) != (self.num_envs, 283):
            raise ValueError(f"Frozen policy observation must be [N, 283], got {observations['policy'].shape}.")

        self.command_term = self.env.command_manager.get_term("base_velocity")
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        self.height_sensor = self.env.scene.sensors["fdm_height_scanner"]
        self.robot = self.env.scene["robot"]
        self._resolve_body_ids()
        self.planner = CorrelatedCommandPlanner(
            self.num_envs, cfg.prediction_horizon, cfg.command, self.device, cfg.seed
        )
        self.planner.reset()

        self.phase = torch.full((self.num_envs,), int(_Phase.SETTLING), device=self.device, dtype=torch.int8)
        self.settle_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.spawn_attempts = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.steps_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.active_commands = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.history_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.history_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.state_history = torch.zeros(self.num_envs, cfg.history_length, 8, device=self.device)
        self.proprio_history = torch.zeros(self.num_envs, cfg.history_length, 96, device=self.device)
        self.history_timestamps = torch.zeros(
            self.num_envs, cfg.history_length, device=self.device, dtype=torch.float64
        )
        self.last_frame_timestamp = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.episode_ids = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.builders: list[EpisodeBuilder | None] = [None] * self.num_envs
        self._frame_maps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._next_episode_id = writer.next_episode_id
        self.completed_episodes = 0
        self._target_episodes = 0
        self.command_term.set_command(torch.zeros(self.num_envs, 3, device=self.device))

    def _resolve_body_ids(self) -> None:
        available = self.contact_sensor.body_names
        missing = [name for name in self.cfg.collision_body_names if name not in available]
        if missing:
            raise RuntimeError(
                "G1 navigation collision links did not resolve exactly. "
                f"Missing={missing}; available bodies={available}."
            )
        self.navigation_body_ids = torch.tensor(
            [available.index(name) for name in self.cfg.collision_body_names], device=self.device
        )
        self.contact_group_ids = (
            torch.tensor([available.index("torso_link")], device=self.device),
            torch.tensor(
                [available.index(name) for name in self.cfg.collision_body_names if name.startswith("left_")],
                device=self.device,
            ),
            torch.tensor(
                [available.index(name) for name in self.cfg.collision_body_names if name.startswith("right_")],
                device=self.device,
            ),
        )
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        missing_feet = [name for name in foot_names if name not in available]
        if missing_feet:
            raise RuntimeError(f"Cannot perform safe-spawn stance check; missing foot bodies {missing_feet}.")
        self.foot_body_ids = torch.tensor([available.index(name) for name in foot_names], device=self.device)

    def _simulation_time(self) -> float:
        return float(self.env.common_step_counter * self.env.step_dt)

    def _collision_and_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
        forces = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w, dim=-1)
        collision = torch.any(forces[:, self.navigation_body_ids] > self.cfg.collision_force_threshold, dim=-1)
        groups = torch.stack(
            [torch.any(forces[:, ids] > self.cfg.collision_force_threshold, dim=-1) for ids in self.contact_group_ids],
            dim=-1,
        )
        return collision, groups

    def _raw_state(self, collision: torch.Tensor) -> torch.Tensor:
        quaternion_wxyz = self.robot.data.root_quat_w
        quaternion_xyzw = quaternion_wxyz[:, (1, 2, 3, 0)]
        return torch.cat(
            (self.robot.data.root_pos_w, quaternion_xyzw, collision.to(torch.float32).unsqueeze(-1)), dim=-1
        )

    def _prepare_height_maps(self, env_ids: torch.Tensor) -> None:
        """Refresh and door-correct all frame maps due at this simulator step."""

        if len(env_ids) == 0:
            return
        # Asynchronous resets can put a frame out of phase with the sensor's
        # update period. Refresh only the environments whose frames are due.
        self.height_sensor.reset(env_ids)
        height, invalid = door_aware_height_map(
            self.height_sensor,
            env_ids,
            shape=(self.cfg.map_height, self.cfg.map_width),
            clip=(self.cfg.map_clip_min, self.cfg.map_clip_max),
            invalid_sentinel=self.cfg.invalid_height_sentinel,
            door_probe_height=self.cfg.map_door_probe_height,
            door_height_threshold=self.cfg.map_door_height_threshold,
        )
        for row, env_id in enumerate(env_ids.tolist()):
            self._frame_maps[env_id] = (height[row], invalid[row])

    def _push_history(self, env_ids: torch.Tensor, raw_state: torch.Tensor, timestamp: float) -> None:
        if len(env_ids) == 0:
            return
        self.state_history[env_ids, 1:] = self.state_history[env_ids, :-1].clone()
        self.proprio_history[env_ids, 1:] = self.proprio_history[env_ids, :-1].clone()
        self.history_timestamps[env_ids, 1:] = self.history_timestamps[env_ids, :-1].clone()
        self.state_history[env_ids, 0] = raw_state[env_ids]
        self.proprio_history[env_ids, 0] = self.observations["policy"][env_ids, :96]
        self.history_timestamps[env_ids, 0] = timestamp
        self.history_count[env_ids] = torch.clamp(self.history_count[env_ids] + 1, max=self.cfg.history_length)

    def _sample_due_history(self, raw_state: torch.Tensor, timestamp: float) -> None:
        eligible = (self.phase == int(_Phase.WARMUP)) | (self.phase == int(_Phase.ACTIVE))
        self.history_elapsed[eligible] += self.cfg.policy_dt
        due = eligible & (self.history_elapsed + 1.0e-7 >= self.cfg.history_timestep)
        env_ids = torch.nonzero(due).flatten()
        self.history_elapsed[env_ids] -= self.cfg.history_timestep
        self._push_history(env_ids, raw_state, timestamp)

    def _set_commands(self, env_ids: torch.Tensor, commands: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.command_term.set_command(commands, env_ids)
        # Policy observation order is fixed by the source task: command occupies 6:9.
        # Patching only that slice avoids applying observation corruption twice.
        self.observations["policy"][env_ids, 6:9] = commands

    def _record_frame(
        self,
        env_id: int,
        *,
        collision: bool,
        contact_groups: torch.Tensor,
        has_outgoing_command: bool,
        terminated: bool,
        truncated: bool,
        reason: TerminationReason,
        timestamp: float,
    ) -> None:
        if int(self.history_count[env_id]) < self.cfg.history_length:
            raise RuntimeError("Attempted to record an FDM frame before its ten-point history was full.")
        builder = self.builders[env_id]
        if builder is None:
            raise RuntimeError("Active environment has no episode builder.")
        height, invalid = self._frame_maps.pop(env_id)
        current_command = self.planner.command[env_id]
        previous_timestamp = float(self.last_frame_timestamp[env_id])
        delta_t = 0.0 if builder.num_frames == 0 else timestamp - previous_timestamp
        importer = self.env.scene.terrain
        origin_id = importer.env_origin_ids[env_id]
        split_id = importer.origin_split_ids[origin_id]
        builder.append(
            state_history_raw=self.state_history[env_id].float(),
            proprio_history=self.proprio_history[env_id].float(),
            history_timestamps=self.history_timestamps[env_id].double(),
            height_map=height.half(),
            height_map_invalid=invalid.bool(),
            command=current_command.float(),
            command_plan=self.planner.plan[env_id].float(),
            has_outgoing_command=torch.tensor(has_outgoing_command),
            timestamp=torch.tensor(timestamp, dtype=torch.float64),
            delta_t=torch.tensor(delta_t, dtype=torch.float32),
            collision_now=torch.tensor(collision),
            contact_groups=contact_groups.bool(),
            terminated=torch.tensor(terminated),
            truncated=torch.tensor(truncated),
            termination_reason=torch.tensor(int(reason), dtype=torch.int8),
            episode_id=self.episode_ids[env_id].cpu(),
            usd_origin_id=origin_id.cpu().to(torch.int32),
            usd_region_split=split_id.cpu(),
        )
        self.last_frame_timestamp[env_id] = timestamp

    def _finish_episode(self, env_id: int) -> None:
        builder = self.builders[env_id]
        if builder is not None and builder.num_frames > 0 and self.completed_episodes < self._target_episodes:
            self.writer.append(builder.finalize())
            self.completed_episodes += 1
        self.builders[env_id] = None

    def _begin_warmup(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.phase[env_ids] = int(_Phase.WARMUP)
        self.steps_remaining[env_ids] = self.cfg.policy_steps_per_command
        self.history_elapsed[env_ids] = 0.0
        self.history_count[env_ids] = 0
        self.state_history[env_ids] = 0.0
        self.proprio_history[env_ids] = 0.0
        self.history_timestamps[env_ids] = 0.0
        self.planner.reset(env_ids)
        self._set_commands(env_ids, self.planner.command[env_ids])

    def _begin_episode_after_warmup(self, env_id: int, contact_groups: torch.Tensor, timestamp: float) -> None:
        env_ids = torch.tensor([env_id], device=self.device)
        self.planner.advance(env_ids)
        self.phase[env_id] = int(_Phase.ACTIVE)
        self.steps_remaining[env_id] = self.cfg.policy_steps_per_command
        self.active_commands[env_id] = 0
        self.episode_ids[env_id] = self._next_episode_id
        self._next_episode_id += 1
        self.builders[env_id] = EpisodeBuilder()
        self._record_frame(
            env_id,
            collision=False,
            contact_groups=contact_groups,
            has_outgoing_command=True,
            terminated=False,
            truncated=False,
            reason=TerminationReason.NONE,
            timestamp=timestamp,
        )
        self._set_commands(env_ids, self.planner.command[env_ids])

    def _handle_resets(self, done: torch.Tensor) -> None:
        env_ids = torch.nonzero(done).flatten()
        if len(env_ids) == 0:
            return
        for env_id in env_ids.tolist():
            if self.phase[env_id] == int(_Phase.ACTIVE):
                builder = self.builders[env_id]
                if builder is not None:
                    builder.mark_last_truncated(TerminationReason.WATCHDOG)
                self._finish_episode(env_id)
        self.policy.reset(done)
        self.phase[env_ids] = int(_Phase.SETTLING)
        self.settle_steps[env_ids] = 0
        self.spawn_attempts[env_ids] += 1
        if torch.any(self.spawn_attempts[env_ids] > self.cfg.spawn_attempts):
            bad = env_ids[self.spawn_attempts[env_ids] > self.cfg.spawn_attempts].tolist()
            raise RuntimeError(f"Safe-spawn rejection exceeded {self.cfg.spawn_attempts} attempts for envs {bad}.")
        self.steps_remaining[env_ids] = 0
        self.history_count[env_ids] = 0
        reset_indices = set(env_ids.tolist())
        self.builders = [
            None if index in reset_indices else value for index, value in enumerate(self.builders)
        ]
        self.planner.reset(env_ids)
        self._set_commands(env_ids, torch.zeros(len(env_ids), 3, device=self.device))

    def _selective_reset(self, env_ids: torch.Tensor, *, rejected_spawn: bool) -> None:
        """Reset selected environments after a rejection or collector truncation."""

        if len(env_ids) == 0:
            return
        self.env._reset_idx(env_ids)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        done[env_ids] = True
        # Recompute after the selective reset. No observation history is advanced.
        self.observations = self.env.observation_manager.compute(update_history=False)
        self.policy.reset(done)
        self.phase[env_ids] = int(_Phase.SETTLING)
        self.settle_steps[env_ids] = 0
        if rejected_spawn:
            self.spawn_attempts[env_ids] += 1
            if torch.any(self.spawn_attempts[env_ids] > self.cfg.spawn_attempts):
                bad = env_ids[self.spawn_attempts[env_ids] > self.cfg.spawn_attempts].tolist()
                raise RuntimeError(
                    f"Safe-spawn rejection exceeded {self.cfg.spawn_attempts} attempts for envs {bad}."
                )
        else:
            self.spawn_attempts[env_ids] = 1
        self.history_count[env_ids] = 0
        self.planner.reset(env_ids)
        self._set_commands(env_ids, torch.zeros(len(env_ids), 3, device=self.device))

    def collect(self, num_episodes: int) -> dict[str, int]:
        """Collect exactly ``num_episodes`` complete trajectories (up to same-step overshoot suppression)."""

        if num_episodes < 1:
            raise ValueError("num_episodes must be positive.")
        self._target_episodes = self.completed_episodes + num_episodes
        max_settle_steps = max(250, self.cfg.initial_settle_steps)
        while self.completed_episodes < self._target_episodes:
            self._frame_maps.clear()
            with torch.inference_mode():
                actions = self.policy.act(self.observations)
                observations, _, terminated, truncated, _ = self.env.step(actions)
            self.observations = observations
            done = terminated | truncated
            collision, groups = self._collision_and_groups()
            timestamp = self._simulation_time()
            raw_state = self._raw_state(collision)
            self._sample_due_history(raw_state, timestamp)

            active_collision = (
                (self.phase == int(_Phase.ACTIVE)) & collision & ~done
            )
            active_collision_ids = torch.nonzero(active_collision).flatten()
            self._prepare_height_maps(active_collision_ids)
            for env_id in active_collision_ids.tolist():
                # Event frames must include the exact collision state even if it is
                # between two regular 20 Hz history ticks.
                if float(self.history_timestamps[env_id, 0]) != timestamp:
                    self._push_history(torch.tensor([env_id], device=self.device), raw_state, timestamp)
                self._record_frame(
                    env_id,
                    collision=True,
                    contact_groups=groups[env_id],
                    has_outgoing_command=False,
                    terminated=False,
                    truncated=False,
                    reason=TerminationReason.COLLISION,
                    timestamp=timestamp,
                )
                self._finish_episode(env_id)
                self.phase[env_id] = int(_Phase.WAITING_FOR_RESET)
                self._set_commands(
                    torch.tensor([env_id], device=self.device), torch.zeros(1, 3, device=self.device)
                )

            warmup_collision = (self.phase == int(_Phase.WARMUP)) & collision & ~done
            self.phase[warmup_collision] = int(_Phase.WAITING_FOR_RESET)

            moving = ((self.phase == int(_Phase.WARMUP)) | (self.phase == int(_Phase.ACTIVE))) & ~collision & ~done
            self.steps_remaining[moving] -= 1
            boundaries = moving & (self.steps_remaining == 0)
            warmup_boundaries = boundaries & (self.phase == int(_Phase.WARMUP))
            active_boundaries = boundaries & (self.phase == int(_Phase.ACTIVE))
            self._prepare_height_maps(torch.nonzero(boundaries).flatten())
            collector_reset_ids: list[int] = []
            for env_id in torch.nonzero(warmup_boundaries).flatten().tolist():
                self._begin_episode_after_warmup(env_id, groups[env_id], timestamp)
            for env_id in torch.nonzero(active_boundaries).flatten().tolist():
                self.active_commands[env_id] += 1
                if self.active_commands[env_id] >= self.cfg.max_episode_commands:
                    self._record_frame(
                        env_id,
                        collision=False,
                        contact_groups=groups[env_id],
                        has_outgoing_command=False,
                        terminated=False,
                        truncated=True,
                        reason=TerminationReason.TIMEOUT,
                        timestamp=timestamp,
                    )
                    self._finish_episode(env_id)
                    self.phase[env_id] = int(_Phase.WAITING_FOR_RESET)
                    collector_reset_ids.append(env_id)
                    self._set_commands(
                        torch.tensor([env_id], device=self.device), torch.zeros(1, 3, device=self.device)
                    )
                else:
                    env_ids = torch.tensor([env_id], device=self.device)
                    self.planner.advance(env_ids)
                    self._record_frame(
                        env_id,
                        collision=False,
                        contact_groups=groups[env_id],
                        has_outgoing_command=True,
                        terminated=False,
                        truncated=False,
                        reason=TerminationReason.NONE,
                        timestamp=timestamp,
                    )
                    self.steps_remaining[env_id] = self.cfg.policy_steps_per_command
                    self._set_commands(env_ids, self.planner.command[env_ids])

            settling = (self.phase == int(_Phase.SETTLING)) & ~done
            self.settle_steps[settling] += 1
            force_norm = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w, dim=-1)
            both_feet = torch.all(force_norm[:, self.foot_body_ids] > self.cfg.collision_force_threshold, dim=-1)
            upright = self.robot.data.projected_gravity_b[:, 2] < -0.7
            root_high_enough = self.robot.data.root_pos_w[:, 2] > 0.3
            safe = settling & ~collision & both_feet & upright & root_high_enough
            ready = safe & (self.settle_steps >= self.cfg.initial_settle_steps)
            self.spawn_attempts[ready] = 0
            self._begin_warmup(torch.nonzero(ready).flatten())

            stuck = settling & (self.settle_steps >= max_settle_steps)
            if torch.any(stuck):
                self._selective_reset(torch.nonzero(stuck).flatten(), rejected_spawn=True)

            if collector_reset_ids:
                self._selective_reset(
                    torch.tensor(collector_reset_ids, device=self.device), rejected_spawn=False
                )

            self._handle_resets(done)

        self.writer.flush()
        return {"episodes": num_episodes, "total_completed": self.completed_episodes}
