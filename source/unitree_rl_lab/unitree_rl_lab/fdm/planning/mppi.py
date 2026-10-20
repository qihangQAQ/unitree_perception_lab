"""Batched receding-horizon MPPI using the frozen height FDM snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from ..config import CommandSamplingCfg


@dataclass
class MPPICfg:
    candidates: int = 128
    iterations: int = 1
    inference_chunk: int = 2048
    temperature: float = 1.0
    noise_std: tuple[float, float, float] = (0.45, 0.12, 0.65)
    noise_correlation: float = 0.6
    goal_terminal_weight: float = 10.0
    goal_path_weight: float = 2.0
    collision_weight: float = 20.0
    high_risk_weight: float = 100.0
    high_risk_threshold: float = 0.55
    smoothness_weight: float = 1.0
    map_penalty: float = 200.0
    # GridPatternCfg(ordering="xy"), scanner x offset 1.75, then row flip.
    map_x_min: float = -0.5
    map_y_max: float = 2.95
    map_resolution: float = 0.1

    def validate(self):
        if self.candidates < 3 or self.iterations < 1 or self.inference_chunk < 1:
            raise ValueError("MPPI requires >=3 candidates, positive iterations and inference chunk.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("MPPI temperature must be finite and positive.")
        if not 0 <= self.noise_correlation < 1 or not 0 < self.high_risk_threshold < 1:
            raise ValueError("Invalid MPPI correlation or risk threshold.")
        if len(self.noise_std) != 3 or any(not math.isfinite(v) or v < 0 for v in self.noise_std):
            raise ValueError("Invalid MPPI noise scale.")
        weights = (self.goal_terminal_weight, self.goal_path_weight, self.collision_weight,
                   self.high_risk_weight, self.smoothness_weight, self.map_penalty)
        if any(not math.isfinite(v) or v < 0 for v in weights) or self.map_resolution <= 0:
            raise ValueError("Invalid MPPI cost/map configuration.")


class MPPIPlanner:
    """Encode once per environment, then score candidate commands in chunks.

    Collision outputs are cumulative event probabilities. We use their monotone
    envelope directly, never multiply them as independent per-step hazards.
    The returned plan is final (bounded and fallback checked) before recording.
    """

    def __init__(self, model, num_envs, *, cfg=None, command_cfg=None, seed=42,
                 forward_only=False, turn_candidates=False, independent_rng=False,
                 record_predictions=False, track_statistics=True):
        self.cfg = cfg or MPPICfg()
        self.cfg.validate()
        command_cfg = command_cfg or CommandSamplingCfg()
        command_cfg.validate()
        self.forward_only = forward_only
        self.turn_candidates = turn_candidates
        self.record_predictions = record_predictions
        self.track_statistics = track_statistics
        if forward_only and (command_cfg.min_forward_speed != 0
                             or command_cfg.min_lateral_speed != 0 or command_cfg.max_lateral_speed != 0):
            raise ValueError("Forward-only planning requires vx >= 0 (including stop) and vy == 0.")
        if turn_candidates and self.cfg.candidates < 8:
            raise ValueError("Navigation turn candidates require at least eight candidates.")
        self.model = model.eval().requires_grad_(False)
        # deepcopy() of a CUDA model loses cuDNN's packed recurrent storage.
        # Repack once per snapshot, not once for every candidate chunk.
        for module in self.model.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
        self.device = next(model.parameters()).device
        self.horizon = model.cfg.horizon
        self.lower = torch.tensor([command_cfg.min_forward_speed, command_cfg.min_lateral_speed,
                                   -command_cfg.max_yaw_rate], device=self.device)
        self.upper = torch.tensor([command_cfg.max_forward_speed, command_cfg.max_lateral_speed,
                                   command_cfg.max_yaw_rate], device=self.device)
        self.sigma = torch.tensor(self.cfg.noise_std, device=self.device)
        if forward_only:
            self.sigma[1] = 0
        self.mean = torch.zeros(num_envs, self.horizon, 3, device=self.device)
        self.previous = torch.zeros(num_envs, 3, device=self.device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.generators = ([torch.Generator(device=self.device).manual_seed(seed + i * 100003)
                            for i in range(num_envs)] if independent_rng else None)
        self.last_pose = torch.zeros(num_envs, self.horizon, 4, device=self.device)
        self.last_risk = torch.zeros(num_envs, device=self.device)
        self.last_invalid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.reset_statistics()

    def reset_statistics(self):
        self.calls = self.decisions = self.fallbacks = self.all_invalid = 0
        self.seconds = 0.0
        self.latencies = []

    def reset(self, ids=None, *, seed=None):
        ids = slice(None) if ids is None else ids
        self.mean[ids] = 0
        self.previous[ids] = 0
        self.initialized[ids] = False
        if seed is not None:
            self.generator.manual_seed(seed)
            if self.generators is not None:
                for index, generator in enumerate(self.generators):
                    generator.manual_seed(seed + index * 100003)

    def seed_streams(self, seeds):
        """Independent trial streams do not depend on which other trials finish."""
        if self.generators is None or len(seeds) != len(self.generators):
            raise ValueError("One seed per independent MPPI stream is required.")
        for generator, seed in zip(self.generators, seeds):
            generator.manual_seed(int(seed))

    def statistics(self):
        latency = torch.tensor(self.latencies or [0.0])
        return {"calls": self.calls, "decisions": self.decisions, "seconds": self.seconds,
                "mean_batch_ms": 1000 * self.seconds / max(1, self.calls),
                "p95_batch_ms": 1000 * float(torch.quantile(latency, .95)),
                "mean_fallbacks": self.fallbacks, "all_invalid_stops": self.all_invalid}

    def _score(self, context, plans, env_rows, goals, invalid_map, previous, map_transform=None):
        costs, risks = [], []
        for start in range(0, len(plans), self.cfg.inference_chunk):
            commands = plans[start:start + self.cfg.inference_chunk]
            rows = env_rows[start:start + self.cfg.inference_chunk]
            prediction = self.model.predict_from_context(context[rows], commands)
            xy = prediction["future_pose"][..., :2]
            probability = prediction["collision_logits"].sigmoid().cummax(dim=1).values
            distance = torch.linalg.vector_norm(xy - goals[rows, None], dim=-1)
            change = torch.diff(torch.cat((previous[rows, None], commands), dim=1), dim=1)
            # NaNs must never become favorable costs or invalid tensor indices.
            finite = torch.isfinite(xy).all(dim=(1, 2)) & torch.isfinite(probability).all(dim=1)
            map_xy = xy
            if map_transform is not None:
                # The ray caster is attached to torso_link, while target poses
                # are relative to the root. Account for waist yaw/translation.
                transform = map_transform[rows]
                cosine, sine = transform[:, 2:3].cos(), transform[:, 2:3].sin()
                map_xy = torch.stack((cosine * xy[..., 0] - sine * xy[..., 1],
                                      sine * xy[..., 0] + cosine * xy[..., 1]), dim=-1) + transform[:, None, :2]
            safe_xy = torch.nan_to_num(map_xy, nan=1e6, posinf=1e6, neginf=-1e6)
            col = torch.round((safe_xy[..., 0] - self.cfg.map_x_min) / self.cfg.map_resolution).long()
            row = torch.round((self.cfg.map_y_max - safe_xy[..., 1]) / self.cfg.map_resolution).long()
            height, width = invalid_map.shape[-2:]
            outside = (col < 0) | (col >= width) | (row < 0) | (row >= height)
            invalid = outside | invalid_map[rows[:, None], 0, row.clamp(0, height - 1), col.clamp(0, width - 1)]
            risk = probability[:, -1]
            cost = (self.cfg.goal_terminal_weight * distance[:, -1]
                    + self.cfg.goal_path_weight * distance.mean(dim=1)
                    + self.cfg.collision_weight * probability.mean(dim=1)
                    + self.cfg.high_risk_weight * (risk >= self.cfg.high_risk_threshold)
                    + self.cfg.smoothness_weight * change.square().sum(dim=-1).mean(dim=1)
                    + self.cfg.map_penalty * invalid.float().mean(dim=1))
            # Outside later in the horizon is penalized. A missing first foothold
            # is inadmissible; if every candidate fails, execute an explicit stop.
            cost = cost.masked_fill(~finite | invalid[:, 0] | ~torch.isfinite(cost), torch.inf)
            costs.append(cost)
            risks.append(torch.nan_to_num(risk, nan=1.0))
        return torch.cat(costs), torch.cat(risks)

    @torch.inference_mode()
    def plan(self, ids, state, proprio, height, invalid_map, goal_local, *, map_transform=None, align_mask=None):
        if not len(ids):
            return self.mean[ids]
        if self.model.training:
            raise RuntimeError("Planning requires an immutable eval-mode FDM snapshot.")
        # Synchronize once per decision batch for useful timing, not per candidate.
        if self.track_statistics and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic() if self.track_statistics else None
        context = self.model.encode_context(state, proprio, height.float())
        batch, population = len(ids), self.cfg.candidates
        nominal = torch.zeros(batch, self.horizon, 3, device=self.device)
        angle = torch.atan2(goal_local[:, 1], goal_local[:, 0])
        nominal[..., 0] = (torch.linalg.vector_norm(goal_local, dim=-1) / 3).clamp(0.1, .8)[:, None]
        nominal[..., 0] *= angle.cos().clamp_min(0)[:, None]
        nominal[..., 2] = (angle * .8).clamp(-1, 1)[:, None]
        nominal = nominal.clamp(self.lower, self.upper)
        shifted = torch.cat((self.mean[ids, 1:], self.mean[ids, -1:]), dim=1)
        mean = torch.where(self.initialized[ids, None, None], shifted, nominal)
        mean = mean.clamp(self.lower, self.upper)
        rows = torch.arange(batch, device=self.device)
        population_rows = rows.repeat_interleave(population)
        last_invalid = torch.zeros(batch, dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.iterations):
            if self.generators is None:
                noise = torch.randn(batch, population, self.horizon, 3, device=self.device, generator=self.generator)
            else:
                noise = torch.stack([torch.randn(population, self.horizon, 3, device=self.device,
                                                 generator=self.generators[i]) for i in ids.tolist()])
            beta = self.cfg.noise_correlation
            for step in range(1, self.horizon):
                noise[:, :, step] = beta * noise[:, :, step - 1] + math.sqrt(1 - beta * beta) * noise[:, :, step]
            candidates = (mean[:, None] + noise * self.sigma).clamp(self.lower, self.upper)
            candidates[:, 0] = mean
            candidates[:, 1] = 0  # Explicit stop candidate; command bounds include zero.
            candidates[:, 2] = nominal
            if self.turn_candidates:
                candidates[:, 3:8] = 0
                rate = self.upper[2]
                candidates[:, 3, :, 2] = rate
                candidates[:, 4, :, 2] = -rate
                steps = torch.arange(self.horizon, device=self.device)[None]
                for candidate, direction in ((5, 1), (6, -1)):
                    turn_angle = torch.remainder(angle * direction, 2 * math.pi)
                    turn_steps = torch.ceil(turn_angle / (rate * self.model.cfg.command_timestep)).clamp(1, self.horizon - 1)
                    turning = steps < turn_steps[:, None]
                    candidates[:, candidate, :, 2] = torch.where(turning, direction * rate, 0.)
                    candidates[:, candidate, :, 0] = torch.where(turning, 0., self.upper[0] * .65)
                candidates[:, 7, :, 0] = self.upper[0] * .65
            if align_mask is not None:
                candidates[align_mask, :, 0, 0] = 0
            costs, risks = self._score(context, candidates.flatten(0, 1), population_rows,
                                       goal_local, invalid_map, self.previous[ids], map_transform)
            costs, risks = costs.view(batch, population), risks.view(batch, population)
            best_cost, best_idx = costs.min(dim=1)
            last_invalid = ~torch.isfinite(best_cost)
            logits = -(costs - best_cost.masked_fill(last_invalid, 0)[:, None]) / self.cfg.temperature
            logits[last_invalid] = 0
            weights = logits.softmax(dim=1)
            proposal = (weights[..., None, None] * candidates).sum(dim=1).clamp(self.lower, self.upper)
            mean_cost, mean_risk = self._score(context, proposal, rows, goal_local, invalid_map, self.previous[ids], map_transform)
            best = candidates[rows, best_idx]
            unsafe_mean = (mean_risk >= self.cfg.high_risk_threshold) & (risks[rows, best_idx] < self.cfg.high_risk_threshold)
            fallback = ~torch.isfinite(mean_cost) | (mean_cost > best_cost) | unsafe_mean
            mean = torch.where(fallback[:, None, None], best, proposal)
            mean[last_invalid] = 0
            if self.track_statistics:
                self.fallbacks += int((fallback & ~last_invalid).sum())
        self.mean[ids] = mean
        self.previous[ids] = mean[:, 0]
        self.initialized[ids] = True
        if self.track_statistics:
            self.all_invalid += int(last_invalid.sum())
        self.last_invalid[ids] = last_invalid
        if self.record_predictions:
            # Reuse encoded observations; save only the selected trajectory.
            selected = self.model.predict_from_context(context, mean)
            self.last_pose[ids] = selected["future_pose"]
            self.last_risk[ids] = selected["collision_logits"].sigmoid().amax(dim=1)
        if self.track_statistics:
            self.decisions += batch
            self.calls += 1
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.monotonic() - started
            self.seconds += elapsed
            self.latencies.append(elapsed)
        return mean
