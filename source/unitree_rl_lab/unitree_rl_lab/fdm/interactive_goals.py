"""CPU-only goal sampling and edge-triggered input for the interactive demo."""

from dataclasses import dataclass
import math
import threading

import numpy as np


@dataclass(frozen=True)
class GoalSamplingCfg:
    radius: float = 5.0
    min_distance: float = 2.0
    max_attempts: int = 256

    def validate(self):
        if not (math.isfinite(self.radius) and math.isfinite(self.min_distance)
                and 0 < self.min_distance < self.radius):
            raise ValueError("Require 0 < --goal-min-distance < --goal-radius, both finite.")
        if self.max_attempts < 1:
            raise ValueError("Goal sampling needs at least one attempt.")


class LocalGoalSampler:
    def __init__(self, scene, cfg=GoalSamplingCfg(), *, seed=0):
        cfg.validate()
        self.scene, self.cfg = scene, cfg
        # An independent stream: planner resets never repeat goal samples.
        self.rng = np.random.default_rng(seed)

    def candidates(self, center):
        center = np.asarray(center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Goal sampling centre must be finite XY coordinates.")
        radius = np.sqrt(self.rng.uniform(self.cfg.min_distance ** 2, self.cfg.radius ** 2,
                                          self.cfg.max_attempts))
        angle = self.rng.uniform(-math.pi, math.pi, self.cfg.max_attempts)
        return center + radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))

    def sample(self, center):
        """Uniform-area rejection sampling; return None after a bounded search."""
        for xy in self.candidates(center):
            try:
                z = self.scene.validate_goal(xy, reject_obstacle_tops=True)
            except ValueError:
                continue
            if math.isfinite(z):
                return (float(xy[0]), float(xy[1]), float(z))
        return None


class SpaceRequest:
    """Only a fresh press during waiting requests a goal; nothing is queued while moving."""

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = self._pressed = self._pending = False

    def enable(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
            if not enabled:
                self._pending = False

    def key(self, pressed):
        with self._lock:
            if pressed and not self._pressed and self._enabled:
                self._pending = True
            self._pressed = bool(pressed)

    def consume(self):
        with self._lock:
            result = self._pending and self._enabled
            self._pending = False
            return result
