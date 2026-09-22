"""Exact discrete clocks for the rollout's multiple sampling rates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlternatingHistoryClock:
    """Generate an average-rate history tick from fixed policy steps.

    At 50 Hz policy and 20 Hz history this yields policy step indices
    ``3, 5, 8, 10, ...``: alternating 0.06/0.04 second intervals and exactly
    ten samples per 0.5-second command interval.
    """

    policy_dt: float = 0.02
    history_dt: float = 0.05
    elapsed: float = 0.0

    def reset(self) -> None:
        self.elapsed = 0.0

    def advance(self) -> bool:
        self.elapsed += self.policy_dt
        if self.elapsed + 1.0e-10 >= self.history_dt:
            self.elapsed -= self.history_dt
            return True
        return False


def history_tick_indices(num_policy_steps: int, policy_dt: float = 0.02, history_dt: float = 0.05) -> list[int]:
    """Return one-based policy-step indices at which history is sampled."""

    clock = AlternatingHistoryClock(policy_dt=policy_dt, history_dt=history_dt)
    return [step for step in range(1, num_policy_steps + 1) if clock.advance()]
