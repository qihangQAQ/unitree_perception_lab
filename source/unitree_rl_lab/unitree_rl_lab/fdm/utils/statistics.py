"""Small, mergeable CPU accumulators for diagnostic reports."""

from __future__ import annotations

import torch


class RunningMoments:
    """Finite-value population statistics, with nulls for empty dimensions."""

    def __init__(self, width: int = 1) -> None:
        self.width = width
        self.count = torch.zeros(width, dtype=torch.long)
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.m2 = torch.zeros(width, dtype=torch.float64)
        self.minimum = torch.full((width,), torch.inf, dtype=torch.float64)
        self.maximum = torch.full((width,), -torch.inf, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1, self.width)
        if not len(values):
            return
        finite = torch.isfinite(values)
        count = finite.sum(dim=0)
        safe = torch.where(finite, values, 0.0)
        mean = safe.sum(dim=0) / count.clamp_min(1)
        m2 = torch.where(finite, (values - mean).square(), 0.0).sum(dim=0)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta.square() * self.count * count / total.clamp_min(1)
        self.mean += delta * count / total.clamp_min(1)
        self.count = total
        self.minimum = torch.minimum(self.minimum, values.masked_fill(~finite, torch.inf).amin(dim=0))
        self.maximum = torch.maximum(self.maximum, values.masked_fill(~finite, -torch.inf).amax(dim=0))

    def result(self) -> dict:
        def export(values):
            output = [float(v) if int(n) else None for v, n in zip(values, self.count)]
            return output[0] if self.width == 1 else output

        counts = self.count.tolist()
        return {
            "count": counts[0] if self.width == 1 else counts,
            "mean": export(self.mean),
            "std": export((self.m2 / self.count.clamp_min(1)).clamp_min(0).sqrt()),
            "min": export(self.minimum),
            "max": export(self.maximum),
        }


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None
