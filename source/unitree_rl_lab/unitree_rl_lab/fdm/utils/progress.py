"""Wall-clock progress for long collection, indexing and training stages."""

from __future__ import annotations

import time
from collections.abc import Callable


class ProgressLogger:
    """Print immediately, periodically, and on completion without buffering stdout."""

    def __init__(self, label: str, total: int, *, unit: str, interval_s: float = 10.0) -> None:
        self.label = label
        self.total = total
        self.unit = unit
        self.interval_s = interval_s
        self.started = time.monotonic()
        self.last_log = self.started
        self.update(0, force=True)

    def update(self, completed: int, *, detail: Callable[[], str] | None = None, force: bool = False) -> None:
        now = time.monotonic()
        if self.interval_s <= 0 or (not force and now - self.last_log < self.interval_s):
            return
        elapsed = now - self.started
        rate = completed / elapsed if elapsed > 0 else 0.0
        rate_text = f"{rate:.3g}" if 0 < rate < 0.01 else f"{rate:.2f}"
        eta = f"{max(0, self.total - completed) / rate:.0f}s" if rate > 0 else "pending"
        suffix = f" {detail()}" if detail is not None else ""
        print(
            f"[FDM] {self.label}: {self.unit}={completed}/{self.total} "
            f"elapsed={elapsed:.1f}s rate={rate_text}/s eta={eta}{suffix}",
            flush=True,
        )
        self.last_log = now
