"""Best-effort available RAM for preflight checks before caching sample tensors."""

import json
from pathlib import Path


def _key_values(path: str | Path) -> dict[str, int]:
    try:
        result = {}
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                result[parts[0].rstrip(":")] = int(parts[1]) * (1024 if parts[-1] == "kB" else 1)
        return result
    except OSError:
        return {}


def _cgroup_directories() -> list[tuple[Path, int]]:
    candidates = [(Path("/sys/fs/cgroup"), 2), (Path("/sys/fs/cgroup/memory"), 1)]
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        version = 2 if not controllers else 1
        if version == 1 and "memory" not in controllers.split(","):
            continue
        # Some container namespaces expose paths above the visible mount root.
        if ".." in Path(relative).parts:
            continue
        base = Path("/sys/fs/cgroup" if version == 2 else "/sys/fs/cgroup/memory")
        directory = base / relative.lstrip("/")
        while directory != base:
            candidates.append((directory, version))
            directory = directory.parent
    return list(dict.fromkeys(candidates))


def memory_snapshot() -> dict:
    """Estimate allocatable RAM, allowing conservative clean inactive file reclaim.

    memory.current includes page cache: limit-current alone can report low RAM
    after shard I/O even when anonymous allocations still fit. Do not count all
    file pages, shared memory, dirty/writeback pages, or swap as available.
    """
    host = _key_values("/proc/meminfo").get("MemAvailable")
    bounds = [] if host is None else [host]
    groups = []
    for directory, version in _cgroup_directories():
        limit_name, current_name = (
            ("memory.max", "memory.current") if version == 2
            else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        )
        try:
            limit = int((directory / limit_name).read_text().strip())
            current = int((directory / current_name).read_text().strip())
        except (OSError, ValueError):
            continue
        stats = _key_values(directory / "memory.stat")

        def stat(name: str) -> int:
            return stats.get(f"total_{name}", stats.get(name, 0)) if version == 1 else stats.get(name, 0)

        file_bytes = stat("file" if version == 2 else "cache")
        inactive = stat("inactive_file")
        reclaimable = max(0, min(inactive, file_bytes - stat("shmem"))
                          - stat("file_dirty" if version == 2 else "dirty")
                          - stat("file_writeback" if version == 2 else "writeback"))
        reclaimable = min(current, reclaimable)
        available = max(0, min(limit, limit - current + reclaimable))
        bounds.append(available)
        groups.append({
            "path": str(directory), "limit": limit, "current": current,
            "anon": stat("anon" if version == 2 else "rss"), "file": file_bytes,
            "inactive_file": inactive, "reclaimable_estimate": reclaimable, "available": available,
        })
    return {
        "host_available": host, "process_rss": _key_values("/proc/self/status").get("VmRSS"),
        "available": min(bounds) if bounds else None, "cgroups": groups,
    }


def available_memory_bytes() -> int | None:
    return memory_snapshot()["available"]


def log_memory(stage: str, *, collector=None, log=None) -> None:
    snapshot = memory_snapshot()
    pending = [builder for builder in collector.builders if builder is not None] if collector else []
    record = {
        "event": "memory", "stage": stage, "bytes": snapshot,
        "pending_episodes": getattr(collector, "pending_episodes", len(pending)),
        "pending_frames": getattr(collector, "pending_frames", sum(builder.num_frames for builder in pending)),
    }
    gib = lambda value: round(value / 2**30, 3) if value is not None else None
    display = {f"{key}_gib": gib(snapshot[key]) for key in ("process_rss", "host_available", "available")}
    display["cgroups"] = [
        {key if key == "path" else f"{key}_gib": value if key == "path" else gib(value)
         for key, value in group.items()} for group in snapshot["cgroups"]
    ]
    print(
        f"[FDM] Memory stage={stage} pending_episodes={record['pending_episodes']} "
        f"pending_frames={record['pending_frames']} {json.dumps(display)}", flush=True,
    )
    if log is not None:
        log.event(record)
