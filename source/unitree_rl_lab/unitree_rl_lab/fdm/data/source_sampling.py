"""Source quotas with a hard per-window repetition cap for stage two."""

from __future__ import annotations

import torch


def select_mixed_pool(dataset, count, *, mppi_fraction, seed, collision_fraction=.35,
                      low_motion_fraction=.10, max_repeats=4):
    if dataset.schema_version != 4 or dataset.sample_pool_report is not None or dataset.cache_mode != "shard":
        raise ValueError("Select one stage-two pool before caching.")
    if count < 1 or max_repeats < 1 or not 0 <= mppi_fraction <= 1:
        raise ValueError("Invalid source pool settings.")
    if min(collision_fraction, low_motion_fraction) < 0 or collision_fraction + low_motion_fraction > 1:
        raise ValueError("Invalid collision/low-motion fractions.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.tensor([item.command_source for item in dataset.indices])
    collision = dataset.collision_flags
    low = dataset.low_motion_flags & ~collision
    requested_mppi = round(count * mppi_fraction)
    selected, reports = [], {}
    for label, requested in ((0, count - requested_mppi), (1, requested_mppi)):
        groups = [torch.nonzero((source == label) & category).flatten()
                  for category in (collision, low, ~collision & ~low)]
        capacities = [len(group) * max_repeats for group in groups]
        quota = min(requested, sum(capacities))
        desired = [round(quota * collision_fraction)]
        desired.append(min(round(quota * low_motion_fraction), quota - desired[0]))
        desired.append(quota - sum(desired))
        allocations = [min(want, capacity) for want, capacity in zip(desired, capacities)]
        # A rare collision stratum cannot be repeated without limit. Move its
        # unused allocation to available strata and report the achieved mix.
        remaining = quota - sum(allocations)
        for index in (2, 1, 0):
            extra = min(remaining, capacities[index] - allocations[index])
            allocations[index] += extra
            remaining -= extra
        source_selected = []
        for group, size in zip(groups, allocations):
            while size:
                take = min(size, len(group))
                source_selected.extend(group[torch.randperm(len(group), generator=generator)[:take]].tolist())
                size -= take
        selected.extend(source_selected)
        reports["random" if label == 0 else "mppi"] = {
            "available": int((source == label).sum()), "requested": requested, "selected": len(source_selected),
            "unique": len(set(source_selected)), "collision": allocations[0], "low_motion_noncollision": allocations[1],
            "collision_fraction": allocations[0] / max(1, len(source_selected)),
            "shortfall": requested - len(source_selected),
        }
    if not selected:
        raise ValueError("No windows available under the requested source quotas.")
    selected.sort()
    natural_count = len(dataset)
    dataset.indices = [dataset.indices[index] for index in selected]
    dataset._cache.clear()
    dataset.sample_pool_report = {
        "source_windows": natural_count, "requested_windows": count, "sampled_windows": len(selected),
        "unique_windows": len(set(selected)), "repeated_windows": len(selected) - len(set(selected)),
        "seed": seed, "max_repeats": max_repeats, "sources": reports,
        "collision_fraction": float(dataset.collision_flags.float().mean()),
        "low_motion_fraction": float(dataset.low_motion_flags.float().mean()),
    }
    print(f"[FDM] Mixed sample pool: {dataset.sample_pool_report}", flush=True)
    return dataset.sample_pool_report
