"""Batched wall-plane paths, sampled once per episode and parameterized by arc length."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PaintingPathCfg:
    spacing: float = 0.01
    height_range: tuple[float, float] = (1.1, 1.4)
    probabilities: tuple[float, float, float, float] = (0.30, 0.20, 0.25, 0.25)
    connector_length: tuple[float, float] = (0.4, 0.7)
    straight_length: tuple[float, float] = (0.35, 0.8)
    wave_length: tuple[float, float] = (0.6, 1.2)
    shape_height: tuple[float, float] = (0.18, 0.28)
    samples_per_piece: int = 193

    def validate(self):
        if self.spacing <= 0 or self.samples_per_piece < 32:
            raise ValueError("Path spacing must be positive and samples_per_piece must be at least 32.")
        lo, hi = self.height_range
        if not (0 < lo < hi):
            raise ValueError("Invalid painting height range.")
        if len(self.probabilities) != 4 or any(p < 0 for p in self.probabilities):
            raise ValueError("Expected four nonnegative primitive probabilities.")
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Primitive probabilities must sum to one.")
        for bounds in (self.connector_length, self.straight_length, self.wave_length, self.shape_height):
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"Invalid geometry bounds: {bounds}")
        if self.shape_height[1] > hi - lo + 1e-6:
            raise ValueError("Shapes must fit inside the configured height band.")


@dataclass
class PaintingPaths:
    points: torch.Tensor  # [N, K, 6]: surface xyz and outward normal
    arc: torch.Tensor  # [N, K], padded with the true total length
    counts: torch.Tensor  # [N], includes both endpoints
    lengths: torch.Tensor
    primitive_counts: torch.Tensor  # [N, 4], counts sampled primitives whose connectors start before L


def _uniform(shape, bounds, device, generator):
    return torch.rand(shape, device=device, generator=generator) * (bounds[1] - bounds[0]) + bounds[0]


def _rounded_square(u: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Closed rounded square, starting on its bottom edge with a +y tangent."""
    radius = 0.2 * side
    straight = side - 2 * radius
    quarter = 0.5 * math.pi * radius
    perimeter = 4 * (straight + quarter)
    distance = u * perimeter
    result = torch.zeros(*distance.shape, 2, device=u.device)
    start = torch.zeros_like(side)
    for edge in range(4):
        t = (distance - start).clamp_min(0).minimum(straight)
        if edge == 0:
            y, z = radius + t, torch.zeros_like(t)
        elif edge == 1:
            y, z = side.expand_as(t), radius + t
        elif edge == 2:
            y, z = side - radius - t, side.expand_as(t)
        else:
            y, z = torch.zeros_like(t), side - radius - t
        mask = (distance >= start) & (distance <= start + straight)
        result = torch.where(mask[..., None], torch.stack((y, z), -1), result)
        start = start + straight
        phi = ((distance - start) / radius).clamp(0, math.pi / 2)
        if edge == 0:
            y, z = side - radius + radius * phi.sin(), radius - radius * phi.cos()
        elif edge == 1:
            y, z = side - radius + radius * phi.cos(), side - radius + radius * phi.sin()
        elif edge == 2:
            y, z = radius - radius * phi.sin(), side - radius + radius * phi.cos()
        else:
            y, z = radius - radius * phi.cos(), radius - radius * phi.sin()
        mask = (distance >= start) & (distance <= start + quarter + 1e-6)
        result = torch.where(mask[..., None], torch.stack((y, z), -1), result)
        start = start + quarter
    result[..., 0] -= radius
    result[:, 0] = 0
    result[:, -1] = 0
    return result


def sample_paths(
    lengths: torch.Tensor,
    heading: torch.Tensor,
    cfg: PaintingPathCfg,
    *,
    capacity: int | None = None,
    generator: torch.Generator | None = None,
) -> PaintingPaths:
    """Generate mixed primitives and smooth height-changing connectors on the tensor device.

    ``heading`` is -1 for right and +1 for left. Analytic pieces meet with a
    horizontal tangent; no later smoothing is allowed to alter the timing budget.
    """
    cfg.validate()
    if lengths.ndim != 1 or heading.shape != lengths.shape or len(lengths) == 0:
        raise ValueError("Expected nonempty, equally shaped length and heading vectors.")
    if not torch.isfinite(lengths).all() or (lengths <= 0).any() or (heading.abs() != 1).any():
        raise ValueError("Lengths must be positive and headings must be +/-1.")
    device, n = lengths.device, len(lengths)
    u = torch.linspace(0, 1, cfg.samples_per_piece, device=device)[None]
    lo, hi = cfg.height_range
    pos = torch.stack((torch.zeros_like(lengths), _uniform((n,), (lo, hi), device, generator)), -1)
    pieces = [pos[:, None].clone()]
    total = torch.zeros_like(lengths)
    primitive_counts = torch.zeros(n, 4, device=device, dtype=torch.long)
    probabilities = torch.tensor(cfg.probabilities, device=device)
    # Every block contributes at least the connector's horizontal displacement.
    max_blocks = math.ceil(float(lengths.max()) / cfg.connector_length[0]) + 1
    for _ in range(max_blocks):
        active = total < lengths
        if not active.any():
            break
        kind = torch.multinomial(probabilities, n, replacement=True, generator=generator)
        primitive_counts.scatter_add_(1, kind[:, None], active.long()[:, None])
        height = _uniform((n, 1), cfg.shape_height, device, generator)
        height = torch.where(kind[:, None] == 0, 0.0, height)
        baseline = lo + torch.rand(n, 1, device=device, generator=generator) * (hi - lo - height)
        advance = _uniform((n, 1), cfg.connector_length, device, generator)
        ease = u**3 * (10 - 15 * u + 6 * u**2)
        connector = torch.stack(
            (pos[:, 0:1] + advance * u, pos[:, 1:2] + (baseline - pos[:, 1:2]) * ease), -1
        )
        total += torch.linalg.vector_norm(torch.diff(connector, dim=1), dim=-1).sum(-1)
        pieces.append(connector[:, 1:])
        pos = connector[:, -1]

        straight_span = _uniform((n, 1), cfg.straight_length, device, generator)
        wave_span = _uniform((n, 1), cfg.wave_length, device, generator)
        radius = height / 2
        angle = 2 * math.pi * u
        y = straight_span * u
        z = torch.zeros_like(y)
        y = torch.where(kind[:, None] == 1, wave_span * u, y)
        z = torch.where(kind[:, None] == 1, radius * (1 - angle.cos()), z)
        y = torch.where(kind[:, None] == 2, radius * angle.sin(), y)
        z = torch.where(kind[:, None] == 2, radius * (1 - angle.cos()), z)
        square = _rounded_square(u, height.clamp_min(cfg.shape_height[0]))
        y = torch.where(kind[:, None] == 3, square[..., 0], y)
        z = torch.where(kind[:, None] == 3, square[..., 1], z)
        primitive = torch.stack((y, z), -1) + pos[:, None]
        total += torch.linalg.vector_norm(torch.diff(primitive, dim=1), dim=-1).sum(-1)
        pieces.append(primitive[:, 1:])
        pos = primitive[:, -1]
    dense_yz = torch.cat(pieces, dim=1)
    dense_yz[..., 0] *= heading[:, None]
    dense_yz[..., 1].clamp_(lo, hi)
    dense = torch.zeros(n, dense_yz.shape[1], 3, device=device)
    dense[..., 1:] = dense_yz
    dense_arc = torch.cat(
        (torch.zeros(n, 1, device=device), torch.linalg.vector_norm(torch.diff(dense, dim=1), dim=-1).cumsum(-1)),
        dim=1,
    )
    counts = torch.ceil(lengths / cfg.spacing).long() + 1
    needed = int(counts.max())
    capacity = needed if capacity is None else capacity
    if capacity < needed:
        raise ValueError(f"Path capacity {capacity} is smaller than required {needed}.")
    queries = (torch.arange(capacity, device=device) * cfg.spacing)[None].minimum(lengths[:, None])
    xyz, _ = interpolate_path(dense, dense_arc, queries)
    points = torch.zeros(n, capacity, 6, device=device)
    points[..., :3] = xyz
    points[..., 3] = -1
    return PaintingPaths(points, queries.contiguous(), counts, lengths.clone(), primitive_counts)


def interpolate_path(points: torch.Tensor, arc: torch.Tensor, queries: torch.Tensor):
    """Interpolate ordered arc queries, including repeated endpoint padding.

    Returns positions and unit tangents. Queries never project to a spatially
    nearest point, so intersections and closed primitives cannot change progress.
    """
    queries = queries.clamp_min(0).minimum(arc[:, -1:]).contiguous()
    upper = torch.searchsorted(arc.contiguous(), queries, right=False).clamp(1, arc.shape[1] - 1)
    lower = upper - 1
    xyz = points[..., :3]
    p0 = xyz.gather(1, lower[..., None].expand(-1, -1, 3))
    p1 = xyz.gather(1, upper[..., None].expand(-1, -1, 3))
    s0, s1 = arc.gather(1, lower), arc.gather(1, upper)
    fraction = ((queries - s0) / (s1 - s0).clamp_min(1e-8)).clamp(0, 1)
    delta = p1 - p0
    tangent = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    return p0 + fraction[..., None] * delta, tangent


def import_surface_path(points: torch.Tensor, cfg: PaintingPathCfg, capacity: int) -> PaintingPaths:
    """Validate and resample one external [K,6] path in the environment's wall frame."""
    cfg.validate()
    if points.ndim != 2 or points.shape[1] != 6 or len(points) < 2 or not torch.isfinite(points).all():
        raise ValueError("An external path must contain at least two finite xyz+normal points.")
    normal = torch.tensor([-1.0, 0.0, 0.0], device=points.device)
    if not torch.allclose(points[:, 3:], normal.expand_as(points[:, 3:]), atol=1e-4, rtol=0):
        raise ValueError("This policy requires a common outward normal (-1,0,0).")
    if points[:, 0].abs().max() > 1e-4:
        raise ValueError("External surface points must lie on wall-frame x=0.")
    if (points[:, 2] < cfg.height_range[0]).any() or (points[:, 2] > cfg.height_range[1]).any():
        raise ValueError("External trajectory is outside the configured height band.")
    distances = torch.linalg.vector_norm(torch.diff(points[:, :3], dim=0), dim=-1)
    keep = torch.cat((torch.ones(1, device=points.device, dtype=torch.bool), distances > 1e-7))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("External path has zero length.")
    distances = torch.linalg.vector_norm(torch.diff(points[:, :3], dim=0), dim=-1)
    arc = torch.cat((torch.zeros(1, device=points.device), distances.cumsum(0)))[None]
    lengths = arc[:, -1]
    counts = torch.ceil(lengths / cfg.spacing).long() + 1
    if int(counts[0]) > capacity:
        raise ValueError("External path exceeds the configured path capacity.")
    queries = (torch.arange(capacity, device=points.device) * cfg.spacing)[None].minimum(lengths[:, None])
    xyz, _ = interpolate_path(points[None], arc, queries)
    output = torch.cat((xyz, normal.expand(1, capacity, 3)), dim=-1)
    return PaintingPaths(output, queries.contiguous(), counts, lengths, torch.zeros(1, 4, device=points.device))
