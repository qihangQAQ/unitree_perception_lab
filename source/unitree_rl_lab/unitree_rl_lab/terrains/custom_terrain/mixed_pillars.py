"""A flat mesh field with randomly scattered boxes, cylinders and triangular prisms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .mixed_pillars_cfg import MeshMixedPillarsTerrainCfg


def _footprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest distance between two ordered, convex XY polygons (zero if overlapping)."""
    edges = np.concatenate((np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    pa, pb = a @ axes.T, b @ axes.T
    separated = (pa.max(axis=0) < pb.min(axis=0)) | (pb.max(axis=0) < pa.min(axis=0))
    if not separated.any():
        return 0.0

    def vertex_edge_distance(points, polygon):
        edge = np.roll(polygon, -1, axis=0) - polygon
        delta = points[:, None] - polygon[None]
        fraction = (delta * edge).sum(axis=-1) / (edge * edge).sum(axis=-1)
        nearest = polygon[None] + fraction.clip(0, 1)[..., None] * edge
        return np.linalg.norm(points[:, None] - nearest, axis=-1).min()

    return float(min(vertex_edge_distance(a, b), vertex_edge_distance(b, a)))


def mixed_pillars_terrain(
    difficulty: float, cfg: MeshMixedPillarsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one tile in [0, size]; difficulty does not alter the random height range."""
    size = np.asarray(cfg.size, dtype=float)
    counts = (cfg.num_boxes, cfg.num_cylinders, cfg.num_triangular_prisms)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Mixed-pillar terrain size must contain two finite positive lengths.")
    if any(not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("Pillar counts must be nonnegative integers.")
    for name in ('height_range', 'box_length_range', 'box_width_range',
                 'cylinder_radius_range', 'triangle_side_range'):
        bounds = np.asarray(getattr(cfg, name), dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"{name} must be a positive, ordered pair.")
    if not np.isfinite(cfg.size_sampling_power) or cfg.size_sampling_power <= 0:
        raise ValueError("size_sampling_power must be finite and positive.")
    if not np.isfinite([cfg.min_gap, cfg.edge_margin, cfg.floor_thickness]).all() \
            or min(cfg.min_gap, cfg.edge_margin) < 0 or cfg.floor_thickness <= 0:
        raise ValueError("Margins must be finite and nonnegative; floor thickness must be positive.")
    if not isinstance(cfg.cylinder_sections, int) or cfg.cylinder_sections < 8:
        raise ValueError("Use at least 8 cylinder sections.")
    if any(not isinstance(n, int) or n < 1 for n in (cfg.max_placement_attempts, cfg.max_layout_attempts)):
        raise ValueError("Placement and layout attempt limits must be positive integers.")

    rng = np.random.default_rng(cfg.seed)

    def sample_cross_section(bounds: tuple[float, float]) -> float:
        # Retain the full size range, but make large cross-sections less common.
        low, high = bounds
        return float(low + (high - low) * rng.random() ** cfg.size_sampling_power)

    kinds = np.repeat(('box', 'cylinder', 'triangular_prism'), counts)
    rng.shuffle(kinds)
    floor = trimesh.creation.box(extents=(*size, cfg.floor_thickness))
    floor.apply_translation((*(size / 2), -cfg.floor_thickness / 2))
    pillars = []

    for kind in kinds:
        height = float(rng.uniform(*cfg.height_range))
        yaw = float(rng.uniform(-np.pi, np.pi))
        if kind == 'box':
            length = sample_cross_section(cfg.box_length_range)
            width = sample_cross_section(cfg.box_width_range)
            mesh = trimesh.creation.box(extents=(length, width, height))
            footprint = np.array([[-length, -width], [length, -width],
                                  [length, width], [-length, width]]) / 2
        else:
            if kind == 'cylinder':
                radius, sections = sample_cross_section(cfg.cylinder_radius_range), cfg.cylinder_sections
            else:
                radius, sections = sample_cross_section(cfg.triangle_side_range) / np.sqrt(3), 3
            # A cylinder with three sides is a closed triangular prism, with a flat top.
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
            angles = np.arange(sections) * (2 * np.pi / sections)
            footprint = radius * np.column_stack((np.cos(angles), np.sin(angles)))

        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        footprint = footprint @ rotation.T
        lower = cfg.edge_margin - footprint.min(axis=0)
        upper = size - cfg.edge_margin - footprint.max(axis=0)
        if (upper < lower).any():
            raise ValueError(f"A {kind} does not fit inside the configured terrain margins.")
        mesh.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0., 0., 1.)))
        mesh.apply_translation((0., 0., height / 2))
        mesh.metadata['pillar_type'] = str(kind)
        pillars.append((mesh, footprint, lower, upper))

    # Place larger footprints first, still at continuously sampled random positions.
    # Retry the entire layout if an unlucky early placement blocks the remaining pillars.
    pillars.sort(key=lambda p: float(np.prod(np.ptp(p[1], axis=0))), reverse=True)
    best_count = 0
    for _ in range(cfg.max_layout_attempts):
        footprints, centers = [], []
        bounds_min = np.empty((len(pillars), 2))
        bounds_max = np.empty_like(bounds_min)
        for _, footprint, lower, upper in pillars:
            for _ in range(cfg.max_placement_attempts):
                center = rng.uniform(lower, upper)
                candidate = footprint + center
                lo, hi = candidate.min(axis=0), candidate.max(axis=0)
                n = len(footprints)
                gaps = np.maximum(np.maximum(bounds_min[:n] - hi, lo - bounds_max[:n]), 0.)
                nearby = np.flatnonzero((gaps * gaps).sum(axis=1) <= cfg.min_gap**2)
                if all(_footprint_distance(candidate, footprints[i]) > cfg.min_gap for i in nearby):
                    bounds_min[n], bounds_max[n] = lo, hi
                    footprints.append(candidate)
                    centers.append(center)
                    break
            else:
                break
        best_count = max(best_count, len(centers))
        if len(centers) == len(pillars):
            break
    else:
        raise ValueError(
            f"Could only place {best_count}/{len(kinds)} pillars with min_gap={cfg.min_gap:g} m. "
            "Increase the terrain size or reduce the counts, pillar sizes or minimum gap."
        )
    meshes = [floor]
    for (mesh, _, _, _), center in zip(pillars, centers):
        mesh.apply_translation((*center, 0.))
        meshes.append(mesh)

    print(f"[Mixed Pillars] {size[0]:g}x{size[1]:g} m; boxes={counts[0]}, cylinders={counts[1]}, "
          f"triangular_prisms={counts[2]}; height={cfg.height_range} m; seed={cfg.seed}.", flush=True)
    # A geometric origin for the preview; it is not a validated robot spawn.
    return meshes, np.array([size[0] / 2, size[1] / 2, 0.])
