# Copyright (c) 2025, The Nav-Suite Project Developers.
# Copyright (c) 2026, Unitree RL Lab contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, metric mesh atlas inspired by FDM's nav-suite test terrains.

The cross pattern is ported from nav_tasks/terrains/single_object.py. The
four-pyramid arrangement, stair/ramp alternatives and DFS maze are adapted
from quad_stairs_terrain.py, stairs_ramp_terrain.py and random_maze_terrain.py.
This is a new composite layout, not a reproduction of the paper's test set.
See fdm/licenses/nav-suite-Apache-2.0.txt.

Coordinates: metres, Z up; X east, Y north; extent [-24, 24] x [-18, 18].
Only numpy and trimesh are needed to build/export the actual collision mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

SIZE = (48.0, 36.0)
TILE_SIZE = 12.0
BASE_Z = -0.06
COLORS = {
    "floor": "#dee3e5",
    "stairs": "#558dbd",
    "ramp": "#e4ad52",
    "platform": "#54a79a",
    "wall": "#bb7767",
    "pillar": "#7c8b9c",
    "rough": "#a293b8",
}
ZONE_SPECS = (
    ("A1", "起点 / 低矮起伏", "Spawn & rough ground"),
    ("A2", "混合柱阵", "Mixed pillars"),
    ("A3", "交叉障碍阵", "Cross pattern"),
    ("A4", "错位窄通道", "Staggered passages"),
    ("B1", "楼梯平台 / 平地绕行", "Stairs or bypass"),
    ("B2", "陡坡 / 常规楼梯", "Steep ramp or stairs"),
    ("B3", "高台阶 / 缓坡", "High steps or gentle ramp"),
    ("B4", "楼梯与墙体", "Stairs & walls"),
    ("C1", "坡道与墙体", "Ramps & walls"),
    ("C2", "多层平台", "Multi-level platform"),
    ("C3", "四向金字塔楼梯", "Four stair pyramids"),
    ("C4", "迷宫 / 短楼梯", "Maze & short stairs"),
)


@dataclass
class Part:
    name: str
    zone: str
    kind: str
    mesh: trimesh.Trimesh


@dataclass
class Atlas:
    seed: int
    parts: list[Part] = field(default_factory=list)
    zones: list[dict] = field(default_factory=list)
    ports: list[dict] = field(default_factory=list)
    spawn: tuple[float, float, float] = (-21.5, 14.5, 0.0)

    def combined_mesh(self) -> trimesh.Trimesh:
        """One static triangle mesh; each obstacle is a closed solid.

        Solids penetrate the single floor slab slightly. There are no stacked
        per-tile floor planes or overlapping boxes inside stair flights.
        This concatenates solids, rather than performing a Boolean union.
        """
        return trimesh.util.concatenate([part.mesh for part in self.parts])

    def metadata(self) -> dict:
        return {
            "name": "fdm_terrain_atlas_v1",
            "seed": self.seed,
            "size_m": list(SIZE),
            "tile_size_m": TILE_SIZE,
            "coordinate_frame": "metres; Z up; X east; Y north; world-centred",
            "spawn_ground_xyz": list(self.spawn),
            "zones": self.zones,
            "ports": self.ports,
            "parts": [
                {"name": p.name, "zone": p.zone, "kind": p.kind,
                 "bounds": p.mesh.bounds.tolist(), "triangles": len(p.mesh.faces)}
                for p in self.parts
            ],
            "triangles": sum(len(p.mesh.faces) for p in self.parts),
            "limitations": [
                "New layout adapted from nav-suite; not the paper's original map.",
                "Geometry checks do not establish G1 traversability or MPPI success.",
                "Preview labels, zone grid and test markers are not collision geometry.",
            ],
        }


def _triangulate_polygon(points: np.ndarray) -> list[list[int]]:
    """Ear-clip a CCW simple polygon, including concave stair profiles."""
    remaining = list(range(len(points)))
    triangles = []
    while len(remaining) > 3:
        for k, b in enumerate(remaining):
            a, c = remaining[k - 1], remaining[(k + 1) % len(remaining)]
            pa, pb, pc = points[[a, b, c]]
            if np.cross(pb - pa, pc - pb) <= 1e-10:
                continue
            other = [i for i in remaining if i not in (a, b, c)]
            q = points[other]
            inside = (
                (np.cross(pb - pa, q - pa) >= -1e-10)
                & (np.cross(pc - pb, q - pb) >= -1e-10)
                & (np.cross(pa - pc, q - pc) >= -1e-10)
            )
            if inside.any():
                continue
            triangles.append([a, b, c])
            remaining.pop(k)
            break
        else:
            raise ValueError("Cannot triangulate terrain profile")
    triangles.append(remaining)
    return triangles


def profile_solid(top: list[tuple[float, float]], width: float) -> trimesh.Trimesh:
    """Extrude an X/Z top profile along Y, with an embedded flat bottom."""
    polygon = np.array([(top[0][0], BASE_Z), (top[-1][0], BASE_Z), *reversed(top)])
    # Remove consecutive duplicates and collinear interior points.
    changed = True
    while changed:
        changed = False
        for i in range(len(polygon)):
            a, b, c = polygon[i - 1], polygon[i], polygon[(i + 1) % len(polygon)]
            if abs(float(np.cross(b - a, c - b))) < 1e-10:
                polygon = np.delete(polygon, i, axis=0)
                changed = True
                break
    triangles = _triangulate_polygon(polygon)
    n = len(polygon)
    vertices = np.array(
        [(x, y, z) for y in (-width / 2, width / 2) for x, z in polygon]
    )
    faces = triangles + [[c + n, b + n, a + n] for a, b, c in triangles]
    for i in range(n):
        j = (i + 1) % n
        faces.extend(([j, i, i + n], [j, i + n, j + n]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def stair_top(start: float, steps: int, tread: float, rise: float,
              base: float = 0.0) -> list[tuple[float, float]]:
    top = [(start, base)]
    for i in range(steps):
        top.extend(((start + i * tread, base + (i + 1) * rise),
                    (start + (i + 1) * tread, base + (i + 1) * rise)))
    return top


def bridge_top(steps=4, rise=0.15, tread=0.35, landing=2.0):
    """Continuous up / landing / down stair bridge."""
    half = steps * tread + landing / 2
    up = stair_top(-half, steps, tread, rise)
    return up + [(-x, z) for x, z in reversed(up)]


def cross_centers(size: float = TILE_SIZE) -> list[tuple[float, float]]:
    """FDM/nav-suite cross_object_pattern, translated to a centred frame."""
    return [(0.0, 0.0), (0.2 * size, 0.2 * size), (-0.2 * size, 0.2 * size),
            (0.2 * size, -0.2 * size), (-0.2 * size, -0.2 * size)]


def maze_edges(seed: int, n: int = 3) -> set[tuple[int, int]]:
    """Adapt nav-suite DFS to logical cells and a local RNG.

    Always consider every unvisited neighbour before backtracking. Unlike the
    source's direction-only branch this visits all logical cells. Add a loop
    so the result has alternative routes rather than a single solution.
    """
    rng = np.random.default_rng(seed)
    visited, stack, openings = {0}, [0], set()
    while stack:
        cell = stack[-1]
        x, y = cell % n, cell // n
        neighbors = [ny * n + nx for nx, ny in
                     ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                     if 0 <= nx < n and 0 <= ny < n and ny * n + nx not in visited]
        if not neighbors:
            stack.pop()
            continue
        nxt = int(rng.choice(neighbors))
        openings.add(tuple(sorted((cell, nxt))))
        visited.add(nxt)
        stack.append(nxt)
    closed = [(a, b) for a in range(n * n) for b in (a + 1, a + n)
              if b < n * n and (b == a + n or a // n == b // n)
              and (a, b) not in openings]
    if closed:
        openings.add(closed[int(rng.integers(len(closed)))])
    return openings


class _Builder:
    def __init__(self, atlas: Atlas, zone: str = "shared", center=(0.0, 0.0)):
        self.atlas, self.zone, self.center = atlas, zone, np.array(center)

    def add(self, name, kind, mesh, xy=(0.0, 0.0), yaw=0.0):
        transform = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
        transform[:2, 3] = self.center + xy
        mesh.apply_transform(transform)
        rgb = tuple(int(COLORS[kind][i:i + 2], 16) for i in (1, 3, 5))
        mesh.visual.face_colors = (*rgb, 255)
        self.atlas.parts.append(Part(f"{self.zone}_{name}", self.zone, kind, mesh))

    def box(self, name, xy, size, height, kind="wall", yaw=0.0):
        mesh = trimesh.creation.box(extents=(*size, height - BASE_Z))
        mesh.apply_translation((0, 0, (height + BASE_Z) / 2))
        self.add(name, kind, mesh, xy, yaw)

    def profile(self, name, top, width, xy=(0, 0), yaw=0, kind="stairs"):
        self.add(name, kind, profile_solid(top, width), xy, yaw)

    def ramp(self, name, start, end, height, width, xy=(0, 0), yaw=0):
        self.profile(name, [(start, 0), (end, height)], width, xy, yaw, "ramp")

    def test(self, start, goal, intent):
        zone = next(z for z in self.atlas.zones if z["id"] == self.zone)
        zone["test"] = {
            "start_ground_xyz": [*(self.center + start), 0.0],
            "goal_ground_xyz": [*(self.center + goal), 0.0],
            "intent": intent,
        }


def _rough_patch(rng: np.random.Generator) -> trimesh.Trimesh:
    """A closed 0.25 m grid patch, with shallow ripples and a tapered margin."""
    x, y = np.linspace(-2, 2, 17), np.linspace(-1.5, 1.5, 13)
    xx, yy = np.meshgrid(x, y)
    taper = np.minimum(np.clip((2 - abs(xx)) / 0.5, 0, 1),
                       np.clip((1.5 - abs(yy)) / 0.5, 0, 1))
    zz = 0.008 + taper * rng.uniform(0.025, 0.075, xx.shape)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).tolist()
    faces = []
    nx, ny = len(x), len(y)
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.extend(([a, a + 1, a + nx + 1], [a, a + nx + 1, a + nx]))
    boundary = (list(range(nx)) + [j * nx + nx - 1 for j in range(1, ny)]
                + list(range(nx * ny - 2, nx * (ny - 1) - 1, -1))
                + [j * nx for j in range(ny - 2, 0, -1)])
    bottom = len(vertices)
    vertices.extend([(vertices[i][0], vertices[i][1], BASE_Z) for i in boundary])
    center = len(vertices)
    vertices.append([0, 0, BASE_Z])
    for k, a in enumerate(boundary):
        nxt = (k + 1) % len(boundary)
        b = boundary[nxt]
        faces.extend(([a, bottom + k, bottom + nxt], [a, bottom + nxt, b],
                      [center, bottom + nxt, bottom + k]))
    return trimesh.Trimesh(vertices, faces, process=True)


def _stairs_and_ramp(b: _Builder, high_steps: bool):
    height = 0.75
    steps, rise = (3, 0.25) if high_steps else (5, 0.15)
    # Two adjacent approaches to one landing, plus a shared gentle descent.
    b.box("landing", (0, 0), (1.8, 5.2), height, "platform")
    start = -0.9 - steps * 0.35
    b.profile("approach_steps", stair_top(start, steps, 0.35, rise), 2.2, (0, 1.5))
    run = 3.0 if high_steps else height / np.tan(np.deg2rad(35))
    b.ramp("approach_ramp", -0.9 - run, -0.9, height, 2.2, (0, -1.5))
    b.profile("exit_ramp", [(0.9, height), (3.9, 0)], 5.2, kind="ramp")
    b.test((-4.6, 0), (4.6, 0),
           "选择两种上台方式，也可从平台外侧平地绕行；共同出口为缓坡")


def _pyramid_mesh(steps=4, tread=0.32, rise=0.15, top_width=0.7):
    """Closed square stair pyramid; rings share vertices, with no stacked boxes."""
    vertices, faces = [], []

    def ring(half, z):
        start = len(vertices)
        vertices.extend([(-half, -half, z), (half, -half, z),
                         (half, half, z), (-half, half, z)])
        return list(range(start, start + 4))

    def connect(a, b):
        for i in range(4):
            j = (i + 1) % 4
            faces.extend(([a[i], a[j], b[j]], [a[i], b[j], b[i]]))

    half = top_width / 2 + (steps - 1) * tread
    current = ring(half, BASE_Z)
    faces.extend(([current[2], current[1], current[0]],
                  [current[3], current[2], current[0]]))
    for step in range(steps):
        outer = ring(half, (step + 1) * rise)
        connect(current, outer)
        if step < steps - 1:
            half -= tread
            inner = ring(half, (step + 1) * rise)
            connect(outer, inner)
            current = inner
        else:
            faces.extend(([outer[0], outer[1], outer[2]], [outer[0], outer[2], outer[3]]))
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return mesh


def _maze(b: _Builder):
    # Structural topology stays fixed across seeds; random pillar/roughness details vary.
    openings = maze_edges(11)
    pitch, half, height, thickness = 2.5, 3.75, 1.05, 0.18
    for y in range(3):
        for x in range(3):
            a = y * 3 + x
            if x < 2 and (a, a + 1) not in openings:
                b.box(f"inner_v_{a}", ((x + 1) * pitch - half, (y + 0.5) * pitch - half),
                      (thickness, pitch), height)
            if y < 2 and (a, a + 3) not in openings:
                b.box(f"inner_h_{a}", ((x + 0.5) * pitch - half, (y + 1) * pitch - half),
                      (pitch, thickness), height)
    # South-west entrance and north-east exit, each 1.8 m wide.
    for side, sign in (("s", -1), ("n", 1)):
        gate = -2.5 if sign == -1 else 2.5
        for suffix, lo, hi in (("l", -half, gate - 0.9), ("r", gate + 0.9, half)):
            if hi > lo:
                b.box(f"outer_{side}_{suffix}", ((lo + hi) / 2, sign * half),
                      (hi - lo, thickness), height)
    for x in (-half, half):
        b.box(f"outer_v_{x}", (x, 0), (thickness, 7.5 - thickness), height)
    # Low, narrow bridge in the centre cell: free room remains on both sides.
    b.profile("short_steps", bridge_top(2, 0.15, 0.35, 0.5), 0.9)
    b.test((-2.5, -4.7), (2.5, 4.7), "迷宫入口到出口；中心有可绕行的 0.30 m 短楼梯")
    next(z for z in b.atlas.zones if z["id"] == b.zone)["maze_openings"] = sorted(openings)


def build_atlas(seed: int = 0) -> Atlas:
    """Build twelve joined 12 x 12 m test zones without launching Isaac Sim."""
    atlas = Atlas(seed=seed)
    rng = np.random.default_rng(seed)
    shared = _Builder(atlas)
    floor = trimesh.creation.box(extents=(*SIZE, 0.12))
    floor.apply_translation((0, 0, -0.06))
    shared.add("floor", "floor", floor)
    for index, (name, title, english) in enumerate(ZONE_SPECS):
        row, col = divmod(index, 4)
        center = (col * TILE_SIZE - 18, 12 - row * TILE_SIZE)
        atlas.zones.append({
            "id": name, "title": title, "english": english, "center": list(center),
            "bounds_xy": [[center[0] - 6, center[1] - 6], [center[0] + 6, center[1] + 6]],
        })
        b = _Builder(atlas, name, center)
        if name == "A1":
            b.add("rough_patch", "rough", _rough_patch(rng), (0.8, -1.2))
            for i, h in enumerate((0.04, 0.06, 0.08, 0.10)):
                b.box(f"low_slat_{i}", (0.1 + i * 0.6, 2.5), (0.28, 2.0), h, "rough")
            b.test((-3.5, 2.5), (4.3, -1.2), "从平坦起点进入低矮起伏区域")
        elif name == "A2":
            positions = [(x, y) for x in (-3.1, 0, 3.1) for y in (-3.1, 0, 3.1)]
            for i, xy in enumerate(positions):
                xy = np.array(xy) + rng.uniform(-0.18, 0.18, 2)
                h = float(rng.uniform(0.65, 1.45))
                if i % 3 == 0:
                    b.box(f"box_{i}", xy, (0.75, 0.85), h, "pillar",
                          float(rng.uniform(-0.4, 0.4)))
                else:
                    sections = 20 if i % 3 == 1 else 3
                    mesh = trimesh.creation.cylinder(
                        radius=0.48 if sections == 20 else 0.68,
                        height=h - BASE_Z, sections=sections)
                    mesh.apply_translation((0, 0, (h + BASE_Z) / 2))
                    b.add(f"prism_{i}", "pillar", mesh, xy, float(rng.uniform(-0.5, 0.5)))
            b.test((-4.6, 1.5), (4.6, -1.5), "在箱体、圆柱、三角柱之间选择通路")
        elif name == "A3":
            for i, xy in enumerate(cross_centers()):
                b.box(f"cross_{i}", xy, (1.1, 1.1), 1.15, "pillar")
            b.test((-4.5, 0), (4.5, 0), "FDM 原始五点交叉布局；比较左右绕行")
        elif name == "A4":
            # Two gates with staggered centres; no lintels over height-scan paths.
            for i, (x, gap_y) in enumerate(((-1.8, -1.25), (1.8, 1.25))):
                for suffix, lo, hi in (("s", -4.2, gap_y - 0.7), ("n", gap_y + 0.7, 4.2)):
                    b.box(f"gate_{i}_{suffix}", (x, (lo + hi) / 2), (0.24, hi - lo), 1.25)
            b.test((-4.6, -1.25), (4.6, 1.25), "两道错位 1.40 m 窄门，外侧仍有绕行空间")
        elif name == "B1":
            b.profile("stair_bridge", bridge_top(5, 0.15, 0.35, 2.3), 3.0)
            b.test((-4.8, 0), (4.8, 0), "直行上 0.75 m 平台再下楼梯，或从南北两侧平地绕行")
        elif name in ("B2", "B3"):
            _stairs_and_ramp(b, high_steps=name == "B3")
        elif name == "B4":
            b.profile("bridge", bridge_top(4, 0.15, 0.35, 1.6), 1.8, (0, 0.8))
            b.box("wall_north", (-0.7, 3.0), (0.25, 2.6), 1.3)
            b.box("wall_south", (0.8, -2.55), (0.25, 3.1), 1.3)
            b.test((-4.6, 0.8), (4.6, 0.8), "楼梯捷径与墙体共同约束绕行路径")
        elif name == "C1":
            b.profile("ramp_bridge", [(-3.9, 0), (-1.2, 0.75), (1.2, 0.75), (3.9, 0)],
                      2.0, kind="ramp")
            b.box("wall_north", (0.6, 2.9), (0.25, 3.4), 1.2)
            b.box("wall_south", (-0.6, -2.9), (0.25, 3.4), 1.2)
            b.test((-4.8, 0), (4.8, 0), "坡顶通行或绕过错位墙体；坡度约 15.5°")
        elif name == "C2":
            # Two levels in one extruded body; side accesses to both tiers.
            top = stair_top(-4.0, 4, 0.35, 0.15)
            top += [(-0.8, 0.6)] + stair_top(-0.8, 4, 0.35, 0.15, base=0.6)
            top += [(1.8, 1.2)] + [(0.2 - x, z) for x, z in reversed(stair_top(-4, 8, 0.3, 0.15))]
            b.profile("two_levels", top, 2.4, kind="platform")
            # Along Y, rising toward the side edge of each corresponding level.
            b.ramp("low_side_ramp", -3.5, -1.2, 0.6, 1.5, (-1.7, 0), np.pi / 2)
            b.ramp("high_side_ramp", -4.5, -1.2, 1.2, 1.0, (1.2, 0), np.pi / 2)
            b.test((-4.8, 0), (4.8, 0), "0.60 / 1.20 m 两级平台，楼梯与侧面坡道连接")
        elif name == "C3":
            # Adapt the reference's four cardinal stair-pyramid positions.
            for i, xy in enumerate(((0, 3.1), (0, -3.1), (-3.1, 0), (3.1, 0))):
                b.add(f"pyramid_{i}", "stairs", _pyramid_mesh(), xy)
            b.test((-4.8, -1.8), (4.8, 1.8), "四个 0.60 m 金字塔楼梯围绕中央平地，可翻越或穿行")
        else:
            _maze(b)

    # Short transverse baffles interrupt straight seam-following shortcuts.
    # Shared portals are sampled flat connectors, not artificial floor plates.
    for col, x in enumerate((-12.0, 0.0, 12.0)):
        for row, y in enumerate((12.0, 0.0, -12.0)):
            offset = 0.8 if (row + col) % 2 == 0 else -0.8
            shared.box(f"v_baffle_{col}_{row}", (x, y + offset), (3.0, 0.22), 0.9)
            atlas.ports.append({
                "zones": [f"{'ABC'[row]}{col + 1}", f"{'ABC'[row]}{col + 2}"],
                "ground_xyz": [x, y - 2.6 * np.sign(offset), 0.0],
                "checked_width_m": 1.8, "crossing_axis": "x",
            })
    for row, y in enumerate((6.0, -6.0)):
        for col, x in enumerate((-18.0, -6.0, 6.0, 18.0)):
            offset = 0.8 if (row + col) % 2 == 0 else -0.8
            shared.box(f"h_baffle_{col}_{row}", (x + offset, y), (0.22, 3.0), 0.9)
            atlas.ports.append({
                "zones": [f"{'ABC'[row]}{col + 1}", f"{'ABC'[row + 1]}{col + 1}"],
                "ground_xyz": [x - 2.6 * np.sign(offset), y, 0.0],
                "checked_width_m": 1.8, "crossing_axis": "y",
            })
    # A physical low boundary keeps the extent obvious in the simulator.
    for y in (-17.91, 17.91):
        shared.box(f"edge_y_{y}", (0, y), (48.0, 0.18), 0.45)
    for x in (-23.91, 23.91):
        shared.box(f"edge_x_{x}", (x, 0), (0.18, 35.64), 0.45)
    return atlas


def sample_heights(atlas: Atlas, xy: np.ndarray) -> np.ndarray:
    """Highest upward-facing surface at XY; no optional ray-tree dependency."""
    return sample_mesh_heights((part.mesh for part in atlas.parts), xy)


def sample_mesh_heights(meshes, xy: np.ndarray) -> np.ndarray:
    """Sample actual imported mesh components as well as preview geometry."""
    xy = np.asarray(xy, dtype=float)
    result = np.full(len(xy), np.nan)
    for mesh in meshes:
        bound = mesh.bounds
        selected = np.flatnonzero(np.all(xy >= bound[0, :2] - 1e-8, axis=1)
                                  & np.all(xy <= bound[1, :2] + 1e-8, axis=1))
        if not len(selected):
            continue
        points = xy[selected]
        heights = result[selected]
        for triangle in mesh.triangles[mesh.face_normals[:, 2] > 1e-8]:
            a, b, c = triangle
            ab, ac = b[:2] - a[:2], c[:2] - a[:2]
            det = np.cross(ab, ac)
            delta = points - a[:2]
            u, v = np.cross(delta, ac) / det, np.cross(ab, delta) / det
            mask = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
            z = a[2] + u[mask] * (b[2] - a[2]) + v[mask] * (c[2] - a[2])
            heights[mask] = np.fmax(heights[mask], z)
        result[selected] = heights
    return result


def check_geometry(atlas: Atlas, resolution=0.1, clearance=0.35) -> dict:
    """Check solids, ports, and conservative flat-ground connectivity.

    The 2-D raster excludes raised surfaces entirely and erodes obstacles by
    ``clearance``. It proves sampled flat bypass connectivity only, not body
    dynamics, height-scan correctness, or reliable locomotion.
    """
    from scipy.ndimage import distance_transform_edt, label

    failures = []
    for p in atlas.parts:
        if not p.mesh.is_watertight or not p.mesh.is_winding_consistent or p.mesh.volume <= 0:
            failures.append(f"Invalid solid: {p.name}")
        if not np.isfinite(p.mesh.vertices).all():
            failures.append(f"Non-finite vertices: {p.name}")
    triangles = sum(len(p.mesh.faces) for p in atlas.parts)
    if triangles > 50_000:
        failures.append(f"Triangle budget exceeded: {triangles}")
    xs = np.arange(-SIZE[0] / 2 + resolution / 2, SIZE[0] / 2, resolution)
    ys = np.arange(-SIZE[1] / 2 + resolution / 2, SIZE[1] / 2, resolution)
    xx, yy = np.meshgrid(xs, ys)
    height = sample_heights(atlas, np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    flat = np.isfinite(height) & (abs(height) < 0.005)
    distance = distance_transform_edt(np.pad(flat, 1))[1:-1, 1:-1] * resolution
    components, _ = label(flat & (distance >= clearance))

    def component(point):
        ix = int(round((point[0] - xs[0]) / resolution))
        iy = int(round((point[1] - ys[0]) / resolution))
        return int(components[iy, ix])

    root = component(atlas.spawn)
    if not root:
        failures.append("Spawn lacks flat-ground clearance")
    points = [("spawn", atlas.spawn)]
    for zone in atlas.zones:
        for key in ("start_ground_xyz", "goal_ground_xyz"):
            points.append((f"{zone['id']}_{key}", zone["test"][key]))
    for i, port in enumerate(atlas.ports):
        point = np.array(port["ground_xyz"][:2])
        across = np.array((0, 1)) if port["crossing_axis"] == "x" else np.array((1, 0))
        samples = point + np.linspace(-0.9, 0.9, 19)[:, None] * across
        if not np.allclose(sample_heights(atlas, samples), 0, atol=1e-6):
            failures.append(f"Port {i} is not flat across its 1.8 m width")
        points.append((f"port_{i}", port["ground_xyz"]))
    for name, point in points:
        if component(point) != root or not root:
            failures.append(f"Flat route disconnected or too narrow: {name}")
    return {
        "passed": not failures, "failures": failures,
        "triangles": triangles, "solids": len(atlas.parts),
        "zones": len(atlas.zones), "shared_ports": len(atlas.ports),
        "flat_route_check_points": len(points),
        "raster_resolution_m": resolution, "ground_clearance_radius_m": clearance,
        "scope": "Static geometry and sampled flat bypasses; no robot simulation",
    }


def atlas_terrain(difficulty: float, cfg):
    """IsaacLab SubTerrain adapter; keep the preview and simulator mesh identical."""
    del difficulty  # Fixed review layout, not a training curriculum.
    if not np.allclose(cfg.size, SIZE):
        raise ValueError(f"FDM atlas requires size={SIZE}, got {cfg.size}")
    atlas = build_atlas(seed=0 if cfg.seed is None else int(cfg.seed))
    # TerrainGenerator expects positive tile-local coordinates, then centres them.
    translation = np.array((*np.array(SIZE) / 2, 0.0))
    mesh = atlas.combined_mesh()
    mesh.apply_translation(translation)
    return [mesh], np.array(atlas.spawn) + translation
