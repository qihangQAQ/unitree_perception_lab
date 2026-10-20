"""Small, deterministic navigation scenes; no simulator or GPU imports.

The visibility graph is an offline scenario check and reference measurement.
Its route is never given to MPPI. Obstacles and the ground mesh share this spec.
"""

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math


BENCHMARK_TASK = "Unitree-G1-29dof-FDM-Benchmark"
SCENE_NAMES = ("empty", "box", "wall")


@dataclass(frozen=True)
class BoxObstacle:
    center: tuple[float, float]
    size: tuple[float, float, float]

    def bounds(self, clearance):
        x, y = self.center
        sx, sy, _ = self.size
        return (x-sx/2-clearance, y-sy/2-clearance,
                x+sx/2+clearance, y+sy/2+clearance)


@dataclass(frozen=True)
class BenchmarkScene:
    name: str
    size: tuple[float, float] = (10., 10.)
    start: tuple[float, float, float] = (-2.5, 0., 0.)
    goal: tuple[float, float] = (2.5, 0.)
    obstacles: tuple[BoxObstacle, ...] = ()
    clearance: float = .4
    version: int = 1

    def contains(self, xy):
        """Footprint stays inside the floor; accepts NumPy or torch batches."""
        hx, hy = (v/2-self.clearance for v in self.size)
        return ((xy[..., 0] >= -hx) & (xy[..., 0] <= hx)
                & (xy[..., 1] >= -hy) & (xy[..., 1] <= hy))

    def free(self, xy):
        valid = self.contains(xy)
        for obstacle in self.obstacles:
            x0, y0, x1, y1 = obstacle.bounds(self.clearance)
            valid = valid & ~((xy[..., 0] >= x0) & (xy[..., 0] <= x1)
                              & (xy[..., 1] >= y0) & (xy[..., 1] <= y1))
        return valid

    def segment_clear(self, a, b):
        """Check the entire segment against expanded boxes, not just endpoints."""
        for obstacle in self.obstacles:
            bounds = obstacle.bounds(self.clearance)
            enter, leave = 0., 1.
            for axis in range(2):
                delta = b[axis]-a[axis]
                low, high = bounds[axis], bounds[axis+2]
                if abs(delta) < 1e-12:
                    if not low <= a[axis] <= high:
                        enter, leave = 1., 0.
                        break
                else:
                    t0, t1 = sorted(((low-a[axis])/delta, (high-a[axis])/delta))
                    enter, leave = max(enter, t0), min(leave, t1)
            if enter <= leave:
                return False
        return True

    def reference_path(self, start=None):
        """Collision-free planar route with a conservative body clearance."""
        start = tuple(self.start[:2] if start is None else start[:2])
        points = [start, self.goal]
        # Slightly offset corners so an edge cannot lie on an obstacle boundary.
        for obstacle in self.obstacles:
            x0, y0, x1, y1 = obstacle.bounds(self.clearance+.001)
            points.extend(((x0, y0), (x0, y1), (x1, y0), (x1, y1)))
        def free_point(point):
            x, y = point
            if not (math.isfinite(x) and math.isfinite(y)
                    and abs(x) <= self.size[0]/2-self.clearance
                    and abs(y) <= self.size[1]/2-self.clearance):
                return False
            return all(not (x0 <= x <= x1 and y0 <= y <= y1)
                       for x0, y0, x1, y1 in (o.bounds(self.clearance) for o in self.obstacles))
        valid = [free_point(p) for p in points]
        if not valid[0] or not valid[1]:
            raise ValueError("Benchmark start/goal lacks floor or obstacle clearance.")
        queue = [(0., 0, (0,))]
        visited = set()
        while queue:
            distance, node, route = heapq.heappop(queue)
            if node in visited:
                continue
            if node == 1:
                return [points[i] for i in route], distance
            visited.add(node)
            for other, point in enumerate(points):
                if other not in visited and valid[other] and self.segment_clear(points[node], point):
                    heapq.heappush(queue, (distance+math.dist(points[node], point), other, route+(other,)))
        raise ValueError("Benchmark has no route with the required body clearance.")

    def metadata(self):
        path, length = self.reference_path()
        blocked = not self.segment_clear(self.start[:2], self.goal)
        if blocked != bool(self.obstacles):
            raise ValueError("Obstacle benchmarks must block the direct start-goal segment.")
        spec = asdict(self)
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return {"terrain_type": "generator", "split": "benchmark", "spec": spec,
                "scene_spec_sha256": digest, "direct_path_blocked": blocked,
                "reference_path": path, "reference_path_length_m": length,
                "reference_method": "planar_visibility_graph_expanded_boxes",
                "reference_used_for_control": False}


def get_scene(name):
    obstacles = {"empty": (), "box": (BoxObstacle((0., 0.), (1., 1., 1.5)),),
                 "wall": (BoxObstacle((0., 0.), (.2, 2., 1.5)),)}
    if name not in obstacles:
        raise ValueError(f"Unknown benchmark scene {name!r}; choose from {SCENE_NAMES}.")
    return BenchmarkScene(name=name, obstacles=obstacles[name])


def scene_meshes(scene):
    """Mesh coordinates match the scene spec (world centered, floor top z=0)."""
    import trimesh
    floor = trimesh.creation.box(extents=(*scene.size, .1))
    floor.apply_translation((0., 0., -.05))
    floor.visual.face_colors = (150, 160, 170, 255)
    meshes = [floor]
    for obstacle in scene.obstacles:
        mesh = trimesh.creation.box(extents=obstacle.size)
        mesh.apply_translation((*obstacle.center, obstacle.size[2]/2))
        mesh.visual.face_colors = (220, 130, 65, 255)
        meshes.append(mesh)
    return meshes
