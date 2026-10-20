"""One generated mesh arena with a fixed start; no USD split or spawn analysis."""

import math
import numpy as np
import torch

from isaaclab.managers import EventTermCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.fdm.benchmarks import get_scene, scene_meshes
from .navigation_env_cfg import NavigationEnvCfg


def benchmark_terrain(difficulty, cfg):
    scene = get_scene(cfg.scene_name)
    if tuple(cfg.size) != scene.size:
        raise ValueError("Benchmark mesh size must match its scene specification.")
    meshes = scene_meshes(scene)
    # TerrainGenerator expects coordinates in [0, size], then centers its mesh.
    center = np.array([scene.size[0]/2, scene.size[1]/2, 0.])
    for mesh in meshes:
        mesh.apply_translation(center)
    return meshes, center


@configclass
class BenchmarkTerrainCfg(SubTerrainBaseCfg):
    function = benchmark_terrain
    scene_name: str = "box"


def reset_benchmark_root(env, env_ids, scene_name):
    if env.num_envs != 1:
        raise ValueError("Generated navigation benchmarks require exactly one environment.")
    scene = get_scene(scene_name)
    robot = env.scene["robot"]
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :2] = torch.tensor(scene.start[:2], device=env.device)
    state[:, 2] += .05
    state[:, 3:7] = torch.tensor([math.cos(scene.start[2]/2), 0., 0., math.sin(scene.start[2]/2)],
                                device=env.device)
    state[:, 7:13] = 0.
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


@configclass
class BenchmarkEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator", collision_group=-1,
            physics_material=self.scene.terrain.physics_material,
            visual_material=None,  # Preserve the floor/obstacle mesh colors.
            terrain_generator=TerrainGeneratorCfg(
                seed=0, size=(10., 10.), num_rows=1, num_cols=1, border_width=0.,
                use_cache=False, curriculum=False, color_scheme="none",
                sub_terrains={"benchmark": BenchmarkTerrainCfg(proportion=1.)}),
            max_init_terrain_level=0, debug_vis=False)
        self.events.reset_base = EventTermCfg(
            func=reset_benchmark_root, mode="reset", params={"scene_name": "box"})
        self.scene.height_scanner.debug_vis = False
        # Show both endpoints and the obstacle from the first rendered frame.
        self.viewer.eye = (6., -8., 8.)
        self.viewer.lookat = (0., 0., .4)


def configure_benchmark(cfg, scene_name):
    scene = get_scene(scene_name)
    cfg.scene.terrain.terrain_generator.size = scene.size
    cfg.scene.terrain.terrain_generator.sub_terrains["benchmark"].scene_name = scene_name
    cfg.events.reset_base.params["scene_name"] = scene_name
