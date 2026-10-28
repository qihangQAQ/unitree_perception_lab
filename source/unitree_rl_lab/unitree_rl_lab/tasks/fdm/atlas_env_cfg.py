"""Single-robot FDM + perception inference on the twelve-zone mesh atlas."""

import copy
import math

import torch
from isaaclab.managers import EventTermCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.fdm.atlas_scene import AtlasScene
from unitree_rl_lab.terrains import FDM_ATLAS_TERRAIN

from .mixed_pillars_env_cfg import MixedPillarsEnvCfg


class AtlasDemoImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        # TerrainGenerator has already centred this exact collision/raycast mesh.
        self.demo_scene = AtlasScene(mesh, seed=self.cfg.terrain_generator.seed)
        if self.cfg.demo_spawn_only:
            self.demo_start = self.demo_scene.choose_spawn(zone=self.cfg.demo_zone, start=self.cfg.demo_start)
            self.demo_goal = self.demo_goal_height = None
        else:
            self.demo_start, self.demo_goal = self.demo_scene.choose_endpoints(
                zone=self.cfg.demo_zone, start=self.cfg.demo_start, goal=self.cfg.demo_goal
            )
            self.demo_goal_height = self.demo_scene.goal_height
        self.demo_start_height = self.demo_scene.start_height
        super().import_mesh(name, mesh)


@configclass
class AtlasDemoImporterCfg(TerrainImporterCfg):
    class_type: type = AtlasDemoImporter
    demo_zone: str | None = None
    demo_start: tuple[float, float, float] | None = None
    demo_goal: tuple[float, float] | None = None
    demo_spawn_only: bool = False


def reset_atlas_root(env, env_ids):
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    x, y, yaw = terrain.demo_start
    state = robot.data.default_root_state[env_ids].clone()
    state[:, 0], state[:, 1] = x, y
    # The default state's Z is nominal base height, not terrain elevation.
    state[:, 2] += terrain.demo_start_height + 0.05
    state[:, 3:7] = torch.tensor(
        (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), device=env.device, dtype=state.dtype
    )
    state[:, 7:13] = 0.0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


@configclass
class AtlasEnvCfg(MixedPillarsEnvCfg):
    def __post_init__(self):
        # Reuse the existing frozen policy interface, sensors and depth appearance.
        super().__post_init__()
        previous = self.scene.terrain
        self.scene.terrain = AtlasDemoImporterCfg(
            prim_path="/World/ground", terrain_type="generator", collision_group=-1,
            terrain_generator=copy.deepcopy(FDM_ATLAS_TERRAIN),
            physics_material=previous.physics_material,
            visual_material=previous.visual_material,
            max_init_terrain_level=0, debug_vis=False,
        )
        self.events.reset_base = EventTermCfg(func=reset_atlas_root, mode="reset")
        self.viewer.eye = (-12.0, -10.0, 10.0)
        self.viewer.lookat = (-18.0, 0.0, 0.4)
