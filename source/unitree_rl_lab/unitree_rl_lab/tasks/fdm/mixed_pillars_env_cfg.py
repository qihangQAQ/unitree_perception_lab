"""Single-robot mixed-pillar demo with the frozen perception actor interface."""

import copy
import importlib
import math

import torch
from isaaclab.managers import EventTermCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.fdm.pillar_scene import PillarScene
from unitree_rl_lab.terrains import MIXED_PILLARS_TERRAIN
from .navigation_env_cfg import NavigationEnvCfg


class PillarDemoImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        # Capture the same world-centered mesh used for collision and ray casting.
        self.demo_scene = PillarScene(mesh)
        self.demo_start, self.demo_goal = self.demo_scene.choose_endpoints(
            seed=self.cfg.terrain_generator.seed, start=self.cfg.demo_start, goal=self.cfg.demo_goal
        )
        super().import_mesh(name, mesh)


@configclass
class PillarDemoImporterCfg(TerrainImporterCfg):
    class_type: type = PillarDemoImporter
    demo_start: tuple[float, float, float] | None = None
    demo_goal: tuple[float, float] | None = None


def reset_pillar_root(env, env_ids):
    robot = env.scene["robot"]
    x, y, yaw = env.scene.terrain.demo_start
    state = robot.data.default_root_state[env_ids].clone()
    state[:, 0], state[:, 1] = x, y
    state[:, 2] += 0.05  # Spawn just above the z=0 floor, using the robot's nominal root height.
    state[:, 3:7] = torch.tensor((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), device=env.device)
    state[:, 7:13] = 0.0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


@configclass
class MixedPillarsEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Appearance is taken from the exact depth task the user selected.
        depth = importlib.import_module(
            "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_depth_env_cfg"
        ).RobotPlayEnvCfg()
        self.scene.num_envs = 1
        self.scene.terrain = PillarDemoImporterCfg(
            prim_path="/World/ground", terrain_type="generator", collision_group=-1,
            terrain_generator=copy.deepcopy(MIXED_PILLARS_TERRAIN),
            physics_material=copy.deepcopy(depth.scene.terrain.physics_material),
            visual_material=copy.deepcopy(depth.scene.terrain.visual_material),
            max_init_terrain_level=0, debug_vis=False,
        )
        self.scene.sky_light = copy.deepcopy(depth.scene.sky_light)
        self.sim.render = copy.deepcopy(depth.sim.render)
        self.events.reset_base = EventTermCfg(func=reset_pillar_root, mode="reset")
        self.observations.policy.enable_corruption = False
        self.scene.height_scanner.debug_vis = False
        self.scene.fdm_height_scanner.debug_vis = False
        self.viewer.eye = (8.0, -10.0, 11.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)
