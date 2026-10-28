"""IsaacLab configuration for the twelve-zone FDM mesh atlas."""

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from unitree_rl_lab.fdm.terrain_atlas import atlas_terrain


@configclass
class MeshFDMAtlasTerrainCfg(SubTerrainBaseCfg):
    function = atlas_terrain
    seed: int | None = 0
