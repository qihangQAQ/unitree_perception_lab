"""Configuration for sharp terrain edges inflated as finite cylinders."""

from isaaclab.utils import configclass

from .edge_cylinder import EdgeCylinder
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class EdgeCylinderCfg(VirtualObstacleCfg):
    class_type: type = EdgeCylinder
    angle_threshold: float = 70.0
    cylinder_radius: float = 0.05
    num_grid_cells: int = 64**3
    adjacent_angle_threshold: float = 30.0
    min_points: int = 2
