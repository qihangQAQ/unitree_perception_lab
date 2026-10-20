"""Configuration for one mixed-pillar obstacle field."""

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .mixed_pillars import mixed_pillars_terrain


@configclass
class MeshMixedPillarsTerrainCfg(SubTerrainBaseCfg):
    function = mixed_pillars_terrain

    # TerrainGenerator supplies its own seed when invoking the function.
    seed: int | None = 0
    num_boxes: int = 10
    num_cylinders: int = 10
    num_triangular_prisms: int = 10
    height_range: tuple[float, float] = (0.5, 1.5)
    box_length_range: tuple[float, float] = (0.3, 1.0)
    box_width_range: tuple[float, float] = (0.3, 1.0)
    cylinder_radius_range: tuple[float, float] = (0.15, 0.5)
    triangle_side_range: tuple[float, float] = (0.3, 0.9)
    size_sampling_power: float = 2.0
    """Cross-section size uses U**power: 1 is uniform; larger values favor thinner pillars."""
    min_gap: float = 0.8
    """Minimum distance between actual XY mesh footprints, in metres."""
    edge_margin: float = 0.5
    floor_thickness: float = 0.1
    cylinder_sections: int = 32
    max_placement_attempts: int = 1000
    max_layout_attempts: int = 20
