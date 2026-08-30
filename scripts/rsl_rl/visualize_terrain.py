#!/usr/bin/env python3

"""Visualize every terrain type and difficulty from a CFG or registered task.

Examples:

.. code-block:: bash

    # Visualize a TerrainGeneratorCfg directly.
    python scripts/rsl_rl/visualize_terrain.py \
        --terrain_cfg unitree_rl_lab.terrains.terrain_generator_cfg:UPGRADE_TERRAIN1

    # Load the terrain and reset ranges from a registered training task.
    python scripts/rsl_rl/visualize_terrain.py \
        --task Unitree-G1-29dof-Velocity-perception-pro

The generated grid has one column per configured sub-terrain and one row per
exact difficulty value. In task mode, the script also draws the XY footprint
of the task's root-state reset event on every terrain tile.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Visualize a complete terrain-type/difficulty grid and optional task reset regions."
)
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument(
    "--terrain_cfg",
    "--terrain-cfg",
    dest="terrain_cfg",
    type=str,
    help="TerrainGeneratorCfg import path in 'python.module:VARIABLE' form.",
)
source_group.add_argument(
    "--task",
    type=str,
    help="Registered Gym task whose terrain and reset configuration should be visualized.",
)
parser.add_argument(
    "--num_levels",
    "--num-levels",
    dest="num_levels",
    type=int,
    default=None,
    help="Number of exact difficulty rows. Defaults to terrain_generator.num_rows.",
)
parser.add_argument(
    "--use_play_cfg",
    "--use-play-cfg",
    dest="use_play_cfg",
    action="store_true",
    help="In task mode, load play_env_cfg_entry_point instead of the training environment CFG.",
)
parser.add_argument(
    "--reset_event",
    "--reset-event",
    dest="reset_event",
    type=str,
    default=None,
    help="Reset event attribute to visualize. By default it is detected from its position-range parameters.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Override the terrain-generator seed for reproducible random terrain details.",
)
parser.add_argument(
    "--debug_vis",
    "--debug-vis",
    dest="debug_vis",
    action="store_true",
    help="Show the TerrainImporter origin markers.",
)
parser.add_argument(
    "--show_virtual_obstacles",
    "--show-virtual-obstacles",
    dest="show_virtual_obstacles",
    action="store_true",
    help="In task mode, retain and visualize supported virtual obstacles from the task terrain importer.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.terrain_cfg is not None and args_cli.use_play_cfg:
    parser.error("--use_play_cfg can only be used together with --task.")
if args_cli.terrain_cfg is not None and args_cli.reset_event is not None:
    parser.error("--reset_event can only be used together with --task.")
if args_cli.num_levels is not None and args_cli.num_levels <= 0:
    parser.error("--num_levels must be greater than zero.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below requires Isaac Sim to be running."""

import copy
import importlib
import math
from dataclasses import dataclass

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainGenerator
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.terrains import TerrainImporterCfg


@dataclass(frozen=True)
class ResetRegion:
    """A rectangular or trapezoidal reset footprint in terrain-local XY."""

    x_range: tuple[float, float]
    y_at_x_min: tuple[float, float]
    y_at_x_max: tuple[float, float]
    is_trapezoid: bool
    is_terrain_specific: bool


@dataclass(frozen=True)
class TaskResetInfo:
    """Task reset configuration needed by the terrain-only visualizer."""

    event_name: str
    function_name: str
    params: dict
    default_root_position: tuple[float, float, float]


def _load_object(import_path: str):
    """Load an object from a ``module:attribute`` import path."""

    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "--terrain_cfg must use 'python.module:VARIABLE' syntax, "
            f"got {import_path!r}."
        )

    module = importlib.import_module(module_name)
    if not hasattr(module, attribute_name):
        available = sorted(
            name
            for name in dir(module)
            if name.endswith("TERRAIN_CFG")
            or name.endswith("TERRAINS_CFG")
            or name.startswith("TERRAIN")
            or name.endswith("TERRAIN1")
        )
        raise AttributeError(
            f"Module {module_name!r} has no attribute {attribute_name!r}. "
            f"Possible terrain CFGs: {available}"
        )
    return getattr(module, attribute_name)


def _event_terms(events_cfg) -> dict[str, object]:
    """Return configured event terms without relying on a particular config class."""

    if events_cfg is None:
        return {}

    event_terms = {}
    for name in dir(events_cfg):
        if name.startswith("_"):
            continue
        try:
            value = getattr(events_cfg, name)
        except Exception:
            continue
        if value is not None and all(hasattr(value, field) for field in ("func", "mode", "params")):
            event_terms[name] = value
    return event_terms


def _find_reset_event(env_cfg) -> tuple[str, object]:
    """Find the root reset event that exposes an XY position range."""

    event_terms = _event_terms(getattr(env_cfg, "events", None))
    if args_cli.reset_event is not None:
        if args_cli.reset_event not in event_terms:
            available = ", ".join(sorted(event_terms)) or "none"
            raise ValueError(
                f"Task has no event {args_cli.reset_event!r}. Available event terms: {available}."
            )
        event_cfg = event_terms[args_cli.reset_event]
        params = getattr(event_cfg, "params", None)
        if not isinstance(params, dict) or not any(
            isinstance(params.get(key), dict) for key in ("pose_range", "root_pos_range")
        ):
            raise ValueError(
                f"Event {args_cli.reset_event!r} does not expose a pose_range or root_pos_range dictionary."
            )
        return args_cli.reset_event, event_cfg

    candidates = []
    for name, event_cfg in event_terms.items():
        if getattr(event_cfg, "mode", None) != "reset":
            continue
        params = getattr(event_cfg, "params", None)
        if isinstance(params, dict) and any(
            isinstance(params.get(key), dict) for key in ("pose_range", "root_pos_range")
        ):
            candidates.append((name, event_cfg))

    if not candidates:
        raise ValueError(
            f"Task {args_cli.task!r} has no reset event with pose_range or root_pos_range parameters."
        )
    if len(candidates) > 1:
        names = ", ".join(name for name, _ in candidates)
        raise ValueError(
            f"Task {args_cli.task!r} has multiple position reset events ({names}); "
            "select one with --reset_event."
        )
    return candidates[0]


def _default_root_position(env_cfg, reset_params: dict) -> tuple[float, float, float]:
    """Read the reset asset's configured initial root position."""

    asset_cfg = reset_params.get("asset_cfg")
    asset_name = getattr(asset_cfg, "name", "robot")
    scene_asset_cfg = getattr(getattr(env_cfg, "scene", None), asset_name, None)
    init_state = getattr(scene_asset_cfg, "init_state", None)
    position = getattr(init_state, "pos", (0.0, 0.0, 0.0))
    if not isinstance(position, (tuple, list)) or len(position) < 3:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in position[:3])


def _load_task_source():
    """Load a task's terrain importer CFG and reset parameters."""

    task_id = args_cli.task.split(":")[-1]
    task_spec = gym.spec(task_id)
    entry_point_key = "env_cfg_entry_point"
    if args_cli.use_play_cfg:
        if task_spec.kwargs.get("play_env_cfg_entry_point") is None:
            raise ValueError(f"Task {task_id!r} has no play_env_cfg_entry_point.")
        entry_point_key = "play_env_cfg_entry_point"

    env_cfg = load_cfg_from_registry(task_id, entry_point_key)
    terrain_importer_cfg = copy.deepcopy(getattr(getattr(env_cfg, "scene", None), "terrain", None))
    if terrain_importer_cfg is None:
        raise ValueError(f"Task {task_id!r} has no scene.terrain configuration.")
    if terrain_importer_cfg.terrain_type != "generator" or terrain_importer_cfg.terrain_generator is None:
        raise ValueError(f"Task {task_id!r} does not use a TerrainGeneratorCfg terrain.")

    event_name, event_cfg = _find_reset_event(env_cfg)
    reset_params = copy.deepcopy(event_cfg.params)
    reset_info = TaskResetInfo(
        event_name=event_name,
        function_name=getattr(event_cfg.func, "__qualname__", repr(event_cfg.func)),
        params=reset_params,
        default_root_position=_default_root_position(env_cfg, reset_params),
    )
    cfg_kind = "play" if args_cli.use_play_cfg else "training"
    label = f"task {task_id} ({cfg_kind} CFG)"
    return terrain_importer_cfg, reset_info, label


def _load_direct_cfg_source():
    """Load a standalone TerrainGeneratorCfg and wrap it in an importer CFG."""

    terrain_generator_cfg = copy.deepcopy(_load_object(args_cli.terrain_cfg))
    if not hasattr(terrain_generator_cfg, "sub_terrains") or not terrain_generator_cfg.sub_terrains:
        raise TypeError(f"{args_cli.terrain_cfg!r} is not a non-empty TerrainGeneratorCfg.")

    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.32, 0.34, 0.38),
            roughness=0.85,
        ),
        debug_vis=False,
    )
    return terrain_importer_cfg, None, args_cli.terrain_cfg


def _make_exact_difficulty_generator(
    base_class: type, difficulties: tuple[float, ...]
) -> type:
    """Create a generator with one named terrain per column and exact difficulty per row."""

    if not isinstance(base_class, type) or not issubclass(base_class, TerrainGenerator):
        raise TypeError(
            f"Terrain generator class must inherit TerrainGenerator, got {base_class!r}."
        )

    class ExactDifficultyTerrainGenerator(base_class):
        selected_difficulties = difficulties

        def _generate_curriculum_terrains(self):
            terrain_items = list(self.cfg.sub_terrains.items())
            if self.cfg.num_cols != len(terrain_items):
                raise ValueError(
                    "Exact terrain grid requires one column per sub-terrain: "
                    f"num_cols={self.cfg.num_cols}, terrain_types={len(terrain_items)}."
                )
            if self.cfg.num_rows != len(self.selected_difficulties):
                raise ValueError(
                    "Exact terrain grid row count does not match the requested difficulties: "
                    f"num_rows={self.cfg.num_rows}, difficulties={self.selected_difficulties}."
                )

            for column, (terrain_name, sub_terrain_cfg) in enumerate(terrain_items):
                for row, difficulty in enumerate(self.selected_difficulties):
                    # Custom generator subclasses may use this temporary key to
                    # associate generated metadata with the named sub-terrain.
                    if hasattr(self, "_pending_subterrain_key"):
                        self._pending_subterrain_key = terrain_name
                    try:
                        mesh, origin = self._get_terrain_mesh(difficulty, sub_terrain_cfg)
                    finally:
                        if hasattr(self, "_pending_subterrain_key"):
                            self._pending_subterrain_key = None
                    self._add_sub_terrain(mesh, origin, row, column, sub_terrain_cfg)

    ExactDifficultyTerrainGenerator.__name__ = f"ExactDifficulty{base_class.__name__}"
    return ExactDifficultyTerrainGenerator


def _configure_grid(terrain_importer_cfg):
    """Convert a copied terrain configuration into the complete organized grid."""

    generator_cfg = terrain_importer_cfg.terrain_generator
    terrain_names = tuple(generator_cfg.sub_terrains.keys())
    num_levels = int(generator_cfg.num_rows if args_cli.num_levels is None else args_cli.num_levels)
    difficulty_low, difficulty_high = (float(value) for value in generator_cfg.difficulty_range)
    difficulties = tuple(
        float(value)
        for value in np.linspace(difficulty_low, difficulty_high, num=num_levels)
    )

    generator_cfg.class_type = _make_exact_difficulty_generator(
        generator_cfg.class_type, difficulties
    )
    generator_cfg.curriculum = True
    generator_cfg.num_rows = num_levels
    generator_cfg.num_cols = len(terrain_names)
    if args_cli.seed is not None:
        generator_cfg.seed = args_cli.seed

    terrain_importer_cfg.num_envs = num_levels * len(terrain_names)
    terrain_importer_cfg.max_init_terrain_level = num_levels - 1
    terrain_importer_cfg.debug_vis = args_cli.debug_vis or args_cli.show_virtual_obstacles

    virtual_obstacles = getattr(terrain_importer_cfg, "virtual_obstacles", None)
    if virtual_obstacles is not None:
        if not args_cli.show_virtual_obstacles:
            terrain_importer_cfg.virtual_obstacles = {}
        elif not virtual_obstacles:
            print("[WARN] The selected task terrain CFG has no virtual obstacles to visualize.")
    elif args_cli.show_virtual_obstacles:
        print("[WARN] The selected terrain importer does not support virtual obstacles.")

    return terrain_names, difficulties


def _as_range(value, field_name: str) -> tuple[float, float]:
    """Validate and normalize a two-value numeric range."""

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain two values, got {value!r}.")
    low, high = float(value[0]), float(value[1])
    return min(low, high), max(low, high)


def _shift_range(value: tuple[float, float], offset: float) -> tuple[float, float]:
    return value[0] + offset, value[1] + offset


def _resolve_reset_regions(
    terrain_names: tuple[str, ...], reset_info: TaskResetInfo
) -> dict[str, ResetRegion]:
    """Resolve default and optional terrain-specific task reset footprints."""

    reset_params = reset_info.params
    root_position_range = reset_params.get("pose_range")
    if not isinstance(root_position_range, dict):
        root_position_range = reset_params.get("root_pos_range")
    if not isinstance(root_position_range, dict):
        raise ValueError(
            f"Reset event {reset_info.event_name!r} has no position-range dictionary."
        )

    default_root_x, default_root_y, _ = reset_info.default_root_position
    default_x = _shift_range(
        _as_range(root_position_range.get("x", (0.0, 0.0)), "reset position x"),
        default_root_x,
    )
    default_y = _shift_range(
        _as_range(root_position_range.get("y", (0.0, 0.0)), "reset position y"),
        default_root_y,
    )

    terrain_ranges = reset_params.get("terrain_specific_pos_range")
    alternative_ranges = reset_params.get("terrain_specific_pose_range")
    if terrain_ranges and alternative_ranges:
        raise ValueError(
            "Reset event defines both terrain_specific_pos_range and "
            "terrain_specific_pose_range; their precedence is ambiguous."
        )
    if terrain_ranges is None:
        terrain_ranges = alternative_ranges
    if terrain_ranges is None:
        terrain_ranges = {}
    if not isinstance(terrain_ranges, dict):
        raise ValueError(
            "terrain_specific_pos_range must be a dictionary when configured, "
            f"got {terrain_ranges!r}."
        )

    regions = {}
    for terrain_name in terrain_names:
        specific = terrain_ranges.get(terrain_name, {})
        if not isinstance(specific, dict):
            raise ValueError(
                f"Terrain-specific reset range for {terrain_name!r} must be a dictionary, "
                f"got {specific!r}."
            )

        has_y_at_x_min = "y_at_x_min" in specific
        has_y_at_x_max = "y_at_x_max" in specific
        if has_y_at_x_min != has_y_at_x_max:
            raise ValueError(
                f"Terrain {terrain_name!r} must define both y_at_x_min and y_at_x_max."
            )

        x_range = _shift_range(
            _as_range(specific.get("x", default_x), f"{terrain_name}.x"),
            default_root_x if "x" in specific else 0.0,
        )
        is_trapezoid = has_y_at_x_min and has_y_at_x_max
        if is_trapezoid:
            if "x" not in specific:
                raise ValueError(f"Trapezoid reset region {terrain_name!r} must define x.")
            if "y" in specific:
                raise ValueError(
                    f"Trapezoid reset region {terrain_name!r} cannot also define y."
                )
            if x_range[1] <= x_range[0]:
                raise ValueError(
                    f"Trapezoid reset region {terrain_name!r} needs an increasing x range."
                )
            y_at_x_min = _shift_range(
                _as_range(specific["y_at_x_min"], f"{terrain_name}.y_at_x_min"),
                default_root_y,
            )
            y_at_x_max = _shift_range(
                _as_range(specific["y_at_x_max"], f"{terrain_name}.y_at_x_max"),
                default_root_y,
            )
        else:
            y_range = _shift_range(
                _as_range(specific.get("y", default_y), f"{terrain_name}.y"),
                default_root_y if "y" in specific else 0.0,
            )
            y_at_x_min = y_range
            y_at_x_max = y_range

        regions[terrain_name] = ResetRegion(
            x_range=x_range,
            y_at_x_min=y_at_x_min,
            y_at_x_max=y_at_x_max,
            is_trapezoid=is_trapezoid,
            is_terrain_specific=terrain_name in terrain_ranges,
        )
    return regions


def _create_reset_region_visualizer(
    terrain_importer,
    terrain_names: tuple[str, ...],
    difficulties: tuple[float, ...],
    regions: dict[str, ResetRegion],
    device: str,
):
    """Draw every resolved reset footprint on every terrain tile."""

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TerrainResetRegions",
        markers={
            "default_fill": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.012),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.45, 1.0),
                    emissive_color=(0.0, 0.05, 0.18),
                    opacity=0.24,
                ),
            ),
            "default_edge": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.025),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.85, 1.0),
                    emissive_color=(0.0, 0.2, 0.3),
                    opacity=0.95,
                ),
            ),
            "specific_fill": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.012),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 1.0, 0.2),
                    emissive_color=(0.02, 0.15, 0.03),
                    opacity=0.26,
                ),
            ),
            "specific_edge": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.03),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.05),
                    emissive_color=(0.35, 0.22, 0.0),
                    opacity=0.95,
                ),
            ),
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)

    translations_data: list[list[float]] = []
    orientations_data: list[list[float]] = []
    scales_data: list[list[float]] = []
    marker_indices_data: list[int] = []
    edge_width = 0.025
    minimum_visible_size = 0.06

    def append_marker(
        tile_origin: tuple[float, float, float],
        local_x: float,
        local_y: float,
        scale_x: float,
        scale_y: float,
        marker_index: int,
        yaw: float = 0.0,
        z_offset: float = 0.025,
    ) -> None:
        translations_data.append(
            [tile_origin[0] + local_x, tile_origin[1] + local_y, tile_origin[2] + z_offset]
        )
        orientations_data.append(
            [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        )
        scales_data.append(
            [
                max(scale_x, minimum_visible_size),
                max(scale_y, minimum_visible_size),
                1.0,
            ]
        )
        marker_indices_data.append(marker_index)

    terrain_origins = terrain_importer.terrain_origins
    if terrain_origins is None:
        raise RuntimeError("The terrain importer did not expose terrain origins.")

    for column, terrain_name in enumerate(terrain_names):
        region = regions[terrain_name]
        x_low, x_high = region.x_range
        size_x = x_high - x_low
        fill_marker_index = 2 if region.is_terrain_specific else 0
        edge_marker_index = 3 if region.is_terrain_specific else 1

        for row, _difficulty in enumerate(difficulties):
            origin = terrain_origins[row, column]
            tile_origin = tuple(float(value.item()) for value in origin)

            fill_slices = 24 if region.is_trapezoid else 1
            slice_length = size_x / fill_slices
            for slice_index in range(fill_slices):
                fraction = (slice_index + 0.5) / fill_slices
                local_x = x_low + fraction * size_x
                lower_y = region.y_at_x_min[0] + fraction * (
                    region.y_at_x_max[0] - region.y_at_x_min[0]
                )
                upper_y = region.y_at_x_min[1] + fraction * (
                    region.y_at_x_max[1] - region.y_at_x_min[1]
                )
                append_marker(
                    tile_origin,
                    local_x,
                    0.5 * (lower_y + upper_y),
                    slice_length + 0.002,
                    upper_y - lower_y,
                    fill_marker_index,
                )

            for edge_x, edge_y_range in (
                (x_low, region.y_at_x_min),
                (x_high, region.y_at_x_max),
            ):
                append_marker(
                    tile_origin,
                    edge_x,
                    0.5 * sum(edge_y_range),
                    edge_width,
                    edge_y_range[1] - edge_y_range[0] + edge_width,
                    edge_marker_index,
                    z_offset=0.03,
                )

            for start_y, end_y in (
                (region.y_at_x_min[0], region.y_at_x_max[0]),
                (region.y_at_x_min[1], region.y_at_x_max[1]),
            ):
                delta_y = end_y - start_y
                append_marker(
                    tile_origin,
                    0.5 * (x_low + x_high),
                    0.5 * (start_y + end_y),
                    math.hypot(size_x, delta_y) + edge_width,
                    edge_width,
                    edge_marker_index,
                    math.atan2(delta_y, size_x) if size_x > 0.0 else 0.0,
                    z_offset=0.03,
                )

    visualizer.visualize(
        translations=torch.tensor(translations_data, dtype=torch.float32, device=device),
        orientations=torch.tensor(orientations_data, dtype=torch.float32, device=device),
        scales=torch.tensor(scales_data, dtype=torch.float32, device=device),
        marker_indices=torch.tensor(marker_indices_data, dtype=torch.long, device=device),
    )
    return visualizer


def _print_summary(
    source_label: str,
    terrain_names: tuple[str, ...],
    difficulties: tuple[float, ...],
    reset_info: TaskResetInfo | None,
    regions: dict[str, ResetRegion] | None,
) -> None:
    """Print the grid legend and resolved reset ranges."""

    print(f"[INFO] Visualizing {source_label}.")
    print("[INFO] Grid axes: rows run along +X; columns run along +Y.")
    print(
        "[INFO] Difficulty rows: "
        + ", ".join(
            f"row {row} -> {difficulty:.4f}"
            for row, difficulty in enumerate(difficulties)
        )
    )
    print(
        "[INFO] Terrain columns: "
        + ", ".join(
            f"col {column} -> {terrain_name}"
            for column, terrain_name in enumerate(terrain_names)
        )
    )

    if reset_info is not None and regions is not None:
        print(
            f"[INFO] Reset event: {reset_info.event_name} ({reset_info.function_name}); "
            f"default root position={reset_info.default_root_position}."
        )
        print(
            "[INFO] Reset colors: blue/cyan = default range; "
            "green/yellow = terrain-specific range."
        )
        print("[INFO] Resolved terrain-local reset footprints:")
        for column, terrain_name in enumerate(terrain_names):
            region = regions[terrain_name]
            source = "terrain-specific" if region.is_terrain_specific else "default"
            if region.is_trapezoid:
                shape = (
                    f"trapezoid x={region.x_range}, y@x_min={region.y_at_x_min}, "
                    f"y@x_max={region.y_at_x_max}"
                )
            else:
                shape = f"rectangle x={region.x_range}, y={region.y_at_x_min}"
            print(f"       col {column}: {terrain_name}: {shape} ({source})")

        configured_names = set(
            (reset_info.params.get("terrain_specific_pos_range") or {}).keys()
        ) | set((reset_info.params.get("terrain_specific_pose_range") or {}).keys())
        unused_names = sorted(configured_names.difference(terrain_names))
        if unused_names:
            print(
                "[WARN] Task reset ranges reference terrains absent from this grid: "
                + ", ".join(unused_names)
            )

        extra_spawn_modes = []
        for key in (
            "terrain_specific_s_curve_path_spawn",
            "terrain_specific_motion_files",
            "terrain_specific_spawn_local_ground",
        ):
            value = reset_info.params.get(key)
            if isinstance(value, dict) and value:
                extra_spawn_modes.append(key)
        if extra_spawn_modes:
            print(
                "[WARN] Task also configures non-footprint spawn behavior not represented by the XY overlay: "
                + ", ".join(extra_spawn_modes)
            )

    print("[INFO] Press Ctrl+C or close Isaac Sim to stop.")


def main() -> None:
    """Create the terrain-only stage and keep it open for inspection."""

    if args_cli.task is not None:
        terrain_importer_cfg, reset_info, source_label = _load_task_source()
    else:
        terrain_importer_cfg, reset_info, source_label = _load_direct_cfg_source()

    terrain_names, difficulties = _configure_grid(terrain_importer_cfg)
    generator_cfg = terrain_importer_cfg.terrain_generator

    sim = SimulationContext(SimulationCfg(dt=0.01, device=args_cli.device))
    total_length = generator_cfg.num_rows * generator_cfg.size[0]
    total_width = generator_cfg.num_cols * generator_cfg.size[1]
    view_scale = max(total_length, total_width, 10.0)
    sim.set_camera_view(
        eye=(-0.8 * total_length, -0.8 * total_width, 0.9 * view_scale),
        target=(0.0, 0.0, 0.0),
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/SkyLight", light_cfg)

    terrain_importer = terrain_importer_cfg.class_type(terrain_importer_cfg)

    regions = None
    reset_region_visualizer = None
    if reset_info is not None:
        regions = _resolve_reset_regions(terrain_names, reset_info)
        reset_region_visualizer = _create_reset_region_visualizer(
            terrain_importer,
            terrain_names,
            difficulties,
            regions,
            sim.device,
        )

    sim.reset()
    _print_summary(source_label, terrain_names, difficulties, reset_info, regions)

    # Keep the generated objects alive while their USD prims are rendered.
    _ = terrain_importer, reset_region_visualizer
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
