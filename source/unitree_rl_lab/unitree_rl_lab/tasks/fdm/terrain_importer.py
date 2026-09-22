"""Minimal split-aware USD terrain importer adapted from Nav-Suite."""

from __future__ import annotations

import math

import torch
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaacsim.core.utils import prims as prim_utils
from pxr import Usd, UsdGeom


class SplitAwareUsdTerrainImporter(TerrainImporter):
    """Tile one merged USD and bind every environment to one spatial split."""

    cfg: "SplitAwareUsdTerrainImporterCfg"

    def configure_env_origins(self, origins=None) -> None:
        if origins is not None or self.cfg.terrain_type != "usd":
            super().configure_env_origins(origins)
            return
        prim_path = f"{self.cfg.prim_path}/terrain"
        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Unable to compute USD bounds: invalid prim {prim_path}.")
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        spacing = self.cfg.usd_uniform_env_spacing
        x = torch.arange(minimum[0] + spacing / 2, maximum[0] - spacing / 2, spacing)
        y = torch.arange(minimum[1] + spacing / 2, maximum[1] - spacing / 2, spacing)
        if x.numel() == 0 or y.numel() == 0:
            raise RuntimeError(
                f"USD bounds {tuple(minimum)}..{tuple(maximum)} contain no {spacing:g} m origin cell."
            )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        all_origins = torch.stack((grid_x.flatten(), grid_y.flatten(), torch.zeros(grid_x.numel())), dim=-1)
        number_of_cells = all_origins.shape[0]
        if number_of_cells < 3:
            raise RuntimeError("A train/validation/test spatial split requires at least three USD cells.")

        train_count = max(1, int(math.floor(number_of_cells * self.cfg.split_ratios[0])))
        validation_count = max(1, int(math.floor(number_of_cells * self.cfg.split_ratios[1])))
        if train_count + validation_count >= number_of_cells:
            train_count = number_of_cells - 2
            validation_count = 1
        split_ranges = {
            "train": (0, train_count),
            "val": (train_count, train_count + validation_count),
            "test": (train_count + validation_count, number_of_cells),
        }
        self.terrain_origins = None
        self.all_usd_origins = all_origins.to(self.device)
        self._split_ranges = split_ranges
        self.usd_bbox = (
            (float(minimum[0]), float(minimum[1]), float(minimum[2])),
            (float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )
        self.split_origin_ids = {
            name: list(range(range_start, range_stop)) for name, (range_start, range_stop) in split_ranges.items()
        }
        split_values = torch.empty(number_of_cells, dtype=torch.int8, device=self.device)
        split_values[:train_count] = 0
        split_values[train_count : train_count + validation_count] = 1
        split_values[train_count + validation_count :] = 2
        self.origin_split_ids = split_values
        self.activate_split(self.cfg.active_split)

    def activate_split(self, split: str) -> None:
        """Rebind all parallel environments to cells from one fixed spatial split."""

        if split not in self._split_ranges:
            raise ValueError(f"Invalid spatial split {split!r}.")
        start, stop = self._split_ranges[split]
        selected_ids = torch.arange(start, stop, dtype=torch.long, device=self.device)
        repetitions = math.ceil(self.cfg.num_envs / len(selected_ids))
        self.env_origin_ids = selected_ids.repeat(repetitions)[: self.cfg.num_envs]
        self.env_origins = self.all_usd_origins[self.env_origin_ids]
        self.cfg.active_split = split


@configclass
class SplitAwareUsdTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = SplitAwareUsdTerrainImporter
    usd_uniform_env_spacing: float = 10.0
    active_split: str = "train"
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if self.active_split not in ("train", "val", "test"):
            raise ValueError(f"Invalid active_split {self.active_split!r}.")
        if abs(sum(self.split_ratios) - 1.0) > 1.0e-6:
            raise ValueError("split_ratios must sum to one.")
        if self.usd_uniform_env_spacing <= 0.0:
            raise ValueError("usd_uniform_env_spacing must be positive.")
        # The base validation requires this for USD even though our override does
        # not use its generic centered environment grid.
        self.env_spacing = self.usd_uniform_env_spacing
