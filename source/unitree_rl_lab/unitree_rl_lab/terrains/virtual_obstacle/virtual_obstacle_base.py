"""Base interface for analytical virtual obstacles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING

import torch
import trimesh
from isaaclab.utils import configclass


@configclass
class VirtualObstacleCfg:
    class_type: type = MISSING


class VirtualObstacleBase(ABC):
    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu"):
        raise NotImplementedError

    @abstractmethod
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def visualize(self):
        """Virtual-obstacle visualization is intentionally disabled for training."""

    def disable_visualizer(self):
        """Virtual-obstacle visualization is intentionally disabled for training."""
