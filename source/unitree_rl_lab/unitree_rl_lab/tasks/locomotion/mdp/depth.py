# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Depth-camera observations and GPU noise models adapted from TRAIN_E1ObstacleRace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.sensors import MultiMeshRayCasterCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RandomRectMaskGPU:
    """Apply temporally persistent rectangular invalid regions to depth images."""

    def __init__(
        self,
        probability: float,
        num_masks: tuple[int, int],
        rect_ratio: tuple[float, float],
        persist_steps: int,
    ):
        self.probability = probability
        self.num_masks = num_masks
        self.rect_ratio = rect_ratio
        self.persist_steps = persist_steps
        self._mask: torch.Tensor | None = None
        self._counter = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = depth.shape
        device = depth.device
        if self._mask is not None and self._mask.shape == depth.shape and self._counter < self.persist_steps:
            self._counter += 1
            return depth * self._mask

        mask = torch.ones_like(depth)
        apply_mask = torch.rand(batch_size, device=device) < self.probability
        if apply_mask.any():
            num_masks = torch.randint(self.num_masks[0], self.num_masks[1] + 1, (batch_size,), device=device)
            max_masks = int(num_masks.max().item())
            rect_height = (
                torch.rand(batch_size, max_masks, device=device) * (self.rect_ratio[1] - self.rect_ratio[0])
                + self.rect_ratio[0]
            )
            rect_width = (
                torch.rand(batch_size, max_masks, device=device) * (self.rect_ratio[1] - self.rect_ratio[0])
                + self.rect_ratio[0]
            )
            rect_height = (rect_height * height).long().clamp(min=1, max=height)
            rect_width = (rect_width * width).long().clamp(min=1, max=width)
            y_start = (torch.rand(batch_size, max_masks, device=device) * (height - rect_height).float()).long()
            x_start = (torch.rand(batch_size, max_masks, device=device) * (width - rect_width).float()).long()
            yy = torch.arange(height, device=device).view(1, 1, height, 1)
            xx = torch.arange(width, device=device).view(1, 1, 1, width)

            for mask_index in range(max_masks):
                valid = (mask_index < num_masks) & apply_mask
                rectangle = (
                    (yy >= y_start[:, mask_index].view(batch_size, 1, 1, 1))
                    & (yy < (y_start[:, mask_index] + rect_height[:, mask_index]).view(batch_size, 1, 1, 1))
                    & (xx >= x_start[:, mask_index].view(batch_size, 1, 1, 1))
                    & (xx < (x_start[:, mask_index] + rect_width[:, mask_index]).view(batch_size, 1, 1, 1))
                    & valid.view(batch_size, 1, 1, 1)
                )
                mask[rectangle] = 0.0

        self._mask = mask
        self._counter = 1
        return depth * mask


class HighReflectionNoiseGPU:
    """Apply temporally persistent multiplicative high-reflection patches."""

    def __init__(
        self,
        probability: float,
        num_patches: tuple[int, int],
        rect_ratio: tuple[float, float],
        coefficient_range: tuple[float, float],
        persist_steps: int,
    ):
        self.probability = probability
        self.num_patches = num_patches
        self.rect_ratio = rect_ratio
        self.coefficient_range = coefficient_range
        self.persist_steps = persist_steps
        self._coefficient_map: torch.Tensor | None = None
        self._counter = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = depth.shape
        device = depth.device
        if (
            self._coefficient_map is not None
            and self._coefficient_map.shape == depth.shape
            and self._counter < self.persist_steps
        ):
            self._counter += 1
            return depth * self._coefficient_map

        coefficient_map = torch.ones_like(depth)
        apply_noise = torch.rand(batch_size, device=device) < self.probability
        if apply_noise.any():
            num_patches = torch.randint(self.num_patches[0], self.num_patches[1] + 1, (batch_size,), device=device)
            max_patches = int(num_patches.max().item())
            patch_height = (
                torch.rand(batch_size, max_patches, device=device) * (self.rect_ratio[1] - self.rect_ratio[0])
                + self.rect_ratio[0]
            )
            patch_width = (
                torch.rand(batch_size, max_patches, device=device) * (self.rect_ratio[1] - self.rect_ratio[0])
                + self.rect_ratio[0]
            )
            patch_height = (patch_height * height).long().clamp(min=1, max=height)
            patch_width = (patch_width * width).long().clamp(min=1, max=width)
            y_start = (torch.rand(batch_size, max_patches, device=device) * (height - patch_height).float()).long()
            x_start = (torch.rand(batch_size, max_patches, device=device) * (width - patch_width).float()).long()
            coefficients = (
                torch.rand(batch_size, max_patches, device=device)
                * (self.coefficient_range[1] - self.coefficient_range[0])
                + self.coefficient_range[0]
            )
            yy = torch.arange(height, device=device).view(1, 1, height, 1)
            xx = torch.arange(width, device=device).view(1, 1, 1, width)

            for patch_index in range(max_patches):
                valid = (patch_index < num_patches) & apply_noise
                rectangle = (
                    (yy >= y_start[:, patch_index].view(batch_size, 1, 1, 1))
                    & (yy < (y_start[:, patch_index] + patch_height[:, patch_index]).view(batch_size, 1, 1, 1))
                    & (xx >= x_start[:, patch_index].view(batch_size, 1, 1, 1))
                    & (xx < (x_start[:, patch_index] + patch_width[:, patch_index]).view(batch_size, 1, 1, 1))
                    & valid.view(batch_size, 1, 1, 1)
                )
                coefficient_map = torch.where(
                    rectangle,
                    coefficients[:, patch_index].view(batch_size, 1, 1, 1),
                    coefficient_map,
                )

        self._coefficient_map = coefficient_map
        self._counter = 1
        return depth * coefficient_map


class FullImageMaskGPU:
    """Drop complete depth frames for a persistent number of observation steps."""

    def __init__(self, probability: float, persist_steps: int):
        self.probability = probability
        self.persist_steps = persist_steps
        self._mask: torch.Tensor | None = None
        self._counter = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        if self._mask is not None and self._mask.shape == depth.shape and self._counter < self.persist_steps:
            self._counter += 1
            return depth * self._mask
        mask = torch.ones_like(depth)
        mask[torch.rand(depth.shape[0], device=depth.device) < self.probability] = 0.0
        self._mask = mask
        self._counter = 1
        return depth * mask


_depth_maskers: dict[int, RandomRectMaskGPU] = {}
_full_image_maskers: dict[int, FullImageMaskGPU] = {}
_high_reflection_noisers: dict[int, HighReflectionNoiseGPU] = {}


def add_depth_noise(
    depth: torch.Tensor,
    distance_noise: float = 0.0,
    gaussian_std: float = 0.0,
    salt_pepper_prob: float = 0.0,
    stripe_prob: float = 0.0,
    edge_drag_prob: float = 0.0,
    min_range: float = 0.0,
    max_range: float = 2.0,
    high_reflection_prob: float = 0.0,
    high_reflection_num_patches: tuple[int, int] = (1, 3),
    high_reflection_rect_ratio: tuple[float, float] = (0.1, 0.25),
    high_reflection_coefficient_range: tuple[float, float] = (1.1, 1.5),
    high_reflection_persist_steps: int = 3,
    mask_prob: float = 0.0,
    mask_num_masks: tuple[int, int] = (1, 3),
    mask_rect_ratio: tuple[float, float] = (0.1, 0.25),
    mask_persist_steps: int = 3,
    full_image_mask_prob: float = 0.0,
    full_image_mask_persist_steps: int = 3,
    full_image_random_prob: float = 0.0,
    env_id: int | None = None,
) -> torch.Tensor:
    """Apply the TRAIN_E1ObstacleRace depth corruption model on GPU."""
    noisy = depth.clone()
    original_shape = noisy.shape
    original_dim = noisy.dim()
    if original_dim == 3:
        noisy = noisy.unsqueeze(1)
    elif original_dim == 4 and original_shape[-1] == 1:
        noisy = noisy.permute(0, 3, 1, 2)
    if noisy.dim() != 4 or noisy.shape[1] != 1:
        raise ValueError(f"Expected depth shape (B,H,W), (B,1,H,W), or (B,H,W,1), got {original_shape}.")

    batch_size, _, height, width = noisy.shape
    device = noisy.device
    if distance_noise > 0:
        noisy += (torch.rand_like(noisy) * 2.0 - 1.0) * distance_noise * noisy.clamp(min=1.0e-3)
    if gaussian_std > 0:
        noisy += torch.randn_like(noisy) * gaussian_std * noisy.clamp(min=1.0e-3)
    if salt_pepper_prob > 0:
        random_values = torch.rand_like(noisy)
        noisy[random_values < salt_pepper_prob / 2.0] = 0.0
        noisy[(random_values >= salt_pepper_prob / 2.0) & (random_values < salt_pepper_prob)] = max_range
    if stripe_prob > 0:
        horizontal = torch.rand(batch_size, 1, height, 1, device=device) < stripe_prob
        vertical = torch.rand(batch_size, 1, 1, width, device=device) < stripe_prob
        noisy = torch.where(horizontal | vertical, torch.zeros_like(noisy), noisy)
    if edge_drag_prob > 0:
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=noisy.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=noisy.dtype).view(1, 1, 3, 3)
        grad_x = F.conv2d(noisy, sobel_x, padding=1)
        grad_y = F.conv2d(noisy, sobel_y, padding=1)
        gradient = torch.sqrt(grad_x.square() + grad_y.square())
        edge_mask = gradient > gradient.mean(dim=(2, 3), keepdim=True) * 2.0
        drag_mask = (torch.rand_like(noisy) < edge_drag_prob) & edge_mask
        noisy = torch.where(drag_mask, torch.roll(noisy, shifts=1, dims=3), noisy)

    if high_reflection_prob > 0:
        key = env_id if env_id is not None else id(noisy)
        if key not in _high_reflection_noisers:
            _high_reflection_noisers[key] = HighReflectionNoiseGPU(
                high_reflection_prob,
                high_reflection_num_patches,
                high_reflection_rect_ratio,
                high_reflection_coefficient_range,
                high_reflection_persist_steps if env_id is not None else 1,
            )
        noisy = _high_reflection_noisers[key](noisy)
    if mask_prob > 0:
        key = env_id if env_id is not None else id(noisy)
        if key not in _depth_maskers:
            _depth_maskers[key] = RandomRectMaskGPU(
                mask_prob,
                mask_num_masks,
                mask_rect_ratio,
                mask_persist_steps if env_id is not None else 1,
            )
        noisy = _depth_maskers[key](noisy)
    if full_image_mask_prob > 0:
        key = env_id if env_id is not None else id(noisy)
        if key not in _full_image_maskers:
            _full_image_maskers[key] = FullImageMaskGPU(
                full_image_mask_prob,
                full_image_mask_persist_steps if env_id is not None else 1,
            )
        noisy = _full_image_maskers[key](noisy)
    if full_image_random_prob > 0:
        replace = torch.rand(batch_size, 1, 1, 1, device=device) < full_image_random_prob
        random_depth = torch.rand_like(noisy) * (max_range - min_range) + min_range
        noisy = torch.where(replace, random_depth, noisy)

    noisy[(noisy < min_range) | (noisy > max_range)] = 0.0
    if original_dim == 3:
        return noisy.squeeze(1)
    if original_dim == 4 and original_shape[-1] == 1:
        return noisy.permute(0, 2, 3, 1)
    return noisy


def raycaster_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("raycaster_camera"),
    distance_noise: float = 0.0,
    gaussian_std: float = 0.0,
    salt_pepper_prob: float = 0.0,
    stripe_prob: float = 0.0,
    edge_drag_prob: float = 0.0,
    min_range: float = 0.0,
    max_range: float | None = None,
    high_reflection_prob: float = 0.0,
    high_reflection_num_patches: tuple[int, int] = (1, 3),
    high_reflection_rect_ratio: tuple[float, float] = (0.1, 0.25),
    high_reflection_coefficient_range: tuple[float, float] = (1.1, 1.5),
    high_reflection_persist_steps: int = 3,
    mask_prob: float = 0.0,
    mask_num_masks: tuple[int, int] = (1, 3),
    mask_rect_ratio: tuple[float, float] = (0.1, 0.25),
    mask_persist_steps: int = 3,
    full_image_mask_prob: float = 0.0,
    full_image_mask_persist_steps: int = 3,
    full_image_random_prob: float = 0.0,
) -> torch.Tensor:
    """Return normalized channels-last depth images from a multi-mesh ray-caster camera."""
    camera: MultiMeshRayCasterCamera = env.scene.sensors[sensor_cfg.name]
    if max_range is None:
        max_range = camera.cfg.max_distance
    depth = add_depth_noise(
        camera.data.output["distance_to_image_plane"],
        distance_noise=distance_noise,
        gaussian_std=gaussian_std,
        salt_pepper_prob=salt_pepper_prob,
        stripe_prob=stripe_prob,
        edge_drag_prob=edge_drag_prob,
        min_range=min_range,
        max_range=max_range,
        high_reflection_prob=high_reflection_prob,
        high_reflection_num_patches=high_reflection_num_patches,
        high_reflection_rect_ratio=high_reflection_rect_ratio,
        high_reflection_coefficient_range=high_reflection_coefficient_range,
        high_reflection_persist_steps=high_reflection_persist_steps,
        mask_prob=mask_prob,
        mask_num_masks=mask_num_masks,
        mask_rect_ratio=mask_rect_ratio,
        mask_persist_steps=mask_persist_steps,
        full_image_mask_prob=full_image_mask_prob,
        full_image_mask_persist_steps=full_image_mask_persist_steps,
        full_image_random_prob=full_image_random_prob,
        env_id=id(env),
    )
    return depth - 1.0
