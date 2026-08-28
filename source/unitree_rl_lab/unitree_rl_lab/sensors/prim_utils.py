# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.physics.tensors.impl.api as physx
import torch
from isaaclab.utils.math import convert_quat
from isaacsim.core.prims import XFormPrim
from pxr import Usd, UsdGeom


def obtain_world_pose_from_view(
    physx_view: XFormPrim | physx.ArticulationView | physx.RigidBodyView,
    env_ids: torch.Tensor,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the world poses of the prim referenced by the prim view.
    Args:
        physx_view: The prim view to get the world poses from.
        env_ids: The environment ids of the prims to get the world poses for.
        clone: Whether to clone the returned tensors (default: False).
    Returns:
        A tuple containing the world positions and orientations of the prims. Orientation is in wxyz format.
    Raises:
        NotImplementedError: If the prim view is not of the correct type.
    """
    if isinstance(physx_view, XFormPrim):
        pos_w, quat_w = physx_view.get_world_poses(env_ids)
    elif isinstance(physx_view, physx.ArticulationView):
        pos_w, quat_w = physx_view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    elif isinstance(physx_view, physx.RigidBodyView):
        pos_w, quat_w = physx_view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    else:
        raise NotImplementedError(f"Cannot get world poses for prim view of type '{type(physx_view)}'.")

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w


def resolve_prim_pose(
    prim: Usd.Prim, ref_prim: Usd.Prim | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve a USD prim pose in world frame or relative to another prim.

    IsaacLab 2.3.0 does not expose this helper, while the multi-mesh ray caster
    needs it to preserve the local offsets of moving visual meshes.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")

    prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetOrthonormalized()
    if ref_prim is not None:
        if not ref_prim.IsValid():
            raise ValueError(f"Ref prim at path '{ref_prim.GetPath().pathString}' is not valid.")
        ref_tf = UsdGeom.Xformable(ref_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetOrthonormalized()
        prim_tf = prim_tf * ref_tf.GetInverse()

    quat = prim_tf.ExtractRotationQuat()
    return tuple(prim_tf.ExtractTranslation()), (quat.real, *quat.imaginary)


def resolve_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Resolve the accumulated world scale of a USD prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(vector.GetLength() for vector in world_transform.ExtractRotationMatrix())
