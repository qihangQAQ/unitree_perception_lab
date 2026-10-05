"""NumPy/ONNX Runtime deployment adapter; no Isaac Sim or PyTorch dependency."""

from __future__ import annotations

import numpy as np


def wall_targets_to_base(surface_points_world, base_position, base_quaternion, distance):
    """Convert five wall points to TCP targets and one common spray axis (wxyz)."""
    surface = np.asarray(surface_points_world, dtype=np.float32)
    if surface.shape != (5, 3):
        raise ValueError("Expected five xyz surface targets.")
    q = np.asarray(base_quaternion, dtype=np.float32)
    if q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
        raise ValueError("Expected a unit wxyz base quaternion.")
    target = surface.copy()
    target[:, 0] -= distance
    values = np.concatenate((target - np.asarray(base_position), np.array([[1.0, 0.0, 0.0]])), axis=0)
    xyz = -q[1:]
    uv = np.cross(xyz, values)
    rotated = values + 2 * (q[0] * uv + np.cross(xyz, uv))
    return rotated[:5].astype(np.float32), rotated[5].astype(np.float32)


class PaintingPolicy:
    """Maintain proprioceptive history and map actions to joint position targets.

    Joint arrays must use deploy.yaml's painting_inference.joint_names order.
    Hardware transport and the world-to-base localization transform belong to
    the caller. ``step`` returns position targets in that same joint order.
    """

    def __init__(self, onnx_path, deploy_config):
        metadata = deploy_config["painting_inference"]
        if metadata.get("interface_version") != 2:
            raise ValueError("PaintingPolicy requires the explicit V2/27-command deployment interface.")
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except ImportError:
            # The ONNX reference evaluator keeps offline validation available
            # in simulator environments that do not bundle ONNX Runtime.
            from onnx.reference import ReferenceEvaluator

            self.session = ReferenceEvaluator(str(onnx_path))
        self.nominal = np.asarray(deploy_config["default_joint_pos"], dtype=np.float32)
        self.scale = np.asarray(deploy_config["actions"]["JointPositionAction"]["scale"], dtype=np.float32)
        self.offset = np.asarray(deploy_config["actions"]["JointPositionAction"]["offset"], dtype=np.float32)
        self.angular_scale = metadata["angular_velocity_scale"]
        self.velocity_scale = metadata["joint_velocity_scale"]
        self.action_clip = metadata["action_clip"]
        self.raw_action_limits = np.asarray(metadata["raw_action_limits"], dtype=np.float32)
        self.target_position_limits = np.asarray(metadata["target_position_limits"], dtype=np.float32)
        if any(x.shape != (29,) for x in (self.nominal, self.scale, self.offset)):
            raise ValueError("Painting joint metadata must have 29 values.")
        if self.raw_action_limits.shape != (29, 2) or self.target_position_limits.shape != (29, 2):
            raise ValueError("Painting V2 requires 29 per-joint raw-action and target-position limits.")
        self.reset()

    def reset(self):
        self.history = np.zeros((5, 93), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.initialized = False

    def step(self, angular_velocity, projected_gravity, joint_position, joint_velocity,
             target_positions_base, spray_direction_base, desired_speed,
             base_velocity_command, base_anchor_position, end_effector_active,
             startup_countdown, remaining_time):
        frame = np.concatenate((
            np.asarray(angular_velocity) * self.angular_scale,
            np.asarray(projected_gravity),
            np.asarray(joint_position) - self.nominal,
            np.asarray(joint_velocity) * self.velocity_scale,
            self.last_action,
        )).astype(np.float32)
        targets = np.asarray(target_positions_base, dtype=np.float32)
        direction = np.asarray(spray_direction_base, dtype=np.float32)
        base_command = np.asarray(base_velocity_command, dtype=np.float32)
        anchor = np.asarray(base_anchor_position, dtype=np.float32)
        if (
            frame.shape != (93,)
            or targets.shape != (5, 3)
            or direction.shape != (3,)
            or base_command.shape != (3,)
            or anchor.shape != (2,)
        ):
            raise ValueError("Invalid painting observation shape.")
        command = np.concatenate((
            targets.ravel(), direction, [desired_speed], base_command, anchor,
            [end_effector_active, startup_countdown, remaining_time],
        )).astype(np.float32)
        if not np.isfinite(frame).all() or not np.isfinite(command).all():
            raise ValueError("Painting observations must be finite.")
        if self.initialized:
            self.history[:-1] = self.history[1:].copy()
            self.history[-1] = frame
        else:
            self.history[:] = frame
            self.initialized = True
        actions = self.session.run(["actions"], {
            "proprio_history": self.history.reshape(1, 465),
            "trajectory_command": command.reshape(1, 27),
        })[0][0]
        self.last_action = np.clip(actions, self.raw_action_limits[:, 0], self.raw_action_limits[:, 1])
        targets = self.offset + self.scale * self.last_action
        return np.clip(targets, self.target_position_limits[:, 0], self.target_position_limits[:, 1])
