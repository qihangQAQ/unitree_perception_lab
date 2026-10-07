"""Convert a trusted UMR G1-29DoF retarget NPZ to this repository's motion CSV.

UMR stores a MuJoCo qpos trajectory: root position, root quaternion in wxyz
order, then robot joints. The CSV consumed by csv_to_npz.py uses root position,
root quaternion in xyzw order, then joints in Unitree SDK order.
"""

import argparse
from pathlib import Path

import numpy as np


G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def convert_motion(input_file: Path) -> tuple[np.ndarray, float]:
    # UMR writes robot_joint_names as an object array, which requires pickle.
    # Only use this script with NPZ files from a trusted UMR run.
    with np.load(input_file, allow_pickle=True) as data:
        required = {"qpos", "fps", "robot_joint_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing UMR fields: {', '.join(sorted(missing))}")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        names = [str(name) for name in data["robot_joint_names"].tolist()]
        fps_values = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
        frame_ids = np.asarray(data["frame_ids"]) if "frame_ids" in data else None

    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != 7 + len(names):
        raise ValueError(f"Expected at least two frames of root pose + {len(names)} joints; got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or infinity")
    if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0:
        raise ValueError(f"Invalid fps: {fps_values}")
    if len(names) != len(G1_JOINT_NAMES) or len(set(names)) != len(names) or set(names) != set(G1_JOINT_NAMES):
        missing_joints = sorted(set(G1_JOINT_NAMES) - set(names))
        extra_joints = sorted(set(names) - set(G1_JOINT_NAMES))
        raise ValueError(f"Expected G1-29DoF joints; missing={missing_joints}, extra={extra_joints}")
    if frame_ids is not None:
        valid_frame_ids = (
            frame_ids.ndim == 1
            and len(frame_ids) == len(qpos)
            and frame_ids[1] > frame_ids[0]
            and np.all(np.diff(frame_ids) == frame_ids[1] - frame_ids[0])
        )
        if not valid_frame_ids:
            raise ValueError("frame_ids must be uniformly spaced and match qpos frames")

    quats_wxyz = qpos[:, 3:7].copy()
    norms = np.linalg.norm(quats_wxyz, axis=1)
    if np.any(np.abs(norms - 1.0) > 0.01):
        raise ValueError("Root quaternions are not unit length")
    quats_wxyz /= norms[:, None]
    for i in range(1, len(quats_wxyz)):
        if np.dot(quats_wxyz[i - 1], quats_wxyz[i]) < 0:
            quats_wxyz[i] *= -1

    joint_order = [names.index(name) for name in G1_JOINT_NAMES]
    csv = np.concatenate((qpos[:, :3], quats_wxyz[:, [1, 2, 3, 0]], qpos[:, 7:][:, joint_order]), axis=1)
    return csv, float(fps_values[0])


def main():
    parser = argparse.ArgumentParser(description="Convert a UMR G1-29DoF qpos NPZ to an Isaac Lab motion CSV.")
    parser.add_argument("-f", "--input_file", type=Path, required=True, help="UMR retarget NPZ file")
    parser.add_argument("-o", "--output_file", type=Path, help="CSV output; defaults to the input path with .csv")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing CSV")
    args = parser.parse_args()

    output_file = args.output_file or args.input_file.with_suffix(".csv")
    if output_file.exists() and not args.force:
        parser.error(f"Output already exists: {output_file}. Use --force to overwrite it.")
    try:
        csv, fps = convert_motion(args.input_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_file, csv, delimiter=",", fmt="%.9g")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {output_file}: {len(csv)} frames, {csv.shape[1]} columns, input FPS {fps:g}")
    training_npz = output_file.with_name(output_file.stem + "_isaaclab.npz")
    print(
        "Next: python scripts/mimic/csv_to_npz.py "
        f"-f {output_file} --input_fps {fps:g} --output_name {training_npz}"
    )


if __name__ == "__main__":
    main()
