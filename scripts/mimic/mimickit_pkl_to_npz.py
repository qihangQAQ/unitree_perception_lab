"""Convert MimicKit G1-29DoF motion PKLs to this project's training NPZ format.

Run this script with the Python environment used for Isaac Lab. Each output NPZ
is saved beside its input PKL with the same filename stem, then played in an
Isaac Sim window. Conversion uses the G1 articulation in csv_to_npz.py to
populate the tracked body states.
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


class BuiltinOnlyUnpickler(pickle.Unpickler):
    """The MimicKit motion format only needs builtin dicts, lists, and numbers."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent pickle references are unsupported")


def load_motion(path: Path) -> tuple[np.ndarray, float]:
    with path.open("rb") as stream:
        motion = BuiltinOnlyUnpickler(stream).load()

    if not isinstance(motion, dict) or not {"fps", "frames"}.issubset(motion):
        raise ValueError("Expected a MimicKit motion dictionary with fps and frames")

    fps = motion["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion fps: {fps!r}")

    try:
        frames = np.asarray(motion["frames"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Motion frames must be a rectangular numeric array") from exc
    if frames.ndim != 2 or frames.shape[0] < 3 or frames.shape[1] != 35:
        raise ValueError(f"Expected at least 3 frames with 3+3+29 columns; got {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("Motion frames contain NaN or infinity")

    return frames, float(fps)


def exp_map_to_xyzw(exp_map: np.ndarray) -> np.ndarray:
    """Convert MimicKit root rotation vectors to unit xyzw quaternions."""
    angle = np.linalg.norm(exp_map, axis=1)
    half_angle = 0.5 * angle
    scale = np.empty_like(angle)
    nonzero = angle > 1.0e-8
    scale[nonzero] = np.sin(half_angle[nonzero]) / angle[nonzero]
    scale[~nonzero] = 0.5 - angle[~nonzero] ** 2 / 48.0

    quat = np.empty((len(exp_map), 4), dtype=np.float64)
    quat[:, :3] = exp_map * scale[:, None]
    quat[:, 3] = np.cos(half_angle)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # Equivalent quaternions can have opposite signs near the pi boundary.
    # Keep adjacent frames on the same branch for the CSV converter's slerp.
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    return quat


def make_csv_motion(frames: np.ndarray) -> np.ndarray:
    return np.column_stack((frames[:, :3], exp_map_to_xyzw(frames[:, 3:6]), frames[:, 6:]))


def validate_npz(path: Path, expected_fps: int) -> int:
    shapes = {
        "joint_pos": (29,),
        "joint_vel": (29,),
        "body_pos_w": (30, 3),
        "body_quat_w": (30, 4),
        "body_lin_vel_w": (30, 3),
        "body_ang_vel_w": (30, 3),
    }
    with np.load(path, allow_pickle=False) as data:
        missing = {"fps", *shapes}.difference(data.files)
        if missing:
            raise ValueError(f"Generated NPZ is missing fields: {', '.join(sorted(missing))}")
        fps = np.asarray(data["fps"]).reshape(-1)
        if fps.size != 1 or fps[0] != expected_fps:
            raise ValueError(f"Generated NPZ has unexpected fps: {fps}")
        frame_count = data["joint_pos"].shape[0]
        if frame_count < 3:
            raise ValueError(f"Generated NPZ has only {frame_count} frames")
        for name, shape in shapes.items():
            values = data[name]
            if values.shape != (frame_count, *shape) or not np.isfinite(values).all():
                raise ValueError(f"Generated NPZ has invalid {name}: {values.shape}")
        quat_norms = np.linalg.norm(data["body_quat_w"], axis=-1)
        if not np.allclose(quat_norms, 1.0, atol=1.0e-3):
            raise ValueError("Generated body quaternions are not unit length")
    return frame_count


def convert(path: Path, output_fps: int, device: str | None, force: bool, preview: bool) -> None:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".pkl" or not path.is_file():
        raise ValueError(f"Input must be an existing .pkl file: {path}")
    output = path.with_suffix(".npz")
    if output.exists() and not force:
        raise ValueError(f"Output already exists: {output}. Use --force to replace it")

    frames, input_fps = load_motion(path)
    csv_motion = make_csv_motion(frames)
    converter = Path(__file__).with_name("csv_to_npz.py")

    with tempfile.TemporaryDirectory(prefix=f".{path.stem}_conversion_", dir=path.parent) as temp_dir:
        temp_path = Path(temp_dir)
        csv_path = temp_path / f"{path.stem}.csv"
        npz_path = temp_path / f"{path.stem}.npz"
        np.savetxt(csv_path, csv_motion, delimiter=",", fmt="%.10g")

        command = [
            sys.executable,
            str(converter),
            "-f",
            str(csv_path),
            "--input_fps",
            str(input_fps),
            "--output_fps",
            str(output_fps),
            "--output_name",
            str(npz_path),
            "--headless",
            "--exit_after_save",
        ]
        if device:
            command.extend(("--device", device))
        subprocess.run(command, check=True)
        if not npz_path.is_file():
            raise ValueError("Isaac Lab exited without creating an NPZ; check its simulator output above")
        frame_count = validate_npz(npz_path, output_fps)
        os.replace(npz_path, output)

    print(f"Saved {output}: {frame_count} frames at {output_fps} FPS")
    if preview:
        print("Opening motion preview; close the Isaac Sim window to continue.", flush=True)
        command = [sys.executable, str(Path(__file__).with_name("replay_npz.py")), "-f", str(output)]
        if device:
            command.extend(("--device", device))
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--input_file", type=Path, nargs="+", required=True, help="MimicKit G1 motion PKL(s)")
    parser.add_argument("--output_fps", type=int, default=50, help="Output frame rate (default: 50)")
    parser.add_argument("--device", type=str, help="Isaac Lab device, for example cuda:0 or cpu")
    parser.add_argument("--force", action="store_true", help="Replace an existing output NPZ")
    parser.add_argument("--no-preview", action="store_true", help="Generate the NPZ without opening a playback window")
    args = parser.parse_args()
    if args.output_fps <= 0:
        parser.error("--output_fps must be positive")

    try:
        for path in args.input_file:
            convert(path, args.output_fps, args.device, args.force, not args.no_preview)
    except (OSError, ValueError, pickle.UnpicklingError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
