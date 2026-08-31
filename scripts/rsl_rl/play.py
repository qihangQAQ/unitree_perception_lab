# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL checkpoint with keyboard velocity control.

The script loads each task's registered play environment configuration so play-only
event, termination, terrain, and visualization settings stay with the task definition.
"""

import argparse
import threading
from dataclasses import dataclass
from typing import Sequence

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# Manual velocity commands: (vx [m/s], vy [m/s], yaw_rate [rad/s]).
# Edit each key independently to tune interactive play control.
W_COMMAND = (1.0, 0.0, 0.0)
S_COMMAND = (-0.5, 0.0, 0.0)
A_COMMAND = (0.0, 0.3, 0.0)
D_COMMAND = (0.0, -0.3, 0.0)
Q_COMMAND = (0.0, 0.0, 1.0)
E_COMMAND = (0.0, 0.0, -1.0)


parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint with keyboard control.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video while playing.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in simulation steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (defaults to one).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    help="Sub-terrain name to spawn on. Must be used together with --level.",
)
parser.add_argument(
    "--level",
    type=int,
    default=None,
    help="Zero-based terrain curriculum level. Must be used together with --terrain.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the published pre-trained checkpoint for the selected task.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if (args_cli.terrain is None) != (args_cli.level is None):
    parser.error("--terrain and --level must be provided together.")
if args_cli.level is not None and args_cli.level < 0:
    parser.error("--level must be greater than or equal to zero.")
if args_cli.num_envs is not None and args_cli.num_envs < 1:
    parser.error("--num_envs must be greater than zero.")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below requires the simulator to be running."""

import os
import time

import carb
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import rsl_rl.runners as rsl_rl_runners
from rsl_rl.utils import string_to_callable
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


@dataclass(frozen=True)
class TerrainSelection:
    """Resolved terrain cell selected from the play-only terrain grid."""

    name: str
    level: int
    column: int
    num_rows: int
    num_cols: int


class FixedVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command term driven by a fixed external command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.manual_command = torch.zeros_like(self.vel_command_b)

    def set_manual_command(self, command: Sequence[float] | torch.Tensor):
        """Apply one ``(vx, vy, yaw_rate)`` command to every environment."""
        command_tensor = torch.as_tensor(command, dtype=self.vel_command_b.dtype, device=self.device)
        if command_tensor.shape != (3,):
            raise ValueError(f"Expected a three-element velocity command, got shape {tuple(command_tensor.shape)}.")
        self.manual_command[:] = command_tensor
        self.vel_command_b.copy_(self.manual_command)

    def reset_manual_command(self):
        """Clear the external command without changing any environment state."""
        self.manual_command.zero_()
        self.vel_command_b.zero_()

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = self.manual_command[env_ids]

    def _update_command(self):
        self.vel_command_b.copy_(self.manual_command)


class FixedVelocityKeyboard:
    """Omniverse keyboard listener for fixed, hold-to-command velocities."""

    _VELOCITY_KEYS = frozenset(("W", "S", "A", "D", "Q", "E"))

    def __init__(self, command_term: FixedVelocityCommand | None, reset_requested: threading.Event):
        import omni.appwindow

        self._command_term = command_term
        self._reset_requested = reset_requested
        self._pressed_keys: set[str] = set()
        self._last_reported_keys: frozenset[str] = frozenset()
        self._lock = threading.Lock()

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("No Omniverse application window is available for keyboard input.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event, *_) -> bool:
        # CHAR events carry a plain string in ``event.input``. Ignore them before
        # reading the enum-style ``name`` used by press/release events.
        if event.type not in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_RELEASE,
        ):
            return True

        key_input = event.input
        key_name = key_input if isinstance(key_input, str) else key_input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name == "R":
                with self._lock:
                    self._pressed_keys.clear()
                self._reset_requested.set()
            elif key_name in self._VELOCITY_KEYS:
                with self._lock:
                    self._pressed_keys.add(key_name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE and key_name in self._VELOCITY_KEYS:
            with self._lock:
                self._pressed_keys.discard(key_name)
        return True

    def apply_command(self):
        """Write the command represented by the currently held keys."""
        if self._command_term is None:
            return
        with self._lock:
            keys = self._pressed_keys.copy()
        command = _velocity_for_pressed_keys(keys)
        self._command_term.set_manual_command(command)

        reported_keys = frozenset(keys)
        if reported_keys != self._last_reported_keys:
            active_keys = "+".join(key for key in ("W", "S", "A", "D", "Q", "E") if key in keys) or "none"
            print(
                f"[INFO] Keyboard command: keys={active_keys}, "
                f"velocity=({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f})."
            )
            self._last_reported_keys = reported_keys

    def reset_command(self):
        """Clear pressed-key state and the command term."""
        with self._lock:
            self._pressed_keys.clear()
        self._last_reported_keys = frozenset()
        if self._command_term is not None:
            self._command_term.reset_manual_command()

    def close(self):
        """Unsubscribe from Omniverse keyboard events."""
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None


def _velocity_for_pressed_keys(keys: set[str]) -> tuple[float, float, float]:
    """Add the independently configured commands for all held WASDQE keys."""
    command = [0.0, 0.0, 0.0]
    if "W" in keys:
        command = [value + delta for value, delta in zip(command, W_COMMAND)]
    if "S" in keys:
        command = [value + delta for value, delta in zip(command, S_COMMAND)]
    if "A" in keys:
        command = [value + delta for value, delta in zip(command, A_COMMAND)]
    if "D" in keys:
        command = [value + delta for value, delta in zip(command, D_COMMAND)]
    if "Q" in keys:
        command = [value + delta for value, delta in zip(command, Q_COMMAND)]
    if "E" in keys:
        command = [value + delta for value, delta in zip(command, E_COMMAND)]
    return command[0], command[1], command[2]


def _configure_terrain_selection(env_cfg) -> TerrainSelection | None:
    """Generate only the selected terrain type and keep its requested level fixed."""
    if args_cli.terrain is None:
        return None
    if args_cli.num_envs not in (None, 1):
        raise ValueError("A fixed --terrain/--level selection supports exactly one environment.")

    scene_cfg = getattr(env_cfg, "scene", None)
    terrain_cfg = getattr(scene_cfg, "terrain", None)
    generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_cfg is None or terrain_cfg.terrain_type != "generator" or generator_cfg is None:
        raise ValueError(f"Task {args_cli.task!r} does not use a generated terrain grid.")

    num_rows = int(generator_cfg.num_rows)
    if args_cli.level >= num_rows:
        raise ValueError(
            f"Terrain level {args_cli.level} is out of range for task {args_cli.task!r}; "
            f"expected 0 through {num_rows - 1}."
        )

    available_names = list(generator_cfg.sub_terrains)
    if args_cli.terrain not in generator_cfg.sub_terrains:
        raise ValueError(
            f"Unknown terrain {args_cli.terrain!r}. Available terrains for this task: "
            f"{', '.join(available_names)}"
        )

    # Keep every curriculum row so --level retains the task's normal difficulty
    # semantics, but prune all unselected columns before TerrainImporter runs.
    selected_cfg = generator_cfg.sub_terrains[args_cli.terrain]
    selected_cfg.proportion = 1.0
    generator_cfg.sub_terrains = {args_cli.terrain: selected_cfg}
    generator_cfg.num_cols = 1
    generator_cfg.curriculum = True
    terrain_cfg.max_init_terrain_level = args_cli.level
    env_cfg.scene.num_envs = 1

    # Prevent the normal terrain curriculum from moving the robot away from the selected cell.
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None and hasattr(curriculum_cfg, "terrain_levels"):
        curriculum_cfg.terrain_levels = None

    return TerrainSelection(args_cli.terrain, args_cli.level, 0, num_rows, 1)


def _pin_terrain_cell(unwrapped_env, selection: TerrainSelection | None):
    """Point every environment origin at the selected terrain cell without resetting assets."""
    if selection is None:
        return
    terrain = unwrapped_env.scene.terrain
    if terrain.terrain_origins is None or not hasattr(terrain, "terrain_levels"):
        raise RuntimeError("The created environment does not expose curriculum terrain origins.")

    terrain.terrain_levels.fill_(selection.level)
    terrain.terrain_types.fill_(selection.column)
    terrain.env_origins[:] = terrain.terrain_origins[selection.level, selection.column]


def _enable_keyboard_velocity_command(env_cfg) -> bool:
    """Replace a compatible base_velocity term with the manual command term."""
    commands_cfg = getattr(env_cfg, "commands", None)
    command_cfg = getattr(commands_cfg, "base_velocity", None)
    ranges = getattr(command_cfg, "ranges", None)
    required_ranges = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    if command_cfg is None or ranges is None or not all(hasattr(ranges, name) for name in required_ranges):
        return False

    command_cfg.class_type = FixedVelocityCommand
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(command_cfg, "rel_standing_envs"):
        command_cfg.rel_standing_envs = 0.0
    if hasattr(command_cfg, "rel_heading_envs"):
        command_cfg.rel_heading_envs = 0.0
    if hasattr(command_cfg, "heading_command"):
        command_cfg.heading_command = False
    if hasattr(ranges, "heading"):
        ranges.heading = None
    return True


def _resolve_checkpoint(agent_cfg: RslRlOnPolicyRunnerCfg) -> str | None:
    """Resolve the checkpoint requested through the standard RSL-RL arguments."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not checkpoint:
            print("[INFO] A published pre-trained checkpoint is unavailable for this task.")
            return None
        return checkpoint
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _get_policy_module(runner):
    """Return the policy module across supported RSL-RL versions."""
    try:
        return runner.alg.policy
    except AttributeError:
        return runner.alg.actor_critic


def _get_policy_normalizer(policy_module):
    """Return the observation normalizer used by the exported policy, if present."""
    if hasattr(policy_module, "actor_obs_normalizer"):
        return policy_module.actor_obs_normalizer
    if hasattr(policy_module, "student_obs_normalizer"):
        return policy_module.student_obs_normalizer
    return None


def _resolve_runner_class(class_name: str):
    """Resolve standard RSL-RL runners and repository-local ``module:class`` runners."""
    if ":" in class_name:
        return string_to_callable(class_name)
    runner_class = getattr(rsl_rl_runners, class_name, None)
    if runner_class is None:
        raise ValueError(f"Unsupported runner class: {class_name}")
    return runner_class


def main():
    """Load a task and checkpoint, export ONNX, then run interactive inference."""
    # Keep play-only behavior in each task's registered RobotPlayEnvCfg.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    policy_obs_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy_obs_cfg is not None and hasattr(policy_obs_cfg, "enable_corruption"):
        policy_obs_cfg.enable_corruption = False

    terrain_selection = _configure_terrain_selection(env_cfg)
    keyboard_velocity_enabled = _enable_keyboard_velocity_command(env_cfg)

    resume_path = _resolve_checkpoint(agent_cfg)
    if resume_path is None:
        return
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _pin_terrain_cell(env.unwrapped, terrain_selection)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording play video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if terrain_selection is not None:
        print(
            f"[INFO] Terrain locked: name={terrain_selection.name!r}, level={terrain_selection.level}, "
            f"column={terrain_selection.column}, grid={terrain_selection.num_rows}x{terrain_selection.num_cols}."
        )

    command_term = None
    if keyboard_velocity_enabled:
        try:
            candidate = env.unwrapped.command_manager.get_term("base_velocity")
        except (AttributeError, KeyError):
            candidate = None
        if isinstance(candidate, FixedVelocityCommand):
            command_term = candidate
        else:
            print("[WARN] base_velocity was not created as FixedVelocityCommand; WASDQE is disabled.")

    reset_requested = threading.Event()
    keyboard_controller = None
    if not getattr(args_cli, "headless", False):
        keyboard_controller = FixedVelocityKeyboard(command_term, reset_requested)
        if command_term is not None:
            print(
                "[INFO] Keyboard: hold W/S for forward/backward, A/D for left/right, "
                "Q/E for yaw, and press R to reset."
            )
            print("[INFO] Click inside the Isaac Sim viewport first so it owns the keyboard focus.")
            print(
                "[INFO] Key commands: "
                f"W={W_COMMAND}, S={S_COMMAND}, A={A_COMMAND}, "
                f"D={D_COMMAND}, Q={Q_COMMAND}, E={E_COMMAND}."
            )
        else:
            print("[INFO] Keyboard: press R to reset. This task has no compatible base_velocity command.")
    else:
        print("[INFO] Headless mode: keyboard input is disabled.")

    try:
        runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
        runner_class = _resolve_runner_class(runner_class_name)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        policy_module = _get_policy_module(runner)
        normalizer = _get_policy_normalizer(policy_module)

        # ONNX export is intentionally the default behavior for every play invocation.
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if hasattr(policy_module, "export_onnx"):
            exported_policy_path = policy_module.export_onnx(export_model_dir, filename="policy.onnx")
        else:
            export_policy_as_onnx(
                policy_module,
                normalizer=normalizer,
                path=export_model_dir,
                filename="policy.onnx",
            )
            exported_policy_path = os.path.join(export_model_dir, "policy.onnx")
        print(f"[INFO] Exported ONNX policy to: {exported_policy_path}")

        dt = env.unwrapped.step_dt
        observations = env.get_observations()
        obs = observations[0] if isinstance(observations, tuple) else observations
        timestep = 0

        while simulation_app.is_running():
            start_time = time.time()
            if keyboard_controller is not None:
                keyboard_controller.apply_command()

            with torch.inference_mode():
                if reset_requested.is_set():
                    reset_requested.clear()
                    if keyboard_controller is not None:
                        keyboard_controller.reset_command()
                    # Only restore the selected terrain metadata. The task owns all robot reset behavior.
                    _pin_terrain_cell(env.unwrapped, terrain_selection)
                    obs, _ = env.reset()
                    policy_module.reset()
                    print("[INFO] Environment reset through the task's reset pipeline.")

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_module.reset(dones)

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
