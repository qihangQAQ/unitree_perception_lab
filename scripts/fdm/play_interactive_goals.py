"""Spawn on flat ground, walk to a random nearby goal, then press SPACE to repeat.

Example:
    python scripts/fdm/play_interactive_goals.py --seed 0 \
        --goal-radius 5 --goal-min-distance 2 --mppi-candidates 256 --rendering_mode balanced
"""

from play_mixed_pillars import main, parse_args


if __name__ == "__main__":
    args = parse_args(interactive=True)
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    try:
        main(args, simulation_app, interactive=True)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
