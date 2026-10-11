"""Open the FDM USD terrain in Isaac Sim without creating a robot or an environment.

Usage:
    python scripts/fdm/view_usd.py
    python scripts/fdm/view_usd.py --usd /path/to/terrain.usd
"""

from __future__ import annotations

import argparse

from _common import require_input_file


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD

    parser = argparse.ArgumentParser(description="View the USD map without robots, policies or training.")
    parser.add_argument(
        "--usd", "--terrain-usd", default=str(DEFAULT_TERRAIN_USD),
        help="USD file to open (default: the FDM training terrain).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        args.usd = require_input_file(args.usd, "--usd")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _view(app, usd_path: str) -> None:
    # Kit-dependent imports must follow AppLauncher.
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
    from pxr import UsdGeom, UsdLux

    print(f"[USD Viewer] Opening: {usd_path}", flush=True)
    omni.timeline.get_timeline_interface().stop()
    # Open the original stage directly, preserving its paths and material bindings.
    if not open_stage(usd_path):
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    while app.is_running() and is_stage_loading():
        app.update()
    if not app.is_running():
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD loading completed without an active stage.")
    # Viewer lighting/camera adjustments stay in the transient session layer.
    # This script never saves changes to the source USD.
    stage.SetEditTarget(stage.GetSessionLayer())
    if not any(prim.HasAPI(UsdLux.LightAPI) for prim in stage.Traverse()):
        light_path = omni.usd.get_stage_next_free_path(stage, "/FDMViewerLight", False)
        light = UsdLux.DomeLight.Define(stage, light_path)
        light.CreateIntensityAttr(2000.0)

    # Let the viewport attach to the newly opened stage before framing it.
    for _ in range(5):
        if not app.is_running():
            return
        app.update()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No viewport available. Run without --headless, or enable Isaac Sim livestreaming.")
    geometry = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Boundable)]
    if not frame_viewport_prims(viewport, prims=geometry):
        print("[USD Viewer] Automatic framing unavailable; select the terrain in Stage and press F.", flush=True)

    print("[USD Viewer] Ready. Close the window or press Ctrl+C to exit.", flush=True)
    # Render and process UI events only; do not create a SimulationContext or play physics.
    while app.is_running():
        app.update()


def main() -> None:
    from isaaclab.app import AppLauncher

    args = _parse_args()
    launcher = AppLauncher(args)
    try:
        _view(launcher.app, args.usd)
    except KeyboardInterrupt:
        pass
    finally:
        launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
