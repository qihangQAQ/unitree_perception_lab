"""Preview one generated reference FDM terrain. Close the window to exit.

Usage:
    python scripts/fdm/view_terrain.py --rendering_mode performance
    python scripts/fdm/view_terrain.py --top-view --difficulty 0.5 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def parse_args():
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.reference_terrains import reference_box_cross_scene

    parser = argparse.ArgumentParser(description="Preview the reference FDM five-box terrain with a free camera.")
    parser.add_argument('--difficulty', type=float, default=.5, help='Box width interpolated from 0.5 to 1 m (0..1).')
    parser.add_argument('--seed', type=int, default=0, help='Fixed seed for obstacle heights (0.5..1.5 m).')
    parser.add_argument('--top-view', action='store_true', help='Start with a top-down view.')
    parser.add_argument('--screenshot', type=Path, help='Optionally save the initial viewport as a PNG.')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        scene = reference_box_cross_scene(args.difficulty, args.seed)
        scene.metadata()  # Check endpoints and the route through this geometry.
        if args.screenshot and args.screenshot.suffix.lower() != '.png':
            raise ValueError('--screenshot must be a .png path.')
        if args.screenshot and args.screenshot.exists():
            raise ValueError(f'Refusing to overwrite screenshot: {args.screenshot}')
    except ValueError as exc:
        parser.error(str(exc))
    return args, scene


def view(app, args, scene):
    import omni.timeline
    import omni.usd
    import trimesh
    from isaaclab.terrains.utils import create_prim_from_mesh
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, UsdGeom, UsdLux
    from unitree_rl_lab.fdm.benchmarks import scene_meshes

    omni.timeline.get_timeline_interface().stop()
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    mesh = trimesh.util.concatenate(scene_meshes(scene))
    create_prim_from_mesh('/World/ground', mesh)
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(2000.)

    # Pure visual markers: green = start, red = goal. No physics is played.
    for name, xy, color in (('Start', scene.start[:2], (.1, .8, .15)),
                            ('Goal', scene.goal, (.95, .1, .1))):
        marker = UsdGeom.Cylinder.Define(stage, f'/World/{name}')
        marker.CreateRadiusAttr(.25)
        marker.CreateHeightAttr(.05)
        marker.CreateAxisAttr(UsdGeom.Tokens.z)
        marker.AddTranslateOp().Set(Gf.Vec3d(*xy, .03))
        marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    for _ in range(5):
        if not app.is_running():
            return
        app.update()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('A viewport is required; launch without --headless or use livestreaming.')
    eye = (0., -.01, 16.) if args.top_view else (9., -12., 11.)
    set_camera_view(eye=eye, target=(0., 0., 0.), camera_prim_path=str(viewport.camera_path))

    print(f'[Terrain Viewer] Ready: {scene.name}, {scene.size[0]:g}x{scene.size[1]:g} m, '
          f'{len(scene.obstacles)} boxes, width={scene.obstacles[0].size[0]:g} m, seed={args.seed}.', flush=True)
    print(f'[Terrain Viewer] Green start={scene.start[:2]}, red goal={scene.goal}. '
          'Free camera; close the window or press Ctrl+C to exit.', flush=True)
    frames, capture = 0, None
    while app.is_running():
        started = time.monotonic()
        app.update()
        frames += 1
        if args.screenshot and frames == 60:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            capture = capture_viewport_to_file(viewport, str(args.screenshot.resolve()))
            print(f'[Terrain Viewer] Screenshot requested: {args.screenshot}', flush=True)
        time.sleep(max(0., 1/30 - (time.monotonic() - started)))


def main():
    from isaaclab.app import AppLauncher

    args, scene = parse_args()
    launcher = AppLauncher(args)
    try:
        view(launcher.app, args, scene)
    except KeyboardInterrupt:
        pass
    finally:
        launcher.app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
