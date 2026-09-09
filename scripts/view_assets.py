"""Display pinned MicroDuck assets without advancing any physics."""
import argparse
from datetime import datetime
import json
import time
import subprocess
import sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from model_io import ROOT, ASSET, prepare_microduck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / ('assets-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    parser.add_argument('--interactive', action='store_true', help='On macOS run with .venv/bin/mjpython')
    parser.add_argument('--seconds', type=float, default=0, help='Optional GUI timeout')
    args = parser.parse_args()
    if not ASSET.exists():
        raise SystemExit('Run scripts/fetch_assets.py first')
    m = mujoco.MjModel.from_xml_string(prepare_microduck())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (d.geom_xpos.min(axis=0) + d.geom_xpos.max(axis=0)) / 2
    cam.lookat[2] = 0.14
    cam.distance, cam.azimuth, cam.elevation = 0.60, 215, -12
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1
    opt.geomgroup[0] = 1
    if args.interactive:
        # Probe in a fresh main-thread process: MuJoCo's viewer dereferences a
        # null primary monitor on displayless macOS sessions (GLFW 3.4).
        probe = subprocess.run([sys.executable, '-c',
            'import glfw,sys; ok=glfw.init(); present=ok and bool(glfw.get_primary_monitor()); glfw.terminate(); sys.exit(0 if present else 1)'],
            capture_output=True, timeout=15)
        if probe.returncode:
            print('No active display; saving offscreen views instead of opening a window.', flush=True)
            args.interactive = False
    if args.interactive:
        from mujoco import viewer as mjviewer
        start = time.monotonic()
        with mjviewer.launch_passive(m, d) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = cam.lookat
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
                viewer.opt.geomgroup[:] = opt.geomgroup
            while viewer.is_running() and (not args.seconds or time.monotonic()-start < args.seconds):
                viewer.sync()
                time.sleep(1/60)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(m, height=960, width=1280) as renderer:
            for name, azimuth in [('front', 215), ('side', 90), ('rear', 35)]:
                cam.azimuth = azimuth
                renderer.update_scene(d, cam, scene_option=opt)
                image = renderer.render()
                Image.fromarray(image).save(args.output / f'microduck_{name}.png')
        report = {'mujoco': mujoco.__version__, 'meshes': m.nmesh, 'materials': m.nmat, 'bodies_excluding_world': m.nbody-1, 'hinges': int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)), 'physics_steps': 0}
        (args.output / 'asset_report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
        print(args.output)


if __name__ == '__main__':
    main()
