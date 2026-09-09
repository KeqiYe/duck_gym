"""Replay native CPU trajectories locally as MP4 and a full asset preview."""

import bootstrap
from bootstrap import ROOT
import argparse
from pathlib import Path
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from duck_gym.render import apply_body_poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("--model-dir", type=Path, default=ROOT / "build/models/standing")
    p.add_argument("--output", type=Path)
    p.add_argument("--fps", type=int, default=25)
    a = p.parse_args()
    out = a.output or a.trajectory.parent
    out.mkdir(parents=True, exist_ok=True)
    record = np.load(a.trajectory)
    states = record["bodies"]
    dt = float(record["control_dt"])
    m = mujoco.MjModel.from_xml_path(str(a.model_dir / "visual.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # Initial fixed lights/camera only; no physics integration.
    cam = mujoco.MjvCamera()
    cam.distance = 0.65
    cam.azimuth = 215
    cam.elevation = -14
    cam.lookat[:] = [0, 0, 0.13]
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 2]] = 1
    frames = 0
    with mujoco.Renderer(m, height=720, width=960) as renderer, imageio.get_writer(
        out / "standing.mp4", fps=a.fps, codec="libx264", quality=8
    ) as writer:
        indices = np.minimum(
            np.rint(np.arange(0, (len(states) - 1) * dt + 1e-9, 1 / a.fps) / dt).astype(int),
            len(states) - 1,
        )
        for index in indices:
            apply_body_poses(m, d, states[index])
            renderer.update_scene(d, cam, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            font = ImageFont.load_default(size=19)
            force = record["forces"][index]
            label = f"Native AVBD | CPU | {index*dt:5.2f} s"
            if np.linalg.norm(force) > 0:
                label += f" | Push ({force[0]:+.3f}, {force[1]:+.3f}) N"
            draw.rectangle((12, 12, 940, 46), fill=(20, 24, 29))
            draw.text((22, 20), label, font=font, fill=(235, 240, 245))
            writer.append_data(np.asarray(frame))
            frames += 1
            if index == 0:
                frame.save(out / "microduck.png")
    (out / "render_report.json").write_text(
        json.dumps(
            dict(
                frames=frames,
                fps=a.fps,
                meshes=m.nmesh,
                materials=m.nmat,
                physics_steps=0,
                source=str(a.trajectory),
                pose_mapping="per-body COM/principal frame to visual geoms",
            ),
            indent=2,
        )
        + "\n"
    )
    print(out / "standing.mp4")


if __name__ == "__main__":
    main()
