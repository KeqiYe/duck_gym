"""Replay native CPU trajectories locally as MP4 and a full asset preview."""

import bootstrap
from bootstrap import ROOT
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from duck_gym.render import apply_body_poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("--model-dir", type=Path, default=ROOT / "build/models/standing")
    p.add_argument(
        "--visual-model", type=Path, help="Exact exported model for this native body layout"
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--follow", action="store_true", help="Follow the base during locomotion")
    p.add_argument(
        "--azimuth",
        type=float,
        default=35,
        help="World camera azimuth; use about 145 for the exported official BAM model",
    )
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--playback-speed", type=float, default=1.0)
    p.add_argument("--label", help="Optional single-line display label")
    p.add_argument("--output-name", default="standing.mp4")
    a = p.parse_args()
    if a.playback_speed<=0 or a.fps<=0:p.error("Playback speed and FPS must be positive")
    out = a.output or a.trajectory.parent
    out.mkdir(parents=True, exist_ok=True)
    record = np.load(a.trajectory)
    states = record["bodies"]
    dt = float(record["control_dt"])
    visual_model = a.visual_model or a.model_dir / "visual.xml"
    m = mujoco.MjModel.from_xml_path(str(visual_model))
    m.vis.headlight.ambient[:] = 0.35
    m.vis.headlight.diffuse[:] = 0.8
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # Initial fixed lights/camera only; no physics integration.
    cam = mujoco.MjvCamera()
    cam.distance = 0.65
    cam.azimuth = a.azimuth
    cam.elevation = -14
    cam.lookat[:] = [0, 0, 0.13]
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 2]] = 1
    frames = 0
    with mujoco.Renderer(m, height=720, width=960) as renderer, imageio.get_writer(
        out / a.output_name, fps=a.fps, codec="libx264", quality=8
    ) as writer:
        # The final state marks the end of the clip, not an extra video frame.
        # Sample [0, duration): 10 seconds at 25 fps produces exactly 250 frames.
        # Preserve the existing single-snapshot preview behavior.
        sample_times = (
            np.arange(0, (len(states) - 1) * dt, a.playback_speed / a.fps)
            if len(states) > 1
            else np.zeros(1)
        )
        indices = np.minimum(np.rint(sample_times / dt).astype(int), len(states) - 1)
        for index in indices:
            apply_body_poses(m, d, states[index])
            if a.follow:
                cam.lookat[:2] = states[index, 1, :2]
            renderer.update_scene(d, cam, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            font = ImageFont.load_default(size=19)
            force = record["forces"][index]
            backend = str(record["backend"]) if "backend" in record else "cpu"
            label = f"Native AVBD | {backend.upper()} | {index*dt:5.2f} s"
            if a.label:label=f"{a.label} | {index*dt:5.2f} s"
            if bool(record.get("command_servo_enabled", False)):
                label += " | PPO + command servo"
            if "command" in record and np.linalg.norm(record["command"]) > 0:
                command = record["command"]
                if "target_series" in record:
                    command = record["target_series"][index]
                elif "command_series" in record:
                    command = record["command_series"][index]
                label += f" | Command ({command[0]:+.3f}, {command[1]:+.3f}) m/s"
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
                playback_speed=a.playback_speed,
                meshes=m.nmesh,
                materials=m.nmat,
                physics_steps=0,
                source=str(a.trajectory),
                trajectory_sha256=hashlib.sha256(a.trajectory.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(visual_model.read_bytes()).hexdigest(),
                camera=dict(
                    azimuth=a.azimuth,
                    elevation=cam.elevation,
                    distance=cam.distance,
                    follow=a.follow,
                ),
                pose_mapping="per-body COM/principal frame to visual geoms",
            ),
            indent=2,
        )
        + "\n"
    )
    print(out / a.output_name)


if __name__ == "__main__":
    main()
