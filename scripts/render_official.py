"""Render saved official baseline qpos on the Mac; no physics integration."""

import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model_io import prepare_microduck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--azimuth", type=float, default=215.0, help="Camera angle relative to initial heading"
    )
    args = p.parse_args()
    record = np.load(args.trajectory)
    out = args.output or args.trajectory.parent
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(prepare_microduck())
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.8
    data = mujoco.MjData(model)
    mapping = {}
    source_names = [str(s).split("/")[-1] for s in record["mj_joint_names"]]
    for i in range(model.njnt):
        name = model.joint(i).name
        source_index = source_names.index(name)  # Fail if any joint is missing.
        width = 7 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else 1
        mapping[int(model.jnt_qposadr[i])] = (int(record["mj_qposadr"][source_index]), width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = 0.65, -14
    q = record["root_pose"][0, 3:7]
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    camera.azimuth = args.azimuth + np.degrees(yaw)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 2]] = 1
    dt = float(record["control_dt"])
    end = (len(record["qpos"]) - 1) * dt
    indices = np.minimum(
        np.rint(np.arange(0, end + 1e-9, 1 / args.fps) / dt).astype(int), len(record["qpos"]) - 1
    )
    font = ImageFont.load_default(size=20)
    with mujoco.Renderer(model, 720, 960) as renderer, imageio.get_writer(
        out / "official.mp4", fps=args.fps, codec="libx264", quality=8
    ) as writer:
        for frame, index in enumerate(indices):
            for dest, (src, width) in mapping.items():
                data.qpos[dest : dest + width] = record["qpos"][index, src : src + width]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            camera.lookat[:] = record["root_pose"][index, :3]
            camera.lookat[2] = 0.13
            renderer.update_scene(data, camera, scene_option=opt)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 948, 46), fill=(20, 24, 29))
            command = record["command"][index]
            draw.text(
                (22, 20),
                f"Official MuJoCo Warp | {index*dt:.2f}s | cmd ({command[0]:+.2f}, {command[1]:+.2f}) m/s",
                font=font,
                fill="white",
            )
            writer.append_data(np.asarray(image))
            if frame == 0:
                image.save(out / "preview.png")
    (out / "render-report.json").write_text(
        json.dumps(
            dict(
                physics_steps=0,
                kinematics="Named joint mapping, complete original asset visuals",
                meshes=model.nmesh,
                materials=model.nmat,
                frames=len(indices),
                fps=args.fps,
            ),
            indent=2,
        )
        + "\n"
    )
    print(out / "official.mp4")


if __name__ == "__main__":
    main()
