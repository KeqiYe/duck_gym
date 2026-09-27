"""Offline kinematic walking references; runtime physics never uses MuJoCo."""

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from prepare_standing import prepare
from model_io import ROOT


def prepare_gait(
    model_dir,
    samples=128,
    speed=0.05,
    period=0.8,
    lift=0.008,
    sway=0.015,
    roll=0.15,
    smooth_velocity=False,
):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(model_dir / "visual.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = meta["qpos"]
    mujoco.mj_forward(model, data)
    sites = [model.site(n).id for n in ("left_foot", "right_foot")]
    nominal = data.site_xpos[sites].copy()
    rotations = data.site_xmat[sites].copy()
    names = meta["joint_names"]
    qadr = [int(model.joint(n).qposadr[0]) for n in names]
    dof = [int(model.joint(n).dofadr[0]) for n in names]
    legs = [
        [k for k, n in enumerate(names) if n.startswith(side + "_")] for side in ("left", "right")
    ]
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float) * speed
    foot_points = []
    for site in sites:
        body = int(model.site_bodyid[site])
        cloud = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body or not model.geom_contype[geom]:
                continue
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            world = (
                model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom]
            )
            cloud.append((world - data.xipos[body]) @ data.ximat[body].reshape(3, 3))
        foot_points.append(np.concatenate(cloud))
    width = max(len(v) for v in foot_points)
    foot_points = np.stack(
        [np.pad(v, ((0, width - len(v)), (0, 0)), mode="edge") for v in foot_points]
    )
    table = []
    max_error = 0.0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for command in directions:
        poses = []
        for index in range(samples):
            phase = index / samples
            data.qpos[:] = meta["qpos"]
            data.qpos[1] += sway * np.sin(2 * np.pi * phase)
            angle = -roll * np.sin(2 * np.pi * phase)
            data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
            desired = nominal.copy()
            for foot in range(2):
                t = (phase + 0.5 * foot) % 1.0
                stance = 0.6
                if t < stance:
                    offset = period * (stance / 2 - t)
                    z = 0
                else:
                    u = (t - stance) / (1 - stance)
                    smooth = u * u * (3 - 2 * u)
                    offset = period * stance * (smooth - 0.5)
                    if smooth_velocity:
                        # Match stance velocity at the swing endpoints.
                        offset -= period * (1 - stance) * (2 * u**3 - 3 * u**2 + u)
                    z = lift * np.sin(np.pi * u) ** 2
                desired[foot, :2] += command * offset
                desired[foot, 2] += z
            for _ in range(50):
                mujoco.mj_forward(model, data)
                changes = []
                for foot, site in enumerate(sites):
                    indices = legs[foot]
                    columns = np.array(dof)[indices]
                    mujoco.mj_jacSite(model, data, jacp, jacr, site)
                    actual = np.empty(4)
                    target = np.empty(4)
                    error = np.empty(3)
                    mujoco.mju_mat2Quat(actual, data.site_xmat[site])
                    mujoco.mju_mat2Quat(target, rotations[foot])
                    mujoco.mju_subQuat(error, target, actual)
                    jac = np.vstack((jacp[:, columns], 0.01 * jacr[:, columns]))
                    residual = np.r_[desired[foot] - data.site_xpos[site], 0.01 * error]
                    dq = np.linalg.solve(
                        jac.T @ jac + 1e-7 * np.eye(len(indices)), jac.T @ residual
                    )
                    changes.append((indices, np.clip(dq, -0.1, 0.1)))
                for indices, dq in changes:
                    positions = np.array(qadr)[indices]
                    data.qpos[positions] = np.clip(
                        data.qpos[positions] + dq,
                        np.array(meta["lower"])[indices],
                        np.array(meta["upper"])[indices],
                    )
            mujoco.mj_forward(model, data)
            max_error = max(
                max_error, float(np.linalg.norm(data.site_xpos[sites] - desired, axis=-1).max())
            )
            poses.append(data.qpos[qadr].copy())
        table.append(poses)
    table = np.asarray(table)
    np.savez_compressed(
        model_dir / "gait.npz",
        targets=table,
        foot_collision_points=foot_points,
        period=period,
        lift=lift,
        speed=speed,
        nominal_feet=nominal,
        foot_bodies=model.site_bodyid[sites],
        foot_local=model.site_pos[sites],
        body_ipos=model.body_ipos,
        body_iquat=model.body_iquat,
    )
    config = dict(
        samples=samples,
        speed=speed,
        period=period,
        lift=lift,
        sway=sway,
        roll=roll,
        smooth_velocity=smooth_velocity,
        max_foot_position_residual_m=max_error,
        max_joint_offset_rad=float(np.abs(table - np.asarray(meta["home"])).max()),
        method="Offline damped least-squares foot pose IK; joint targets only at runtime",
    )
    (model_dir / "gait.json").write_text(json.dumps(config, indent=2) + "\n")
    print(config)
    return config


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=ROOT / "build/models/standing")
    a = p.parse_args()
    prepare_gait(a.model_dir)
