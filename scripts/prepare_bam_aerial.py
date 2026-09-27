"""Prepare fixed-foot crouch IK and exact convex support of complete visuals.

MuJoCo is used only for kinematics/asset conversion. No dynamics step occurs.
Robot masses, joint limits and BAM parameters are retained verbatim.
"""

import bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import ConvexHull


def prepare(folder, output):
    cfg = json.loads((folder / "config.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(folder / "ground-only.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = cfg["default_qpos"]
    mujoco.mj_forward(m, d)
    home = d.qpos.copy()
    feet = [m.site(f"robot/{s}_foot").id for s in ("left", "right")]
    positions, rotations = d.site_xpos[feet].copy(), d.site_xmat[feet].copy()
    com0 = d.subtree_com[1].copy()
    joints = [m.joint(n).id for n in cfg["joint_names"]]
    legs = [
        i
        for i, n in enumerate(cfg["joint_names"])
        if "head" not in n and "neck" not in n
    ]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    columns, qcolumns = np.r_[0, va[legs]], np.r_[0, qa[legs]]
    jp, jr, jc = (np.zeros((3, m.nv)) for _ in range(3))
    reports = []
    for depth in np.arange(-0.015, 0.0501, 0.005):
        d.qpos[:] = home
        d.qpos[2] -= depth
        for _ in range(180):
            mujoco.mj_forward(m, d)
            matrices, residuals = [], []
            for f, site in enumerate(feet):
                mujoco.mj_jacSite(m, d, jp, jr, site)
                actual, target, error = np.empty(4), np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(actual, d.site_xmat[site])
                mujoco.mju_mat2Quat(target, rotations[f])
                mujoco.mju_subQuat(error, target, actual)
                matrices.extend([jp[:, columns], 0.02 * jr[:, columns]])
                residuals.extend([positions[f] - d.site_xpos[site], 0.02 * error])
            mujoco.mj_jacSubtreeCom(m, d, jc, 1)
            matrices.append(jc[:1, columns])
            residuals.append(com0[:1] - d.subtree_com[1, :1])
            jac, err = np.vstack(matrices), np.concatenate(residuals)
            delta = np.linalg.solve(
                jac.T @ jac + 1e-8 * np.eye(len(columns)), jac.T @ err
            )
            d.qpos[qcolumns] += np.clip(delta, -0.05, 0.05)
            d.qpos[qa] = np.clip(
                d.qpos[qa], m.jnt_range[joints, 0], m.jnt_range[joints, 1]
            )
        mujoco.mj_forward(m, d)
        reports.append(
            dict(
                depth_m=float(depth),
                foot_error_m=float(
                    np.linalg.norm(d.site_xpos[feet] - positions, axis=-1).max()
                ),
                com_x_error_m=float(abs(d.subtree_com[1, 0] - com0[0])),
                joint_offsets=(d.qpos[qa] - home[qa]).tolist(),
            )
        )
    d.qpos[:] = home
    mujoco.mj_forward(m, d)
    body_points = {}
    foot_ids = {f["body"] for f in cfg["feet"]}
    for g in range(m.ngeom):
        bid = int(m.geom_bodyid[g])
        if (
            bid == 0
            or bid in foot_ids
            or m.geom_group[g] != 2
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mesh = m.geom_dataid[g]
        verts = m.mesh_vert[
            m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]
        ]
        world = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        principal = (world - d.xipos[bid]) @ d.ximat[bid].reshape(3, 3)
        body_points.setdefault(bid, []).append(principal)
    for f in cfg["feet"]:
        body_points[f["body"]] = [np.asarray(f["points"])]
    points = []
    ids = sorted(body_points)
    for bid in ids:
        verts = np.unique(np.concatenate(body_points[bid]), axis=0)
        verts = verts[ConvexHull(verts).vertices]
        points.append(verts)
    count = max(map(len, points))
    padded = np.zeros((len(points), count, 3))
    for i, pts in enumerate(points):
        padded[i, : len(pts)] = pts
        padded[i, len(pts) :] = pts[0]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "geometry.npz",
        points=padded,
        counts=[len(p) for p in points],
        bodies=ids,
        foot_slots=[ids.index(f["body"]) for f in cfg["feet"]],
    )
    (output / "crouch.json").write_text(
        json.dumps(
            dict(
                method="Fixed feet and COM projection IK; dynamic feasibility unverified",
                poses=reports,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(output=str(output), geometry_shape=padded.shape, poses=reports),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    prepare(a.model_dir, a.output)
