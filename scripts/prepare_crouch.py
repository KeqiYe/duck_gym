"""Generate sagittal crouch poses with original feet and COM projection fixed."""

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco


def prepare_crouch(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(folder / "visual.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = meta["qpos"]
    mujoco.mj_forward(m, d)
    sites = [m.site(n).id for n in ("left_foot", "right_foot")]
    positions = d.site_xpos[sites].copy()
    rotations = d.site_xmat[sites].copy()
    com0 = d.subtree_com[1].copy()
    names = meta["joint_names"]
    legs = [i for i, n in enumerate(names) if n.startswith(("left", "right"))]
    qadr = np.array([int(m.joint(n).qposadr[0]) for n in names])
    dofs = np.array([int(m.joint(n).dofadr[0]) for n in names])
    columns = np.r_[0, dofs[legs]]
    qcolumns = np.r_[0, qadr[legs]]
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    jc = np.zeros((3, m.nv))
    poses = []
    reports = []
    depths = np.r_[[-0.02, -0.015, -0.01, -0.005], np.linspace(0, 0.05, 11)]
    for depth in depths:
        d.qpos[:] = meta["qpos"]
        d.qpos[2] -= depth
        for _ in range(150):
            mujoco.mj_forward(m, d)
            matrices = []
            residuals = []
            for f, site in enumerate(sites):
                mujoco.mj_jacSite(m, d, jp, jr, site)
                actual = np.empty(4)
                target = np.empty(4)
                error = np.empty(3)
                mujoco.mju_mat2Quat(actual, d.site_xmat[site])
                mujoco.mju_mat2Quat(target, rotations[f])
                mujoco.mju_subQuat(error, target, actual)
                matrices.extend([jp[:, columns], 0.02 * jr[:, columns]])
                residuals.extend([positions[f] - d.site_xpos[site], 0.02 * error])
            mujoco.mj_jacSubtreeCom(m, d, jc, 1)
            matrices.append(jc[:1, columns])
            residuals.append(com0[:1] - d.subtree_com[1, :1])
            jac = np.vstack(matrices)
            residual = np.concatenate(residuals)
            dq = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(len(columns)), jac.T @ residual)
            d.qpos[qcolumns] += np.clip(dq, -0.05, 0.05)
            d.qpos[qadr] = np.clip(d.qpos[qadr], meta["lower"], meta["upper"])
        mujoco.mj_forward(m, d)
        err = float(np.linalg.norm(d.site_xpos[sites] - positions, axis=-1).max())
        rotation_error = float(np.abs(d.site_xmat[sites] - rotations).max())
        row = dict(
            depth_m=float(depth),
            foot_error_m=err,
            foot_rotation_matrix_error=rotation_error,
            com_x_error_m=float(abs(d.subtree_com[1, 0] - com0[0])),
            base_translation_x_m=float(d.qpos[0] - meta["qpos"][0]),
            joint_offsets=(d.qpos[qadr] - np.array(meta["home"])).tolist(),
        )
        reports.append(row)
        poses.append(d.qpos[qadr].copy())
    np.savez_compressed(folder / "crouch.npz", depths=depths, poses=poses)
    (folder / "crouch.json").write_text(
        json.dumps(
            dict(
                method="Offline IK with fixed foot poses and COM projection; no dynamics feasibility claim",
                poses=reports,
            ),
            indent=2,
        )
        + "\n"
    )
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    a = p.parse_args()
    print(json.dumps(prepare_crouch(a.model_dir), indent=2))
