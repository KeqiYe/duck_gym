"""Compile original assets and upstream STAND2 configuration; no dynamics step."""

import json
from pathlib import Path
import argparse
import numpy as np
import mujoco
from model_io import ROOT, prepare_microduck, export_model

# microduck_rl@1e79c29c97d8b38aee9eefde77a545860ba7658e,
# src/mjlab_microduck/robot/microduck_constants.py: HOME_FRAME (STAND2).
HOME = dict(
    left_hip_yaw=0,
    right_hip_yaw=0,
    left_hip_roll=-0.0873,
    right_hip_roll=0.0873,
    left_hip_pitch=-0.4579,
    right_hip_pitch=0.4579,
    left_knee=-0.0049,
    right_knee=0.0049,
    left_ankle=0.4530,
    right_ankle=-0.4530,
    neck_pitch=0.3491,
    head_pitch=0.3491,
    head_yaw=0,
    head_roll=0,
)


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    xml = prepare_microduck()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    names = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    assert set(names) == set(HOME)
    for name in names:
        d.qpos[m.joint(name).qposadr[0]] = HOME[name]
    mujoco.mj_forward(m, d)
    lowest = np.inf
    for g in range(m.ngeom):
        if not m.geom_contype[g] or m.geom_bodyid[g] == 0:
            continue
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        mesh = m.geom_dataid[g]
        start = m.mesh_vertadr[mesh]
        count = m.mesh_vertnum[mesh]
        xyz = m.mesh_vert[start : start + count] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lowest = min(lowest, xyz[:, 2].min())
    # Geometric placement with 0.1 mm clearance; not a modified robot parameter.
    d.qpos[2] += 0.0001 - lowest
    mujoco.mj_forward(m, d)
    export_model(m, d, output / "microduck.duck")
    (output / "visual.xml").write_text(xml)
    metadata = dict(
        joint_names=names,
        home=[HOME[n] for n in names],
        root_iquat=m.body_iquat[1].tolist(),
        root_height=float(d.xipos[1, 2]),
        mass=float(m.body_mass.sum()),
        kp=0.55,
        torque_limit=0.96,
        lower=[float(m.joint(n).range[0]) for n in names],
        upper=[float(m.joint(n).range[1]) for n in names],
        qpos=d.qpos.tolist(),
        upstream_commit="1e79c29c97d8b38aee9eefde77a545860ba7658e",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output, flush=True)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "build/models/standing")
    prepare(p.parse_args().output)
