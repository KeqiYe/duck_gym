"""Offline joint damping/armature compensation; no external body actuation."""

import json
from pathlib import Path
import numpy as np
import mujoco


def add_motor_feedforward(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(folder / "visual.xml"))
    dof = [int(m.joint(n).dofadr[0]) for n in meta["joint_names"]]
    with np.load(folder / "gait.npz") as data:
        record = {k: data[k] for k in data.files}
    q = record["targets"]
    h = float(record["period"]) / q.shape[1]
    dq = (np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)) / (2 * h)
    ddq = (np.roll(q, -1, axis=1) - 2 * q + np.roll(q, 1, axis=1)) / h**2
    torque = (
        m.dof_damping[dof] * dq
        + m.dof_armature[dof] * ddq
        + m.dof_frictionloss[dof] * np.tanh(dq / 0.1)
    )
    target = q + np.clip(torque, -meta["torque_limit"], meta["torque_limit"]) / meta["kp"]
    record.update(
        kinematic_targets=q,
        reference_velocity=dq,
        reference_acceleration=ddq,
        motor_feedforward_torque=torque,
        targets=np.clip(target, meta["lower"], meta["upper"]),
    )
    np.savez_compressed(folder / "gait.npz", **record)
    report = json.loads((folder / "gait.json").read_text())
    report["motor_feedforward"] = dict(
        method="Original joint damping*dq + armature*ddq + smoothed dry friction; converted to PD target offset",
        robot_parameters_changed=False,
        full_inverse_dynamics=False,
        max_abs_requested_torque_nm=float(np.abs(torque).max()),
        target_clipped_fraction=float(
            np.mean((target < np.array(meta["lower"])) | (target > np.array(meta["upper"])))
        ),
    )
    (folder / "gait.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
