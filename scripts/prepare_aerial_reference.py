"""Kinematic crouch/extension prior with original motor-term compensation."""

import json
from pathlib import Path
import numpy as np
import mujoco
from prepare_crouch import prepare_crouch


def prepare_aerial_reference(folder):
    folder = Path(folder)
    rows = prepare_crouch(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    home = np.array(meta["home"])

    def pose(depth):
        row = min(rows, key=lambda r: abs(r["depth_m"] - depth))
        if row["foot_error_m"] > 0.001:
            raise ValueError("Aerial prior IK failed")
        return home + row["joint_offsets"]

    times = np.array([0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.05, 1.5, 4.0])
    knots = np.array(
        [home, home, pose(0.03), pose(0.03), pose(-0.01), pose(-0.01), pose(0.025), home, home]
    )
    t = np.arange(0, 4.0001, 0.005)
    interval = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    h = times[interval + 1] - times[interval]
    u = np.clip((t - times[interval]) / h, 0, 1)
    delta = knots[interval + 1] - knots[interval]
    q = knots[interval] + (u * u * (3 - 2 * u))[:, None] * delta
    dq = (6 * u * (1 - u) / h)[:, None] * delta
    ddq = ((6 - 12 * u) / h**2)[:, None] * delta
    m = mujoco.MjModel.from_xml_path(str(folder / "visual.xml"))
    dof = [int(m.joint(n).dofadr[0]) for n in meta["joint_names"]]
    torque = (
        m.dof_damping[dof] * dq
        + m.dof_armature[dof] * ddq
        + m.dof_frictionloss[dof] * np.tanh(dq / 0.1)
    )
    targets = np.clip(
        q + np.clip(torque, -meta["torque_limit"], meta["torque_limit"]) / meta["kp"],
        meta["lower"],
        meta["upper"],
    )
    np.savez_compressed(
        folder / "aerial_reference.npz",
        times=t,
        targets=targets,
        kinematic_targets=q,
        motor_feedforward_torque=torque,
    )
    report = dict(
        method="Fixed-feet/COM-projection crouch IK + known joint motor terms; targets only; no root pose or velocity forcing",
        motion_knots_seconds=times.tolist(),
        dynamic_feasibility="Unverified prior; evaluate native physical trajectories",
        max_requested_feedforward_nm=float(abs(torque).max()),
        robot_parameters_changed=False,
    )
    (folder / "aerial_reference.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    print(prepare_aerial_reference(p.parse_args().model_dir))
