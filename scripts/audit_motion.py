"""Audit saved physical trajectories independently of the training reward."""

import bootstrap
from bootstrap import ROOT
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from duck_gym.motion import MotionGeometry


def audit(trajectory, model_dir, warmup=2.0):
    r = np.load(trajectory)
    meta = json.loads((Path(model_dir) / "metadata.json").read_text())
    geom = MotionGeometry(Path(model_dir) / "motion.npz", "cpu", torch.float64)
    dt = float(r["control_dt"])
    start = min(round(warmup / dt), len(r["bodies"]) - 1)
    bodies = torch.tensor(r["bodies"][start:], dtype=torch.float64)
    force = torch.tensor(r["ground_forces"][start:], dtype=torch.float64)
    clear, velocities, rms_speeds = [], [], []
    for chunk in bodies.split(64):
        height, velocity, _, rms = geom.support(chunk, feet_only=True, pointwise_rms=True)
        clear.append(height)
        velocities.append(velocity)
        rms_speeds.append(rms)
    clearance = torch.cat(clear)
    velocity = torch.cat(velocities)
    foot_force = force[:, geom.feet, 2]
    contact = foot_force > 0.05 * meta["mass"] * 9.81
    slip = velocity[..., :2].norm(dim=-1)
    head_ids = [i for i, n in enumerate(meta["joint_names"]) if n.startswith(("head", "neck"))]
    joints = r["joints"][start:, :, 0]
    head_delta = joints[:, head_ids] - np.array(meta["home"])[head_ids]
    flight = ~contact
    durations = []
    for f in range(2):
        transitions = np.diff(np.r_[False, flight[:, f].numpy(), False].astype(int))
        durations.append(
            ((np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)) * dt).tolist()
        )
    support_slip = slip[contact]
    support_rms = torch.cat(rms_speeds)[contact]
    report = dict(
        source=str(trajectory),
        trajectory_sha256=hashlib.sha256(Path(trajectory).read_bytes()).hexdigest(),
        warmup_seconds=warmup,
        measured_seconds=max(0, (len(bodies) - 1) * dt),
        diagnostic_definitions={
            "loaded_contact": "last-substep vertical foot force > 5% total weight",
            "support_velocity": "v_COM + omega cross r at mean vertices within 0.5 mm of lowest support point",
            "geometric_air": "lowest original collision vertex > 1 mm",
            "thresholds": "diagnostic classifications, not acceptance tolerances",
        },
        foot_contact_duty=contact.double().mean(0).tolist(),
        double_support_fraction=float(contact.all(-1).double().mean()),
        single_support_fraction=float((contact.sum(-1) == 1).double().mean()),
        geometric_air_fraction=(clearance > 0.001).double().mean(0).tolist(),
        max_clearance_m=clearance.amax(0).tolist(),
        loaded_support_slip_mean_mps=float(support_slip.mean()) if support_slip.numel() else None,
        loaded_support_slip_p95_mps=(
            float(support_slip.quantile(0.95)) if support_slip.numel() else None
        ),
        loaded_pointwise_rms_slip_mean_mps=(
            float(support_rms.mean()) if support_rms.numel() else None
        ),
        loaded_pointwise_rms_slip_p95_mps=(
            float(support_rms.quantile(0.95)) if support_rms.numel() else None
        ),
        air_intervals_seconds=durations,
        head_joint_mean_offset_rad=dict(
            zip((meta["joint_names"][i] for i in head_ids), head_delta.mean(0).tolist())
        ),
        head_joint_max_abs_offset_rad=dict(
            zip((meta["joint_names"][i] for i in head_ids), np.abs(head_delta).max(0).tolist())
        ),
        action_saturation_fraction=float((np.abs(r["actions"][start:]) >= 0.99).mean()),
        accepted=False,
        visual_review="required; speed alone does not establish walking",
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("--model-dir", type=Path, default=ROOT / "build/models/motion")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.trajectory, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "air_intervals_seconds"}, indent=2))
