"""Independently audit continuous flight segments from native trajectory snapshots.

This is a conservative contact/geometry check, not a physical feasibility or
real-robot certification. Initial state is never counted as a takeoff interval.
"""

import bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from duck_gym.motion import MotionGeometry, pitch_increment, contact_free_interval


def flight_segments(airborne, increments, dt):
    """Each sample i describes interval (i-1, i]; never merge separate hops."""
    mask = np.asarray(airborne, dtype=bool).copy()
    if not len(mask):
        return []
    mask[0] = False
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        result.append(
            dict(
                first_sample=int(start),
                end_sample_exclusive=int(end),
                start_seconds=float((start - 1) * dt),
                end_seconds=float((end - 1) * dt),
                duration_seconds=float((end - start) * dt),
                signed_pitch_rotation_rad=float(np.asarray(increments)[start:end].sum()),
                contact_after_observed=bool(end < len(mask)),
            )
        )
    return result


def audit_aerial(trajectory, model_dir):
    path, model = Path(trajectory), Path(model_dir)
    with np.load(path) as r:
        b = torch.tensor(r["bodies"], dtype=torch.float64)
        force = torch.tensor(r["ground_forces"], dtype=torch.float64)
        diag = torch.tensor(r["diagnostics"], dtype=torch.float64)
        external = r["forces"].copy()
        dt = float(r["control_dt"])
    geom = MotionGeometry(model / "motion.npz", "cpu", torch.float64)
    clearance = torch.cat([geom.support(chunk)[0].amin(-1) for chunk in b.split(32)])
    free = contact_free_interval(diag, force, clearance)
    delta = torch.zeros(len(b), dtype=torch.float64)
    delta[1:] = pitch_increment(b[:-1, 1, 3:7], b[1:, 1, 3:7])
    segments = flight_segments(free.numpy(), delta.numpy(), dt)
    com, com_velocity, _ = geom.centroidal(b)
    nonfoot = force.clone()
    nonfoot[:, geom.feet] = 0
    nonfoot[:, 0] = 0
    bad = nonfoot.abs().sum(dim=(1, 2)) > 1e-4
    for segment in segments:
        start, end = segment["first_sample"], segment["end_sample_exclusive"]
        segment["peak_full_geometry_clearance_m"] = float(clearance[start:end].max())
        segment["com_vertical_velocity_at_interval_start_mps"] = float(com_velocity[start - 1, 2])
        segment["upward_at_interval_start"] = bool(com_velocity[start - 1, 2] > 0)
        segment["peak_com_rise_from_initial_m"] = float(com[start:end, 2].max() - com[0, 2])
        segment["nonfoot_contact_after_flight"] = bool(bad[end:].any())
        segment["feet_contact_on_next_sample"] = (
            (force[end, geom.feet, 2] > 1e-4).tolist() if end < len(b) else None
        )
    return dict(
        trajectory=str(path),
        trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        motion_geometry_sha256=hashlib.sha256((model / "motion.npz").read_bytes()).hexdigest(),
        control_dt=dt,
        completed_seconds=(len(b) - 1) * dt,
        zero_recorded_external_force=bool(np.all(external == 0)),
        max_recorded_initial_body_speed_mps=float(b[0, 1:, 7:10].norm(dim=-1).max()),
        max_recorded_initial_body_angular_speed_radps=float(b[0, 1:, 10:13].norm(dim=-1).max()),
        nonfoot_contact_samples=int(bad.sum()),
        first_nonfoot_contact_seconds=float(torch.nonzero(bad)[0, 0]) * dt if bad.any() else None,
        contact_free_segments=segments,
        max_abs_single_flight_rotation_rad=max(
            (abs(x["signed_pitch_rotation_rad"]) for x in segments), default=0
        ),
        accepted=False,
        limitations=[
            "All-substep contact-candidate maximum and positive original-mesh clearance define conservative flight; proximity contacts can shorten it.",
            "Pitch rotation integrates quaternion increments about world y; tumbling about other axes is not a somersault.",
            "Last-substep forces cannot exclude transient nonfoot contact within a landing interval.",
            "No self-collision or real actuator model; full rotation alone is insufficient for acceptance.",
            "A contact-free interval can also occur while collapsing; inspect COM takeoff velocity and landing before calling it a jump.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_aerial(args.trajectory, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
