"""One uninterrupted grounded-start skill rollout, retaining failure frames."""

import bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from duck_gym.bam_skill import BamSkillEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--initial-noise",
        type=float,
        default=0.0,
        help="Joint and roll/pitch reset amplitude in radians",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Repeat crouch phase without resetting physical state",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    metadata = json.loads(a.policy.with_suffix(".json").read_text())
    e = BamSkillEnv(
        a.model_dir,
        a.preparation,
        metadata["task"],
        backend=a.backend,
        fp64=a.fp64,
        iterations=a.iterations,
        seed=a.seed,
    )
    e.reset(joint_noise=a.initial_noise, tilt_noise=a.initial_noise)
    e.reset_on_done = False
    policy = torch.jit.load(str(a.policy), map_location=e.device).eval()
    if a.cycles < 1 or (a.cycles != 1 and metadata["task"]["skill"] != "crouch"):
        p.error("Multiple cycles only supported for crouch")
    records = []

    def capture(env):
        com, v = env.centroidal()
        records.append(
            dict(
                **{
                    k: getattr(env, k)[0].detach().cpu().numpy().copy()
                    for k in ("bodies", "joints", "root_quat", "contact", "diagnostics")
                },
                targets=(env.home + env.action)[0].cpu().numpy().copy(),
                com=com[0].cpu().numpy().copy(),
                com_velocity=v[0].cpu().numpy().copy(),
                clearance=env.clearances()[0].cpu().numpy().copy(),
            )
        )

    capture(e)
    done = False
    for step in range(e.max_episode_length * a.cycles):
        with torch.no_grad():
            _, _, done, _ = e.step(
                policy(e.get_observations()["actor"]), substep_callback=capture
            )
        if bool(e.invalid[0]):
            break
        if bool(done[0]):
            if step + 1 == e.max_episode_length * a.cycles:
                break
            e.episode_length_buf.zero_()
            e.pending.zero_()
    rows = {k: np.stack([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(
        a.output / "trajectory.npz",
        **rows,
        control_dt=e.dt,
        backend=a.backend,
        forces=np.zeros((len(records), 3)),
    )
    q = rows["root_quat"]
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
    report = dict(
        backend=a.backend,
        fp64=a.fp64,
        iterations=a.iterations,
        seed=a.seed,
        initial_noise_rad=a.initial_noise,
        cycles=a.cycles,
        physical_resets=0,
        policy_metadata=metadata,
        policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),
        duration=(len(records) - 1) * e.dt,
        invalid=bool(e.invalid[0]),
        numerical_invalid=bool(e.numerical_invalid[0]),
        max_energy_excess_j=float(e.max_energy_excess[0]),
        positive_motor_work_j=float(e.positive_motor_work[0]),
        took_off=bool(e.took_off[0]),
        landed=bool(e.landed[0]),
        flight_seconds=float(e.flight_time[0]),
        air_rotation_rad=float(e.air_rotation[0]),
        takeoff_vz_mps=float(e.takeoff_vz[0]),
        max_upward_mps=float(e.max_upward[0]),
        peak_com_height_m=float(e.peak_height[0]),
        landing_contact_seconds=float(e.landing_time[0]),
        max_tilt_rad=float(tilt.max()),
        final_tilt_rad=float(tilt[-1]),
        final_com_velocity=rows["com_velocity"][-1].tolist(),
        minimum_nonfoot_height=float(
            rows["clearance"][:, e.geom_nonfeet.cpu().numpy()].min()
        ),
        accepted=False,
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
