"""Independent, simultaneous walking rollouts; no hidden resets after failure.

Uses the established speed criteria, with additional head motion statistics.
All cases run identical source/model/policy; random initial states are generated
by NumPy so CPU and CUDA receive the same perturbations.
"""

import bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from duck_gym.bam_env import BamEnv
from duck_gym.command_servo import VelocityCommandServo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--record-physics", action="store_true")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--warmup", type=float, default=2)
    p.add_argument("--noise-seeds", type=int, nargs="*", default=[])
    p.add_argument(
        "--command-servo",
        type=Path,
        default=bootstrap.ROOT / "configs/velocity_command_servo_native600.json",
    )
    p.add_argument("--switch-every", type=float)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    directions = dict(
        forward=[0.05, 0, 0],
        backward=[-0.05, 0, 0],
        left=[0, 0.05, 0],
        right=[0, -0.05, 0],
    )
    cases = [dict(name=n, direction=n, seed=None, noise=0.0) for n in directions]
    if a.switch_every:
        cases = [dict(name="sequence", direction="forward", seed=None, noise=0.0)]
    else:
        cases += [
            dict(name=f"seed-{s}-{n}", direction=n, seed=s, noise=0.01)
            for s in a.noise_seeds
            for n in directions
        ]
    e = BamEnv(
        a.model_dir,
        num_envs=len(cases),
        backend=a.backend,
        fp64=a.fp64,
        seconds=a.seconds + a.warmup + 1,
        iterations=a.iterations,
    )
    e.reset()
    initial = e.home.cpu().numpy().copy()
    root = np.zeros((len(cases), 6))
    for i, c in enumerate(cases):
        if c["seed"] is not None:
            rng = np.random.default_rng(c["seed"])
            initial[i] += rng.uniform(-c["noise"], c["noise"], 14)
            root[i, 3:] = rng.uniform(-c["noise"], c["noise"], 3)
    e.native.reset(
        torch.ones_like(e.pending),
        e.home.new_tensor(initial).contiguous(),
        e.home.new_tensor(root).contiguous(),
    )
    e.refresh()
    policy = torch.jit.load(str(a.policy), map_location=e.device).eval()
    metadata = json.loads(a.policy.with_suffix(".json").read_text())
    servo_cfg = json.loads(a.command_servo.read_text())
    servo = VelocityCommandServo(e.command, servo_cfg)
    servo.reset(e.yaw())
    targets = e.home.new_tensor([directions[c["direction"]] for c in cases])
    keys = ["bodies", "joints", "root_quat", "contact", "diagnostics"]
    history = {
        k: []
        for k in keys
        + [
            "sole_height",
            "self_collision_substep_max",
            "foot_site_velocity",
            "command_series",
            "target_series",
        ]
    }
    physics = {k: [] for k in keys}
    stop = np.full(len(cases), -1, dtype=int)

    def capture(current):
        for k in physics:
            physics[k].append(getattr(current, k).detach().cpu().numpy().copy())

    if a.record_physics:
        capture(e)
    steps = round((a.seconds + a.warmup) / e.step_dt)
    for k in range(steps + 1):
        if a.switch_every:
            targets[:] = e.home.new_tensor(
                list(directions.values())[int(k * e.step_dt / a.switch_every) % 4]
            )
        e.command.copy_(servo.command(targets, e.yaw(), e.bodies[:, 1, 12], e.step_dt))
        values = {key: getattr(e, key) for key in keys}
        values.update(
            sole_height=e.sole_height,
            self_collision_substep_max=torch.stack(
                [e.self_contacts, e.self_penetration], -1
            ),
            foot_site_velocity=e.site_vel,
            command_series=e.command,
            target_series=targets,
        )
        for key, value in values.items():
            history[key].append(value.detach().cpu().numpy().copy())
        if k == steps:
            break
        with torch.no_grad():
            action = policy(e.get_observations()["actor"])
            if metadata.get("clip_actions") is not None:
                action = action.clamp(
                    -metadata["clip_actions"], metadata["clip_actions"]
                )
            e.pending.zero_()  # retain first failure index; never reset physical state
            _, _, done, _ = e.step(
                action, substep_callback=capture if a.record_physics else None
            )
            failed = done.cpu().numpy().astype(bool)
            stop[(stop < 0) & failed] = k + 1
            servo.observe(e.origin_world_velocity, e.step_dt)
    history = {k: np.asarray(v) for k, v in history.items()}
    physics = {k: np.asarray(v) for k, v in physics.items()}
    report = dict(
        backend=a.backend,
        fp64=a.fp64,
        iterations=a.iterations,
        physics_dt=e.dt,
        control_dt=e.step_dt,
        model=str(a.model_dir),
        policy=str(a.policy),
        policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),
        policy_metadata=metadata,
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                a.model_dir / "model.duck",
                a.model_dir / "config.json",
                a.command_servo,
            ]
        },
        cases={},
        accepted=False,
        record_physics=a.record_physics,
        command_servo=servo_cfg,
    )
    for i, c in enumerate(cases):
        last = stop[i] if stop[i] >= 0 else steps
        r = {k: v[: last + 1, i] for k, v in history.items()}
        out = a.output / c["name"]
        out.mkdir()
        np.savez_compressed(
            out / "trajectory.npz",
            **r,
            control_dt=e.step_dt,
            backend=a.backend,
            forces=np.zeros((last + 1, 3)),
            initial_joints=initial[i],
            initial_root_perturbation=root[i],
            **(
                {
                    "physics_dt": e.dt,
                    **{
                        "physics_" + k: v[: last * e.decimation + 1, i]
                        for k, v in physics.items()
                    },
                }
                if a.record_physics
                else {}
            ),
        )
        start = round(a.warmup / e.step_dt)
        window = round(1 / e.step_dt)
        duration = last * e.step_dt
        measured = max(0.0, duration - a.warmup)
        q = r["root_quat"]
        yaw = np.unwrap(
            np.arctan2(
                2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
            )
        )
        max_yaw = float(abs(yaw - yaw[0]).max())
        cmd = np.asarray(directions[c["direction"]][:2])
        mean = (
            (r["bodies"][-1, 1, :2] - r["bodies"][start, 1, :2]) / measured
            if measured > 0
            else None
        )
        velocities = (
            (r["bodies"][start + window :, 1, :2] - r["bodies"][start:-window, 1, :2])
            / (window * e.step_dt)
            if last > start + window
            else None
        )
        error = (
            float(np.linalg.norm(velocities - cmd, axis=-1).max())
            if velocities is not None
            else None
        )
        head = r["joints"][:, e.head_ids, 0] - e.home[i, e.head_ids].cpu().numpy()
        headv = r["joints"][:, e.head_ids, 1]
        contacts = r["contact"][start:]
        single = np.where(
            contacts[:, 0] & ~contacts[:, 1],
            1,
            np.where(contacts[:, 1] & ~contacts[:, 0], -1, 0),
        )
        single = single[single != 0]
        result = dict(
            **c,
            duration=duration,
            terminated=bool(stop[i] >= 0),
            average_velocity=mean.tolist() if mean is not None else None,
            max_yaw_change=max_yaw,
            max_one_second_velocity_error=error,
            diagnostic_speed_pass=bool(
                not a.switch_every
                and stop[i] < 0
                and mean is not None
                and np.linalg.norm(mean - cmd) <= 0.01
                and error is not None
                and error <= 0.02
                and max_yaw <= 0.3
            ),
            head_max_abs_rad=abs(head).max(0).tolist(),
            head_rms_rad=np.sqrt((head * head).mean(0)).tolist(),
            head_velocity_rms_rad_s=np.sqrt((headv * headv).mean(0)).tolist(),
            max_anchor_error=float(r["diagnostics"][:, 0].max()),
            max_penetration=float(r["diagnostics"][:, 1].max()),
            max_self_penetration_all_substeps=float(
                r["self_collision_substep_max"][:, 1].max()
            ),
            alternating_single_support_transitions=int(
                np.count_nonzero(np.diff(single))
            ),
        )
        if a.switch_every:
            period = round(a.switch_every / e.step_dt)
            result["command_segments"] = [
                dict(
                    start_seconds=j * e.step_dt,
                    end_seconds=min(j + period, last) * e.step_dt,
                    command=r["target_series"][j].tolist(),
                    average_velocity=(
                        (
                            r["bodies"][min(j + period, last), 1, :2]
                            - r["bodies"][j, 1, :2]
                        )
                        / ((min(j + period, last) - j) * e.step_dt)
                    ).tolist(),
                )
                for j in range(0, last, period)
            ]
        report["cases"][c["name"]] = result
        print(json.dumps(result), flush=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
