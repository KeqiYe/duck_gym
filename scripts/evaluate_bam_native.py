"""Closed-loop native migration diagnostic for an exported official policy."""

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
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fp64", action="store_true")
    p.add_argument(
        "--record-physics",
        action="store_true",
        help="Save every physical substep for independent mesh collision review",
    )
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--reset-joint-noise", type=float, default=0.0)
    p.add_argument("--reset-tilt-noise", type=float, default=0.0)
    p.add_argument(
        "--directions",
        nargs="+",
        choices=("forward", "backward", "left", "right"),
        default=("forward", "backward", "left", "right"),
    )
    p.add_argument(
        "--switch-every", type=float, help="Diagnostic command cycle interval, in seconds"
    )
    p.add_argument(
        "--command-servo", type=Path, help="Explicit experimental outer command controller config"
    )
    args = p.parse_args()
    if args.seconds <= 0 or args.warmup < 0:
        p.error("Require positive evaluation duration and nonnegative warmup")
    if args.switch_every is not None and args.switch_every <= 0:
        p.error("Command switch interval must be positive")
    args.output.mkdir(parents=True)
    metadata = json.loads(args.policy.with_suffix(".json").read_text())
    env = BamEnv(
        args.model_dir,
        backend=args.backend,
        fp64=args.fp64,
        iterations=args.iterations,
        seconds=args.seconds + args.warmup + 1,
        seed=args.seed,
        heading_hold=metadata.get("heading_hold", False),
    )
    switch_steps = round(args.switch_every / env.step_dt) if args.switch_every is not None else None
    if switch_steps is not None and (
        switch_steps < 1 or abs(switch_steps * env.step_dt - args.switch_every) > 1e-8
    ):
        p.error("Command switch interval must be an integer number of control steps")
    policy = torch.jit.load(str(args.policy), map_location=env.device).eval()
    servo_cfg = json.loads(args.command_servo.read_text()) if args.command_servo else None
    servo = VelocityCommandServo(env.command, servo_cfg) if servo_cfg else None
    if servo is not None:
        env.heading_hold = False
    assert metadata["joint_names"] == [name.split("/")[-1] for name in env.cfg["joint_names"]]
    if metadata["action_scale"] != 1.0:
        raise ValueError("This adapter requires the upstream action scale of 1")
    report = dict(
        backend=args.backend,
        fp64=args.fp64,
        iterations=args.iterations,
        record_physics=args.record_physics,
        seed=args.seed,
        reset_joint_noise=args.reset_joint_noise,
        reset_tilt_noise=args.reset_tilt_noise,
        command_switch_seconds=args.switch_every,
        command_servo=servo_cfg,
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        policy_metadata=metadata,
        model_sha256={
            name: hashlib.sha256((args.model_dir / name).read_bytes()).hexdigest()
            for name in ("model.duck", "config.json", "ground-only.xml")
        },
        source_sha256={
            name: hashlib.sha256((bootstrap.ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/evaluate_bam_native.py",
                "python/duck_gym/bam_env.py",
                "python/duck_gym/actuator.py",
                "python/duck_gym/command_servo.py",
                "src/cuda/kernel.cuh",
                "src/cuda/convex.cuh",
            )
        },
        diagnostic_sampling="Body state and generic diagnostics: last substep; self_collision_substep_max: maximum across all physics substeps",
        model_scope=env.cfg["scope"],
        actual_upstream_task=False,
        accepted=False,
        directions={},
    )
    directions = dict(
        forward=(0.05, 0.0), backward=(-0.05, 0.0), left=(0.0, 0.05), right=(0.0, -0.05)
    )
    for name in args.directions:
        cmd = directions[name]
        env.reset(joint_noise=args.reset_joint_noise, tilt_noise=args.reset_tilt_noise)
        if servo is not None:
            servo.reset(env.yaw())
        env.command[0, :2] = torch.tensor(cmd, device=env.device)
        rows = []
        joints = []
        contacts = []
        sole = []
        quats = []
        diagnostics = []
        self_collision = []
        foot_velocity = []
        command_series = []
        target_series = []
        physics = {key: [] for key in ("bodies", "contact", "joints", "root_quat")}

        def record_substep(current):
            for key in physics:
                physics[key].append(getattr(current, key)[0].detach().cpu().numpy().copy())

        if args.record_physics:
            record_substep(env)
        done = False
        for step in range(round((args.seconds + args.warmup) / env.step_dt) + 1):
            target = cmd
            if args.switch_every is not None:
                offset = list(directions).index(name)
                active = (offset + step // switch_steps) % 4
                target = list(directions.values())[active]
                env.command[0, :2] = torch.tensor(
                    list(directions.values())[active], device=env.device
                )
            if servo is not None:
                goal = env.command.new_tensor([[*target, 0.0]])
                env.command.copy_(servo.command(goal, env.yaw(), env.bodies[:, 1, 12], env.step_dt))
            command_series.append(env.command[0].cpu().numpy().copy())
            target_series.append([*target, 0.0])
            rows.append(env.bodies[0].cpu().numpy().copy())
            joints.append(env.joints[0].cpu().numpy().copy())
            contacts.append(env.contact[0].cpu().numpy().copy())
            sole.append(env.sole_height[0].cpu().numpy().copy())
            quats.append(env.root_quat[0].cpu().numpy().copy())
            diagnostics.append(env.diagnostics[0].cpu().numpy().copy())
            self_collision.append([float(env.self_contacts[0]), float(env.self_penetration[0])])
            foot_velocity.append(env.site_vel[0].cpu().numpy().copy())
            if step == round((args.seconds + args.warmup) / env.step_dt) or done:
                break
            with torch.no_grad():
                action = policy(env.get_observations()["actor"])
                if metadata["clip_actions"] is not None:
                    action = action.clamp(-metadata["clip_actions"], metadata["clip_actions"])
                _, _, dones, _ = env.step(
                    action, substep_callback=record_substep if args.record_physics else None
                )
                if servo is not None:
                    servo.observe(env.origin_world_velocity, env.step_dt)
            done = bool(dones[0])
        rows, joints, contacts, sole, quats = map(np.asarray, (rows, joints, contacts, sole, quats))
        path = args.output / name
        path.mkdir()
        np.savez_compressed(
            path / "trajectory.npz",
            bodies=rows,
            joints=joints,
            contact=contacts,
            sole_height=sole,
            control_dt=env.step_dt,
            backend=args.backend,
            command=cmd,
            forces=np.zeros((len(rows), 3)),
            root_quat=quats,
            diagnostics=np.asarray(diagnostics),
            self_collision_substep_max=np.asarray(self_collision),
            foot_site_velocity=np.asarray(foot_velocity),
            command_series=np.asarray(command_series),
            target_series=np.asarray(target_series),
            command_servo_enabled=servo is not None,
            **(
                {
                    "physics_dt": env.dt,
                    **{"physics_" + key: np.asarray(value) for key, value in physics.items()},
                }
                if args.record_physics
                else {}
            ),
        )
        duration = (len(rows) - 1) * env.step_dt
        start = round(args.warmup / env.step_dt)
        measured_seconds = max(0.0, duration - start * env.step_dt)
        mean = (
            (rows[-1, 1, :2] - rows[start, 1, :2]) / measured_seconds
            if measured_seconds > 0
            else None
        )
        yaw = np.unwrap(
            np.arctan2(
                2 * (quats[:, 0] * quats[:, 3] + quats[:, 1] * quats[:, 2]),
                1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2),
            )
        )
        window = round(1 / env.step_dt)
        velocities = (
            (rows[start + window :, 1, :2] - rows[start:-window, 1, :2]) / (window * env.step_dt)
            if len(rows) > start + window
            else None
        )
        window_error = (
            float(np.linalg.norm(velocities - np.asarray(cmd), axis=-1).max())
            if velocities is not None
            else None
        )
        steady_contact = contacts[start:]
        slip = np.asarray(foot_velocity)[start:, :, :2]
        support_samples = steady_contact.sum(0)
        slip_rms = np.sqrt(
            (np.square(slip).sum(-1) * steady_contact).sum(0) / np.maximum(support_samples, 1)
        )
        support_side = np.where(
            steady_contact[:, 0] & ~steady_contact[:, 1],
            1,
            np.where(steady_contact[:, 1] & ~steady_contact[:, 0], -1, 0),
        )
        single_support = support_side[support_side != 0]
        alternating_transitions = int(np.count_nonzero(np.diff(single_support)))
        result = dict(
            duration=duration,
            measured_seconds=measured_seconds,
            terminated=done,
            command=list(cmd),
            average_velocity=(
                mean.tolist() if mean is not None and args.switch_every is None else None
            ),
            max_yaw_change=float(np.max(np.abs(yaw - yaw[0]))),
            max_one_second_velocity_error=window_error if args.switch_every is None else None,
            alternating_single_support_transitions=alternating_transitions,
            stance_foot_site_xy_speed_rms=slip_rms.tolist(),
            max_anchor_error=float(np.asarray(diagnostics)[:, 0].max()),
            max_penetration=float(np.asarray(diagnostics)[:, 1].max()),
            max_self_penetration_all_substeps=float(np.asarray(self_collision)[:, 1].max()),
            self_contact_control_intervals=int(np.count_nonzero(np.asarray(self_collision)[1:, 0])),
            unconverged_last_substep_samples=int(
                np.count_nonzero(np.asarray(diagnostics)[1:, 3] == 0)
            ),
            diagnostic_speed_pass=bool(
                args.switch_every is None
                and not done
                and measured_seconds >= args.seconds - 1e-6
                and mean is not None
                and np.linalg.norm(mean - np.asarray(cmd)) <= 0.01
                and window_error is not None
                and window_error <= 0.02
                and np.max(np.abs(yaw - yaw[0])) <= 0.3
            ),
            no_contact_fraction=(~contacts).mean(0).tolist(),
            full_sole_above_1mm_fraction=(sole > 0.001).mean(0).tolist(),
            head_offset_mean=(joints[:, env.head_ids, 0] - env.home[0, env.head_ids].cpu().numpy())
            .mean(0)
            .tolist(),
        )
        if args.switch_every is not None:
            period = switch_steps
            result["command_segments"] = [
                dict(
                    start_seconds=i * env.step_dt,
                    end_seconds=min(i + period, len(rows) - 1) * env.step_dt,
                    command=target_series[i][:2],
                    average_velocity=(
                        (rows[min(i + period, len(rows) - 1), 1, :2] - rows[i, 1, :2])
                        / ((min(i + period, len(rows) - 1) - i) * env.step_dt)
                    ).tolist(),
                )
                for i in range(0, len(rows) - 1, period)
            ]
        report["directions"][name] = result
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, result, flush=True)


if __name__ == "__main__":
    main()
