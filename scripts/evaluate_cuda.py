"""Evaluate a saved GPU policy on native CPU or CUDA; record complete body poses."""

import bootstrap
from bootstrap import ROOT
import argparse
import json
import hashlib
import os
from pathlib import Path
import time
import platform
import sys
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from duck_gym.tensor_env import TensorEnv

PROTOCOL = json.loads((ROOT / "configs/locomotion_evaluation.json").read_text())


def locomotion_metrics(positions, yaw, target, dt, warmup=2.0, failed=False):
    """The 30 s protocol uses physical displacement and every full 1 s window."""
    segment = np.asarray(positions, dtype=np.float64)[round(warmup / dt) :]
    count = max(0, len(segment) - 1)
    mean = (segment[-1] - segment[0]) / (count * dt) if count else np.full(2, np.nan)
    window = max(1, round(PROTOCOL["window_seconds"] / dt))
    averaged = (segment[window:] - segment[:-window]) / (window * dt)
    error = float(np.linalg.norm(mean - target))
    window_error = (
        float(np.linalg.norm(averaged - target, axis=-1).max()) if len(averaged) else float("inf")
    )
    measured = count * dt
    criteria = (
        not failed
        and error <= PROTOCOL["mean_speed_error_mps"]
        and window_error <= PROTOCOL["window_speed_error_mps"]
        and np.isfinite(yaw).all()
        and float(np.abs(yaw).max()) <= PROTOCOL["max_yaw_rad"]
    )
    return dict(
        measured_seconds=measured,
        target_mps=np.asarray(target).tolist(),
        mean_velocity_mps=mean.tolist(),
        mean_error_mps=error,
        max_1s_window_error_mps=window_error,
        criteria_met_for_measured_duration=bool(criteria),
        passed=bool(criteria and measured >= PROTOCOL["measured_seconds"] - 1e-9),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=PROTOCOL["measured_seconds"])
    p.add_argument("--direction", choices=["forward", "backward", "left", "right"])
    p.add_argument(
        "--batch-policy",
        action="store_true",
        help="Use a full policy batch instead of per-env inference",
    )
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--seed", type=int, default=456)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error("--seconds must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    env_cfg = config["environment"].copy()
    task = env_cfg["task"]
    speed = PROTOCOL["target_speed_mps"]
    directions = dict(forward=[speed, 0], backward=[-speed, 0], left=[0, speed], right=[0, -speed])
    selection = args.direction or env_cfg.get("command_direction", "all")
    names = (
        ([selection] if selection != "all" else list(directions))
        if task == "locomotion"
        else ["standing"]
    )
    nenv = len(names)
    env_cfg.update(
        num_envs=nenv,
        backend=args.backend,
        cpu_threads=args.cpu_threads,
        auto_reset=False,
        randomize=args.randomize,
        seed=args.seed,
    )
    bundle = args.checkpoint.parent / config.get("model_bundle", "model")
    model_dir = args.model_dir or (bundle if bundle.is_dir() else Path(config["model_dir"]))
    if args.model_dir is None and bundle.is_dir():
        for name, expected in config.get("bundle_sha256", {}).items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checkpoint model bundle changed: {name}")
    env = TensorEnv(model_dir, **env_cfg)
    runner = OnPolicyRunner(env, config["training"], log_dir=None, device=env.device)
    # rsl-rl 2.3.3 load lacks map_location; load on CPU first for portable inference.
    saved = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    runner.alg.policy.load_state_dict(saved["model_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(saved["obs_norm_state_dict"])
    policy = runner.get_inference_policy(device=env.device)
    if task == "locomotion":
        env.commands[:] = torch.tensor(
            [directions[n] for n in names], device=env.device, dtype=env.dtype
        )
    warmup = PROTOCOL["warmup_seconds"] if task == "locomotion" else 0.0
    total = round((args.seconds + warmup) / env.control_dt)
    failure_poll_steps = max(1, round(1.0 / env.control_dt))
    obs, _ = env.get_observations()
    b, j, d = env.native.state()
    history = dict(
        bodies=[b],
        forces=[env.forces.clone()],
        ground_forces=[env.native.contact_forces()],
        actions=[env.actions.clone()],
        joints=[j],
        targets=[env.home.expand(nenv, -1).clone()],
        diagnostics=[d],
        yaw=[],
        tilt=[],
    )
    failed = torch.zeros(nenv, device=env.device, dtype=torch.bool)
    failure_steps = torch.full((nenv,), -1, device=env.device, dtype=torch.long)
    start = time.perf_counter()
    with torch.inference_mode():
        for k in range(total):
            if task == "standing":
                t = k * env.control_dt
                env.forces.zero_()
                for onset, direction in [
                    (5, [1, 0, 0]),
                    (10, [-1, 0, 0]),
                    (15, [0, 1, 0]),
                    (20, [0, -1, 0]),
                ]:
                    if onset <= t < onset + 0.2 - 1e-9:
                        env.forces[0] = torch.tensor(direction, device=env.device) * (
                            0.05 * env.meta["mass"] * 9.81
                        )
            action = (
                policy(obs)
                if args.batch_policy
                else torch.cat([policy(obs[e : e + 1]) for e in range(nenv)], dim=0)
            )
            obs, reward, done, info = env.step(action)
            b, j, d, g, v, w, heading = env.state()
            new = info["failed"] & ~failed
            failure_steps = torch.where(new, k + 1, failure_steps)
            failed |= info["failed"]
            sample = dict(
                bodies=b,
                forces=env.forces.clone(),
                ground_forces=env.native.contact_forces(),
                actions=env.actions.clone(),
                joints=j,
                targets=info["applied_targets"],
                diagnostics=d,
                yaw=torch.atan2(heading[:, 1], heading[:, 0]),
                tilt=torch.acos((-g[:, 2]).clamp(-1, 1)),
            )
            for key, value in sample.items():
                history[key].append(value)
            # Keep owned snapshots on their device, then transfer once. Only
            # poll early termination once per simulated second on CUDA.
            if (k + 1) % failure_poll_steps == 0 and bool(failed.all()):
                break
            if (k + 1) % 250 == 0:
                print(f"Evaluated {(k+1)*env.control_dt:.2f}s", flush=True)
    record = {key: torch.stack(value).cpu().numpy() for key, value in history.items()}
    failure_times = [
        step * env.control_dt if step >= 0 else None for step in failure_steps.cpu().tolist()
    ]
    states, yaw, tilt = record["bodies"], record["yaw"], record["tilt"]
    completed = len(states) - 1
    results = []
    for e, name in enumerate(names):
        passed = failure_times[e] is None and completed >= total
        stop = (
            min(completed, round(failure_times[e] / env.control_dt))
            if failure_times[e] is not None
            else completed
        )
        row = dict(
            direction=name,
            failed_at=failure_times[e],
            max_tilt_rad=float(tilt[:stop, e].max()),
            max_yaw_rad=float(np.abs(yaw[:stop, e]).max()),
        )
        row["completed_seconds"] = stop * env.control_dt
        if task == "locomotion":
            target = env.commands[e].cpu().numpy()
            row.update(
                locomotion_metrics(
                    states[: stop + 1, e, 1, :2],
                    yaw[:stop, e],
                    target,
                    env.control_dt,
                    warmup,
                    failed=not passed,
                )
            )
        else:
            row["max_horizontal_displacement_m"] = float(
                np.linalg.norm(states[: stop + 1, e, 1, :2] - states[0, e, 1, :2], axis=-1).max()
            )
            passed = passed and row["max_horizontal_displacement_m"] <= 0.05
            row["criteria_met_for_measured_duration"] = bool(passed)
            row["passed"] = bool(passed and args.seconds >= 30)
        # Incomplete evaluations use JSON null, never non-standard NaN/Infinity.
        for key, value in list(row.items()):
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
            if isinstance(value, list):
                row[key] = [v if np.isfinite(v) else None for v in value]
        results.append(row)
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / "trajectory.npz",
            bodies=states[: stop + 1, e],
            ground_forces=record["ground_forces"][: stop + 1, e],
            actions=record["actions"][: stop + 1, e],
            joints=record["joints"][: stop + 1, e],
            joint_names=np.asarray(env.meta["joint_names"]),
            targets=record["targets"][: stop + 1, e],
            diagnostics=record["diagnostics"][: stop + 1, e],
            command=env.commands[e].cpu().numpy(),
            forces=record["forces"][: stop + 1, e],
            control_dt=env.control_dt,
            backend=args.backend,
        )
    report = dict(
        checkpoint=str(args.checkpoint),
        model_sha256=hashlib.sha256((model_dir / "microduck.duck").read_bytes()).hexdigest(),
        model_dir=str(model_dir),
        fp64=env_cfg.get("fp64", False),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        backend=args.backend,
        num_envs=nenv,
        policy_batch_size=nenv if args.batch_policy else 1,
        cpu_threads=args.cpu_threads if args.backend == "cpu" else None,
        seconds=args.seconds,
        warmup_seconds=warmup,
        required_measured_seconds=PROTOCOL["measured_seconds"] if task == "locomotion" else 30,
        protocol=PROTOCOL if task == "locomotion" else None,
        runtime={
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/evaluate_cuda.py",
                "python/duck_gym/tensor_env.py",
                "src/cuda/kernel.cuh",
                "src/cuda/model.hpp",
                "src/cuda/cpu_bindings.cpp",
                "src/cuda/extension.cu",
                "configs/locomotion_evaluation.json",
            )
        },
        velocity_measurement="Physical displacement; all full sliding 1 s windows",
        seed=args.seed,
        randomize=args.randomize,
        wall_seconds=time.perf_counter() - start,
        results=results,
        passed=all(r["passed"] for r in results),
    )
    binary = Path(
        env.ext.__file__ if args.backend == "cuda" else sys.modules["_duck_reference"].__file__
    )
    report["native_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    if task == "locomotion":
        report["accepted"] = report["passed"] and PROTOCOL["status"] == "confirmed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
