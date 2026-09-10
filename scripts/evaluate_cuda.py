"""Evaluate a saved GPU policy on native CPU or CUDA; record complete body poses."""

import bootstrap
from bootstrap import ROOT
import argparse
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from duck_gym.tensor_env import TensorEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--direction", choices=["forward", "backward", "left", "right"])
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
    directions = dict(forward=[0.05, 0], backward=[-0.05, 0], left=[0, 0.05], right=[0, -0.05])
    names = (
        ([args.direction] if args.direction else list(directions))
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
    warmup = 2.0 if task == "locomotion" else 0.0
    total = round((args.seconds + warmup) / env.control_dt)
    obs, _ = env.get_observations()
    states = [env.state()[0].cpu().numpy()]
    forces = [env.forces.cpu().numpy().copy()]
    ground_forces = [env.native.contact_forces().cpu().numpy()]
    actions = [env.actions.cpu().numpy().copy()]
    diagnostics = [env.native.state()[2].cpu().numpy()]
    vels = []
    yaw = []
    tilt = []
    failed = torch.zeros(nenv, device=env.device, dtype=torch.bool)
    failure_times = [None] * nenv
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
            obs, reward, done, info = env.step(policy(obs))
            b, j, d, g, v, w, heading = env.state()
            new = info["failed"] & ~failed
            failed |= info["failed"]
            for e in new.nonzero().flatten().tolist():
                failure_times[e] = (k + 1) * env.control_dt
            states.append(b.cpu().numpy())
            forces.append(env.forces.cpu().numpy().copy())
            ground_forces.append(env.native.contact_forces().cpu().numpy())
            actions.append(env.actions.cpu().numpy().copy())
            diagnostics.append(d.cpu().numpy())
            vels.append(b[:, 1, 7:9].cpu().numpy())
            yaw.append(torch.atan2(heading[:, 1], heading[:, 0]).cpu().numpy())
            tilt.append(torch.acos((-g[:, 2]).clamp(-1, 1)).cpu().numpy())
            if failed.all():
                break
            if (k + 1) % 250 == 0:
                print(f"Evaluated {(k+1)*env.control_dt:.2f}s", flush=True)
    states = np.asarray(states)
    # Integrate the real trajectory over each control interval. Sampling only
    # the last 1 ms velocity of a 20 ms interval can alias contact oscillations.
    vels = np.diff(states[:, :, 1, :2], axis=0) / env.control_dt
    yaw = np.asarray(yaw)
    tilt = np.asarray(tilt)
    begin = round(warmup / env.control_dt)
    results = []
    for e, name in enumerate(names):
        passed = failure_times[e] is None and len(vels) >= total
        stop = (
            min(len(vels), round(failure_times[e] / env.control_dt))
            if failure_times[e] is not None
            else len(vels)
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
            segment = vels[begin:stop, e]
            mean = segment.mean(0) if len(segment) else np.full(2, np.nan)
            window = round(1 / env.control_dt)
            averaged = [
                segment[i : i + window].mean(0) for i in range(0, len(segment) - window + 1, window)
            ]
            error = float(np.linalg.norm(mean - target))
            window_error = (
                float(np.linalg.norm(np.asarray(averaged) - target, axis=-1).max())
                if averaged
                else float("inf")
            )
            row.update(
                measured_seconds=len(segment) * env.control_dt,
                target_mps=target.tolist(),
                mean_velocity_mps=mean.tolist(),
                mean_error_mps=error,
                max_1s_window_error_mps=window_error,
            )
            passed = passed and error <= 0.01 and window_error <= 0.02 and row["max_yaw_rad"] <= 0.3
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
            ground_forces=np.asarray(ground_forces)[: stop + 1, e],
            actions=np.asarray(actions)[: stop + 1, e],
            diagnostics=np.asarray(diagnostics)[: stop + 1, e],
            command=env.commands[e].cpu().numpy(),
            forces=np.asarray(forces)[: stop + 1, e],
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
        seconds=args.seconds,
        warmup_seconds=warmup,
        required_measured_seconds=30,
        velocity_measurement="Body position difference over each control interval",
        seed=args.seed,
        randomize=args.randomize,
        wall_seconds=time.perf_counter() - start,
        results=results,
        passed=all(r["passed"] for r in results),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
