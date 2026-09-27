"""Run an exported, normalized MicroDuck ONNX actor on native BAM/AVBD dynamics."""

import bootstrap
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from duck_gym.bam_env import BamEnv


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--physics-dt", type=float)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--command", type=float, nargs=3, default=(2.2, 0.0, 0.0))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        p.error("Duration must be finite and positive")
    if args.iterations < 1 or not all(math.isfinite(v) for v in args.command):
        p.error("Require positive iterations and a finite command")
    contract = json.loads(args.contract.read_text())
    if sha256(args.policy) != contract["policy_sha256"]:
        raise ValueError("Policy SHA256 does not match the pinned contract")
    if (not contract["normalizer_embedded"] or contract["action_scale"] != 1
            or contract["action_clipping"]):
        raise ValueError("This adapter requires embedded normalization, unit scale and no clipping")
    cfg = json.loads((args.model_dir / "config.json").read_text())
    if [n.split("/")[-1] for n in cfg["joint_names"]] != contract["joint_names"]:
        raise ValueError("Policy and engine joint order differ")
    if not np.allclose(cfg["default_qpos"][7:], contract["home"], atol=1e-7, rtol=0):
        raise ValueError("Policy and engine HOME differ")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    actor = ort.InferenceSession(str(args.policy), sess_options=options,
                                providers=["CPUExecutionProvider"])
    inputs, outputs = actor.get_inputs(), actor.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape != [1, 61]
            or inputs[0].type != "tensor(float)" or len(outputs) != 1
            or outputs[0].name != "actions" or outputs[0].shape != [1, 14]
            or outputs[0].type != "tensor(float)"):
        raise ValueError("Expected obs float32[1,61] -> actions float32[1,14]")
    env = BamEnv(args.model_dir, backend=args.backend, fp64=args.fp64,
                 iterations=args.iterations, physics_dt=args.physics_dt,
                 auto_reset=False, seconds=args.seconds + 1, seed=args.seed,
                 heading_hold=False)
    steps = round(args.seconds / env.step_dt)
    if abs(steps * env.step_dt - args.seconds) > 1e-8:
        p.error("Duration must be a whole number of control steps")
    if abs(env.step_dt - 1 / contract["control_hz"]) > 1e-9:
        raise ValueError("Policy control frequency differs from the engine")
    args.output.mkdir(parents=True, exist_ok=False)
    env.reset()
    env.command[0] = env.command.new_tensor(args.command)
    trajectory = {k: [] for k in ("bodies", "joints", "contact", "sole_height",
                                   "root_quat", "diagnostics")}
    physics = {k: [] for k in ("bodies", "joints", "diagnostics")}
    observations, actions, collisions = [], [], []
    finite = True

    def record_substep(current):
        nonlocal finite
        for key in physics:
            value = getattr(current, key)[0].detach().cpu().numpy().copy()
            finite = finite and bool(np.isfinite(value).all())
            physics[key].append(value)

    def record_control():
        for key in trajectory:
            trajectory[key].append(getattr(env, key)[0].detach().cpu().numpy().copy())
        collisions.append([float(env.self_contacts[0]), float(env.self_penetration[0])])

    record_substep(env)
    record_control()
    done = False
    failure_reason = None
    start = time.monotonic()
    for step in range(steps):
        obs = env.get_observations()["actor"].detach().cpu().numpy()
        if not np.isfinite(obs).all():
            finite, done, failure_reason = False, True, "nonfinite observation"
            break
        action = actor.run(["actions"], {"obs": obs})[0]
        if not np.isfinite(action).all():
            finite, done, failure_reason = False, True, "nonfinite policy output"
            break
        observations.append(obs[0].copy())
        actions.append(action[0].copy())
        # The ONNX includes normalization. Raw outputs are HOME-relative targets;
        # do not clip, normalize again, neutralize the head, or add a command servo.
        with torch.no_grad():
            _, _, dones, _ = env.step(torch.as_tensor(action, device=env.device),
                                      substep_callback=record_substep)
        record_control()
        done = bool(dones[0]) or not finite
        if done:
            failure_reason = "native environment failure" if finite else "nonfinite physics state"
        if (step + 1) % round(1 / env.step_dt) == 0 or done:
            print(json.dumps(dict(sim_seconds=(step + 1) * env.step_dt, terminated=done,
                                  root_xyz=trajectory["bodies"][-1][1, :3].tolist(),
                                  diagnostics=trajectory["diagnostics"][-1].tolist())), flush=True)
        if done:
            break
    elapsed = time.monotonic() - start
    trajectory = {k: np.asarray(v) for k, v in trajectory.items()}
    physics = {k: np.asarray(v) for k, v in physics.items()}
    duration = len(actions) * env.step_dt
    np.savez_compressed(args.output / "trajectory.npz", **trajectory,
                        **{"physics_" + k: v for k, v in physics.items()},
                        control_dt=env.step_dt, physics_dt=env.dt, backend=args.backend,
                        command=args.command, forces=np.zeros((len(actions) + 1, 3)),
                        command_servo_enabled=False, observations=np.asarray(observations),
                        actions=np.asarray(actions), self_collision_substep_max=collisions)
    q = trajectory["root_quat"]
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 1]**2 + q[:, 2]**2), -1, 1))
    diagnostic_max = physics["diagnostics"].max(axis=0)
    report = dict(
        contract=contract, backend=args.backend, fp64=args.fp64, iterations=args.iterations,
        seed=args.seed, policy_path=str(args.policy.resolve()), onnxruntime=ort.__version__,
        inference_providers=actor.get_providers(), control_dt=env.step_dt, physics_dt=env.dt,
        actuator_delay_seconds=env.lag * env.dt, command=args.command,
        requested_seconds=args.seconds, duration_seconds=duration, wall_seconds=elapsed,
        terminated=done, failure_reason=failure_reason, finite=finite,
        completed=(not done and len(actions) == steps),
        resets_during_recording=0, command_servo=False, neutral_head=False,
        model_scope=cfg["scope"],
        simulation_scope="Native AVBD/BAM; nominal fixed battery/friction/delay, no sensor noise or domain randomization",
        root_displacement_m=(trajectory["bodies"][-1, 1, :3] - trajectory["bodies"][0, 1, :3]).tolist(),
        maximum_tilt_degrees=float(np.rad2deg(tilt).max()),
        tilt_sampling="control frames; inspect physics_bodies for substep extrema",
        physics_diagnostic_max=diagnostic_max.tolist(),
        unconverged_physics_steps=int(np.count_nonzero(physics["diagnostics"][1:, 3] == 0)),
        diagnostic_columns=["anchor_error_m", "penetration_m", "contact_count", "converged",
                            "axis_error", "numerical_failure"],
        maximum_self_penetration_m=float(np.asarray(collisions)[:, 1].max()),
        model_sha256={name: sha256(args.model_dir / name) for name in ("config.json", "model.duck")},
        source_sha256={name: sha256(bootstrap.ROOT / name) for name in (
            "scripts/evaluate_onnx_native.py", "python/duck_gym/bam_env.py",
            "python/duck_gym/actuator.py", "src/cuda/kernel.cuh", "src/cuda/convex.cuh")},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
