"""30 s continuous evaluation, without auto-reset, with recorded force pulses."""

import bootstrap
from bootstrap import ROOT
import argparse
from pathlib import Path
import json
import hashlib
import platform
import subprocess
import time
import numpy as np
import torch
from duck_gym import StandingEnv


def evaluate(
    model_dir, output, checkpoint=None, seconds=30.0, pushes=True, seed=123, randomize=False
):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    overrides = {}
    if checkpoint:
        config = json.loads((checkpoint.parent / "config.json").read_text())
        overrides = {
            k: v
            for k, v in config["environment"].items()
            if k in ("action_scale", "dt", "substeps")
        }
    env = StandingEnv(
        model_dir,
        **overrides,
        num_envs=1,
        threads=1,
        seed=seed,
        randomize=randomize,
        auto_reset=False,
        episode_seconds=seconds
    )
    policy = lambda obs: torch.zeros((1, env.num_actions))
    if checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        config = json.loads((checkpoint.parent / "config.json").read_text())["training"]
        runner = OnPolicyRunner(env, config, log_dir=None, device="cpu")
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device="cpu")
    states = [env.native.body_state()[0]]
    forces = [np.zeros(3)]
    diags = []
    tilts = []
    heights = []
    failed_at = None
    pulses = (
        [(5.0, [1, 0, 0]), (10.0, [-1, 0, 0]), (15.0, [0, 1, 0]), (20.0, [0, -1, 0])]
        if pushes
        else []
    )
    strength = 0.05 * env.meta["mass"] * 9.81
    start = time.perf_counter()
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for k in range(round(seconds / env.control_dt)):
            t = k * env.control_dt
            force = np.zeros(3)
            for onset, direction in pulses:
                if onset <= t < onset + 0.2 - 1e-9:
                    force = strength * np.array(direction)
            env.forces[0] = force
            obs, rew, done, info = env.step(policy(obs))
            b, j, g, v, w = env.state()
            states.append(b[0])
            forces.append(force)
            diags.append(info["diagnostics"][0])
            tilts.append(float(np.arccos(np.clip(-g[0, 2], -1, 1))))
            heights.append(float(b[0, 1, 2]))
            if info["failed"][0]:
                failed_at = (k + 1) * env.control_dt
                break
    states = np.asarray(states)
    diags = np.asarray(diags)
    displacement = float(np.linalg.norm(states[:, 1, :2] - states[0, 1, :2], axis=1).max())
    report = dict(
        passed=failed_at is None
        and displacement <= 0.05
        and (len(states) - 1) * env.control_dt >= seconds - 1e-8,
        simulated_seconds=(len(states) - 1) * env.control_dt,
        failed_at=failed_at,
        seed=seed,
        randomize=randomize,
        checkpoint=str(checkpoint) if checkpoint else None,
        wall_seconds=time.perf_counter() - start,
        max_tilt_rad=max(tilts),
        height_range_m=[min(heights), max(heights)],
        max_horizontal_displacement_m=displacement,
        max_joint_error_m=float(diags[:, 0].max()),
        max_penetration_m=float(diags[:, 1].max()),
        unconverged_control_samples=int((diags[:, 3] == 0).sum()),
        physics_dt=env.dt,
        solver_iterations=200,
        control_dt=env.control_dt,
        force_newtons=strength,
        pulse_seconds=0.2,
        pulses=pulses,
        failure_thresholds=dict(
            tilt_rad=0.5,
            height_error_m=0.04,
            horizontal_displacement_m=0.05,
            joint_error_m=0.005,
            penetration_m=0.005,
        ),
    )
    report["host"] = platform.node()
    report["model_sha256"] = hashlib.sha256(
        (Path(model_dir) / "microduck.duck").read_bytes()
    ).hexdigest()
    report["checkpoint_sha256"] = (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
    )
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    report["source_sha256"] = {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        for f in tracked
        if f and (ROOT / f).is_file()
    }
    report["native_sha256"] = hashlib.sha256(
        Path(__import__("_duck_cpu").__file__).read_bytes()
    ).hexdigest()
    report["push_recovery_samples"] = [
        dict(
            onset=t,
            tilt_before_rad=tilts[max(0, round(t / env.control_dt) - 1)],
            tilt_2s_after_rad=tilts[round((t + 2.2) / env.control_dt) - 1],
        )
        for t, direction in pulses
        if t + 2.2 <= (len(states) - 1) * env.control_dt
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "trajectory.npz",
        bodies=states,
        forces=forces,
        diagnostics=diags,
        control_dt=env.control_dt,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2), flush=True
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=ROOT / "build/models/standing")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--no-pushes", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--randomize", action="store_true")
    a = p.parse_args()
    r = evaluate(
        a.model_dir, a.output, a.checkpoint, a.seconds, not a.no_pushes, a.seed, a.randomize
    )
    raise SystemExit(0 if r["passed"] else 1)
