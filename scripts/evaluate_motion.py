"""Unreset physical replay with motion diagnostics, never speed-only acceptance."""

import bootstrap
from bootstrap import ROOT
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from duck_gym.motion_env import MotionEnv
from evaluate_cuda import locomotion_metrics
from audit_motion import audit
from audit_aerial import audit_aerial


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--direction", choices=["forward", "backward", "left", "right"])
    p.add_argument("--seconds", type=float)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        p.error("seconds must be finite and positive")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    model = args.checkpoint.parent / config["model_bundle"]
    for name, sha in config["bundle_sha256"].items():
        assert hashlib.sha256((model / name).read_bytes()).hexdigest() == sha, name
    env_cfg = config["environment"].copy()
    env_cfg.update(
        backend=args.backend,
        auto_reset=False,
        randomize=args.randomize,
        seed=args.seed,
        num_envs=1,
        cpu_threads=1,
    )
    skill = env_cfg["skill"]
    names = (
        [args.direction]
        if args.direction
        else (["forward", "backward", "left", "right"] if skill == "walking" else [skill])
    )
    seconds = args.seconds or (
        32 if skill == "walking" else env_cfg["motion_config"]["flip"]["episode_seconds"]
    )
    if round(seconds / (env_cfg["dt"] * env_cfg["substeps"])) < 1:
        p.error("seconds must include at least one control interval")
    torch.set_num_threads(1)
    results = []
    start = time.perf_counter()
    for name in names:
        env = MotionEnv(model, **env_cfg)
        if skill == "walking":
            env.commands[:] = torch.tensor(
                {
                    "forward": [0.05, 0],
                    "backward": [-0.05, 0],
                    "left": [0, 0.05],
                    "right": [0, -0.05],
                }[name],
                device=env.device,
            )
        runner = OnPolicyRunner(
            env, copy.deepcopy(config["training"]), log_dir=None, device=env.device
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        runner.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = env.get_observations()
        b, j, d = env.native.state()
        records = dict(
            bodies=[b[0]],
            joints=[j[0]],
            diagnostics=[d[0]],
            forces=[env.forces[0].clone()],
            ground_forces=[env.native.contact_forces()[0]],
            actions=[env.actions[0].clone()],
            targets=[env.home.clone()],
        )
        yaw = []
        failed_at = None
        with torch.inference_mode():
            for k in range(round(seconds / env.control_dt)):
                obs, reward, done, info = env.step(policy(obs))
                b, j, d, g, v, w, h = env.state()
                sample = dict(
                    bodies=b[0],
                    joints=j[0],
                    diagnostics=d[0],
                    forces=env.forces[0].clone(),
                    ground_forces=env.native.contact_forces()[0],
                    actions=env.actions[0].clone(),
                    targets=info["applied_targets"][0],
                )
                for key, value in sample.items():
                    records[key].append(value)
                yaw.append(torch.atan2(h[0, 1], h[0, 0]))
                if bool(info["failed"][0]):
                    failed_at = (k + 1) * env.control_dt
                    break
        data = {k: torch.stack(v).cpu().numpy() for k, v in records.items()}
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "trajectory.npz",
            **data,
            control_dt=env.control_dt,
            command=env.commands[0].cpu().numpy(),
            backend=args.backend,
            joint_names=np.array(env.meta["joint_names"])
        )
        report = dict(
            direction=name,
            failed_at=failed_at,
            completed_seconds=(len(data["bodies"]) - 1) * env.control_dt,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            backend=args.backend,
            randomize=args.randomize,
            seed=args.seed,
            accepted=False,
        )
        if skill == "walking":
            report["speed_only_check"] = locomotion_metrics(
                data["bodies"][:, 1, :2],
                torch.stack(yaw).cpu().numpy(),
                env.commands[0].cpu().numpy(),
                env.control_dt,
                failed=failed_at is not None,
            )
            report["motion"] = audit(folder / "trajectory.npz", model, warmup=min(2, seconds))
        else:
            report["aerial"] = {
                k: float(getattr(env, k)[0])
                for k in [
                    "air_rotation",
                    "rotation_frontier",
                    "flight_time",
                    "took_off",
                    "landed",
                    "landing_hold",
                    "invalid_contact",
                    "peak_height",
                    "peak_air_clearance",
                ]
            }
            report["aerial_detector_version"] = env.cfg.get("aerial_detector_version", 1)
            report["independent_aerial_audit"] = audit_aerial(folder / "trajectory.npz", model)
            report["scope"] = (
                "Grounded initial state, zero external force; no aerial or landing acceptance claimed automatically."
            )
        report = json_finite(report)
        (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        results.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "motion"}, indent=2), flush=True)
    final = dict(
        results=results,
        wall_seconds=time.perf_counter() - start,
        accepted=False,
        training_config_sha256=hashlib.sha256(
            (args.checkpoint.parent / "config.json").read_bytes()
        ).hexdigest(),
        physics_binary_sha256=hashlib.sha256(
            Path(
                env.ext.__file__
                if args.backend == "cuda"
                else __import__("_duck_reference").__file__
            ).read_bytes()
        ).hexdigest(),
        torch_version=torch.__version__,
        evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "python/duck_gym/motion.py",
                "python/duck_gym/motion_env.py",
                "python/duck_gym/tensor_env.py",
                "scripts/audit_motion.py",
                "scripts/audit_aerial.py",
            ]
        },
    )
    (args.output / "report.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
