"""Train a CPU policy using rsl-rl PPO and native AVBD physics."""

import bootstrap
from bootstrap import ROOT
import argparse
from datetime import datetime
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from duck_gym import StandingEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / ("standing-" + datetime.now().strftime("%Y%m%d-%H%M%S")),
    )
    p.add_argument("--model-dir", type=Path, default=ROOT / "build/models/standing")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--solver-iterations", type=int)
    p.add_argument("--action-scale", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.iterations < 1:
        p.error("--iterations must be positive")
    if (args.output / "config.json").exists():
        p.error("Use a new output directory; --resume reads an existing checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prior = json.loads((args.resume.parent / "config.json").read_text()) if args.resume else None
    cfg = prior["training"] if prior else json.loads((ROOT / "configs/standing.json").read_text())
    env_cfg = dict(dt=0.001, substeps=20, episode_seconds=30, randomize=True)
    if prior:
        env_cfg.update(prior["environment"])
    env_cfg.update(num_envs=args.num_envs, threads=args.threads, seed=args.seed)
    env_cfg["iterations"] = args.solver_iterations or env_cfg.get("iterations", 200)
    env_cfg["action_scale"] = (
        args.action_scale
        if args.action_scale is not None
        else env_cfg.get("action_scale", 0.15 if prior else 0.05)
    )
    env = StandingEnv(args.model_dir, **env_cfg)
    run_cfg = dict(
        training=cfg,
        environment=env_cfg,
        model_dir=str(args.model_dir.resolve()),
        command=__import__("sys").argv,
        host=platform.node(),
        platform=platform.platform(),
        torch=torch.__version__,
        backend="cpu",
        actor_initialization="checkpoint" if prior else "zero_residual",
        native_sha256=hashlib.sha256(
            Path(__import__("_duck_cpu").__file__).read_bytes()
        ).hexdigest(),
        dependency_versions={
            name: __import__("importlib.metadata", fromlist=["version"]).version(name)
            for name in ("rsl-rl-lib", "torch", "numpy", "pybind11")
        },
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    run_cfg["source_sha256"] = {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        for f in tracked
        if f and (ROOT / f).is_file()
    }
    run_cfg["model_sha256"] = hashlib.sha256(
        (args.model_dir / "microduck.duck").read_bytes()
    ).hexdigest()
    (args.output / "config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")
    runner = OnPolicyRunner(env, cfg, log_dir=str(args.output), device="cpu")
    if args.resume:
        runner.load(str(args.resume))
        runner.current_learning_iteration += 1
    else:
        # Residual control starts at the verified nominal PD pose.
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight)
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)
    before = [v.detach().clone() for v in runner.alg.policy.actor.parameters()]
    start = time.perf_counter()
    runner.learn(args.iterations)
    change = (
        sum(
            float((a - b).detach().square().sum())
            for a, b in zip(before, runner.alg.policy.actor.parameters())
        )
        ** 0.5
    )
    report = dict(
        wall_seconds=time.perf_counter() - start,
        learning_iterations=args.iterations,
        actor_parameter_l2_change=change,
        checkpoint=str(args.output / f"model_{runner.current_learning_iteration}.pt"),
    )
    assert change > 0 and np.isfinite(change), "PPO did not update the actor"
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
