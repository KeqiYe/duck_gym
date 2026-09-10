"""Train standing or four-direction locomotion on native CUDA physics."""

import bootstrap
from bootstrap import ROOT
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
from datetime import datetime
import torch
from rsl_rl.runners import OnPolicyRunner
from duck_gym.tensor_env import TensorEnv
from prepare_standing import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["standing", "locomotion"], default="standing")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--solver-iterations", type=int)
    p.add_argument("--action-scale", type=float)
    p.add_argument("--foot-clearance", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gait", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--velocity-filter-seconds", type=float)
    p.add_argument("--load-transfer", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tracking-sigma", type=float)
    p.add_argument("--tracking-weight", type=float)
    p.add_argument("--heading-weight", type=float)
    p.add_argument("--noise-std", type=float)
    p.add_argument("--fp64", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "DUCK_RUN_DIR", ROOT / "runs" / datetime.now().strftime("train-%Y%m%d-%H%M%S")
            )
        )
        / "train",
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--dt", type=float)
    p.add_argument("--substeps", type=int)
    args = p.parse_args()
    if not torch.cuda.is_available() or "DUCK_CUDA_BUILD" not in os.environ:
        p.error("Use scripts/remote/run.py to sync, build, and train on a CUDA host")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_dir = Path(os.environ["DUCK_CUDA_BUILD"]).parents[1] / "build/models/standing"
    prepare(model_dir)
    cfg = json.loads((ROOT / "configs/standing.json").read_text())
    cfg["save_interval"] = 50
    cfg["policy"].update(
        actor_hidden_dims=[128, 128], critic_hidden_dims=[128, 128], init_noise_std=0.3
    )
    env_cfg = dict(
        gait=bool(args.gait),
        foot_clearance=bool(args.foot_clearance),
        tracking_sigma=args.tracking_sigma,
        load_transfer=bool(args.load_transfer),
        velocity_filter_seconds=args.velocity_filter_seconds or 0.0,
        tracking_weight=args.tracking_weight if args.tracking_weight is not None else 4.0,
        heading_weight=args.heading_weight if args.heading_weight is not None else 0.4,
        task=args.task,
        num_envs=args.num_envs,
        iterations=args.solver_iterations if args.solver_iterations is not None else 100,
        seed=args.seed,
        action_scale=args.action_scale if args.action_scale is not None else 0.3,
        fp64=bool(args.fp64),
        dt=args.dt if args.dt is not None else 0.001,
        substeps=args.substeps if args.substeps is not None else 20,
    )
    if args.resume:
        old = json.loads((args.resume.parent / "config.json").read_text())
        cfg = old["training"]
        if old["environment"]["task"] != args.task:
            raise ValueError("Cannot load different observation/action task")
        for key in ("gait", "foot_clearance", "load_transfer"):
            if getattr(args, key) is None:
                env_cfg[key] = old["environment"].get(key, False)
        if args.velocity_filter_seconds is None:
            env_cfg["velocity_filter_seconds"] = old["environment"].get(
                "velocity_filter_seconds", 0.0
            )
        if args.tracking_sigma is None:
            env_cfg["tracking_sigma"] = old["environment"].get("tracking_sigma")
        for key in ("tracking_weight", "heading_weight"):
            if getattr(args, key) is None:
                env_cfg[key] = old["environment"].get(key, env_cfg[key])
        if args.action_scale is None:
            env_cfg["action_scale"] = old["environment"]["action_scale"]
        for k in ("fp64", "dt", "substeps"):
            if getattr(args, k) is None:
                env_cfg[k] = old["environment"][k]
        if args.solver_iterations is None:
            env_cfg["iterations"] = old["environment"]["iterations"]
    needs_gait = any(env_cfg[k] for k in ("gait", "foot_clearance", "load_transfer"))
    if needs_gait:
        from prepare_gait import prepare_gait

        old_bundle = args.resume.parent / old.get("model_bundle", "model") if args.resume else None
        if old_bundle and (old_bundle / "gait.npz").is_file():
            for name in ("gait.npz", "gait.json"):
                shutil.copy2(old_bundle / name, model_dir / name)
        else:
            prepare_gait(model_dir)
    env = TensorEnv(model_dir, **env_cfg)
    bundle = args.output / "model"
    bundle.mkdir()
    for name in ["microduck.duck", "metadata.json"] + (
        ["gait.npz", "gait.json"] if needs_gait else []
    ):
        shutil.copy2(model_dir / name, bundle / name)
    config = dict(
        resume=str(args.resume) if args.resume else None,
        noise_std_override=args.noise_std,
        training=cfg,
        environment=env_cfg,
        model_dir=str(model_dir),
        model_bundle="model",
        bundle_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()
        },
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        model_sha256=hashlib.sha256((model_dir / "microduck.duck").read_bytes()).hexdigest(),
        source_manifest=(
            str(Path(os.environ["DUCK_RUN_DIR"]) / "manifest.json")
            if "DUCK_RUN_DIR" in os.environ
            else None
        ),
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    runner = OnPolicyRunner(env, cfg, log_dir=str(args.output), device="cuda:0")
    if args.resume:
        from checkpoint_io import extend_observations

        adapted = extend_observations(args.resume, runner, args.output / "resume_adapted.pt")
        runner.load(str(adapted))
        runner.current_learning_iteration += 1
        ratio = old["environment"]["action_scale"] / env_cfg["action_scale"]
        if ratio != 1:
            with torch.no_grad():
                for parameter in runner.alg.policy.actor[-1].parameters():
                    parameter.mul_(ratio)
                    runner.alg.optimizer.state.pop(parameter, None)
                runner.alg.policy.std.mul_(ratio)
            runner.alg.optimizer.state.pop(runner.alg.policy.std, None)
    else:
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight)
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)
    if args.noise_std is not None:
        if args.noise_std <= 0:
            raise ValueError("noise std must be positive")
        with torch.no_grad():
            runner.alg.policy.std.fill_(args.noise_std)
        runner.alg.optimizer.state.pop(runner.alg.policy.std, None)
    before = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
    start = time.perf_counter()
    runner.learn(args.iterations)
    torch.cuda.synchronize()
    change = (
        sum(
            float((a - b).square().sum().detach())
            for a, b in zip(before, runner.alg.policy.actor.parameters())
        )
        ** 0.5
    )
    report = dict(
        iterations=args.iterations,
        wall_seconds=time.perf_counter() - start,
        actor_parameter_l2_change=change,
        checkpoint=str(args.output / f"model_{runner.current_learning_iteration}.pt"),
    )
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    if not change > 0:
        raise RuntimeError("PPO actor did not update")


if __name__ == "__main__":
    main()
