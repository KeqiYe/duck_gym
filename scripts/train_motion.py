"""PPO for audited walking/jump/flip tasks, preserving native physical parameters."""

import bootstrap
from bootstrap import ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch
from rsl_rl.runners import OnPolicyRunner
from duck_gym.motion_env import MotionEnv
from prepare_standing import prepare
from prepare_motion import prepare_motion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill", choices=["walking", "jump", "flip"], default="walking")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--solver-iterations", type=int, default=50)
    p.add_argument("--action-scale", type=float, default=0.8)
    p.add_argument("--noise-std", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--command-direction",
        default="all",
        choices=["all", "forward", "backward", "left", "right"],
    )
    p.add_argument("--motion-config", type=Path)
    p.add_argument("--recipe", choices=["periodic", "bias", "feedforward"], default="periodic")
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--warm-start-walking",
        type=Path,
        help="Initialize actor/normalizer from a legacy walking checkpoint; new rewards and a fresh critic/optimizer",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.warm_start_walking and (args.resume or args.skill != "walking"):
        p.error("walking warm start is mutually exclusive with resume and aerial skills")
    if not torch.cuda.is_available() or "DUCK_RUN_DIR" not in os.environ:
        p.error("Launch with scripts/remote/run.py on the CUDA host")
    out = args.output or Path(os.environ["DUCK_RUN_DIR"]) / "train"
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = out / "prepared"
    prepare(model)
    prepare_motion(model)
    config = json.loads((args.motion_config or ROOT / "configs/motion_training.json").read_text())
    if args.resume and args.motion_config is None:
        previous_config = json.loads((args.resume.parent / "config.json").read_text())
        config = previous_config["environment"]["motion_config"]
    elif args.recipe in ("bias", "feedforward"):
        config["walking"].update(config["walking_bias_recipe"])
        if args.recipe == "feedforward":
            config["walking"]["reference"] = "ik_feedforward"
    uses_reference = args.skill == "walking" and config["walking"].get("reference") in (
        "ik",
        "ik_feedforward",
    )
    warm_config = None
    if args.warm_start_walking:
        warm_config = json.loads((args.warm_start_walking.parent / "config.json").read_text())
        old_env = warm_config["environment"]
        if old_env["task"] != "locomotion" or not old_env["gait"]:
            raise ValueError("Warm start requires a legacy gait policy")
        for key, actual in [
            ("action_scale", args.action_scale),
            ("iterations", args.solver_iterations),
            ("command_direction", args.command_direction),
            ("dt", 0.001),
            ("substeps", 20),
            ("fp64", False),
        ]:
            if old_env[key] != actual:
                raise ValueError(f"Warm-start {key} must match source {old_env[key]}")
        config["walking"]["reference"] = "ik"
        config["walking"]["velocity_filter_seconds"] = old_env["velocity_filter_seconds"]
        uses_reference = True
    if uses_reference and not warm_config:
        from prepare_gait import prepare_gait

        prepare_gait(
            model,
            period=config["walking"]["period"],
            lift=config["walking"]["swing_height_m"],
            smooth_velocity=config["walking"].get("reference") == "ik_feedforward",
        )
        if config["walking"].get("reference") == "ik_feedforward":
            from prepare_feedforward import add_motor_feedforward

            add_motor_feedforward(model)
    if warm_config:
        old_model = args.warm_start_walking.parent / warm_config["model_bundle"]
        for name, sha in warm_config["bundle_sha256"].items():
            if hashlib.sha256((old_model / name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Warm-start bundle hash mismatch: {name}")
        if (model / "microduck.duck").read_bytes() != (old_model / "microduck.duck").read_bytes():
            raise ValueError("Warm start requires the identical original physical model")
        for name in ("gait.npz", "gait.json"):
            shutil.copy2(old_model / name, model / name)
        gait = json.loads((model / "gait.json").read_text())
        config["walking"].update(period=gait["period"], swing_height_m=gait["lift"])
    uses_aerial_reference = (
        args.skill != "walking" and config["flip"].get("reference") == "crouch_feedforward"
    )
    if uses_aerial_reference:
        from prepare_aerial_reference import prepare_aerial_reference

        prepare_aerial_reference(model)
    env_cfg = dict(
        skill=args.skill,
        motion_config=config,
        num_envs=args.num_envs,
        iterations=args.solver_iterations,
        action_scale=args.action_scale,
        seed=args.seed,
        command_direction=args.command_direction,
        dt=0.001,
        substeps=20,
        fp64=False,
    )
    training = json.loads((ROOT / "configs/standing.json").read_text())
    training["policy"].update(
        actor_hidden_dims=[128, 128], critic_hidden_dims=[128, 128], init_noise_std=args.noise_std
    )
    if args.resume:
        old = json.loads((args.resume.parent / "config.json").read_text())
        expected = {k: v for k, v in old["environment"].items() if k not in ("num_envs", "seed")}
        actual = {k: v for k, v in env_cfg.items() if k not in ("num_envs", "seed")}
        if actual != expected:
            raise ValueError(
                "Resume requires identical task, dynamics, rewards and action semantics"
            )
        training = old["training"]
    env = MotionEnv(model, **env_cfg)
    bundle = out / "model"
    bundle.mkdir()
    for name in (
        ["microduck.duck", "metadata.json", "motion.npz", "motion.json"]
        + (["gait.npz", "gait.json"] if uses_reference else [])
        + (["aerial_reference.npz", "aerial_reference.json"] if uses_aerial_reference else [])
    ):
        shutil.copy2(model / name, bundle / name)
    record = dict(
        environment=env_cfg,
        training=training,
        model_bundle="model",
        bundle_sha256={
            x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in bundle.iterdir()
        },
        resume=str(args.resume) if args.resume else None,
        warm_start_walking=str(args.warm_start_walking) if args.warm_start_walking else None,
        warm_start_checkpoint_sha256=(
            hashlib.sha256(args.warm_start_walking.read_bytes()).hexdigest()
            if args.warm_start_walking
            else None
        ),
        source_manifest=str(Path(os.environ["DUCK_RUN_DIR"]) / "manifest.json"),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    (out / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    runner = OnPolicyRunner(env, training, log_dir=str(out), device=env.device)
    if args.resume:
        runner.load(str(args.resume))
        runner.current_learning_iteration += 1
    elif args.warm_start_walking:
        from checkpoint_io import extend_observations

        adapted = extend_observations(args.warm_start_walking, runner, out / "warm_start.pt")
        state = torch.load(adapted, map_location=env.device, weights_only=False)
        actor = {
            k.removeprefix("actor."): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("actor.")
        }
        runner.alg.policy.actor.load_state_dict(actor)
        runner.obs_normalizer.load_state_dict(state["obs_norm_state_dict"])
        # The old value function predicts a different reward; deliberately keep
        # the new critic and optimizer. This is transfer, not resumed training.
        (out / "warm_start_semantics.json").write_text(
            json.dumps(
                dict(
                    actor_and_observation_normalization_transferred=True,
                    critic_and_optimizer="fresh",
                    rewards="new motion rewards",
                    velocity_observation="new displacement-based filter; old used last-substep velocity",
                    acceptance="requires fresh physical evaluation",
                ),
                indent=2,
            )
            + "\n"
        )
    else:
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight)
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)
    before = [x.detach().clone() for x in runner.alg.policy.actor.parameters()]
    start = time.perf_counter()
    runner.learn(args.iterations)
    torch.cuda.synchronize()
    delta = (
        sum(
            float((x.detach() - y).square().sum())
            for x, y in zip(runner.alg.policy.actor.parameters(), before)
        )
        ** 0.5
    )
    report = dict(
        iterations=args.iterations,
        wall_seconds=time.perf_counter() - start,
        actor_parameter_l2_change=delta,
        checkpoint=str(out / f"model_{runner.current_learning_iteration}.pt"),
    )
    (out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    if delta <= 0:
        raise RuntimeError("PPO actor did not update")


if __name__ == "__main__":
    main()
