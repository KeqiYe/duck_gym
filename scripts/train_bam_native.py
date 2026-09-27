"""rsl-rl 5 PPO migration experiment; native physics only, no MuJoCo stepping."""

import bootstrap
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from duck_gym.bam_env import BamEnv


class TrainingEnv(BamEnv):
    def get_observations(self):
        return TensorDict(super().get_observations(), batch_size=[self.num_envs])


class NativeRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        super().save(
            path,
            dict(
                infos or {},
                env_state={
                    "common_step_counter": self.env.common_step_counter,
                    "source_iteration": self.env.source_iteration,
                },
            ),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--actor-checkpoint", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=bootstrap.ROOT / "configs/native_bam_training.json"
    )
    p.add_argument(
        "--resume-native",
        action="store_true",
        help="Restore actor, critic, optimizer, normalization, and curriculum counters",
    )
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if "DUCK_RUN_DIR" not in os.environ or not torch.cuda.is_available():
        p.error("Use the remote runner and pinned venv-official")
    out = Path(os.environ["DUCK_RUN_DIR"])
    # Build with this Torch runtime, never load the benchmark/Torch 2.8 binary.
    os.environ.setdefault("DUCK_CUDA_BUILD", str(out.parents[1] / "build/cuda-official"))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    task = json.loads(args.config.read_text())
    import mjlab.tasks
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import load_rl_cfg

    cfg = asdict(load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck"))
    cfg.update(
        logger="tensorboard",
        upload_model=False,
        seed=args.seed,
        save_interval=task["save_interval"],
        max_iterations=args.iterations,
    )
    cfg["algorithm"].update(learning_rate=task["learning_rate"], entropy_coef=task["entropy_coef"])
    for key in ("actor", "critic"):
        for option in ("cnn_cfg", "distribution_cfg"):
            if cfg[key].get(option) is None:
                cfg[key].pop(option, None)
        if cfg[key].get("rnn_type") is None:
            for option in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                cfg[key].pop(option, None)
    env = TrainingEnv(
        args.model_dir,
        num_envs=args.num_envs,
        backend="cuda",
        fp64=task["fp64"],
        iterations=task["solver_iterations"],
        auto_reset=True,
        seconds=task["episode_seconds"],
        seed=args.seed,
        reward_enabled=True,
        source_iteration=task["source_iteration"],
        training=task,
    )
    record = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        task=task,
        runner=deepcopy(cfg),
        actor_sha256=hashlib.sha256(args.actor_checkpoint.read_bytes()).hexdigest(),
        model_sha256={
            name: hashlib.sha256((args.model_dir / name).read_bytes()).hexdigest()
            for name in ("model.duck", "config.json")
        },
        torch=torch.__version__,
        actor_observations=61,
        critic_observations=env.get_observations()["critic"].shape[-1],
        accepted=False,
        physics="Native AVBD CPU/CUDA shared core, CUDA execution",
    )
    (out / "native_training.json").write_text(json.dumps(record, indent=2) + "\n")
    runner = NativeRunner(env, cfg, log_dir=str(out / "train"), device="cuda:0")
    restored = runner.load(
        str(args.actor_checkpoint),
        load_cfg=None if args.resume_native else {"actor": True},
        strict=True,
        map_location="cuda:0",
    )
    if args.resume_native:
        state = restored["env_state"]
        env.common_step_counter = int(state["common_step_counter"])
        env.source_iteration = int(state["source_iteration"])
        # Checkpoints are written after the numbered update has completed.
        runner.current_learning_iteration += 1
        record["restored_env_state"] = state
        record["first_update_index"] = runner.current_learning_iteration
        record["optimizer_restored"] = True
        (out / "native_training.json").write_text(json.dumps(record, indent=2) + "\n")
    distribution = runner.alg.actor.distribution
    if distribution.std_type != "scalar":
        raise ValueError("Expected the pinned scalar Gaussian distribution")
    if not args.resume_native:
        with torch.no_grad():
            distribution.std_param.fill_(task["initial_action_std"])
    before = torch.cat([v.detach().flatten() for v in runner.alg.actor.parameters()]).clone()
    start = time.monotonic()
    runner.learn(args.iterations, init_at_random_ep_len=True)
    after = torch.cat([v.detach().flatten() for v in runner.alg.actor.parameters()])
    change = (after - before).norm().item()
    if not torch.isfinite(after).all() or change == 0:
        raise RuntimeError("Training did not produce finite changed actor parameters")
    deploy = runner.alg.actor.as_onnx(verbose=False).cpu().eval()
    with torch.no_grad():
        sample = deploy.get_dummy_inputs()
        if env.neutral_head:
            from duck_gym.policy_wrappers import NeutralHeadPolicy
            deploy = NeutralHeadPolicy(deploy, env.head_ids).eval()
        traced = torch.jit.trace(deploy, sample)
        traced.save(str(out / "policy.pt"))
    (out / "policy.json").write_text(
        json.dumps(
            dict(
                joint_names=[n.split("/")[-1] for n in env.cfg["joint_names"]],
                action_scale=1.0,
                clip_actions=cfg.get("clip_actions"),
                normalization="Embedded upstream deployment wrapper",
                training_run=str(out),
                heading_hold=env.heading_hold,
                actual_upstream_task=False,
                neutral_head=env.neutral_head,
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "training_complete.json").write_text(
        json.dumps(
            dict(
                iterations=args.iterations,
                final_update_index=runner.current_learning_iteration,
                wall_seconds=time.monotonic() - start,
                actor_parameter_change_l2=change,
                accepted=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
