"""Train phase-conditioned skills with rsl-rl 5 and current BAM physics."""

import bootstrap
import argparse
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from duck_gym.bam_skill import BamSkillEnv


class TrainingEnv(BamSkillEnv):
    def get_observations(self):
        return TensorDict(super().get_observations(), batch_size=[self.num_envs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if "DUCK_RUN_DIR" not in os.environ or not torch.cuda.is_available():
        p.error("Use the remote runner")
    out = Path(os.environ["DUCK_RUN_DIR"])
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    import mjlab.tasks
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import load_rl_cfg

    cfg = asdict(load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck"))
    cfg.update(
        logger="tensorboard",
        upload_model=False,
        seed=a.seed,
        save_interval=100,
        max_iterations=a.iterations,
    )
    cfg["algorithm"].update(learning_rate=0.0003, entropy_coef=0.005)
    for key in ("actor", "critic"):
        for option in ("cnn_cfg", "distribution_cfg"):
            if cfg[key].get(option) is None:
                cfg[key].pop(option, None)
        if cfg[key].get("rnn_type") is None:
            for option in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                cfg[key].pop(option, None)
    task = json.loads(a.config.read_text())
    env = TrainingEnv(
        a.model_dir,
        a.preparation,
        task,
        num_envs=a.num_envs,
        backend="cuda",
        iterations=200,
        fp64=False,
        seed=a.seed,
    )
    original_cfg = deepcopy(cfg)
    runner = OnPolicyRunner(env, cfg, log_dir=str(out / "train"), device="cuda:0")
    if a.resume:
        runner.load(
            str(a.resume), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
        )
    else:
        torch.nn.init.zeros_(runner.alg.actor.mlp[-1].weight)
        torch.nn.init.zeros_(runner.alg.actor.mlp[-1].bias)
    with torch.no_grad():
        runner.alg.actor.distribution.std_param.fill_(task.get("initial_std", 0.3))
    record = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        task=task,
        runner=original_cfg,
        observations={k: v.shape[-1] for k, v in env.get_observations().items()},
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                a.config,
                a.model_dir / "model.duck",
                a.model_dir / "config.json",
                a.preparation / "geometry.npz",
                a.preparation / "crouch.json",
            ]
        },
        accepted=False,
    )
    (out / "skill_training.json").write_text(json.dumps(record, indent=2) + "\n")
    if task.get("reference_trajectory"):
        reference = a.preparation / task["reference_trajectory"]
        record["input_sha256"][str(reference)] = hashlib.sha256(
            reference.read_bytes()
        ).hexdigest()
        (out / "skill_training.json").write_text(json.dumps(record, indent=2) + "\n")
    before = torch.cat(
        [p.detach().flatten() for p in runner.alg.actor.parameters()]
    ).clone()
    start = time.monotonic()
    runner.learn(a.iterations, init_at_random_ep_len=False)
    after = torch.cat([p.detach().flatten() for p in runner.alg.actor.parameters()])
    delta = float((after - before).norm())
    if not torch.isfinite(after).all() or delta == 0:
        raise RuntimeError("Invalid PPO update")
    deploy = runner.alg.actor.as_onnx(verbose=False).cpu().eval()
    with torch.no_grad():
        torch.jit.trace(deploy, deploy.get_dummy_inputs()).save(str(out / "policy.pt"))
    (out / "training_complete.json").write_text(
        json.dumps(
            dict(
                iterations=a.iterations,
                actor_change_l2=delta,
                wall_seconds=time.monotonic() - start,
                accepted=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
