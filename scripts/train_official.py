"""Run the pinned, unmodified upstream walking task and PPO with local logs."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if "DUCK_RUN_DIR" not in os.environ:
        parser.error("Use the approved remote runner")
    out = Path(os.environ["DUCK_RUN_DIR"])
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["MUJOCO_GL"] = "egl"
    import warp as wp

    wp.config.kernel_cache_dir = str(out.parents[1] / "cache/warp-official")
    import mjlab.tasks
    import mjlab_microduck.tasks
    from mjlab.scripts.train import TrainConfig, run_train

    task = "Mjlab-Velocity-Flat-MicroDuck"
    cfg = TrainConfig.from_task(task)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.iterations
    cfg.agent.seed = args.seed
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    if args.resume:
        (out / "resume-input").symlink_to(args.resume.resolve().parent, target_is_directory=True)
        cfg.agent.resume = True
        cfg.agent.load_run = "^resume-input$"
        cfg.agent.load_checkpoint = "^" + re.escape(args.resume.name) + "$"
    source_root = Path(__file__).resolve().parents[1]
    record = dict(
        task=task,
        upstream=json.loads((source_root / "configs/official_baseline.json").read_text()),
        overrides=vars(args) | {"resume": str(args.resume) if args.resume else None},
        source_manifest=str(out / "manifest.json"),
        reward_action_actuator_curriculum="Unmodified pinned upstream task",
        logging="Local tensorboard, no model upload",
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "official_run.json").write_text(json.dumps(record, indent=2) + "\n")
    start = time.monotonic()
    run_train(task, cfg, out / "train")
    (out / "training_complete.json").write_text(
        json.dumps(
            dict(
                requested_iterations=args.iterations,
                wall_seconds=time.monotonic() - start,
                accepted=False,
                note="Independent directional trajectories and video review required",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
