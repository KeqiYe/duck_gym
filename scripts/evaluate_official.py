"""Independent four-direction evaluation of the pinned upstream PPO baseline."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--export-only", action="store_true")
    p.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="Diagnostic speed; delivery target remains 0.05 m/s",
    )
    args = p.parse_args()
    if args.seconds <= 0 or args.warmup < 0 or args.speed <= 0 or "DUCK_RUN_DIR" not in os.environ:
        p.error("Positive duration and remote runner required")
    out = Path(os.environ["DUCK_RUN_DIR"])
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    import warp as wp
    from tensordict import TensorDict

    wp.config.kernel_cache_dir = str(out.parents[1] / "cache/warp-official")
    import mjlab.tasks
    import mjlab_microduck.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    task = "Mjlab-Velocity-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = args.seconds + args.warmup + 1
    # Keep upstream reset/model/observation randomization. Remove external pushes
    # for this first directional diagnostic, and freeze command curricula.
    cfg.events = {k: v for k, v in cfg.events.items() if v.mode != "interval"}
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = twist.rel_forward_envs = twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = twist.rel_world_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.resampling_time_range = (1e9, 1e9)
    for name in ("head_pose", "body_pose"):
        cfg.commands[name].ranges = tuple((0.0, 0.0) for _ in cfg.commands[name].ranges)
        cfg.commands[name].resampling_time_range = (1e9, 1e9)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    agent_cfg = load_rl_cfg(task)
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(wrapper, asdict(agent_cfg), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    # Use upstream's own deployment wrapper so normalization is embedded.
    # TorchScript permits local CPU inference with the existing Torch runtime.
    deploy = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    sample_obs = wrapper.get_observations()
    with torch.no_grad():
        sample_action = policy(sample_obs).cpu()
        sample_input = sample_obs["actor"].cpu()
        traced = torch.jit.trace(deploy, deploy.get_dummy_inputs())
        local_action = traced(sample_input)
        torch.testing.assert_close(local_action, sample_action, atol=2e-5, rtol=2e-5)
        traced.save(str(out / "policy.pt"))
    np.savez(
        out / "policy-check.npz", observations=sample_input.numpy(), actions=sample_action.numpy()
    )
    from mjlab.rl.exporter_utils import get_base_metadata

    export_metadata = get_base_metadata(env, str(args.checkpoint))
    export_metadata.update(
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        clip_actions=agent_cfg.clip_actions,
        normalization="Embedded upstream deployment wrapper",
    )
    (out / "policy.json").write_text(json.dumps(export_metadata, indent=2) + "\n")
    if args.export_only:
        env.close()
        return
    robot = env.scene["robot"]
    head_ids = [i for i, name in enumerate(robot.joint_names) if "head" in name or "neck" in name]
    home = robot.data.default_joint_pos[0].cpu().numpy().copy()
    dt = env.step_dt
    report = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        backend="Official MuJoCo Warp/BAM",
        seed=args.seed,
        control_dt=dt,
        requested_speed=args.speed,
        auto_reset=False,
        reset_and_sensor_randomization="Pinned upstream play configuration",
        evaluation_changes="Fixed velocity/neutral pose commands, no interval pushes or curricula",
        accepted=False,
        note="Speed checks are diagnostics; gait/video review is separate",
        directions={},
    )
    for name, velocity in dict(
        forward=(args.speed, 0.0),
        backward=(-args.speed, 0.0),
        left=(0.0, args.speed),
        right=(0.0, -args.speed),
    ).items():
        twist.ranges.lin_vel_x = (velocity[0], velocity[0])
        twist.ranges.lin_vel_y = (velocity[1], velocity[1])
        twist.ranges.ang_vel_z = (0.0, 0.0)
        obs, _ = wrapper.reset()
        # Manager configs may be copied: explicitly set its persistent command.
        cmd = env.command_manager.get_term("twist")
        cmd.vel_command_b[:] = torch.tensor([*velocity, 0.0], device=env.device)
        cmd.vel_command_w.copy_(cmd.vel_command_b)
        cmd.is_standing_env[:] = cmd.is_heading_env[:] = cmd.is_forward_env[:] = False
        cmd.is_world_env[:] = False
        obs = wrapper.get_observations()
        rows = {
            key: [] for key in ("qpos", "root_pose", "joint_pos", "contact", "command", "actions")
        }

        def capture(action):
            values = dict(
                qpos=env.sim.data.qpos,
                root_pose=robot.data.root_link_pose_w,
                joint_pos=robot.data.joint_pos,
                contact=env.scene["feet_ground_contact"].data.found,
                command=cmd.command,
                actions=action,
            )
            for key, value in values.items():
                rows[key].append(value[0].detach().cpu().numpy().copy())

        capture(torch.zeros((1, wrapper.num_actions), device=env.device))
        done = False
        # Delay buffers are updated again during the next direction's reset.
        # no_grad keeps those persistent tensors mutable outside this block.
        with torch.no_grad():
            for step in range(round((args.warmup + args.seconds) / dt)):
                action = policy(obs)
                obs, _, dones, _ = wrapper.step(action)
                capture(action)
                if bool(dones[0]):
                    done = True
                    break
        data = {k: np.asarray(v) for k, v in rows.items()}
        path = out / name
        path.mkdir()
        model = env.sim.mj_model
        np.savez_compressed(
            path / "trajectory.npz",
            **data,
            control_dt=dt,
            joint_names=np.array(robot.joint_names),
            mj_joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),
            mj_qposadr=model.jnt_qposadr,
            backend="official-mujoco-warp"
        )
        start = round(args.warmup / dt)
        xy = data["root_pose"][:, :2]
        duration = (len(xy) - 1) * dt
        measured_seconds = max(0.0, duration - start * dt)
        avg = (xy[-1] - xy[start]) / measured_seconds if measured_seconds > 0 else None
        window = round(1 / dt)
        windows = (
            (xy[start + window :] - xy[start:-window]) / (window * dt)
            if len(xy) > start + window
            else None
        )
        # Commands are body-relative; report velocity rotated by initial yaw too.
        q = data["root_pose"][0, 3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        measured = rot @ avg if avg is not None else None
        quats = data["root_pose"][:, 3:7]
        yaws = np.arctan2(
            2 * (quats[:, 0] * quats[:, 3] + quats[:, 1] * quats[:, 2]),
            1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2),
        )
        max_yaw = float(np.max(np.abs(np.unwrap(yaws) - yaws[0])))
        maxerr = (
            float(np.max(np.linalg.norm(windows @ rot.T - velocity, axis=1)))
            if windows is not None
            else None
        )
        terms = [
            key
            for key in env.termination_manager.active_terms
            if bool(env.termination_manager.get_term(key)[0])
        ]
        result = dict(
            duration=duration,
            measured_seconds=measured_seconds,
            terminated=done,
            termination_terms=terms,
            initial_yaw=float(yaw),
            command=list(velocity),
            max_yaw_change=max_yaw,
            head_joint_names=[robot.joint_names[i] for i in head_ids],
            mean_head_offset=(data["joint_pos"][:, head_ids] - home[head_ids]).mean(0).tolist(),
            average_velocity_initial_heading=measured.tolist() if measured is not None else None,
            max_one_second_velocity_error=maxerr,
            no_contact_fraction=(data["contact"] == 0).mean(axis=0).tolist(),
            diagnostic_speed_pass=bool(
                not done
                and duration >= args.warmup + args.seconds - 1e-6
                and measured is not None
                and np.linalg.norm(measured - velocity) <= 0.01
                and maxerr is not None
                and maxerr <= 0.02
                and max_yaw <= 0.3
            ),
        )
        report["directions"][name] = result
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, result, flush=True)
    env.close()


if __name__ == "__main__":
    main()
