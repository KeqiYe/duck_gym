"""Compare migrated reward terms with actual pinned upstream callables."""

import bootstrap
import json
import os
from pathlib import Path
import torch
import warp as wp
import mjlab.tasks
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from duck_gym.walking_reward import WalkingReward


def main():
    out = Path(os.environ["DUCK_RUN_DIR"])
    wp.config.kernel_cache_dir = str(out.parents[1] / "cache/warp-official")
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg.scene.num_envs = 8
    cfg.seed = 19
    env = ManagerBasedRlEnv(cfg, "cuda:0")
    env.reset()
    robot = env.scene["robot"]
    rd = robot.data
    migrated = WalkingReward(rd.default_joint_pos, robot.joint_names, rd.soft_joint_pos_limits)
    sites = env.reward_manager.get_term_cfg("foot_slip").params["asset_cfg"].site_ids
    swing = env.reward_manager.get_term_cfg("foot_swing_height").func
    maximum = {}
    for step in range(40):
        env.step(torch.randn((8, 14), device=env.device) * 0.25)
        contact = env.scene["feet_ground_contact"].data
        # Copy the same pre-call accumulator states. env.step also evaluates
        # upstream rewards, so the reference and migrated calls start equal.
        migrated.head_ema = getattr(
            env, "_head_bias_ema", torch.zeros_like(migrated.head_ema)
        ).clone()
        migrated.peak_height = swing.peak_heights.clone()
        f = dict(
            command=env.command_manager.get_command("twist"),
            lin_vel=rd.root_link_lin_vel_b,
            ang_vel=rd.root_link_ang_vel_b,
            gravity=rd.projected_gravity_b,
            world_ang_vel=rd.root_link_ang_vel_w,
            angular_momentum=env.scene["robot/root_angmom"].data,
            joint_pos=rd.joint_pos,
            action=env.action_manager.action,
            previous_action=env.action_manager.prev_action,
            contact=contact.found > 0,
            contact_time=contact.current_contact_time,
            air_time=contact.current_air_time,
            foot_height=env.scene["foot_height_scan"].data.heights,
            foot_vel=rd.site_lin_vel_w[:, sites],
            episode_steps=env.episode_length_buf,
            head_command=env.command_manager.get_command("head_pose"),
            self_collisions=env.scene["self_collision"].data.found.sum(-1).float(),
        )
        actual = migrated.terms(f, env.step_dt)
        for name, value in actual.items():
            term = env.reward_manager.get_term_cfg(name)
            expected = term.func(env, **term.params)
            error = float((value - expected).abs().max())
            maximum[name] = max(maximum.get(name, 0.0), error)
            torch.testing.assert_close(value, expected, atol=2e-6, rtol=1e-5, msg=name)
    # Verify the copied weight schedules against the pinned task configuration.
    base = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    for iteration in (0, 499, 500, 600, 750, 1000, 1250, 1500, 2500):
        expected = {
            name: term.weight for name, term in base.rewards.items() if name != "body_pose_tracking"
        }
        for key in ("action_rate_weight", "head_pose_bias_weight"):
            params = base.curriculum[key].params
            for stage in params["weight_stages"]:
                if iteration * 24 >= stage["step"]:
                    expected[params["reward_name"]] = stage["weight"]
        assert expected == WalkingReward.weights(iteration), (iteration, expected)
    result = dict(
        samples=320,
        term_max_errors=maximum,
        weight_schedules=True,
        passed=True,
        scope="Reward equations on identical upstream features and accumulator states; native feature/timing parity is separate",
    )
    (out / "reward-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
