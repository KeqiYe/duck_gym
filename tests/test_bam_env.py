"""Native actor/reward feature frames against independent MuJoCo kinematics."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import mujoco
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
from model_io import export_model, rot, inv
from duck_gym.bam_env import BamEnv
from duck_gym.walking_reward import WalkingReward

MODEL_DIR = Path(os.environ.get("DUCK_OFFICIAL_MODEL", bootstrap.ROOT / "build/models/official"))


@unittest.skipUnless(
    (MODEL_DIR / "config.json").exists(),
    "Set DUCK_OFFICIAL_MODEL to a prepared official diagnostic model",
)
class FeatureTests(unittest.TestCase):
    def test_step_refinement_preserves_control_and_actuator_delay(self):
        coarse=BamEnv(MODEL_DIR,fp64=True)
        fine=BamEnv(MODEL_DIR,fp64=True,physics_dt=.0025)
        self.assertEqual(fine.decimation,2*coarse.decimation)
        self.assertAlmostEqual(fine.step_dt,coarse.step_dt)
        self.assertAlmostEqual(fine.lag*fine.dt,coarse.lag*coarse.dt)
        with self.assertRaises(ValueError):BamEnv(MODEL_DIR,physics_dt=.0015)

    def test_neutral_head_changes_targets_without_freezing_physical_joints(self):
        cfg = json.loads((bootstrap.ROOT / 'configs/native_bam_neutral_head.json').read_text())
        env = BamEnv(MODEL_DIR, fp64=True, training=cfg)
        actions = torch.zeros(1,14)
        actions[:,env.head_ids] = .5
        env.step(actions)
        torch.testing.assert_close(env.action[:,env.head_ids],torch.zeros(1,4,dtype=torch.float64))
        torch.testing.assert_close(env.target_history[:,:,env.head_ids],env.home[None,:,env.head_ids].expand(env.lag+1,-1,-1))
        self.assertGreater(float((env.joints[:,env.head_ids,0]-env.home[:,env.head_ids]).abs().max()),0)

    def test_neutral_head_deployment_matches_action_mapping(self):
        from duck_gym.policy_wrappers import NeutralHeadPolicy
        policy = torch.nn.Linear(61,14)
        wrapped = NeutralHeadPolicy(policy,[5,6,7,8])
        sample = torch.randn(4,61)
        expected = policy(sample).detach()
        expected[:,5:9] = 0
        actual = torch.jit.trace(wrapped,sample)(sample)
        torch.testing.assert_close(actual,expected)

    def test_head_stability_penalizes_motion_even_with_zero_mean_pose(self):
        cfg = json.loads((bootstrap.ROOT / "configs/native_bam_head_stability.json").read_text())
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True, training=cfg, reward_enabled=True)
        # Equal and opposite offsets must both incur a position cost. The
        # additional velocity term distinguishes swinging through neutral.
        original = env.refresh
        def refresh():
            original()
            env.joints[:, env.head_ids, 0] = env.home[:, env.head_ids]
            env.joints[:, env.head_ids, 1] = 0
            env.joints[:, env.head_ids[2], 0] += env.home.new_tensor([0.2, -0.2])
            env.joints[1, env.head_ids[2], 1] = 2
        env.refresh = refresh
        _, _, _, info = env.step(torch.zeros(2, 14))
        self.assertAlmostEqual(float(info['log']['Reward/head_position_l2']), .04, places=7)
        self.assertAlmostEqual(float(info['log']['Reward/head_velocity_l2']), 2, places=7)

    def test_transient_self_contact_survives_control_interval_and_reset_is_local(self):
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True, reward_enabled=True)
        # A contact on an earlier physics substep must not disappear from the
        # control reward/diagnostics just because the final substep is clear.
        calls = [0]

        def transient():
            value = env.home.new_zeros(2, 2)
            if calls[0] == 0:
                value[0] = value.new_tensor([1, 0.0001])
            calls[0] += 1
            return value

        env.native.self_contact_stats = transient
        env.step(torch.zeros_like(env.action))
        self.assertEqual(calls[0], env.decimation)
        np.testing.assert_array_equal(env.reward_features()["self_collisions"], [1, 0])
        self.assertAlmostEqual(float(env.self_penetration[0]), 0.0001)
        env.reset(torch.tensor([False, True]))
        np.testing.assert_array_equal(env.self_contacts, [1, 0])
        env.step(torch.zeros_like(env.action))
        self.assertFalse(env.self_contacts.any())
        self.assertFalse(env.self_penetration.any())

    def test_random_kinematic_features(self):
        model_dir = MODEL_DIR
        cfg = json.loads((model_dir / "config.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(model_dir / "ground-only.xml"))
        data = mujoco.MjData(model)
        rng = np.random.default_rng(28)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "config.json").write_text(json.dumps(cfg))
            for _ in range(5):
                data.qpos[:] = cfg["default_qpos"]
                data.qpos[3:7] = rng.normal(size=4)
                data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
                data.qpos[7:] += rng.uniform(-0.2, 0.2, 14)
                data.qvel[:] = rng.uniform(-2, 2, model.nv)
                export_model(model, data, folder / "model.duck")
                mujoco.mj_subtreeVel(model, data)
                env = BamEnv(folder, fp64=True)
                np.testing.assert_allclose(
                    env.gravity[0], rot(inv(data.xquat[1]), [0, 0, -1]), atol=2e-12
                )
                np.testing.assert_allclose(env.ang_vel[0], data.qvel[3:6], atol=2e-12)
                np.testing.assert_allclose(
                    env.lin_vel[0], rot(inv(data.xquat[1]), data.qvel[:3]), atol=2e-12
                )
                np.testing.assert_allclose(
                    env.reward_features()["angular_momentum"][0], data.subtree_angmom[1], atol=2e-12
                )
                for index, side in enumerate(("left", "right")):
                    site = model.site(f"robot/{side}_foot").id
                    np.testing.assert_allclose(
                        env.site_pos[0, index], data.site_xpos[site], atol=2e-12
                    )
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jp, jr, site)
                    np.testing.assert_allclose(env.site_vel[0, index], jp @ data.qvel, atol=2e-12)
                    geom = model.geom(f"robot/{side}_foot_collision")
                    mesh = int(geom.dataid[0])
                    vertices = model.mesh_vert[
                        model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh]
                        + model.mesh_vertnum[mesh]
                    ]
                    world = (
                        vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
                    )
                    self.assertAlmostEqual(
                        float(env.sole_height[0, index]), float(world[:, 2].min()), delta=2e-8
                    )

    def test_reward_reset_and_observation_lag(self):
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True, reward_enabled=True)
        env.command[:, 0] = 0.05
        for _ in range(3):
            velocity = env.joints[:, :, 1].clone()
            obs, reward, done, info = env.step(torch.full((2, 14), 0.01))
            torch.testing.assert_close(obs["actor"][:, 20:34], velocity.float())
            self.assertTrue(bool(torch.isfinite(reward).all()))
            self.assertEqual(sum(k.startswith("Reward/") for k in info["log"]), 15)
            self.assertFalse(bool(done.any()))
        preserved = env.reward_model.head_ema[1].clone()
        joints = env.joints[1].clone()
        env.reset(torch.tensor([True, False]))
        torch.testing.assert_close(env.reward_model.head_ema[1], preserved)
        torch.testing.assert_close(env.joints[1], joints)
        self.assertEqual(float(env.reward_model.head_ema[0].abs().sum()), 0.0)
        self.assertEqual(float(env.target_history[:, 0].sub(env.home[0]).abs().sum()), 0.0)
        self.assertEqual(int(env.episode_length_buf[0]), 0)
        self.assertEqual(int(env.episode_length_buf[1]), 3)

    def test_low_speed_kernel_preserves_vertical_tolerance(self):
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True)
        reward = WalkingReward(
            env.home,
            env.cfg["joint_names"],
            env.soft_limits,
            linear_variance=0.0025,
            vertical_variance=0.1,
        )
        features = env.reward_features()
        features["command"][:] = env.home.new_tensor([0.05, 0.0, 0.0])
        features["lin_vel"][:] = env.home.new_tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.1]])
        scores = reward.terms(features, 0.02)["track_linear_velocity"]
        torch.testing.assert_close(scores, torch.exp(env.home.new_tensor([-1.0, -0.1])))
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_filtered_velocity_is_privileged_and_resets_independently(self):
        cfg = json.loads((bootstrap.ROOT / "configs/native_bam_mean_velocity.json").read_text())
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True, reward_enabled=True, training=cfg)
        env.step(torch.zeros((2, 14)))
        observation = env.get_observations()
        self.assertEqual(observation["actor"].shape, (2, 61))
        self.assertEqual(observation["critic"].shape, (2, 79))
        preserved = env.mean_velocity_world[1].clone()
        env.reset(torch.tensor([True, False]))
        torch.testing.assert_close(env.mean_velocity_world[1], preserved)
        self.assertEqual(float(env.mean_velocity_world[0].abs().sum()), 0.0)

    def test_offline_collision_audit_detects_injected_overlap(self):
        from audit_bam_contacts import build_audit_model
        from duck_gym.render import apply_body_poses

        model, pairs = build_audit_model(MODEL_DIR)
        env = BamEnv(MODEL_DIR, fp64=True)
        state = env.bodies[0].numpy().copy()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        apply_body_poses(model, data, state)
        guards = np.flatnonzero(model.geom_contype == 2)
        a, b = map(int, guards[1:3])
        self.assertIn((a, b), pairs)
        state[model.geom_bodyid[b], :3] += data.geom_xpos[a] - data.geom_xpos[b] + [0.002, 0, 0]
        apply_body_poses(model, data, state)
        mujoco.mj_collision(model, data)
        self.assertTrue(
            any(
                c.geom1 == a and c.geom2 == b and c.dist < -0.001 for c in data.contact[: data.ncon]
            )
        )

    def test_heading_reward_distinguishes_drift_and_resets_per_environment(self):
        cfg = json.loads((bootstrap.ROOT / "configs/native_bam_heading.json").read_text())
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True, reward_enabled=True, training=cfg)
        env.target_yaw += env.home.new_tensor([0.0, 0.2])
        observation = env.get_observations()
        self.assertEqual(observation["actor"].shape, (2, 61))
        self.assertEqual(observation["critic"].shape, (2, 80))
        score = torch.exp(-env.heading_error().square() / cfg["heading_reward_variance"])
        torch.testing.assert_close(score, torch.exp(env.home.new_tensor([0.0, -1.0])))
        preserved = env.target_yaw[1].clone()
        env.reset(torch.tensor([True, False]))
        torch.testing.assert_close(env.target_yaw[1], preserved)
        self.assertAlmostEqual(float(env.heading_error()[0]), 0)
        _, _, _, info = env.step(torch.zeros((2, 14)))
        self.assertIn("Reward/track_heading", info["log"])

    def test_heading_controller_wraps_error_without_moving_bodies(self):
        env = BamEnv(MODEL_DIR, num_envs=2, fp64=True, heading_hold=True)
        env.reset()
        root = torch.zeros((2, 6), dtype=torch.float64)
        root[:, 5] = torch.tensor([0.2, -0.2])
        env.native.reset(torch.tensor([True, True]), env.home, root)
        env.refresh()
        before = env.bodies.clone()
        env.update_heading_command()
        torch.testing.assert_close(
            env.command[:, 2], torch.tensor([-0.1, 0.1], dtype=torch.float64), atol=1e-8, rtol=1e-8
        )
        torch.testing.assert_close(env.bodies, before)

    def test_support_band_speed_cannot_cancel_opposite_sliding(self):
        from audit_bam_contacts import support_band_rms

        body = np.zeros(13)
        body[3] = 1
        points = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
        body[7] = 0.01
        self.assertAlmostEqual(support_band_rms(body, points), 0.01)
        body[7] = 0
        body[12] = 1
        self.assertAlmostEqual(support_band_rms(body, points), 0.02)


if __name__ == "__main__":
    unittest.main()
