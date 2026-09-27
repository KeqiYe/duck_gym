"""Motion validity: actual support geometry, phase state, airborne rotation."""

import sys
from pathlib import Path
import unittest
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bootstrap
from duck_gym.motion_env import MotionEnv
from duck_gym.motion import MotionGeometry, pitch_increment, rotate, contact_free_interval
from audit_aerial import flight_segments


class MotionTests(unittest.TestCase):
    def test_descending_contact_loss_is_not_rewarded_as_upward_launch(self):
        for vertical_velocity, expected in [(-0.5, False), (0.5, True)]:
            env = self.env("jump")
            self.assertGreaterEqual(env.cfg["aerial_detector_version"], 3)
            b, j, d = env.native.state()
            b[:, 1:, 2] += 0.1
            b[:, 1:, 9] = vertical_velocity
            d[:] = 0
            # Synthetic sensor snapshots test the reward classifier only;
            # this is deliberately not a physical takeoff demonstration.
            env.native.state = lambda: (b.clone(), j.clone(), d.clone())
            env.native.contact_forces = lambda: torch.zeros(b.shape[:2] + (3,))
            env.native.step = lambda *args: None
            _, _, _, info = env.step(torch.zeros(3, 14))
            self.assertEqual(env.took_off.tolist(), [expected] * 3)
            self.assertEqual(bool(info["log"]["reward/launch"] > 0), expected)

    def test_separate_hops_cannot_be_counted_as_one_flip(self):
        result = flight_segments(
            [True, False, True, True, False, True, True], [9, 0, 1, 2, 4, 2, 1], 0.02
        )
        self.assertEqual(len(result), 2)
        self.assertEqual([x["signed_pitch_rotation_rad"] for x in result], [3.0, 3.0])
        self.assertEqual([x["duration_seconds"] for x in result], [0.04, 0.04])
        self.assertTrue(result[0]["contact_after_observed"])
        self.assertFalse(result[1]["contact_after_observed"])

    def test_zero_last_force_is_not_proof_of_flight(self):
        diag = torch.zeros(3, 6)
        force = torch.zeros(3, 16, 3)
        clearance = torch.ones(3) * 0.01
        diag[0, 2] = 1
        force[1, 1, 2] = 1
        self.assertEqual(
            contact_free_interval(diag, force, clearance).tolist(), [False, False, True]
        )

    def env(self, skill="walking"):
        return MotionEnv(
            ROOT / "build/models/motion",
            skill=skill,
            num_envs=3,
            backend="cpu",
            cpu_threads=3,
            randomize=False,
            iterations=50,
            auto_reset=False,
        )

    def test_rotation_unwrap(self):
        angles = torch.linspace(0, 2 * torch.pi, 101, dtype=torch.float64)
        q = torch.stack([torch.cos(angles / 2), angles * 0, torch.sin(angles / 2), angles * 0], -1)
        delta = pitch_increment(q[:-1], q[1:])
        self.assertAlmostEqual(float(delta.sum()), 2 * np.pi, places=10)
        self.assertAlmostEqual(float(pitch_increment(q[1:], q[:-1]).sum()), -2 * np.pi, places=10)
        torch.testing.assert_close(delta, pitch_increment(q[:-1], -q[1:]))

    def test_support_point_velocity(self):
        env = self.env()
        b = env.native.state()[0]
        b[..., 7:13] = 0
        b[:, env.geometry.feet, 10] = 2
        height, velocity, point = env.geometry.support(b, feet_only=True)
        expected = torch.linalg.cross(
            b[:, env.geometry.feet, 10:13], point - b[:, env.geometry.feet, :3]
        )
        torch.testing.assert_close(velocity, expected)
        self.assertGreater(float(velocity.abs().sum()), 0)

    def test_spinning_foot_slip_cannot_cancel_at_centroid(self):
        env = self.env()
        geom = MotionGeometry(ROOT / "build/models/motion/motion.npz", "cpu", torch.float64)
        b = env.native.state()[0].double()
        b[..., 7:13] = 0
        b[:, geom.feet, 12] = 2
        _, mean, _ = geom.support(b, feet_only=True)
        b[:, geom.feet, 7:10] = -mean
        _, mean, _, rms = geom.support(b, feet_only=True, pointwise_rms=True)
        self.assertLess(float(mean.abs().max()), 1e-12)
        self.assertGreater(float(rms.min()), 0.001)

    def test_support_hull_preserves_original_foot_extrema(self):
        geom = MotionGeometry(ROOT / "build/models/motion/motion.npz", "cpu", torch.float64)
        full = torch.tensor(
            np.load(ROOT / "build/models/standing/gait.npz")["foot_collision_points"],
            dtype=torch.float64,
        )
        directions = torch.randn(
            64, 2, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64
        )
        old = torch.einsum("fvi,efi->efv", full, directions).amin(-1)
        new = torch.einsum("fvi,efi->efv", geom.points[geom.foot_slots], directions).amin(-1)
        torch.testing.assert_close(old, new, rtol=0, atol=1e-10)

    def test_bias_state_is_observed_and_reset_per_env(self):
        cfg = json.loads((ROOT / "configs/motion_training.json").read_text())
        cfg["walking"]["head_bias_seconds"] = 1.0
        env = MotionEnv(
            ROOT / "build/models/motion",
            motion_config=cfg,
            num_envs=3,
            backend="cpu",
            randomize=False,
            auto_reset=False,
            iterations=50,
        )
        obs, _, _, _ = env.step(torch.zeros(3, 14))
        torch.testing.assert_close(obs[:, -4:], env.head_bias.float())
        before = env.head_bias.clone()
        env.reset(torch.tensor([True, False, False]))
        self.assertEqual(float(env.head_bias[0].abs().sum()), 0)
        torch.testing.assert_close(before[1:], env.head_bias[1:])

    def test_no_motion_rewards_from_nominal_reset(self):
        env = self.env("flip")
        obs, _ = env.get_observations()
        self.assertTrue(torch.isfinite(obs).all())
        self.assertEqual(float(env.air_rotation.abs().sum()), 0)
        env.step(torch.zeros(3, 14))
        self.assertFalse(env.took_off.any())
        self.assertFalse(env.landed.any())
        self.assertEqual(float(env.air_rotation.abs().sum()), 0)
        self.assertEqual(float(env.forces.abs().sum()), 0)

    def test_mask_reset_and_batch_independence(self):
        env = self.env()
        for _ in range(3):
            env.step(torch.zeros(3, 14))
        before = env.native.state()[0].clone()
        env.air_time[:] = 0.5
        env.reset(torch.tensor([True, False, False]))
        torch.testing.assert_close(before[1:], env.native.state()[0][1:], rtol=0, atol=0)
        self.assertEqual(float(env.air_time[0].sum()), 0)
        self.assertEqual(float(env.air_time[1].sum()), 1)
        self.assertTrue(torch.isfinite(env.get_observations()[0]).all())


if __name__ == "__main__":
    unittest.main()
