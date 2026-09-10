"""Portable shared-core and VecEnv contract tests; no CUDA required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
from bootstrap import ROOT
import unittest
import tempfile
import numpy as np
import mujoco
from unittest.mock import patch
import torch
from duck_gym.tensor_env import TensorEnv, inverse_rotate


class TensorTests(unittest.TestCase):
    def env(self, **kw):
        return TensorEnv(
            ROOT / "build/models/standing",
            num_envs=3,
            backend="cpu",
            randomize=False,
            iterations=50,
            **kw
        )

    def test_isolation_reset_ownership(self):
        for precision in (False, True):
            e = self.env(fp64=precision)
            initial = e.native.state()[0].clone()
            actions = torch.zeros(3, 14)
            actions[1, 0] = 0.1
            with patch("mujoco.mj_step", side_effect=AssertionError("Foreign physics used")):
                e.step(actions)
            before = e.native.state()[0].clone()
            torch.testing.assert_close(before[0], before[2], rtol=0, atol=0)
            self.assertFalse(torch.equal(before[0], before[1]))
            e.reset(torch.tensor([True, False, False]))
            after = e.native.state()[0]
            torch.testing.assert_close(after[0], initial[0], rtol=0, atol=0)
            torch.testing.assert_close(after[1:], before[1:], rtol=0, atol=0)
            after.fill_(123)
            self.assertFalse(torch.equal(after, e.native.state()[0]))

    def test_timeout_terminal_observation(self):
        e = self.env(task="locomotion", episode_seconds=0.04)
        e.episode_length_buf[:] = torch.tensor([1, 0, 0])
        obs, reward, done, info = e.step(torch.zeros(3, 14))
        self.assertEqual(obs.shape, (3, 58))
        self.assertEqual(reward.shape, (3,))
        self.assertEqual(done.tolist(), [True, False, False])
        self.assertEqual(info["time_outs"].tolist(), [True, False, False])
        self.assertEqual(e.episode_length_buf.tolist(), [0, 1, 1])
        self.assertFalse(torch.equal(obs[0], info["terminal_observation"][0]))

    @unittest.skipUnless(
        (ROOT / "build/models/standing/gait.npz").exists(), "Generate gait reference first"
    )
    def test_gait_reference_does_not_move_physics(self):
        e = self.env(task="locomotion", gait=True)
        before = e.native.state()[0].clone()
        e.episode_length_buf.fill_(100)
        target = e.reference()
        self.assertFalse(torch.equal(target, e.home.expand_as(target)))
        torch.testing.assert_close(e.native.state()[0], before, rtol=0, atol=0)
        e.episode_length_buf += 40
        torch.testing.assert_close(e.reference(), target, rtol=1e-4, atol=1e-5)

    def test_parallel_cpu_matches_serial(self):
        a = self.env(cpu_threads=1)
        b = self.env(cpu_threads=3)
        actions = torch.linspace(-0.1, 0.1, 42).reshape(3, 14)
        for _ in range(4):
            a.step(actions)
            b.step(actions)
        for x, y in zip(a.native.state(), b.native.state()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_physics_invariant_to_batch_size(self):
        batch = self.env(auto_reset=False)
        single = TensorEnv(
            ROOT / "build/models/standing",
            num_envs=1,
            backend="cpu",
            randomize=False,
            auto_reset=False,
            iterations=50,
        )
        for step in range(20):
            action = (0.1 * torch.sin(torch.arange(14) + step * 0.1))[None]
            single.step(action)
            batch.step(action.expand(3, -1))
        for a, b in zip(single.native.state(), batch.native.state()):
            torch.testing.assert_close(a[0], b[1], rtol=0, atol=0)

    @unittest.skipUnless(
        (ROOT / "build/models/standing/gait.npz").exists(), "Generate gait reference first"
    )
    def test_full_sole_projection(self):
        e = self.env(gait=True, foot_clearance=True)
        poses = e.native.state()[0][:, e.foot_bodies]
        q = poses[:, :, 3:7]
        points = e.foot_points
        full = inverse_rotate(
            (q * torch.tensor([1, -1, -1, -1]))[:, :, None, :].expand(-1, -1, points.shape[1], -1),
            points[None].expand(3, -1, -1, -1),
        )
        projected = torch.einsum(
            "fvi,efi->efv", points, inverse_rotate(q, torch.tensor([0.0, 0, 1.0]).expand(3, 2, -1))
        )
        torch.testing.assert_close(full[:, :, :, 2], projected, rtol=1e-5, atol=1e-7)
        obs, reward, _, _ = e.step(torch.zeros(3, 14))
        self.assertTrue(torch.isfinite(obs).all() and torch.isfinite(reward).all())

    def test_contact_force_has_newton_units(self):
        from validate_cpu import cases
        from model_io import export_model
        from _duck_reference import CpuBatch64

        case = next(c for c in cases() if c[0] == "box_rest")
        model = mujoco.MjModel.from_xml_string(case[1])
        data = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "box.duck"
            export_model(model, data, path)
            sim = CpuBatch64(str(path), 1, 0.001, 200)
            sim.step(np.zeros((1, 0)), np.zeros((1, 3)), 1000, 0.0, 1.0)
            force = sim.contact_forces().sum(axis=1)[0]
            np.testing.assert_allclose(
                force, [0, 0, float(model.body_mass.sum()) * 9.81], atol=0.001, rtol=0
            )
            sim.reset(np.ones(1, dtype=bool), np.zeros((1, 0)), np.zeros((1, 6)))
            self.assertEqual(float(np.abs(sim.contact_forces()).sum()), 0)

    def test_filtered_velocity_observation_and_reset(self):
        e = self.env(task="locomotion", velocity_filter_seconds=0.8, auto_reset=False)
        obs, _, _, _ = e.step(torch.zeros(3, 14))
        self.assertEqual(obs.shape, (3, 60))
        expected = e.native.state()[0][:, 1, 7:9] * e.velocity_alpha
        torch.testing.assert_close(obs[:, -2:], expected, rtol=1e-5, atol=1e-8)
        old = e.filtered_velocity.clone()
        e.reset(torch.tensor([True, False, False]))
        self.assertEqual(float(e.filtered_velocity[0].abs().sum()), 0)
        torch.testing.assert_close(e.filtered_velocity[1:], old[1:], rtol=0, atol=0)

    def test_fixed_direction_survives_resets(self):
        e = self.env(task="locomotion", command_direction="backward", episode_seconds=0.02)
        expected = torch.tensor([[-0.05, 0.0]]).expand(3, -1)
        torch.testing.assert_close(e.commands, expected)
        e.step(torch.zeros(3, 14))
        torch.testing.assert_close(e.commands, expected)

    def test_numeric_failure_isolated_and_resettable(self):
        e = self.env(auto_reset=False)
        target = e.home.repeat(3, 1)
        target[1, 0] = float("nan")
        e.native.step(target, e.forces, 1, 0.55, 0.96)
        self.assertEqual(e.native.state()[2][:, 5].tolist(), [0, 1, 0])
        e.reset(torch.tensor([False, True, False]))
        self.assertEqual(e.native.state()[2][:, 5].tolist(), [0, 0, 0])

    def test_no_autoreset_does_not_hide_failure(self):
        e = self.env(auto_reset=False)
        q = e.home.repeat(3, 1)
        root = torch.zeros(3, 6)
        root[0, 3] = 0.7
        e.native.reset(torch.ones(3, dtype=torch.bool), q, root)
        obs, reward, done, info = e.step(torch.zeros(3, 14))
        self.assertTrue(bool(info["failed"][0]))
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(info["time_outs"][0]))
        self.assertEqual(e.episode_length_buf[0], 1)
        self.assertTrue(torch.isfinite(obs).all())


if __name__ == "__main__":
    unittest.main()
