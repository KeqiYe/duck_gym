"""Boundary tests: native isolation, FK convention, reset masks, visual transforms."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
from bootstrap import ROOT
import json
import unittest
import tempfile
import numpy as np
import torch
import mujoco
from _duck_cpu import CpuBatch
from duck_gym import StandingEnv
from duck_gym.env import quat_mul
from duck_gym.render import apply_body_poses

MODEL = ROOT / "build/models/standing"


class BoundaryTests(unittest.TestCase):
    def test_fk_and_render(self):
        e = StandingEnv(MODEL, num_envs=2, randomize=False)
        m = mujoco.MjModel.from_xml_path(str(MODEL / "visual.xml"))
        d = mujoco.MjData(m)
        d.qpos[:] = e.meta["qpos"]
        angles = e.home + np.linspace(-0.03, 0.03, e.num_actions)
        for name, q in zip(e.meta["joint_names"], angles):
            d.qpos[m.joint(name).qposadr[0]] = q
        mujoco.mj_forward(m, d)
        e.native.reset([0], angles[None, :], np.zeros((1, 6)))
        state = e.native.body_state()[0]
        np.testing.assert_allclose(e.native.joint_state()[0, :, 0], angles, atol=1e-12)
        np.testing.assert_allclose(state[:, :3], d.xipos, atol=1e-12)
        reference = d.geom_xpos.copy()
        mat = d.geom_xmat.copy()
        apply_body_poses(m, d, state)
        np.testing.assert_allclose(d.geom_xpos, reference, atol=1e-12)
        np.testing.assert_allclose(d.geom_xmat, mat, atol=1e-12)
        # Move one link independently: rendering must retain maximal-coordinate error.
        body = m.nbody - 1
        state[body, 0] += 0.012
        apply_body_poses(m, d, state)
        mask = m.geom_bodyid == body
        np.testing.assert_allclose(d.geom_xpos[mask], reference[mask] + [0.012, 0, 0], atol=1e-12)
        np.testing.assert_allclose(d.geom_xpos[~mask], reference[~mask], atol=1e-12)

    def test_isolation_and_parallel(self):
        a = StandingEnv(MODEL, num_envs=2, threads=1, randomize=False)
        b = StandingEnv(MODEL, num_envs=2, threads=2, randomize=False)
        snapshot = a.native.body_state().copy()
        action = torch.zeros(2, a.num_actions)
        action[1, 0] = 0.5
        a.step(action)
        b.step(action)
        np.testing.assert_array_equal(a.native.body_state(), b.native.body_state())
        other = a.native.body_state()[1].copy()
        a.reset([0])
        np.testing.assert_array_equal(a.native.body_state()[0], snapshot[0])
        np.testing.assert_array_equal(a.native.body_state()[1], other)
        copy = a.native.body_state()
        copy[:] = 99
        self.assertFalse((a.native.body_state() == 99).all())
        with self.assertRaises(ValueError):
            a.step(torch.full_like(action, float("nan")))
        with self.assertRaises(ValueError):
            a.native.step(np.zeros((1, 1)), 1, 0.55, 0.96, np.zeros((2, 3)))

    def test_force_units_and_validation(self):
        # 1 kg free body: +9.81 N cancels gravity exactly.
        model = "DUCK_MODEL 1 2 0 0\n0 0 -9.81 0\nworld 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0\nbase 1 .1 .1 .1 0 0 1 1 0 0 0 0 0 0 0 0 0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".duck") as f:
            f.write(model)
            f.flush()
            batch = CpuBatch(f.name, 2, 0.001, 200, 2)
            batch.step(np.zeros((2, 0)), 100, 0, 1, np.array([[0, 0, 9.81], [1, 0, 9.81]]))
            b = batch.body_state()
            np.testing.assert_allclose(b[:, 1, 2], 1, atol=1e-12)
            self.assertAlmostEqual(b[1, 1, 0], 0.00505, places=10)
            self.assertAlmostEqual(b[1, 1, 7], 0.1, places=10)
            with self.assertRaises(ValueError):
                batch.step(np.zeros((2, 0)), 1, 0, 1, np.full((2, 3), np.nan))

    def test_timeout_and_native_only(self):
        e = StandingEnv(MODEL, num_envs=2, randomize=False, episode_seconds=0.04)
        original = mujoco.mj_step
        mujoco.mj_step = lambda *a, **k: self.fail("MuJoCo advanced training physics")
        try:
            e.episode_length_buf[0] = 1
            obs, reward, done, info = e.step(torch.zeros((2, e.num_actions)))
            self.assertEqual(obs.shape, (2, 52))
            self.assertEqual(obs.dtype, torch.float32)
            self.assertEqual(done.tolist(), [True, False])
            self.assertEqual(info["time_outs"].tolist(), [True, False])
            self.assertEqual(e.episode_length_buf.tolist(), [0, 1])
            self.assertEqual(info["terminal_observation"].shape, (1, 52))
        finally:
            mujoco.mj_step = original


if __name__ == "__main__":
    unittest.main()
