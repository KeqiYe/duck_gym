"""Shared native MPR versus independent MuJoCo narrow phase, in meter units."""

import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
import _duck_reference as native


def vertices(extents):
    return np.asarray(list(itertools.product([-1, 1], repeat=3))) * extents


class ConvexTests(unittest.TestCase):
    def test_axis_aligned_depth_and_margin(self):
        box = vertices([0.01] * 3)
        poses = np.array([[0, 0, 0, 1, 0, 0, 0], [0.025, 0, 0, 1, 0, 0, 0]])
        for query in [native.convex_query32, native.convex_query64]:
            self.assertEqual(query(box, box, poses)["status"], 0)
            poses[1, 0] = 0.019
            c = query(box, box, poses)
            self.assertEqual(c["status"], 1)
            self.assertAlmostEqual(c["depth"], 0.001, delta=2e-6)
            np.testing.assert_allclose(c["normal"], [-1, 0, 0], atol=1e-6)
            poses[1, 0] = 0.021
            c = query(box, box, poses, 0.003)
            self.assertEqual(c["status"], 1)
            self.assertAlmostEqual(c["depth"], 0.002, delta=2e-6)
            poses[1, 0] = 0.025

    def test_random_rotated_boxes_and_separating_translation(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="box" size=".015 .025 .04"/></body><body><freejoint/><geom type="box" size=".03 .02 .015"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        va, vb = vertices([0.015, 0.025, 0.04]), vertices([0.03, 0.02, 0.015])
        rng = np.random.default_rng(87)
        hits = 0
        for _ in range(1000):
            poses = np.zeros((2, 7))
            for i in range(2):
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                poses[i] = np.r_[rng.uniform(-0.045, 0.045, 3), q]
            data.qpos[:] = poses.ravel()
            mujoco.mj_forward(model, data)
            expected = any(c.dist < -1e-8 for c in data.contact[: data.ncon])
            hits += expected
            for query in [native.convex_query32, native.convex_query64]:
                c = query(va, vb, poses)
                self.assertNotEqual(c["status"], -1)
                self.assertEqual(c["status"] == 1, expected, (poses, c))
                if c["status"]:
                    self.assertGreater(c["depth"], 0)
                    self.assertTrue(np.isfinite(c["position"]).all())
                    self.assertAlmostEqual(np.linalg.norm(c["normal"]), 1, delta=1e-5)
                    # MPR penetration direction need not be the minimum SAT axis,
                    # but moving by its depth must separate the original boxes.
                    shifted = poses.copy()
                    shifted[0, :3] += np.asarray(c["normal"]) * (c["depth"] + 5e-6)
                    data.qpos[:] = shifted.ravel()
                    mujoco.mj_forward(model, data)
                    self.assertFalse(any(x.dist < -5e-6 for x in data.contact[: data.ncon]), c)
        self.assertGreater(hits, 100)
        self.assertLess(hits, 900)

    def test_coincident_centers_and_invalid_inputs(self):
        box = vertices([0.01] * 3)
        poses = np.array([[0, 0, 0, 1, 0, 0, 0]] * 2, dtype=float)
        for query in [native.convex_query32, native.convex_query64]:
            c = query(box, box, poses)
            self.assertEqual(c["status"], 1)
            self.assertAlmostEqual(c["depth"], 0.02, delta=2e-6)
            with self.assertRaises(ValueError):
                query(box, box, poses, -1)
            with self.assertRaises(ValueError):
                query(box * np.nan, box, poses)


if __name__ == "__main__":
    unittest.main()
