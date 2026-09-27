"""Physical invariants of native two-body contacts, independent of walking reward."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
from model_io import export_model, rot, inv
import _duck_reference as native


class BodyContactTests(unittest.TestCase):
    def simulation(self, directory, friction=0, ground=False):
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><option gravity="0 0 0"/><worldbody><body pos="-.021 0 .1"><freejoint/><geom type="sphere" size=".02" mass="1" friction="{friction} 0 0"/></body><body pos=".021 0 .1"><freejoint/><geom type="sphere" size=".02" mass="2" friction="{friction} 0 0"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        data.qvel[:3] = [0.2, 0.05 if friction else 0, 0]
        data.qvel[6:9] = [-0.1, 0, 0]
        path = Path(directory) / "pair.duck"
        export_model(model, data, path, self_contacts=True)
        return path

    def test_equal_opposite_impulse_and_momentum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.simulation(directory)
            for cls, dtype, tol in [
                (native.CpuBatch64, np.float64, 2e-5),
                (native.CpuBatch32, np.float32, 2e-4),
            ]:
                env = cls(str(path), 2, 0.001, 400, 1)
                initial = env.state()[0].copy()
                momentum0 = initial[0, 1, 7:10] + 2 * initial[0, 2, 7:10]
                saw_force = False
                for _ in range(100):
                    env.step(np.zeros((2, 0), dtype=dtype), np.zeros((2, 3), dtype=dtype), 1, 0, 1)
                    b, j, d = env.state()
                    self.assertFalse(d[:, 5].any())
                    forces = env.contact_forces()
                    np.testing.assert_allclose(forces.sum(1), 0, atol=tol)
                    saw_force |= np.linalg.norm(forces) > 1e-3
                    np.testing.assert_allclose(
                        b[0, 1, 7:10] + 2 * b[0, 2, 7:10], momentum0, atol=tol
                    )
                    self.assertGreater(np.linalg.norm(b[0, 1, :3] - b[0, 2, :3]), 0.04 - 2e-5)
                    np.testing.assert_array_equal(b[0], b[1])
                self.assertTrue(saw_force)
                self.assertLess(
                    abs(b[0, 1, 7] - b[0, 2, 7]), 0.002 if dtype == np.float64 else 0.02
                )

    def test_friction_opposes_relative_surface_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.simulation(directory, friction=0.8)
            env = native.CpuBatch64(str(path), 1, 0.001, 400, 1)
            seen = False
            for _ in range(25):
                before = env.state()[0][0]
                env.step(np.zeros((1, 0)), np.zeros((1, 3)), 1, 0, 1)
                b, _, d = env.state()
                f = env.contact_forces()[0]
                if np.linalg.norm(f[1]) > 1e-4:
                    seen = True
                    # Total energy cannot grow in a zero-gravity inelastic collision.
                    energy = lambda b: sum(
                        0.5 * m * np.dot(b[i, 7:10], b[i, 7:10])
                        + 0.5 * (0.4 * m * 0.02**2) * np.dot(b[i, 10:13], b[i, 10:13])
                        for i, m in [(1, 1), (2, 2)]
                    )
                    self.assertLessEqual(energy(b[0]), energy(before) + 2e-5)
                    np.testing.assert_allclose(f[1] + f[2], 0, atol=1e-10)
                self.assertFalse(d[:, 5].any())
            self.assertTrue(seen)

    def test_both_hinge_loads_match_contact_virtual_work(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 0"/><worldbody>'
            '<body pos="-.03 0 .1"><joint axis="0 0 1"/><geom type="sphere" pos=".02 .01 0" size=".02" mass="1" friction="0 0 0"/></body>'
            '<body pos=".03 0 .1"><joint axis="0 0 1"/><geom type="sphere" pos="-.02 -.01 0" size=".02" mass="1" friction="0 0 0"/></body>'
            "</worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hinges.duck"
            export_model(model, data, path, self_contacts=True)
            env = native.CpuBatch64(str(path), 1, 0.001, 1000, 1)
            initial = env.state()[0][0]
            n = initial[1, :3] - initial[2, :3]
            n /= np.linalg.norm(n)
            # Spheres are centered at their COMs; detected surface points in
            # principal frames remain the contact's material points this step.
            local = {
                1: rot(inv(initial[1, 3:7]), -0.02 * n),
                2: rot(inv(initial[2, 3:7]), 0.02 * n),
            }
            env.step_motor(np.zeros((1, 2, 3)), np.zeros((1, 3)))
            state = env.state()[0][0]
            force = env.contact_forces()[0]
            self.assertGreater(np.linalg.norm(force), 0.01)
            np.testing.assert_allclose(force.sum(0), 0, atol=1e-10)
            self.assertFalse(env.contact_forces(ground_only=True).any())
            np.testing.assert_array_equal(env.contact_counts(), [[0, 1]])
            loads = env.generalized_loads(np.zeros((1, 3)))[0, :, 1]
            for i in (1, 2):
                local_anchor = rot(inv(initial[i, 3:7]), data.xanchor[i - 1] - initial[i, :3])
                lever = rot(state[i, 3:7], local[i] - local_anchor)
                expected = np.cross(lever, force[i])[2]
                self.assertAlmostEqual(loads[i - 1], expected, delta=2e-8)

    def test_truncated_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.duck"
            for data in ("", "DUCK_MODEL 2", "DUCK_MODEL 2 3 0 2\n0 0 0 0\n"):
                path.write_text(data)
                with self.assertRaises((ValueError, RuntimeError)):
                    native.CpuBatch64(str(path), 1, 0.001, 200, 1)

    def test_legacy_solver_does_not_silently_ignore_pairs(self):
        import _duck_cpu

        with tempfile.TemporaryDirectory() as directory:
            path = self.simulation(directory)
            # The legacy solver is a separate reference implementation.
            # Its public API rejects v2 pair dynamics rather than pretending parity.
            with self.assertRaisesRegex(ValueError, "shared CpuBatch"):
                _duck_cpu.CpuBatch(str(path), 1, 0.001, 200, 1)


if __name__ == "__main__":
    unittest.main()
