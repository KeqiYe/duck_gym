"""Independent geometry and false-positive flight checks for BAM aerial work."""

import os
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
from duck_gym.bam_aerial import BamAerialEnv
from duck_gym.render import apply_body_poses

MODEL = Path(
    os.environ.get(
        "DUCK_OFFICIAL_MODEL",
        bootstrap.ROOT / "runs/gpu-20260911-132540-398052/official-model",
    )
)
GEOMETRY = bootstrap.ROOT / "runs/aerial-preparation-20260913/geometry.npz"


@unittest.skipUnless(
    (MODEL / "config.json").exists() and GEOMETRY.exists(),
    "Prepared BAM geometry required",
)
class AerialTests(unittest.TestCase):
    def env(self, n=1):
        return BamAerialEnv(MODEL, GEOMETRY, num_envs=n, fp64=True)

    def test_support_matches_full_mesh_on_random_body_poses(self):
        e = self.env()
        m = mujoco.MjModel.from_xml_path(str(MODEL / "ground-only.xml"))
        d = mujoco.MjData(m)
        rng = np.random.default_rng(76)
        b = e.bodies[0].numpy().copy()
        for _ in range(3):
            b[:, 3:7] = rng.normal(size=(len(b), 4))
            b[:, 3:7] /= np.linalg.norm(b[:, 3:7], axis=1)[:, None]
            b[:, 2] = rng.uniform(0.1, 0.3, len(b))
            e.bodies = torch.tensor(b[None])
            actual = e.clearances()[0].numpy()
            mujoco.mj_forward(m, d)
            apply_body_poses(m, d, b)
            for slot in e.geom_nonfeet.tolist():
                bid = int(e.geom_bodies[slot])
                expected = float("inf")
                for g in range(m.ngeom):
                    if (
                        m.geom_bodyid[g] != bid
                        or m.geom_group[g] != 2
                        or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                    ):
                        continue
                    mesh = m.geom_dataid[g]
                    v = m.mesh_vert[
                        m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh]
                        + m.mesh_vertnum[mesh]
                    ]
                    expected = min(
                        expected,
                        float(
                            (
                                v @ d.geom_xmat[g].reshape(3, 3)[2] + d.geom_xpos[g, 2]
                            ).min()
                        ),
                    )
                self.assertAlmostEqual(actual[slot], expected, delta=1e-7)

    def test_falling_clear_feet_are_not_upward_takeoff(self):
        e = self.env(2)
        # Synthetic diagnostic states only, never used to train or qualify a skill.
        e.bodies[:, :, 2] += 0.2
        e.bodies[:, :, 9] = -1
        e.initial_energy = e.mechanical_energy().clone()
        e.diagnostics.zero_()
        e.contact.zero_()
        e.observe_substep(e)
        self.assertFalse(bool(e.took_off.any()))
        self.assertFalse(bool(e.flight_time.any()))
        e.bodies[:, :, 9] = 1
        e.airborne.zero_()
        e.observe_substep(e)
        self.assertTrue(bool(e.took_off.all()))
        saved = e.flight_time[1].clone()
        e.reset(torch.tensor([True, False]))
        self.assertFalse(bool(e.took_off[0]))
        self.assertTrue(bool(e.took_off[1]))
        torch.testing.assert_close(e.flight_time[1], saved)

    def test_nonfoot_ground_crossing_is_failure(self):
        e = self.env()
        e.bodies[:, :, 2] -= 0.2
        e.observe_substep(e)
        self.assertTrue(bool(e.invalid[0]))
        self.assertFalse(bool(e.took_off[0]))

    def test_proximity_candidates_do_not_imply_ground_support(self):
        e = self.env()
        for _ in range(10):
            e.step(torch.zeros(1, 14))
        self.assertGreater(float(e.native.contact_counts()[0, 0]), 0)
        # Diagnostic-only synthetic state: retain proximity candidates but
        # move both full soles 1 mm clear with upward velocity and zero force.
        e.bodies[:, :, 2] += 0.001 - float(e.clearances()[:, e.geom_feet].min())
        e.bodies[:, :, 9] = 0.1
        e.contact.zero_()
        e.diagnostics.zero_()
        e.initial_energy = e.mechanical_energy().clone()
        e.observe_substep(e)
        self.assertTrue(bool(e.took_off[0]))

    def test_ground_counts_and_energy_rejection(self):
        e = self.env()
        e.step(torch.zeros(1, 14))
        torch.testing.assert_close(
            e.native.contact_counts().sum(-1), e.diagnostics[:, 2]
        )
        e.bodies[:, :, 9] = 20
        e.observe_substep(e)
        self.assertTrue(bool(e.numerical_invalid[0]))


if __name__ == "__main__":
    unittest.main()
