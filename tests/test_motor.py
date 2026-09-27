"""Explicit actuator interface: analytical inertia, friction and batch safety."""

import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
from model_io import export_model, prepare_microduck, rot, inv
from _duck_reference import CpuBatch64


class MotorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "motor.duck")
        model = mujoco.MjModel.from_xml_string(
            """<mujoco>
          <option gravity="0 0 0"/>
          <worldbody><body><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
          <joint name="motor" axis="0 0 1" armature=".002"/>
          </body></worldbody></mujoco>"""
        )
        export_model(model, mujoco.MjData(model), self.path)
        self.native = CpuBatch64(self.path, 3, 0.001, 1000, 1)
        self.force = np.zeros((3, 3))

    def tearDown(self):
        self.tmp.cleanup()

    def test_torque_acceleration_and_dry_friction(self):
        motor = np.zeros((3, 1, 3))
        motor[:, 0, 0] = [0.02, -0.02, 0.02]
        motor[2, 0, 1] = 0.04
        for _ in range(100):
            self.native.step_motor(motor, self.force)
        _, joints, diag = self.native.state()
        expected = 0.02 / 0.012 * 0.1
        np.testing.assert_allclose(joints[:2, 0, 1], [expected, -expected], atol=2e-4)
        self.assertLess(abs(joints[2, 0, 0]), 1e-5)
        self.assertFalse(diag[:, 5].any())

    def test_bad_environment_requires_reset(self):
        motor = np.zeros((3, 1, 3))
        motor[1, 0, 1] = -1
        self.native.step_motor(motor, self.force)
        np.testing.assert_array_equal(self.native.state()[2][:, 5], [0, 1, 0])
        self.native.reset(np.array([False, True, False]), np.zeros((3, 1)), np.zeros((3, 6)))
        self.assertFalse(self.native.state()[2][:, 5].any())

    def test_pd_does_not_keep_previous_motor_torque(self):
        motor = np.zeros((3, 1, 3))
        motor[0, 0, 0] = 0.02
        self.native.step_motor(motor, self.force)
        before = self.native.state()[1][0, 0, 1]
        self.native.step(np.zeros((3, 1)), self.force, 20, 0.0, 1.0)
        after = self.native.state()[1][0, 0, 1]
        self.assertAlmostEqual(before, after, delta=1e-6)

    def test_rnea_bias_against_mujoco_random_states(self):
        model = mujoco.MjModel.from_xml_string(prepare_microduck(ground=False))
        data = mujoco.MjData(model)
        rng = np.random.default_rng(81)
        hinge = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        offset = rot(inv(model.body_iquat[1]), model.body_ipos[1])[None, :]
        for _ in range(30):
            mujoco.mj_resetData(model, data)
            data.qpos[3:7] = rng.normal(size=4)
            data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
            data.qpos[model.jnt_qposadr[hinge]] = rng.uniform(-0.7, 0.7, len(hinge))
            data.qvel[:] = rng.uniform(-5, 5, model.nv)
            export_model(model, data, self.path)
            native = CpuBatch64(self.path, 1, 0.001, 50, 1)
            loads = native.generalized_loads(offset)[0]
            np.testing.assert_allclose(
                loads[:, 0], data.qfrc_bias[model.jnt_dofadr[hinge]], atol=2e-12, rtol=2e-12
            )
            np.testing.assert_array_equal(loads[:, 1:], 0)

    def test_contact_generalized_load_matches_virtual_work(self):
        model = mujoco.MjModel.from_xml_string(
            """<mujoco><worldbody>
          <geom type="plane" size="1 1 .1" friction="0 0 0"/>
          <body pos="0 0 .23"><inertial pos=".1 0 0" mass="1" diaginertia=".002 .002 .002"/>
          <joint name="hinge" axis="0 1 0" armature=".001"/>
          <geom type="sphere" pos=".3 0 0" size=".02" friction="0 0 0"/>
          </body></worldbody></mujoco>"""
        )
        data = mujoco.MjData(model)
        data.qpos[0] = np.pi / 4
        export_model(model, data, self.path)
        native = CpuBatch64(self.path, 1, 0.001, 1000, 1)
        for _ in range(30):
            previous_quat = native.state()[0][0, 1, 3:7].copy()
            native.step_motor(np.zeros((1, 1, 3)), np.zeros((1, 3)))
        body = native.state()[0][0, 1]
        force = native.contact_forces()[0, 1]
        self.assertGreater(force[2], 0.1)
        # A vertical force at this x coordinate does work -x*Fz*dtheta
        # under a positive rotation about the fixed world y hinge.
        # The solver holds the detected sphere feature over this step. Use
        # that feature under the hinge's virtual rotation, rather than a new
        # post-integration sphere/plane contact that was never solved.
        feature = np.array([0.3, 0, 0]) + rot(inv(previous_quat), [0, 0, -0.02])
        point_x = rot(body[3:7], feature)[0]
        load = native.generalized_loads(np.zeros((1, 3)))[0, 0, 1]
        self.assertAlmostEqual(load, -point_x * force[2], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
