"""Acceptance must reject short runs, transient drift, and resets/failures."""

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_cuda import locomotion_metrics


class EvaluationTests(unittest.TestCase):
    def trajectory(self, seconds=32):
        time = np.arange(round(seconds / 0.02) + 1) * 0.02
        return np.column_stack((0.05 * time, np.zeros_like(time)))

    def measure(self, positions, **kw):
        yaw = kw.pop("yaw", np.zeros(len(positions) - 1))
        return locomotion_metrics(positions, yaw, np.array([0.05, 0]), 0.02, **kw)

    def test_full_duration_required(self):
        self.assertTrue(self.measure(self.trajectory())["passed"])
        short = self.measure(self.trajectory(7))
        self.assertTrue(short["criteria_met_for_measured_duration"])
        self.assertFalse(short["passed"])
        self.assertFalse(self.measure(np.zeros_like(self.trajectory()))["passed"])

    def test_sliding_windows_reject_cancelling_drift(self):
        velocity = np.full((1600, 2), [0.05, 0.0])
        velocity[100:125, 0] += 0.08
        velocity[125:150, 0] -= 0.08
        positions = np.vstack((np.zeros(2), velocity.cumsum(0) * 0.02))
        report = self.measure(positions)
        self.assertLess(report["mean_error_mps"], 1e-8)
        self.assertGreater(report["max_1s_window_error_mps"], 0.03)
        self.assertFalse(report["passed"])

    def test_failure_heading_and_incomplete_warmup(self):
        positions = self.trajectory()
        self.assertFalse(self.measure(positions, failed=True)["passed"])
        yaw = np.zeros(len(positions) - 1)
        yaw[10] = 0.31
        self.assertFalse(self.measure(positions, yaw=yaw)["passed"])
        short = self.measure(self.trajectory(0.5), failed=True)
        self.assertEqual(short["measured_seconds"], 0)
        self.assertFalse(short["passed"])


if __name__ == "__main__":
    unittest.main()
