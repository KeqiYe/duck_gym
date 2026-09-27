"""Frame, saturation and reset contracts for the optional command controller."""

import json
import math
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from duck_gym.command_servo import VelocityCommandServo


class CommandServoTests(unittest.TestCase):
    def make(self, **changes):
        cfg = json.loads((ROOT / "configs/velocity_command_servo.json").read_text())
        cfg.update(changes)
        return VelocityCommandServo(torch.zeros(2, 3, dtype=torch.float64), cfg)

    def test_world_target_rotates_to_body_and_wraps_yaw(self):
        servo = self.make(kp=0.0, ki=0.0)
        yaw = torch.tensor([math.pi / 2, -math.pi + 0.1], dtype=torch.float64)
        servo.reset(torch.tensor([math.pi / 2, math.pi - 0.1], dtype=torch.float64))
        target = torch.tensor([[0.05, 0, 0], [0, 0, 0]], dtype=torch.float64)
        result = servo.command(target, yaw, torch.zeros_like(yaw), 0.02)
        torch.testing.assert_close(result[0], result.new_tensor([0, -0.05, 0]))
        self.assertAlmostEqual(float(result[1, 2]), -0.8)
        torch.testing.assert_close(target[0], target.new_tensor([0.05, 0, 0]))

    def test_saturation_releases_and_reset_clears_history(self):
        servo = self.make()
        yaw = torch.zeros(2, dtype=torch.float64)
        target = torch.tensor([[10, -10, 0], [-10, 10, 0]], dtype=torch.float64)
        for _ in range(100):
            result = servo.command(target, yaw, yaw, 0.02)
            self.assertTrue(bool((result.abs() <= result.new_tensor([0.4, 0.3, 1])).all()))
        servo.observe(torch.ones_like(target), 0.5)
        self.assertGreater(float(servo.velocity.abs().sum()), 0)
        servo.reset(yaw)
        result = servo.command(torch.zeros_like(target), yaw, yaw, 0.02)
        torch.testing.assert_close(result, torch.zeros_like(result))
        torch.testing.assert_close(servo.velocity, torch.zeros_like(servo.velocity))

    def test_velocity_filter_has_time_step_invariant_constant_response(self):
        coarse, fine = self.make(), self.make()
        velocity = torch.tensor([[0.05, -0.02, 0], [-0.03, 0.01, 0]], dtype=torch.float64)
        coarse.observe(velocity, 0.1)
        for _ in range(5):
            fine.observe(velocity, 0.02)
        torch.testing.assert_close(coarse.velocity, fine.velocity)

    def test_command_ramp_bounds_vector_acceleration_and_resets(self):
        servo = self.make(kp=0.0, ki=0.0, command_acceleration_limit=0.05)
        yaw = torch.zeros(2, dtype=torch.float64)
        goal = torch.tensor([[0.05, 0, 0], [0.03, 0.04, 0]], dtype=torch.float64)
        previous = torch.zeros_like(goal)
        for _ in range(50):
            current = servo.command(goal, yaw, yaw, 0.02)
            self.assertTrue(bool(((current - previous).norm(dim=-1) <= 0.001 + 1e-12).all()))
            previous = current
        torch.testing.assert_close(current, goal)
        servo.reset(yaw)
        torch.testing.assert_close(servo.filtered_target, torch.zeros_like(servo.filtered_target))


if __name__ == "__main__":
    unittest.main()
