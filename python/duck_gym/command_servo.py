"""Optional high-level velocity-command feedback; never edits physical state."""

import math
import torch


class VelocityCommandServo:
    def __init__(self, reference, cfg):
        self.cfg = cfg
        self.velocity = torch.zeros_like(reference)
        self.integral = torch.zeros_like(reference[:, :2])
        self.yaw_target = torch.zeros_like(reference[:, 0])
        self.filtered_target = torch.zeros_like(reference[:, :2])

    def reset(self, yaw):
        self.velocity.zero_()
        self.integral.zero_()
        self.yaw_target.copy_(yaw)
        self.filtered_target.zero_()

    def observe(self, velocity, dt):
        self.velocity.lerp_(velocity, -math.expm1(-dt / self.cfg["velocity_tau"]))

    def command(self, target_world, yaw, yaw_rate, dt):
        desired = target_world[:, :2]
        acceleration = self.cfg.get("command_acceleration_limit")
        if acceleration is not None:
            if acceleration <= 0:
                raise ValueError("Command acceleration limit must be positive")
            delta = desired - self.filtered_target
            scale = (acceleration * dt / delta.norm(dim=-1).clamp_min(1e-12)).clamp(max=1)
            self.filtered_target.add_(delta * scale[:, None])
            desired = self.filtered_target
        error = desired - self.velocity[:, :2]
        self.integral.add_(error * dt).clamp_(
            -self.cfg["integral_limit"], self.cfg["integral_limit"]
        )
        demand = desired + self.cfg["kp"] * error + self.cfg["ki"] * self.integral
        c, s = yaw.cos(), yaw.sin()
        result = torch.zeros_like(target_world)
        result[:, 0] = c * demand[:, 0] + s * demand[:, 1]
        result[:, 1] = -s * demand[:, 0] + c * demand[:, 1]
        limits = result.new_tensor(self.cfg["command_limits"])
        clipped = result[:, :2].clamp(-limits[:2], limits[:2])
        # Back-calculation prevents accumulating an unattainable request.
        excess = result[:, :2] - clipped
        excess_world = torch.stack(
            (c * excess[:, 0] - s * excess[:, 1], s * excess[:, 0] + c * excess[:, 1]), dim=-1
        )
        if self.cfg["ki"] > 0:
            self.integral.sub_(
                excess_world * min(1.0, dt / self.cfg["antiwindup_tau"]) / self.cfg["ki"]
            )
        result[:, :2] = clipped
        angle = self.yaw_target - yaw
        result[:, 2] = (
            self.cfg["heading_kp"] * torch.atan2(angle.sin(), angle.cos())
            - self.cfg["heading_kd"] * yaw_rate
        ).clamp(-limits[2], limits[2])
        return result
