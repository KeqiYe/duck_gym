"""Flat/neutral-command reward math from pinned mjlab 1.3.0 / MicroDuck.

Derived from Apache-2.0 upstream sources named in official_baseline.json.
Copyright 2026 Pollen Robotics (MicroDuck task). Licenses are retained in
third_party/licenses/{MicroDuck,mjlab}-Apache-2.0.txt. Modified here to
separate supplied features from the physics backend and permit explicit
low-speed experimental variances.
All physical state is supplied by the caller; this module never advances it.
Validation against the actual upstream reward implementations is required
before using this narrower task for training.
"""

import torch


class WalkingReward:
    def __init__(self, home, joint_names, soft_limits, linear_variance=0.1, vertical_variance=None):
        vertical_variance = linear_variance if vertical_variance is None else vertical_variance
        if linear_variance <= 0 or vertical_variance <= 0:
            raise ValueError("Velocity variances must be positive")
        self.linear_variance = linear_variance
        self.vertical_variance = vertical_variance
        self.home = home
        self.soft_limits = soft_limits
        self.head = [i for i, n in enumerate(joint_names) if "head" in n or "neck" in n]
        self.legs = [i for i in range(len(joint_names)) if i not in self.head]
        stand, walk = [], []
        for i in self.legs:
            name = joint_names[i]
            key = next(
                k for k in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle") if k in name
            )
            stand.append(
                dict(hip_yaw=0.1, hip_roll=0.05, hip_pitch=0.15, knee=0.15, ankle=0.1)[key]
            )
            walk.append(dict(hip_yaw=0.3, hip_roll=0.05, hip_pitch=0.4, knee=0.4, ankle=0.25)[key])
        self.stand = home.new_tensor(stand)
        self.walk = home.new_tensor(walk)
        self.head_ema = torch.zeros_like(home[:, self.head])
        self.peak_height = home.new_zeros((len(home), 2))

    def reset(self, mask):
        self.head_ema[mask] = 0
        self.peak_height[mask] = 0

    def terms(self, f, dt):
        command = f["command"]
        speed = command[:, :2].norm(dim=-1) + command[:, 2].abs()
        active = (speed > 0.01).to(self.home.dtype)
        std = torch.where((speed < 0.01)[:, None], self.stand, self.walk)
        pose_error = f["joint_pos"] - self.home
        height = f["foot_height"]
        foot_speed2 = f["foot_vel"][:, :, :2].square().sum(-1)
        first = (f["contact_time"] > 0) & (f["contact_time"] < dt + 1e-8)
        self.peak_height = torch.where(
            ~f["contact"], torch.maximum(self.peak_height, height), self.peak_height
        )
        swing_cost = ((self.peak_height / 0.02 - 1).square() * first).sum(-1) * active
        self.peak_height = torch.where(first, 0.0, self.peak_height)
        head_error = pose_error[:, self.head] - f["head_command"]
        self.head_ema[f["episode_steps"] <= 1] = 0
        alpha = min(1.0, dt / 1.0)
        self.head_ema = (1 - alpha) * self.head_ema + alpha * head_error
        return {
            "track_linear_velocity": torch.exp(
                -(f["lin_vel"][:, :2] - command[:, :2]).square().sum(-1) / self.linear_variance
                - f["lin_vel"][:, 2].square() / self.vertical_variance
            ),
            "track_angular_velocity": torch.exp(
                -(
                    (f["ang_vel"][:, 2] - command[:, 2]).square()
                    + f["ang_vel"][:, :2].square().sum(-1)
                )
                / 0.5
            ),
            "upright": torch.exp(-f["gravity"][:, :2].square().sum(-1) / 0.05),
            "pose": torch.exp(-(pose_error[:, self.legs] / std).square().mean(-1)),
            "body_ang_vel": f["world_ang_vel"][:, :2].square().sum(-1),
            "angular_momentum": f["angular_momentum"].square().sum(-1),
            "dof_pos_limits": (
                (self.soft_limits[:, :, 0] - f["joint_pos"]).clamp(min=0)
                + (f["joint_pos"] - self.soft_limits[:, :, 1]).clamp(min=0)
            ).sum(-1),
            "action_rate_l2": (f["action"] - f["previous_action"]).square().sum(-1),
            "air_time": ((f["air_time"] > 0.125) & (f["air_time"] < 0.3)).sum(-1) * active,
            "foot_clearance": ((height - 0.02).abs() * foot_speed2.sqrt()).sum(-1) * active,
            "foot_swing_height": swing_cost,
            "foot_slip": (foot_speed2 * f["contact"]).sum(-1) * active,
            "self_collisions": f["self_collisions"],
            "head_pose_tracking": torch.exp(-(head_error / 0.5).square()).mean(-1),
            "head_pose_bias": -self.head_ema.abs().mean(-1),
        }

    @staticmethod
    def weights(iteration):
        # Same upstream action-rate and head-bias schedule; transfer must pass
        # the source iteration, not restart a mature actor at curriculum zero.
        action = -0.1
        for i, w in ((500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)):
            if iteration >= i:
                action = w
        head = 0.0
        for i, w in ((600, 1.0), (1000, 2.0), (1500, 3.0)):
            if iteration >= i:
                head = w
        return dict(
            track_linear_velocity=2.0,
            track_angular_velocity=2.0,
            upright=2.0,
            pose=1.0,
            body_ang_vel=-0.05,
            angular_momentum=-0.02,
            dof_pos_limits=-1.0,
            action_rate_l2=action,
            air_time=3.0,
            foot_clearance=-2.0,
            foot_swing_height=-0.25,
            foot_slip=-0.1,
            self_collisions=-1.0,
            head_pose_tracking=2.0,
            head_pose_bias=head,
        )
