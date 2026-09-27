"""Phase-conditioned residual PPO for crouching and aerial skills.

References supply joint targets only. All physical state transitions use BAM
and native AVBD; every episode is initialized at rest on the ground.
"""

import json
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from .bam_aerial import BamAerialEnv


class BamSkillEnv(BamAerialEnv):
    def __init__(self, model_dir, preparation, task, **kwargs):
        self.skill_ready = False
        self.task = task
        self.reset_on_done = True
        super().__init__(
            model_dir,
            Path(preparation) / "geometry.npz",
            seconds=task["episode_seconds"],
            physics_dt=task.get("physics_dt"),
            **kwargs,
        )
        rows = json.loads((Path(preparation) / "crouch.json").read_text())["poses"]

        def pose(depth):
            row = min(rows, key=lambda r: abs(r["depth_m"] - depth))
            if row["foot_error_m"] > 0.001:
                raise ValueError("Invalid IK reference")
            return row["joint_offsets"]

        if task.get("reference_trajectory"):
            reference_path = Path(preparation) / task["reference_trajectory"]
            if (
                task.get("reference_sha256")
                and hashlib.sha256(reference_path.read_bytes()).hexdigest()
                != task["reference_sha256"]
            ):
                raise ValueError("Dynamic reference does not match policy metadata")
            trajectory = np.load(reference_path)
            times, targets, depths = (
                trajectory["times"],
                trajectory["target_offsets"],
                trajectory["depths"],
            )
            if (
                targets.shape != (len(times), 14)
                or depths.shape != times.shape
                or len(times) < 2
                or not np.all(np.diff(times) > 0)
                or not all(np.isfinite(v).all() for v in (times, targets, depths))
            ):
                raise ValueError("Invalid dynamic target reference")
            if abs(times[0]) > 1e-9 or times[-1] < task["episode_seconds"] - 1e-9:
                raise ValueError("Dynamic reference must cover the whole skill episode")
            self.reference_times = self.home.new_tensor(times)
            self.reference_poses = self.home.new_tensor(targets)
            self.reference_depths = self.home.new_tensor(depths)
        else:
            self.reference_times = self.home.new_tensor(task["reference_times"])
            self.reference_poses = self.home.new_tensor(
                [pose(d) for d in task["reference_depths"]]
            )
            self.reference_depths = self.home.new_tensor(task["reference_depths"])
        self.residual = self.home.new_zeros(self.num_envs, 14)
        self.old_residual = self.residual.clone()
        self.nominal_root_height = self.bodies[:, 1, 2] - self.clearances()[
            :, self.geom_feet
        ].amin(-1)
        self.skill_ready = True
        self.reset()

    def reference(self, next_step=False):
        t = (self.episode_length_buf.to(self.dtype) + int(next_step)) * self.step_dt
        index = (torch.searchsorted(self.reference_times, t, right=True) - 1).clamp(
            0, len(self.reference_times) - 2
        )
        u = (
            (t - self.reference_times[index])
            / (self.reference_times[index + 1] - self.reference_times[index])
        ).clamp(0, 1)
        u = u * u * (3 - 2 * u)
        return self.reference_poses[index] * (1 - u[:, None]) + self.reference_poses[
            index + 1
        ] * u[:, None], self.reference_depths[index] * (1 - u) + self.reference_depths[
            index + 1
        ] * u

    def get_observations(self):
        obs = super().get_observations()
        if not self.skill_ready:
            return obs
        ref, depth = self.reference()
        phase = self.episode_length_buf.to(self.dtype) / self.max_episode_length
        extra = torch.cat(
            [phase[:, None], torch.sin(phase[:, None] * 2 * math.pi), ref], -1
        ).float()
        obs["actor"] = torch.cat([obs["actor"], extra], -1)
        _, v = self.centroidal()
        privileged = torch.cat(
            [
                v,
                self.bodies[:, 1, 2:3],
                self.took_off[:, None],
                self.landed[:, None],
                self.flight_time[:, None],
                self.air_rotation[:, None],
                depth[:, None],
            ],
            -1,
        ).float()
        obs["critic"] = torch.cat([obs["critic"], extra, privileged], -1)
        return obs

    def reset(self, mask=None, **kwargs):
        kwargs.setdefault("joint_noise", self.task.get("reset_joint_noise", 0.0))
        kwargs.setdefault("tilt_noise", self.task.get("reset_tilt_noise", 0.0))
        obs = super().reset(mask, **kwargs)
        if self.skill_ready:
            if mask is None:
                mask = torch.ones_like(self.pending)
            self.residual[mask] = 0
            self.old_residual[mask] = 0
            return self.get_observations()
        return obs

    def step(self, actions, *, substep_callback=None):
        old_flight = self.flight_time.clone()
        old_takeoff = self.took_off.clone()
        old_rotation = self.air_rotation.clone()
        ref, _ = self.reference(next_step=True)
        self.old_residual.copy_(self.residual)
        self.residual.copy_(
            self.task["residual_scale"] * torch.tanh(actions.to(self.dtype))
        )
        target = ref + self.residual
        if self.task.get("landing_feedback"):
            cfg = self.task["landing_feedback"]
            pitch = torch.atan2(self.gravity[:, 0], -self.gravity[:, 2])
            kp = self.home.new_tensor(cfg["kp"])
            kd = self.home.new_tensor(cfg["kd"])
            correction = (kp * pitch + kd * self.ang_vel[:, 1]) * self.landed
            target[:, 2] += correction
            target[:, 4] += 0.5 * correction
            target[:, 11] -= correction
            target[:, 13] -= 0.5 * correction
        _, _, _, info = super().step(target, substep_callback=substep_callback)
        ref, depth = self.reference()
        com, v = self.centroidal()
        upright = (-self.gravity[:, 2]).clamp(-1, 1)
        pose_error = self.joints[:, :, 0] - self.home - ref
        t = self.episode_length_buf * self.step_dt
        flight = self.took_off & ~self.landed
        terms = dict(
            pose=4 * torch.exp(-pose_error.square().mean(-1) / 0.09),
            height=2
            * torch.exp(
                -(
                    (self.bodies[:, 1, 2] - (self.nominal_root_height - depth)) / 0.025
                ).square()
            ),
            upright=3 * ((upright + 1) / 2).pow(4),
            head=-2
            * (self.joints[:, self.head_ids, 0] - self.home[:, self.head_ids])
            .square()
            .sum(-1),
            action_rate=-0.1 * (self.residual - self.old_residual).square().sum(-1),
            velocity=-0.05 * v[:, :2].square().sum(-1)
            - self.task.get("joint_velocity_weight", 0.01)
            * self.joints[:, :, 1].square().sum(-1),
            support=0.5 * self.contact.sum(-1),
        )
        event = self.home.new_zeros(self.num_envs)
        if self.task["skill"] != "crouch":
            launch = (
                (t >= self.task["launch_window"][0])
                & (t <= self.task["launch_window"][1])
                & ~self.landed
            )
            terms["launch"] = (
                self.task["launch_weight"] * launch * v[:, 2].clamp(-0.5, 2)
            )
            terms["pose"] *= torch.where(
                launch,
                self.home.new_tensor(self.task.get("launch_pose_scale", 1.0)),
                self.home.new_tensor(1.0),
            )
            terms["height"] *= ~flight
            terms["support"] *= ~flight
            event += 2 * (self.took_off & ~old_takeoff) + 4 * (
                self.flight_time - old_flight
            )
            if self.task["skill"] in ("forward", "backward"):
                sign = 1 if self.task["skill"] == "forward" else -1
                terms["upright"] *= ~flight
                event += 0.5 * sign * (self.air_rotation - old_rotation)
            landing = self.landed & self.contact.all(-1) & ~self.invalid
            terms["landing"] = (
                8
                * landing
                * ((upright + 1) / 2).pow(4)
                * torch.exp(-v.square().sum(-1) / 0.04)
            )
        reward = sum(terms.values()) * self.step_dt + event - self.invalid.float()
        timeout = self.episode_length_buf >= self.max_episode_length
        done = self.invalid | timeout
        info.update(
            time_outs=timeout & ~self.invalid,
            terminal_observation=self.get_observations(),
            log={
                **{f"Reward/{k}": val.mean() for k, val in terms.items()},
                "Skill/takeoff_fraction": self.took_off.float().mean(),
                "Skill/flight_seconds": self.flight_time.mean(),
                "Skill/invalid": self.invalid.float().mean(),
                "Skill/max_upward": self.max_upward.mean(),
            },
        )
        if self.reset_on_done:
            self.reset(done)
        else:
            self.pending.copy_(done)
        return self.get_observations(), reward.float(), done.long(), info
