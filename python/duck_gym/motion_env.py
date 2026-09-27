"""Style-aware walking and grounded-start aerial skill learning on native AVBD.

All constants in motion_training.json are experimental reward hyperparameters,
not user acceptance tolerances. No state forcing or external takeoff impulses.
"""

import json
import math
from pathlib import Path
import torch
from .tensor_env import TensorEnv, multiply
from .motion import MotionGeometry, pitch_increment, contact_free_interval


class MotionEnv(TensorEnv):
    def __init__(self, model_dir, skill="walking", motion_config=None, **kwargs):
        if skill not in ("walking", "flip", "jump"):
            raise ValueError(skill)
        self.skill = skill
        path = Path(__file__).resolve().parents[2] / "configs/motion_training.json"
        self.motion_config = motion_config or json.loads(path.read_text())
        cfg = self.motion_config["walking" if skill == "walking" else "flip"]
        self.cfg = cfg
        kwargs.update(
            task="locomotion",
            gait=cfg.get("reference") in ("ik", "ik_feedforward"),
            foot_clearance=False,
            load_transfer=False,
        )
        kwargs["velocity_filter_seconds"] = cfg.get("velocity_filter_seconds", 0.0)
        if skill != "walking":
            kwargs["episode_seconds"] = cfg["episode_seconds"]
        self.motion_ready = False
        super().__init__(model_dir, **kwargs)
        self.geometry = MotionGeometry(self.model_dir / "motion.npz", self.device, self.dtype)
        if self.skill != "walking" and self.cfg.get("reference") == "crouch_feedforward":
            import numpy as np

            with np.load(self.model_dir / "aerial_reference.npz") as record:
                self.aerial_reference = torch.tensor(
                    record["targets"], device=self.device, dtype=self.dtype
                )
                self.aerial_reference_dt = float(record["times"][1] - record["times"][0])
        self.head_ids = [
            i for i, n in enumerate(self.meta["joint_names"]) if n.startswith(("head", "neck"))
        ]
        self.leg_ids = [i for i in range(14) if i not in self.head_ids]
        self.leg_std = self.home.new_tensor(
            [
                (
                    0.3
                    if "yaw" in self.meta["joint_names"][i]
                    else 0.15 if "roll" in self.meta["joint_names"][i] else 0.4
                )
                for i in self.leg_ids
            ]
        )
        self.gait_period = self.motion_config["walking"]["period"]
        n, d, t = self.num_envs, self.device, self.dtype
        self.air_time = torch.zeros(n, 2, device=d, dtype=t)
        self.head_bias = torch.zeros(n, 4, device=d, dtype=t)
        self.air_rotation = torch.zeros(n, device=d, dtype=t)
        self.rotation_frontier = torch.zeros(n, device=d, dtype=t)
        self.flight_time = torch.zeros(n, device=d, dtype=t)
        self.in_flight = torch.zeros(n, device=d, dtype=torch.bool)
        self.took_off = torch.zeros_like(self.in_flight)
        self.landed = torch.zeros_like(self.in_flight)
        self.invalid_contact = torch.zeros_like(self.in_flight)
        self.landing_hold = torch.zeros(n, device=d, dtype=t)
        self.peak_height = torch.zeros(n, device=d, dtype=t)
        self.peak_air_clearance = torch.zeros(n, device=d, dtype=t)
        self.motion_ready = True
        self.reset(torch.ones(n, device=d, dtype=torch.bool))

    def reset(self, mask):
        super().reset(mask)
        if self.skill != "walking" and self.cfg.get("aerial_detector_version", 1) >= 2:
            self.commands = torch.where(mask[:, None], 0, self.commands)
        if not self.motion_ready:
            return
        for key in (
            "air_time",
            "head_bias",
            "air_rotation",
            "rotation_frontier",
            "flight_time",
            "in_flight",
            "took_off",
            "landed",
            "invalid_contact",
            "landing_hold",
            "peak_height",
            "peak_air_clearance",
        ):
            value = getattr(self, key)
            m = mask[:, None] if value.ndim == 2 else mask
            setattr(self, key, torch.where(m, torch.zeros_like(value), value))

    def observations(self, state):
        obs = super().observations(state)
        if not self.motion_ready:
            return obs
        extra = [self.air_time]
        if self.cfg.get("head_bias_seconds", 0) > 0:
            extra.append(self.head_bias)
        if self.skill != "walking":
            extra.extend(
                [
                    self.air_rotation[:, None],
                    self.flight_time[:, None],
                    self.took_off[:, None].to(self.dtype),
                    self.landed[:, None].to(self.dtype),
                    self.landing_hold[:, None],
                    self.invalid_contact[:, None].to(self.dtype),
                    self.rotation_frontier[:, None],
                    self.peak_height[:, None],
                ]
            )
            if self.cfg.get("aerial_detector_version", 1) >= 2:
                extra.extend(
                    [
                        self.in_flight[:, None].to(self.dtype),
                        self.peak_air_clearance[:, None],
                        self.episode_length_buf[:, None].to(self.dtype) / self.max_episode_length,
                    ]
                )
        return torch.cat([obs, *extra], dim=-1).float()

    def reference(self):
        if not hasattr(self, "aerial_reference"):
            return super().reference()
        index = (
            (self.episode_length_buf.to(self.dtype) + 1)
            * self.control_dt
            / self.aerial_reference_dt
        )
        index = index.clamp(0, len(self.aerial_reference) - 1)
        low = index.long()
        high = (low + 1).clamp_max(len(self.aerial_reference) - 1)
        fraction = (index - low)[:, None]
        return self.aerial_reference[low] * (1 - fraction) + self.aerial_reference[high] * fraction

    def step(self, actions):
        previous = self.state()
        previous_b, previous_j = previous[:2]
        a = actions.to(self.dtype).clamp(-1, 1).contiguous()
        target = (
            (self.reference() + self.action_scale * a).clamp(self.lower, self.upper).contiguous()
        )
        self.native.step(
            target,
            self.forces.contiguous(),
            self.substeps,
            self.meta["kp"],
            self.meta["torque_limit"],
        )
        state = self.state()
        b, j, diag, gravity, velocity, omega, heading = state
        force = self.native.contact_forces().to(self.dtype)
        clearance, foot_velocity, support = self.geometry.support(b, feet_only=True)
        foot_force = force[:, self.geometry.feet, 2]
        weight = self.meta["mass"] * 9.81
        contact = foot_force > 0.05 * weight
        self.air_time = torch.where(contact, 0, self.air_time + self.control_dt)
        # Displacement over the whole control interval avoids last-substep aliasing.
        measured = (b[:, 1, :2] - previous_b[:, 1, :2]) / self.control_dt
        self.filtered_velocity = (
            1 - self.velocity_alpha
        ) * self.filtered_velocity + self.velocity_alpha * measured
        head_delta = j[:, self.head_ids, 0] - self.home[self.head_ids]
        if self.cfg.get("head_bias_seconds", 0) > 0:
            alpha = 1 - math.exp(-self.control_dt / self.cfg["head_bias_seconds"])
            self.head_bias = (1 - alpha) * self.head_bias + alpha * head_delta
        nonfoot = force[:, 1:, 2].sum(-1) - foot_force.sum(-1)
        slip = foot_velocity[..., :2].square().sum(-1)
        terms = {}
        failed = (
            (diag[:, 0] > 0.005)
            | (diag[:, 1] > 0.005)
            | (diag[:, 5] > 0)
            | ~torch.isfinite(b).all(dim=(1, 2))
        )
        height = b[:, 1, 2] - self.meta["root_height"]
        if self.skill == "walking":
            c = self.cfg
            phase = (
                (self.episode_length_buf[:, None] + 1) * self.control_dt / c["period"]
                + b.new_tensor([0, 0.5])
            ) % 1
            u = ((phase - c["stance_fraction"]) / (1 - c["stance_fraction"])).clamp(0, 1)
            swing = phase > c["stance_fraction"]
            wanted_height = c["swing_height_m"] * torch.sin(torch.pi * u).square()
            error = (self.filtered_velocity - self.commands).square().sum(-1)
            terms["velocity"] = c["tracking_weight"] * torch.exp(
                -error / c["tracking_sigma_mps"] ** 2
            )
            terms["upright"] = c["upright_weight"] * torch.exp(
                -20 * gravity[:, :2].square().sum(-1)
            )
            terms["height"] = torch.exp(-2000 * height.square())
            terms["head_pose"] = c["head_pose_weight"] * torch.exp(
                -head_delta.square() / c.get("head_pose_sigma_rad", 0.25) ** 2
            ).mean(-1)
            bias = self.head_bias if c.get("head_bias_seconds", 0) > 0 else head_delta
            terms["head_bias"] = -c["head_pose_l2_weight"] * bias.square().sum(-1)
            terms["leg_pose"] = c["leg_pose_weight"] * torch.exp(
                -((j[:, self.leg_ids, 0] - self.home[self.leg_ids]) / self.leg_std)
                .square()
                .mean(-1)
            )
            terms["heading"] = -c["heading_weight"] * (1 - heading[:, 0])
            terms["swing"] = c["swing_weight"] * (
                torch.exp(-((clearance - wanted_height) / c.get("swing_sigma_m", 0.004)).square())
                * swing
            ).sum(-1)
            terms["stance"] = c["stance_weight"] * (
                (~swing) * torch.exp(-slip / 0.03**2) * contact
            ).sum(-1)
            terms["slip"] = -c["slip_weight"] * (slip.clamp_max(0.25) * contact).sum(-1) / 0.05**2
            terms["unload"] = -0.5 * (swing * (foot_force / weight).clamp(0, 3).square()).sum(-1)
            terms["body_motion"] = -0.05 * omega.square().sum(-1) - velocity[:, 2].square()
            terms["action_rate"] = -c["action_rate_weight"] * (a - self.actions).square().sum(-1)
            terms["joint_velocity"] = -c["joint_velocity_weight"] * j[..., 1].square().sum(-1)
            failed |= (
                (gravity[:, 2] > -math.cos(0.5)) | (height.abs() > 0.04) | (nonfoot > 0.05 * weight)
            )
        else:
            c = self.cfg
            all_clearance = self.geometry.support(b)[0].amin(-1)
            airborne = (force[:, 1:, :].abs().sum(dim=(1, 2)) < c["contact_force_epsilon_n"]) & (
                all_clearance > 0
            )
            strict_flight = c.get("aerial_detector_version", 1) >= 2
            if strict_flight:
                airborne = contact_free_interval(diag, force, all_clearance)
            first_takeoff = airborne & (~self.in_flight if strict_flight else ~self.took_off)
            if strict_flight:
                for key in (
                    "air_rotation",
                    "rotation_frontier",
                    "flight_time",
                    "peak_air_clearance",
                ):
                    setattr(self, key, torch.where(first_takeoff, 0, getattr(self, key)))
                self.landed &= ~first_takeoff
            upward_launch_required = c.get("aerial_detector_version", 1) >= 3
            launch_event = first_takeoff
            if upward_launch_required:
                # A real contact-free interval can still be a collapsing
                # robot retracting its feet. Require upward COM velocity at
                # the beginning of that first proven free interval.
                previous_com_velocity = self.geometry.centroidal(previous_b)[1]
                launch_event = first_takeoff & (previous_com_velocity[:, 2] > 0)
                self.took_off = torch.where(first_takeoff, launch_event, self.took_off)
            else:
                self.took_off |= airborne
            # A single uninterrupted aerial interval, never sum across hops.
            first_landing = self.in_flight & ~airborne & self.took_off
            self.landed |= first_landing
            self.invalid_contact |= nonfoot > c["contact_force_epsilon_n"]
            q0 = multiply(previous_b[:, 1, 3:7], self.iquat.expand(self.num_envs, -1))
            q1 = multiply(b[:, 1, 3:7], self.iquat.expand(self.num_envs, -1))
            direction = 1 if c["direction"] == "forward" else -1
            delta = direction * pitch_increment(q0, q1)
            in_segment = airborne & ~self.landed & ~self.invalid_contact
            if upward_launch_required:
                in_segment &= self.took_off
            rotation_interval = in_segment if strict_flight else in_segment & self.in_flight
            self.air_rotation += torch.where(rotation_interval, delta, 0)
            self.flight_time += in_segment.to(self.dtype) * self.control_dt
            self.in_flight = airborne
            frontier = torch.maximum(self.rotation_frontier, self.air_rotation)
            progress = frontier - self.rotation_frontier
            self.rotation_frontier = frontier
            self.peak_height = torch.maximum(self.peak_height, height)
            self.peak_air_clearance = torch.maximum(
                self.peak_air_clearance, torch.where(in_segment, all_clearance, 0)
            )
            rotation_error = (self.air_rotation - c["rotation_target_rad"]).abs()
            rotation_score = torch.exp(-(rotation_error / c["landing_angle_scale_rad"]).square())
            if self.skill == "jump":
                rotation_score = torch.exp(
                    -(self.air_rotation / c["landing_angle_scale_rad"]).square()
                )
            upright = ((-gravity[:, 2] + 1) / 2).clamp(0, 1)
            standing = (
                torch.exp(-(height / c["landing_height_scale_m"]).square()) * upright.square()
            )
            settled = (
                self.landed
                & contact.all(-1)
                & (gravity[:, 2] < -math.cos(0.3))
                & ~self.invalid_contact
            )
            self.landing_hold = torch.where(settled, self.landing_hold + self.control_dt, 0)
            terms["launch"] = (launch_event & ~self.invalid_contact).to(
                self.dtype
            ) / self.control_dt
            terms["jump_height"] = (
                2
                * height.clamp(0, c["jump_height_scale_m"])
                / c["jump_height_scale_m"]
                * in_segment
            )
            if c.get("launch_shaping", False):
                # Dense exploration signal before takeoff. This is training
                # shaping only: standing taller never establishes a jump.
                com, com_velocity, _ = self.geometry.centroidal(b)
                previous_com = self.geometry.centroidal(previous_b)[0]
                terms["launch_velocity"] = 2 * com_velocity[:, 2].clamp(0, 1.5) * ~self.landed
                terms["com_rise"] = (
                    2
                    * ((com[:, 2] - previous_com[:, 2]) / self.control_dt).clamp(-1.5, 1.5)
                    * ~self.landed
                )
            terms["air_rotation"] = (
                2 * progress.clamp(0, c["rotation_target_rad"]) / self.control_dt
                if self.skill == "flip"
                else -0.1 * omega.square().sum(-1)
            )
            terms["landing"] = 10 * rotation_score * standing * self.landed * ~self.invalid_contact
            if strict_flight and self.skill == "jump":
                terms["landing"] *= (self.peak_air_clearance / c["jump_height_scale_m"]).clamp(0, 1)
            terms["head_pose"] = -0.1 * head_delta.square().sum(-1)
            terms["action_rate"] = -0.02 * (a - self.actions).square().sum(-1)
            terms["off_axis"] = -0.1 * omega[:, [0, 2]].square().sum(-1)
            failed |= self.invalid_contact | (b[:, 1, 2] < 0)
        reward = sum(terms.values())
        reward = torch.where(failed, -5.0, reward) * self.control_dt
        self.episode_length_buf += 1
        self.actions = a
        self.returns += reward.float()
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failed
        done = failed | timeout
        obs = self.observations(state)
        logs = {"reward/" + k: v.mean() for k, v in terms.items()}
        logs.update(
            {
                "motion/head_error_rad": head_delta.abs().mean(),
                "motion/contact_slip_mps": (slip.sqrt() * contact).sum()
                / contact.sum().clamp_min(1),
                "motion/clearance_m": clearance.clamp_min(0).mean(),
                "motion/speed_error_mps": (self.filtered_velocity - self.commands)
                .norm(dim=-1)
                .mean(),
                "motion/failed_fraction": failed.float().mean(),
            }
        )
        if self.skill != "walking":
            logs.update(
                {
                    "flip/max_rotation_rad": self.air_rotation.max(),
                    "flip/max_flight_s": self.flight_time.max(),
                    "flip/max_height_m": self.peak_height.max(),
                    "flip/landed_fraction": self.landed.float().mean(),
                }
            )
        info = dict(
            observations={},
            time_outs=timeout,
            failed=failed,
            diagnostics=diag,
            applied_targets=target,
            terminal_observation=obs,
            terminal_mask=done,
            log=logs,
        )
        if self.auto_reset:
            self.reset(done)
            obs = self.get_observations()[0]
        return obs, reward.float(), done, info
