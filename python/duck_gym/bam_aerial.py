"""BAM aerial diagnostics, using the same native dynamics as walking.

Every rollout starts on the ground at rest. Contact/clearance, upward COM
velocity and rotation are sampled at every physical substep. Nonfoot visual
ground penetration is a failure even where the original collision mask omits
that body. Such a trajectory cannot be used as a flight reference.
"""

from pathlib import Path
import numpy as np
import torch
from .bam_env import BamEnv
from .motion import pitch_increment
from .tensor_env import inverse_rotate


class BamAerialEnv(BamEnv):
    def __init__(self, model_dir, geometry, **kwargs):
        super().__init__(model_dir, **kwargs)
        r = np.load(Path(geometry))
        counts = r["counts"]
        self.geom_points = self.home.new_tensor(
            np.concatenate([p[:n] for p, n in zip(r["points"], counts)])
        )
        self.point_slots = torch.as_tensor(
            np.repeat(np.arange(len(counts)), counts),
            device=self.device,
            dtype=torch.long,
        )
        self.geom_bodies = torch.as_tensor(
            r["bodies"], device=self.device, dtype=torch.long
        )
        self.geom_feet = torch.as_tensor(
            r["foot_slots"], device=self.device, dtype=torch.long
        )
        self.geom_nonfeet = torch.tensor(
            [i for i in range(len(r["bodies"])) if i not in r["foot_slots"]],
            device=self.device,
        )
        self.reset()

    def centroidal(self):
        w = self.mass[None, :, None] / self.mass.sum()
        return (self.bodies[:, :, :3] * w).sum(1), (self.bodies[:, :, 7:10] * w).sum(1)

    def mechanical_energy(self):
        local_w = inverse_rotate(self.bodies[:, :, 3:7], self.bodies[:, :, 10:13])
        kinetic = 0.5 * (
            self.mass[None] * self.bodies[:, :, 7:10].square().sum(-1)
        ).sum(-1)
        kinetic += 0.5 * (self.inertia[None] * local_w.square()).sum((1, 2))
        kinetic += (
            0.5
            * self.cfg["parameters"]["armature"]
            * self.joints[:, :, 1].square().sum(-1)
        )
        potential = 9.81 * (self.mass[None] * self.bodies[:, :, 2]).sum(-1)
        return kinetic + potential

    def clearances(self):
        b = self.bodies[:, self.geom_bodies]
        up = b.new_tensor([0, 0, 1]).expand(*b.shape[:2], 3)
        local = inverse_rotate(b[:, :, 3:7], up)
        z = (self.geom_points[None] * local[:, self.point_slots]).sum(-1) + b[
            :, self.point_slots, 2
        ]
        result = self.home.new_full(
            (self.num_envs, len(self.geom_bodies)), float("inf")
        )
        return result.scatter_reduce_(
            1, self.point_slots[None].expand(self.num_envs, -1), z, reduce="amin"
        )

    def reset(self, mask=None, **kwargs):
        # Base __init__ does not reset when training=None.
        obs = super().reset(mask, **kwargs)
        if not hasattr(self, "geom_points"):
            return obs
        if mask is None:
            mask = torch.ones_like(self.pending)
        for name in ("airborne", "took_off", "landed", "invalid", "numerical_invalid"):
            if not hasattr(self, name):
                setattr(self, name, torch.zeros_like(self.pending))
            getattr(self, name)[mask] = False
        for name in (
            "flight_time",
            "air_rotation",
            "takeoff_vz",
            "max_upward",
            "peak_height",
            "landing_time",
            "max_anchor",
            "max_penetration",
            "valid_time",
            "positive_motor_work",
            "max_energy_excess",
        ):
            if not hasattr(self, name):
                setattr(self, name, self.home.new_zeros(self.num_envs))
            getattr(self, name)[mask] = 0
        com, _ = self.centroidal()
        if not hasattr(self, "initial_com"):
            self.initial_com = com.clone()
            self.previous_quat = self.root_quat.clone()
        self.initial_com[mask] = com[mask]
        self.previous_quat[mask] = self.root_quat[mask]
        if not hasattr(self, "initial_energy"):
            self.initial_energy = self.mechanical_energy().clone()
            self.energy_previous_velocity = self.joints[:, :, 1].clone()
        self.initial_energy[mask] = self.mechanical_energy()[mask]
        self.energy_previous_velocity[mask] = self.joints[mask, :, 1]
        return obs

    def failure_mask(self):
        return self.invalid

    def observe_substep(self, env):
        com, velocity = self.centroidal()
        height = self.clearances()
        self.max_anchor = torch.maximum(self.max_anchor, self.diagnostics[:, 0])
        self.max_penetration = torch.maximum(
            self.max_penetration, self.diagnostics[:, 1]
        )
        speed = 0.5 * (self.energy_previous_velocity + self.joints[:, :, 1])
        self.positive_motor_work += (self.previous_applied * speed).clamp_min(0).sum(
            -1
        ) * self.dt
        self.energy_previous_velocity = self.joints[:, :, 1].clone()
        excess = (
            self.mechanical_energy() - self.initial_energy - self.positive_motor_work
        )
        self.max_energy_excess = torch.maximum(self.max_energy_excess, excess)
        # Positive work is deliberately an upper bound: friction and negative
        # motor work cannot add energy. This 1% numerical screening allowance
        # is an experimental guard, not a validated accuracy tolerance.
        allowance = 0.01 * (self.initial_energy.abs() + self.positive_motor_work) + 1e-4
        self.numerical_invalid |= (
            (excess > allowance)
            | (self.diagnostics[:, 5] > 0)
            | (self.diagnostics[:, 0] > 0.005)
            | (self.diagnostics[:, 1] > 0.005)
            | (self.diagnostics[:, 4] > 0.02)
            | (~torch.isfinite(self.bodies).all(dim=(1, 2)))
        )
        self.invalid |= (
            height[:, self.geom_nonfeet].amin(-1) < -0.0001
        ) | self.numerical_invalid
        self.valid_time += (~self.invalid) * self.dt
        # Candidate contacts extend 3 mm above the plane and can carry zero
        # force. They must not suppress genuine low jumps. Require no actual
        # ground foot force and a positive full-sole gap above numerical noise.
        # This model permits ground support only on the two feet.
        free = (~self.contact.any(-1)) & (height[:, self.geom_feet].amin(-1) > 1e-5)
        leaving = (
            free
            & ~self.airborne
            & ~self.took_off
            & ~self.invalid
            & (velocity[:, 2] > 0)
        )
        self.takeoff_vz = torch.where(leaving, velocity[:, 2], self.takeoff_vz)
        self.took_off |= leaving
        self.landed |= self.airborne & ~free & self.took_off
        flight = free & self.took_off & ~self.landed & ~self.invalid
        self.air_rotation += torch.where(
            flight & self.airborne,
            pitch_increment(self.previous_quat, self.root_quat),
            0,
        )
        self.flight_time += flight * self.dt
        self.landing_time += (
            self.landed & self.contact.all(-1) & ~self.invalid
        ) * self.dt
        self.airborne = free
        self.previous_quat = self.root_quat.clone()
        self.max_upward = torch.maximum(
            self.max_upward, torch.where(~self.invalid, velocity[:, 2], 0)
        )
        self.peak_height = torch.maximum(
            self.peak_height,
            torch.where(~self.invalid, com[:, 2] - self.initial_com[:, 2], 0),
        )

    def step(self, actions, *, substep_callback=None):
        def callback(env):
            self.observe_substep(env)
            if substep_callback is not None:
                substep_callback(env)

        result = super().step(actions, substep_callback=callback)
        self.invalid |= self.self_penetration > 0.005
        return result
