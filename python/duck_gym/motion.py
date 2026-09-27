"""Physical motion diagnostics shared by training and independent evaluation."""

import numpy as np
import torch
from .tensor_env import inverse_rotate, multiply


def rotate(q, v):
    return inverse_rotate(q * q.new_tensor([1, -1, -1, -1]), v)


def pitch_increment(previous, current):
    """Signed incremental rotation about world y, without Euler pitch wrapping."""
    delta = multiply(current, previous * previous.new_tensor([1, -1, -1, -1]))
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    norm = delta[..., 1:].norm(dim=-1)
    scale = 2 * torch.atan2(norm, delta[..., 0].clamp_min(0)) / norm.clamp_min(1e-9)
    return delta[..., 2] * scale


def contact_free_interval(diagnostics, forces, clearance):
    """Conservative flight proof: no contact candidate in ANY physics substep.

    Candidate contacts include the solver's proximity band, so this may shorten
    flight. A zero last-substep force alone can be contact chatter, not a jump.
    """
    return (
        (diagnostics[..., 2] == 0)
        & (forces[..., 1:, :].abs().sum(dim=(-1, -2)) < 1e-4)
        & (clearance > 0)
    )


class MotionGeometry:
    def __init__(self, path, device, dtype):
        r = np.load(path)
        self.points = torch.tensor(r["points"], device=device, dtype=dtype)
        self.moments = self.points[..., :, None] * self.points[..., None, :]
        self.valid = torch.tensor(r["valid"], device=device)
        self.bodies = torch.tensor(r["bodies"], device=device, dtype=torch.long)
        self.foot_slots = torch.tensor(r["foot_slots"], device=device, dtype=torch.long)
        self.feet = torch.tensor(r["foot_bodies"], device=device, dtype=torch.long)
        self.mass = torch.tensor(r["body_mass"], device=device, dtype=dtype)
        self.inertia = torch.tensor(r["body_inertia"], device=device, dtype=dtype)

    def support(self, body, feet_only=False, pointwise_rms=False):
        slots = self.foot_slots if feet_only else torch.arange(len(self.bodies), device=body.device)
        pose = body[:, self.bodies[slots]]
        points, valid = self.points[slots], self.valid[slots]
        up = pose.new_tensor([0, 0, 1]).expand(*pose.shape[:2], 3)
        local_up = inverse_rotate(pose[..., 3:7], up)
        z = torch.einsum("gvi,egi->egv", points, local_up) + pose[..., None, 2]
        clearance = z.amin(-1)
        # Average velocity over the actual lowest support band, not foot COM.
        weight = ((z <= clearance[..., None] + 0.0005) & valid[None]).to(body.dtype)
        center = torch.einsum("egv,gvi->egi", weight, points) / weight.sum(-1, keepdim=True)
        offset = rotate(pose[..., 3:7], center)
        velocity = pose[..., 7:10] + torch.linalg.cross(pose[..., 10:13], offset)
        if pointwise_rms:
            covariance = (
                torch.einsum("egv,gvij->egij", weight, self.moments[slots])
                / weight.sum(-1)[..., None, None]
            )
            covariance -= center[..., :, None] * center[..., None, :]
            # v_point is affine in local r. Include variance so opposite
            # slipping points cannot cancel at a spinning foot's centroid.
            omega = pose[..., 10:13]
            x = torch.zeros_like(omega)
            x[..., 0] = 1
            y = torch.zeros_like(omega)
            y[..., 1] = 1
            cx = inverse_rotate(pose[..., 3:7], torch.linalg.cross(x, omega))
            cy = inverse_rotate(pose[..., 3:7], torch.linalg.cross(y, omega))
            variance = torch.einsum("egi,egij,egj->eg", cx, covariance, cx) + torch.einsum(
                "egi,egij,egj->eg", cy, covariance, cy
            )
            rms = (velocity[..., :2].square().sum(-1) + variance).clamp_min(0).sqrt()
            return clearance, velocity, pose[..., :3] + offset, rms
        return clearance, velocity, pose[..., :3] + offset

    def centroidal(self, body):
        total = self.mass.sum()
        com = (body[..., :3] * self.mass[None, :, None]).sum(1) / total
        velocity = (body[..., 7:10] * self.mass[None, :, None]).sum(1) / total
        omega_local = inverse_rotate(body[..., 3:7], body[..., 10:13])
        spin = rotate(body[..., 3:7], omega_local * self.inertia)
        orbital = torch.linalg.cross(body[..., :3] - com[:, None], body[..., 7:10])
        angular = (spin + orbital * self.mass[None, :, None]).sum(1)
        return com, velocity, angular
