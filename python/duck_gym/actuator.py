"""Portable Torch implementation of the pinned BAM XL330/m6 mjlab equations.

Derived from BAM (Marc Duclusaud & Grégoire Passault, copyright 2025),
Apache-2.0, commit 62bd8ce12154340be97e06f7f41a0ca8f116d967.
See https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967
for the original source; license: third_party/licenses/BAM-Apache-2.0.txt.
Modified here for explicit CPU/CUDA motor-channel input and output.
The parameter dictionary must come from the pinned xl330/m6.json; no fitted
or guessed hardware constants are substituted here.
"""

import math
import torch


def xl330_m6(
    target,
    position,
    velocity,
    previous_motor_torque,
    previous_applied_torque,
    external_torque,
    *,
    parameters,
    vin,
    vin_drop_gain,
    vin_min,
    kp_fw,
    kp_scale,
    kd_scale,
    friction_scale,
    force_limit,
):
    """Return (native motor channels, raw motor torque, effective supply).

    Joint tensors have shape [E,J]; supply and gains broadcast across joints.
    external_torque is previous-step -bias + constraint - joint dry-friction
    force, in generalized hinge coordinates. It must come from the native
    dynamics when this actuator is used with the native engine.
    previous_motor_torque is BEFORE the MuJoCo actuator force-range clamp;
    previous_applied_torque is AFTER it. Their uses differ in upstream BAM.
    The caller owns history and reset semantics and calls this every physics dt.
    """
    p = parameters
    supply = torch.clamp(
        vin - vin_drop_gain * previous_motor_torque.abs().sum(-1, keepdim=True), min=vin_min
    )
    speed = velocity * kd_scale
    kt, resistance = p["kt"], p["R"]
    error_gain = (4096 / (2 * math.pi)) / (256 * 885)
    duty = (target - position) * kp_fw * kp_scale * error_gain
    center = kt * speed / supply
    span = resistance * 1.75 / supply
    duty = torch.clamp(torch.clamp(duty, min=center - span, max=center + span), -1.0, 1.0)
    raw_torque = kt * supply * duty / resistance - kt**2 * speed / resistance
    stribeck = torch.exp(-torch.pow(velocity.abs() / p["dtheta_stribeck"], p["alpha"]))
    mot, ext = previous_applied_torque, external_torque
    budget = (
        p["friction_base"]
        + stribeck * p["friction_stribeck"]
        + (ext * p["load_friction_external"] - mot * p["load_friction_motor"]).abs()
        + stribeck
        * (
            ext * p["load_friction_external_stribeck"] - mot * p["load_friction_motor_stribeck"]
        ).abs()
        + stribeck
        * torch.where(
            mot.abs() > ext.abs(),
            p["load_friction_external_quad"] * ext.square(),
            p["load_friction_motor_quad"] * mot.square(),
        )
    ) * friction_scale
    # Deliberately follows bam.mjlab, whose quadratic term differs from the
    # numpy Model.compute_frictions sign/equality masking at this pinned commit.
    applied = torch.clamp(raw_torque, min=-force_limit, max=force_limit)
    damping = torch.full_like(applied, p["friction_viscous"])
    return torch.stack((applied, budget, damping), dim=-1), raw_torque, supply
