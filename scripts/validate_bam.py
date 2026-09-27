"""Compare the portable actuator directly with the pinned BAM implementation."""

import bootstrap
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from bam.actuator import TorchBackend
from bam.model import load_model
from bam.mjlab import BamActuator
from duck_gym.actuator import xl330_m6


def main():
    params_path = files("bam") / "params/xl330/m6.json"
    params = json.loads(params_path.read_text())
    results = []
    for device in ("cpu", "cuda"):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(123)
            n, j = 64, 14
            make = lambda *s: torch.randn(s, device=device, dtype=dtype)
            pos, target, speed = make(n, j), make(n, j), 25 * make(n, j)
            speed[0] = 0
            prev, applied, ext = make(n, j), make(n, j), make(n, j)
            ext[0] = applied[0]  # Boundary where upstream NumPy and mjlab differ.
            vin = 6.5 + 1.7 * torch.sigmoid(make(n, 1))
            drop = 0.2 * torch.sigmoid(make(n, 1))
            kp, kd, friction = 1 + 0.1 * make(n, 1), 1 + 0.1 * make(n, 1), 1 + 0.1 * make(n, 1)
            limit = 8.2 * params["kt"] / params["R"]
            packed, raw, effective = xl330_m6(
                target,
                pos,
                speed,
                prev,
                applied,
                ext,
                parameters=params,
                vin=vin,
                vin_drop_gain=drop,
                vin_min=6.0,
                kp_fw=200.0,
                kp_scale=kp,
                kd_scale=kd,
                friction_scale=friction,
                force_limit=limit,
            )
            model = load_model(motor_name="xl330", model="m6")
            act = model.actuator
            act.backend = TorchBackend()
            act.vin = torch.clamp(vin - drop * prev.abs().sum(-1, keepdim=True), min=6.0)
            act.kp = 200.0 * kp
            voltage = act.compute_control(target, pos, speed * kd, 0.001)
            raw_reference = act.compute_torque(voltage, True, pos, speed * kd)
            stribeck = torch.exp(
                -torch.pow(speed.abs() / params["dtheta_stribeck"], params["alpha"])
            )
            budget = (
                BamActuator._compute_friction_budget(
                    SimpleNamespace(_bam_model=model), applied, ext, stribeck
                )
                * friction
            )
            expected = torch.stack(
                (
                    raw_reference.clamp(-limit, limit),
                    budget,
                    torch.full_like(raw_reference, params["friction_viscous"]),
                ),
                -1,
            )
            atol = 2e-6 if dtype == torch.float32 else 1e-12
            torch.testing.assert_close(raw, raw_reference, atol=atol, rtol=1e-6)
            torch.testing.assert_close(packed, expected, atol=atol, rtol=1e-6)
            torch.testing.assert_close(effective, act.vin, atol=atol, rtol=1e-6)
            results.append(
                dict(
                    device=device,
                    dtype=str(dtype),
                    samples=n * j,
                    max_raw_torque_error=float((raw - raw_reference).abs().max()),
                    max_packed_error=float((packed - expected).abs().max()),
                    passed=True,
                )
            )
    out = Path(os.environ["DUCK_RUN_DIR"])
    report = dict(
        parameters_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
        cases=results,
        scope="Equations only, supplied loads; does not validate native load estimation",
    )
    (out / "bam-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
