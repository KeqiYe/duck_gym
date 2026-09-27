"""CUDA/shared-host body collision invariants and cross-stream diagnostics."""

import bootstrap
import json
import os
from pathlib import Path
import numpy as np
import torch
import mujoco
from build_cuda import build
from model_io import export_model


def main():
    out = Path(os.environ["DUCK_RUN_DIR"]) / "validation"
    out.mkdir()
    ext = build()
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody><body pos="-.021 0 .1"><freejoint/><geom type="sphere" size=".02" mass="1" friction="0 0 0"/></body><body pos=".021 0 .1"><freejoint/><geom type="sphere" size=".02" mass="2" friction="0 0 0"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 0.2
    data.qvel[6] = -0.1
    path = out / "spheres.duck"
    export_model(model, data, path, self_contacts=True)
    results = []
    for fp64 in (False, True):
        dtype = torch.float64 if fp64 else torch.float32
        batch = ext.CudaBatch(str(path), 64, 0.001, 400, 0, fp64)
        q = torch.zeros(64, 0, device="cuda", dtype=dtype)
        f = torch.zeros(64, 3, device="cuda", dtype=dtype)
        max_momentum = 0.0
        max_penetration = 0.0
        seen = False
        for _ in range(100):
            batch.step(q, f, 1, 0.0, 1.0)
            b, j, d = batch.state()
            force = batch.contact_forces()
            stats = batch.self_contact_stats()
            assert not bool(d[:, 5].any())
            assert torch.equal(b, b[:1].expand_as(b))
            assert torch.equal(force.sum(1), torch.zeros_like(force[:, 0]))
            assert not bool(batch.contact_forces(ground_only=True).any())
            seen |= bool(stats[:, 0].any())
            max_momentum = max(
                max_momentum, float((b[:, 1, 7:10] + 2 * b[:, 2, 7:10]).norm(dim=-1).max())
            )
            max_penetration = max(max_penetration, float(d[:, 1].max()))
        assert seen
        assert max_momentum < (2e-5 if fp64 else 2e-4)
        ref = batch.reference(
            torch.zeros(0, dtype=dtype), torch.zeros(3, dtype=dtype), 100, 0.0, 1.0
        )
        error = float((b[0].cpu() - ref[0]).abs().max())
        assert error < (1e-8 if fp64 else 5e-5), error
        with torch.cuda.stream(torch.cuda.Stream()):
            stats = batch.self_contact_stats()
            ground = batch.contact_forces(ground_only=True)
        torch.cuda.synchronize()
        assert torch.isfinite(stats).all() and not ground.any()
        f[1, 0] = float("nan")
        batch.step(q, f, 1, 0.0, 1.0)
        fault = batch.state()[2][:, 5]
        assert int(fault.sum()) == 1 and bool(fault[1])
        mask = torch.zeros(64, device="cuda", dtype=torch.bool)
        mask[1] = True
        batch.reset(mask, q, torch.zeros(64, 6, device="cuda", dtype=dtype))
        assert not batch.state()[2][:, 5].any()
        results.append(
            dict(
                fp64=fp64,
                envs=64,
                steps=100,
                maximum_total_momentum=max_momentum,
                maximum_penetration=max_penetration,
                host_parity_max_error=error,
            )
        )
    (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
