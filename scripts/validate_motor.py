"""CUDA motor-interface regression against shared CPU math and MuJoCo RNEA."""

import json
import os
from pathlib import Path
import numpy as np
import torch
import mujoco
from build_cuda import build
from model_io import export_model, prepare_microduck, rot, inv


def main():
    out = Path(os.environ["DUCK_RUN_DIR"]) / "motor-validation"
    out.mkdir()
    ext = build()
    results = []
    for fp64 in (True, False):
        dtype = torch.float64 if fp64 else torch.float32
        tolerance = 2e-10 if fp64 else 2e-5
        for torque, friction, damping in (
            (0.02, 0, 0),
            (-0.02, 0, 0),
            (0.02, 0.04, 0),
            (0.02, 0, 0.01),
        ):
            model = mujoco.MjModel.from_xml_string(
                f"""<mujoco><option gravity="0 0 0"/>
              <worldbody><body><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
              <joint name="motor" axis="0 0 1" armature=".002" frictionloss="{friction}" damping="{damping}"/>
              </body></worldbody></mujoco>"""
            )
            path = out / "single.duck"
            export_model(model, mujoco.MjData(model), path, {"motor": torque})
            batch = ext.CudaBatch(str(path), 3, 0.001, 1000, 0, fp64)
            motor = (
                torch.tensor([torque, friction, damping], device="cuda", dtype=dtype)
                .expand(3, 1, 3)
                .contiguous()
            )
            force = torch.zeros(3, 3, device="cuda", dtype=dtype)
            for _ in range(100):
                batch.step_motor(motor, force)
            actual = batch.state()
            expected = batch.reference(
                torch.zeros(1, dtype=dtype), torch.zeros(3, dtype=dtype), 100, 0.0, 0.96
            )
            for field in (0, 1):
                for env in range(3):
                    torch.testing.assert_close(
                        actual[field][env].cpu(), expected[field], atol=tolerance, rtol=tolerance
                    )
            assert not actual[2][:, 5].any()
            if friction == 0 and damping == 0:
                assert abs(float(actual[1][0, 0, 1]) - torque / 0.012 * 0.1) < 2e-4
            results.append(
                dict(
                    fp64=fp64,
                    torque=torque,
                    friction=friction,
                    damping=damping,
                    cpu_joint_difference=float((actual[1][0].cpu() - expected[1]).abs().max()),
                    passed=True,
                )
            )
        # Fail one environment without contaminating its neighbors, then reset.
        motor[1, 0, 0] = float("nan")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            batch.step_motor(motor, force)
        assert batch.state()[2][:, 5].tolist() == [0.0, 1.0, 0.0]
        batch.reset(
            torch.tensor([False, True, False], device="cuda"),
            torch.zeros(3, 1, device="cuda", dtype=dtype),
            torch.zeros(3, 6, device="cuda", dtype=dtype),
        )
        assert not batch.state()[2][:, 5].any()
        model = mujoco.MjModel.from_xml_string(prepare_microduck(ground=False))
        data = mujoco.MjData(model)
        rng = np.random.default_rng(81)
        offset = torch.tensor(
            rot(inv(model.body_iquat[1]), model.body_ipos[1])[None, :], device="cuda", dtype=dtype
        )
        maxerr = 0.0
        for k in range(10):
            mujoco.mj_resetData(model, data)
            data.qpos[3:7] = rng.normal(size=4)
            data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
            data.qpos[7:] = rng.uniform(-0.7, 0.7, 14)
            data.qvel[:] = rng.uniform(-5, 5, model.nv)
            path = out / f"rnea-{k}.duck"
            export_model(model, data, path)
            batch = ext.CudaBatch(str(path), 1, 0.001, 50, 0, fp64)
            loads = batch.generalized_loads(offset)[0, :, 0].double().cpu().numpy()
            error = float(np.max(np.abs(loads - data.qfrc_bias[6:])))
            maxerr = max(maxerr, error)
            assert error < (2e-12 if fp64 else 1e-6), (fp64, error)
        results.append(dict(fp64=fp64, rnea_max_error=maxerr, random_states=10, passed=True))
    (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
