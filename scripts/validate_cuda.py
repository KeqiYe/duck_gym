"""Validate native CUDA against original CPU FP64 and same-math CPU FP32."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import mujoco
from build_cuda import build
from model_io import ROOT, export_model, prepare_microduck
from validate_cpu import cases
from prepare_standing import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(os.environ["DUCK_RUN_DIR"]) / "validation")
    p.add_argument("--precision", choices=["both", "fp32", "fp64"], default="both")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ext = build()
    workspace = Path(os.environ["DUCK_CUDA_BUILD"]).parents[1]
    cpu_build = workspace / "build/cpu-release"
    subprocess.run(
        ["cmake", "-S", str(ROOT), "-B", str(cpu_build), "-DCMAKE_BUILD_TYPE=Release"], check=True
    )
    subprocess.run(["cmake", "--build", str(cpu_build), "--parallel", "4"], check=True)
    subprocess.run(["ctest", "--test-dir", str(cpu_build), "--output-on-failure"], check=True)
    selected = list(cases()) + [
        ("microduck_release", prepare_microduck(), {}, {}, 0.1 if args.quick else 1.0, 0.02)
    ]
    results = []
    for name, xml, initial, torques, duration, tolerance in selected:
        path = args.output / name
        path.mkdir(exist_ok=True)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        for key, value in initial.items():
            if key == "vx":
                d.qvel[0] = value
            elif key == "omega":
                d.qvel[3:6] = value
            else:
                d.qpos[m.joint(key).qposadr[0]] = value
        export_model(m, d, path / "model.duck", torques)
        steps = round(duration / 0.001)
        cpu = subprocess.run(
            [
                str(cpu_build / "duck_sim"),
                str(path / "model.duck"),
                "--steps",
                str(steps),
                "--iterations",
                "200",
                "--output",
                str(path / "cpu.csv"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        records = np.genfromtxt(path / "cpu.csv", delimiter=",", names=True)
        cpu_positions = np.stack([records[k] for k in ["x", "y", "z"]], -1).reshape(
            steps + 1, m.nbody - 1, 3
        )
        cpu_quat = np.stack([records[k] for k in ["qw", "qx", "qy", "qz"]], -1).reshape(
            steps + 1, m.nbody - 1, 4
        )
        case = dict(name=name, cpu=json.loads(cpu.stdout), precisions=[])
        for fp64 in [True, False] if args.precision == "both" else [args.precision == "fp64"]:
            dtype = torch.float64 if fp64 else torch.float32
            batch = ext.CudaBatch(str(path / "model.duck"), 2, 0.001, args.iterations, 0, fp64)
            q = torch.zeros((2, len(batch.joint_names)), device="cuda", dtype=dtype)
            force = torch.zeros((2, 3), device="cuda", dtype=dtype)
            maximum = rotation_error = joint_error = penetration = 0
            start = time.perf_counter()
            for k in range(0, steps, 10):
                n = min(10, steps - k)
                batch.step(q, force, n, 0.0, 0.96)
                b, j, diag = batch.state()
                torch.cuda.synchronize()
                assert not diag[:, 5].any(), (name, diag)
                assert (
                    torch.isfinite(b).all()
                    and torch.isfinite(j).all()
                    and torch.isfinite(diag).all()
                )
                quat = b[0, 1:, 3:7].double().cpu().numpy()
                quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
                rotation_error = max(
                    rotation_error,
                    float(
                        (
                            2 * np.arccos(np.clip(np.abs((quat * cpu_quat[k + n]).sum(-1)), 0, 1))
                        ).max()
                    ),
                )
                joint_error = max(joint_error, float(diag[:, 0].max()))
                penetration = max(penetration, float(diag[:, 1].max()))
                torch.testing.assert_close(b[0], b[1], rtol=0, atol=0)
                maximum = max(
                    maximum,
                    float(
                        np.linalg.norm(
                            b[0, 1:, :3].cpu().numpy() - cpu_positions[k + n], axis=-1
                        ).max()
                    ),
                )
            reference = batch.reference(q[0].cpu(), force[0].cpu(), steps, 0.0, 0.96)
            same_math = float((b[0, :, :3].cpu() - reference[0][:, :3]).abs().max())
            # Numerical migration gates, separate from the independent MuJoCo tolerance.
            allowed = 0.002 if fp64 else 0.01
            passed = (
                maximum < allowed
                and same_math < allowed
                and rotation_error < 0.05
                and joint_error < 0.005
                and penetration < 0.005
            )
            row = dict(
                dtype=str(dtype),
                position_error_vs_original_cpu_m=maximum,
                position_error_vs_same_dtype_cpu_m=same_math,
                allowed_m=allowed,
                rotation_error_rad=rotation_error,
                joint_error_m=joint_error,
                penetration_m=penetration,
                wall_seconds=time.perf_counter() - start,
                passed=passed,
            )
            case["precisions"].append(row)
            print(name, row, flush=True)
        results.append(case)
    model_dir = workspace / "build/models/standing"
    meta = prepare(model_dir)
    control_tests = []
    for fp64 in (False, True):
        dtype = torch.float64 if fp64 else torch.float32
        for limit in (0.96, 0.01):
            sim = ext.CudaBatch(
                str(model_dir / "microduck.duck"), 2, 0.001, args.iterations, 0, fp64
            )
            target = torch.tensor(meta["home"], device="cuda", dtype=dtype).repeat(2, 1)
            target[:, 0] += 0.04
            forces = torch.tensor([[0.02, 0, 0], [0.02, 0, 0]], device="cuda", dtype=dtype)
            sim.step(target, forces, 100, 0.55, limit)
            actual = sim.state()
            reference = sim.reference(target[0].cpu(), forces[0].cpu(), 100, 0.55, limit)
            position = float((actual[0][0, :, :3].cpu() - reference[0][:, :3]).abs().max())
            joint = float((actual[1][0].cpu() - reference[1]).abs().max())
            assert torch.isfinite(actual[0]).all() and not actual[2][:, 5].any()
            assert position < 0.002 and joint < 0.02, (fp64, limit, position, joint)
            control_tests.append(
                dict(
                    fp64=fp64,
                    torque_limit=limit,
                    position_error_m=position,
                    joint_state_error=joint,
                )
            )
    n = 4
    batch = ext.CudaBatch(str(model_dir / "microduck.duck"), n, 0.001, 200, 0, False)
    q = torch.tensor(meta["home"], device="cuda").repeat(n, 1)
    f = torch.zeros(n, 3, device="cuda")
    roots = torch.zeros(n, 6, device="cuda")
    batch.reset(torch.ones(n, dtype=torch.bool, device="cuda"), q, roots)
    initial = batch.state()[0].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        target = q.clone()
        target[1, 0] += 0.01
        batch.step(target, f, 10, 0.55, 0.96)
    # The extension serializes its persistent state across streams with an event.
    before = batch.state()[0].clone()
    mask = torch.tensor([True, False, False, False], device="cuda")
    batch.reset(mask, q, roots)
    after = batch.state()[0]
    torch.testing.assert_close(after[0], initial[0], rtol=0, atol=1e-7)
    torch.testing.assert_close(after[1:], before[1:], rtol=0, atol=0)
    after.fill_(123)
    assert not (batch.state()[0] == 123).all()
    bad = q.clone()
    bad[2, 0] = float("nan")
    batch.step(bad, f, 1, 0.55, 0.96)
    assert batch.state()[2][2, 5] == 1
    batch.reset(torch.ones(n, dtype=torch.bool, device="cuda"), q, roots)
    assert not batch.state()[2][:, 5].any()
    report = dict(
        precision=args.precision,
        iterations=args.iterations,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cases=results,
        control_tests=control_tests,
        boundary_tests_passed=True,
        passed=all(v["passed"] for c in results for v in c["precisions"]),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
