"""Gate MicroDuck CUDA specialization on trajectory parity, then time both paths.

Run on a configured Linux CUDA host through scripts/remote/run.py. The default
model is prepare_standing's complete MicroDuck; --model-dir may be repeated to
add exported official model.duck/config.json bundles, including self collisions.
This compares an implementation optimization against the generic AVBD core. It
does not establish absolute physical accuracy or a speed advantage over Newton.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("DUCK_RUN_DIR", "runs")) / "microduck-specialized",
    )
    p.add_argument("--model-dir", type=Path, action="append", default=[])
    p.add_argument("--precision", choices=["both", "fp32", "fp64"], default="both")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--validation-envs", type=int, default=8)
    p.add_argument("--validation-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--atol-fp32", type=float, default=1e-6)
    p.add_argument("--rtol-fp32", type=float, default=1e-6)
    p.add_argument("--atol-fp64", type=float, default=1e-12)
    p.add_argument("--rtol-fp64", type=float, default=1e-12)
    p.add_argument("--skip-timing", action="store_true")
    p.add_argument("--shared-memory", action="store_true")
    p.add_argument("--timing-precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--envs", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--chunk-steps", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--warmup-seconds", type=float, default=10.0)
    a = p.parse_args()
    positive = [
        a.dt,
        a.iterations,
        a.validation_steps,
        a.steps,
        a.chunk_steps,
        a.repetitions,
        a.warmup_steps,
    ]
    if (
        min(positive) <= 0
        or a.validation_envs < 3
        or any(n < 1 or n > 65536 for n in a.envs)
    ):
        p.error(
            "Positive step counts and dt, at least 3 validation envs, and 1..65536 timing envs required"
        )
    if a.steps % a.chunk_steps:
        p.error("--steps must be divisible by --chunk-steps")
    if min(a.atol_fp32, a.rtol_fp32, a.atol_fp64, a.rtol_fp64, a.warmup_seconds) < 0:
        p.error("Tolerances and warmup seconds must be nonnegative")
    if not a.skip_timing and a.precision not in ("both", a.timing_precision):
        p.error("The timing precision must also be validated")
    return a


def save(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def gpu_info():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def exclusive():
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in physical:
        raise RuntimeError("Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            physical,
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    others = [
        int(s)
        for s in rows.splitlines()
        if s.strip().isdigit() and int(s) != os.getpid()
    ]
    if others:
        raise RuntimeError(f"Competing GPU processes: {others}")


def model_spec(folder, prepared=None):
    """Normalize metadata without changing physical parameters or source assets."""
    import mujoco
    from model_io import inv, rot

    if prepared is not None:
        meta = prepared
        model = folder / "microduck.duck"
        visual = mujoco.MjModel.from_xml_path(str(folder / "visual.xml"))
        offset = rot(inv(visual.body_iquat[1]), visual.body_ipos[1]).tolist()
        metadata_path = folder / "metadata.json"
        scope = "Original complete standing model; ground contacts only"
    else:
        metadata_path = folder / "config.json"
        cfg = json.loads(metadata_path.read_text())
        model = folder / "model.duck"
        meta = dict(joint_names=cfg["joint_names"], home=cfg["default_qpos"][7:])
        # This is an interface-parity test with diagnostic PD inputs, not a BAM
        # policy replay. The torque cap comes from this model's own configuration.
        meta.update(kp=0.55, torque_limit=cfg["force_limit"])
        offset = cfg["root_com_offset"]
        scope = cfg.get("scope", "Provided official model")
    return dict(
        model=str(model.resolve()),
        metadata=str(metadata_path.resolve()),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        home=meta["home"],
        joint_names=meta["joint_names"],
        kp=meta["kp"],
        torque_limit=meta["torque_limit"],
        root_offset=offset,
        scope=scope,
        contact_response_scope="All collision pairs present in the unmodified model",
    )


class Parity:
    def __init__(self, precision, args):
        self.atol = getattr(args, "atol_" + precision)
        self.rtol = getattr(args, "rtol_" + precision)
        self.max_error = {}
        self.comparisons = 0
        self.physical_max = {}
        self.ground_seen = False
        self.self_seen = False

    def compare(self, name, generic, specialized, exact=False):
        import torch

        if generic.shape != specialized.shape:
            raise AssertionError(
                f"{name}: shape mismatch {generic.shape}, {specialized.shape}"
            )
        if not (
            bool(torch.isfinite(generic).all())
            and bool(torch.isfinite(specialized).all())
        ):
            raise AssertionError(f"{name}: nonfinite values")
        delta = (generic - specialized).abs()
        error = float(delta.max()) if delta.numel() else 0.0
        self.max_error[name] = max(self.max_error.get(name, 0.0), error)
        self.comparisons += 1
        tolerance = 0.0 if exact else self.atol + self.rtol * generic.abs()
        if bool((delta > tolerance).any()):
            raise AssertionError(
                f"{name}: max absolute error {error:.12g}, atol={0 if exact else self.atol}, rtol={0 if exact else self.rtol}"
            )

    def states(self, pair, label, allowed_failed=()):
        import torch

        samples = [batch.state() for batch in pair]
        for name, section in (
            ("position_m", slice(0, 3)),
            ("quaternion", slice(3, 7)),
            ("velocity_m_s", slice(7, 10)),
            ("angular_velocity_rad_s", slice(10, 13)),
        ):
            self.compare(
                "body_" + name, samples[0][0][..., section], samples[1][0][..., section]
            )
        self.compare("joint_angle_rad", samples[0][1][..., 0], samples[1][1][..., 0])
        self.compare(
            "joint_velocity_rad_s", samples[0][1][..., 1], samples[1][1][..., 1]
        )
        self.compare(
            "diagnostics_continuous",
            samples[0][2][:, [0, 1, 4]],
            samples[1][2][:, [0, 1, 4]],
        )
        self.compare(
            "diagnostics_discrete",
            samples[0][2][:, [2, 3, 5]],
            samples[1][2][:, [2, 3, 5]],
            exact=True,
        )
        for which, sample in zip(("generic", "specialized"), samples):
            failures = tuple(
                torch.nonzero(sample[2][:, 5], as_tuple=False).flatten().cpu().tolist()
            )
            if failures != tuple(allowed_failed):
                raise AssertionError(
                    f"{label}/{which}: unexpected failed envs {failures}; expected {allowed_failed}"
                )
            for key, index in (
                ("joint_anchor_m", 0),
                ("penetration_m", 1),
                ("axis_error_rad", 4),
            ):
                tag = which + "_" + key
                self.physical_max[tag] = max(
                    self.physical_max.get(tag, 0.0), float(sample[2][:, index].max())
                )
        return samples

    def outputs(self, pair, offsets):
        for method, kwargs in (
            ("contact_forces", {}),
            ("contact_forces", {"ground_only": True}),
            ("self_contact_stats", {}),
            ("contact_counts", {}),
        ):
            tensors = [getattr(batch, method)(**kwargs) for batch in pair]
            tag = method + ("_ground" if kwargs else "")
            if method in ("contact_counts", "self_contact_stats"):
                self.compare(
                    tag + "_count", tensors[0][:, 0], tensors[1][:, 0], exact=True
                )
                self.compare(
                    tag, tensors[0], tensors[1], exact=method == "contact_counts"
                )
            else:
                self.compare(tag, tensors[0], tensors[1])
            if kwargs:
                self.ground_seen |= bool(tensors[0].abs().max() > 1e-6)
            if method == "self_contact_stats":
                self.self_seen |= bool(tensors[0][:, 0].any())
        loads = [batch.generalized_loads(offsets) for batch in pair]
        self.compare("generalized_loads", *loads)

    def report(self):
        return dict(
            atol=self.atol,
            rtol=self.rtol,
            comparisons=self.comparisons,
            maximum_absolute_errors=self.max_error,
            physical_diagnostics_maxima=self.physical_max,
            active_ground_response_seen=self.ground_seen,
            active_self_response_seen=self.self_seen,
        )


def make_pair(ext, spec, n, precision, a):
    pair = [
        ext.CudaBatch(
            spec["model"], n, a.dt, a.iterations, 0, precision == "fp64", specialized=s,
            shared_memory=s and a.shared_memory,
        )
        for s in (False, True)
    ]
    assert list(pair[0].joint_names) == spec["joint_names"] == list(pair[1].joint_names)
    return pair


def validate(ext, spec, precision, a):
    import numpy as np
    import torch

    start = time.perf_counter()
    dtype = torch.float64 if precision == "fp64" else torch.float32
    n, nj = a.validation_envs, len(spec["home"])
    pair = make_pair(ext, spec, n, precision, a)
    home = (
        torch.tensor(spec["home"], dtype=dtype, device="cuda")
        .expand(n, nj)
        .contiguous()
    )
    zeros = torch.zeros(n, 3, dtype=dtype, device="cuda")
    roots = torch.zeros(n, 6, dtype=dtype, device="cuda")
    offsets = (
        torch.tensor(spec["root_offset"], dtype=dtype, device="cuda")
        .expand(n, 3)
        .contiguous()
    )
    mask = torch.ones(n, dtype=torch.bool, device="cuda")
    rng = np.random.default_rng(a.seed)
    q_random = home + torch.tensor(
        rng.uniform(-0.01, 0.01, (n, nj)), dtype=dtype, device="cuda"
    )
    roots_random = roots.clone()
    roots_random[:, :2] = torch.tensor(
        rng.uniform(-0.005, 0.005, (n, 2)), dtype=dtype, device="cuda"
    )
    roots_random[:, 2] = torch.tensor(
        rng.uniform(0.001, 0.004, n), dtype=dtype, device="cuda"
    )
    roots_random[:, 3:] = torch.tensor(
        rng.uniform(-0.01, 0.01, (n, 3)), dtype=dtype, device="cuda"
    )
    phase = torch.tensor(rng.uniform(0, 2 * np.pi, (n, nj)), dtype=dtype, device="cuda")
    monitor = Parity(precision, a)
    cases = []

    def observe(label, failures=()):
        result = monitor.states(pair, label, failures)
        if not failures:
            monitor.outputs(pair, offsets)
        return result

    observe("constructor")
    for name in ("standing", "random_motion_push", "raised_release", "step_motor"):
        qr = q_random if name in ("random_motion_push", "step_motor") else home
        rr = (
            roots_random.clone()
            if name in ("random_motion_push", "step_motor")
            else roots.clone()
        )
        if name == "raised_release":
            rr[:, 2] = 0.01
        for batch in pair:
            batch.reset(mask, qr, rr)
        observe(name + "/reset")
        for step in range(a.validation_steps):
            sim_time = step * a.dt
            target = home
            force = zeros.clone()
            if name in ("random_motion_push", "step_motor"):
                target = home + 0.02 * torch.sin(2 * np.pi * sim_time + phase)
                force[:, 0] = 0.02 * np.sin(2 * np.pi * sim_time)
                force[:, 1] = 0.015 * np.cos(2 * np.pi * sim_time)
            if name == "step_motor":
                # Derive ONE input from generic state and give the identical
                # tensor to both backends; controller divergence cannot hide or
                # manufacture a physics difference.
                joint = pair[0].state()[1]
                motor = torch.zeros(n, nj, 3, dtype=dtype, device="cuda")
                motor[..., 0] = (
                    spec["kp"] * (target - joint[..., 0]) - 0.01 * joint[..., 1]
                ).clamp(-spec["torque_limit"], spec["torque_limit"])
                motor[..., 1] = 0.005
                motor[..., 2] = 0.005
                for batch in pair:
                    batch.step_motor(motor, force)
            else:
                for batch in pair:
                    batch.step(target, force, 1, spec["kp"], spec["torque_limit"])
            observe(f"{name}/{step + 1}")
        cases.append(
            dict(
                name=name,
                physics_steps=a.validation_steps,
                duration_seconds=a.validation_steps * a.dt,
                every_physics_step_checked=True,
            )
        )
        print(
            f"parity {Path(spec['model']).parent.name} {precision} {name}: "
            f"{a.validation_steps} physics steps checked",
            flush=True,
        )
    if not monitor.ground_seen:
        raise AssertionError(
            "No actual ground contact response exercised; increase --validation-steps"
        )

    # Partial reset must preserve every nonselected environment bit for bit.
    before = observe("before_partial_reset")
    partial = torch.arange(n, device="cuda") % 2 == 0
    for batch in pair:
        batch.reset(partial, home, roots)
    after = observe("after_partial_reset")
    for which in range(2):
        for field in range(3):
            torch.testing.assert_close(
                after[which][field][~partial],
                before[which][field][~partial],
                atol=0,
                rtol=0,
            )
    fresh = make_pair(ext, spec, n, precision, a)
    for which, batch in enumerate(fresh):
        batch.reset(mask, home, roots)
        initial = batch.state()
        for field in range(3):
            torch.testing.assert_close(
                after[which][field][partial], initial[field][partial], atol=0, rtol=0
            )
    del fresh

    # The persistent-state event must serialize update/readback across streams.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for batch in pair:
            batch.step(home, zeros, 3, spec["kp"], spec["torque_limit"])
    observe("cross_stream")
    fault_id = 1
    bad_motor = torch.zeros(n, nj, 3, dtype=dtype, device="cuda")
    bad_motor[fault_id, 0, 0] = float("nan")
    for batch in pair:
        batch.step_motor(bad_motor, zeros)
    observe("nan_motor_isolation", (fault_id,))
    recover = torch.zeros(n, dtype=torch.bool, device="cuda")
    recover[fault_id] = True
    for batch in pair:
        batch.reset(recover, home, roots)
        batch.step(home, zeros, 5, spec["kp"], spec["torque_limit"])
    observe("nan_recovery")

    # Both CPU references begin from the file's state, irrespective of current
    # CUDA state. Strict host-host parity is a separate gate; CPU/GPU errors are
    # reported separately because contact reduction and libm differ by backend.
    cpu_steps = min(10, a.validation_steps)
    cpu_target = home[0].cpu().contiguous()
    cpu_force = zeros[0].cpu().contiguous()
    references = [
        batch.reference(
            cpu_target, cpu_force, cpu_steps, spec["kp"], spec["torque_limit"]
        )
        for batch in pair
    ]
    for field in range(3):
        monitor.compare(
            "cpu_reference_" + str(field), references[0][field], references[1][field]
        )
    pristine = make_pair(ext, spec, n, precision, a)
    gpu_cpu = []
    for which, batch in enumerate(pristine):
        batch.step(home, zeros, cpu_steps, spec["kp"], spec["torque_limit"])
        samples = batch.state()
        gpu_cpu.append(
            {
                name: float((value[0].cpu() - expected).abs().max())
                for name, value, expected in zip(
                    ("body", "joint", "diagnostics"), samples, references[which]
                )
            }
        )
    monitor.states(pristine, "reference_chunk")
    return dict(
        precision=precision,
        model=spec["model"],
        num_envs=n,
        cases=cases,
        physics_step_calls_per_env=4 * a.validation_steps + 9,
        successful_physics_steps_by_env=[
            4 * a.validation_steps + 9 - int(env == fault_id) for env in range(n)
        ],
        extra_reference_steps_per_env=cpu_steps,
        cpu_reference_steps=cpu_steps,
        cpu_gpu_difference_diagnostic_only=gpu_cpu,
        cpu_host_host_parity_required=True,
        boundary_checks=[
            "partial_reset_preserves_unselected",
            "reset_matches_fresh",
            "cross_stream",
            "nan_motor_isolation",
            "reset_recovers_failed_environment",
        ],
        parity=monitor.report(),
        wall_seconds=time.perf_counter() - start,
        passed=True,
    )


def benchmark(ext, spec, n, a):
    import torch

    exclusive()
    dtype = torch.float64 if a.timing_precision == "fp64" else torch.float32
    pair = make_pair(ext, spec, n, a.timing_precision, a)
    target = (
        torch.tensor(spec["home"], dtype=dtype, device="cuda")
        .expand(n, -1)
        .contiguous()
    )
    force = torch.zeros(n, 3, dtype=dtype, device="cuda")
    roots = torch.zeros(n, 6, dtype=dtype, device="cuda")
    mask = torch.ones(n, dtype=torch.bool, device="cuda")
    monitor = Parity(a.timing_precision, a)

    def rollout(batch):
        for _ in range(a.steps // a.chunk_steps):
            batch.step(target, force, a.chunk_steps, spec["kp"], spec["torque_limit"])

    def trial(batch):
        batch.reset(mask, target, roots)
        torch.cuda.synchronize()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start = time.perf_counter()
        begin.record()
        rollout(batch)
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / 1000, time.perf_counter() - start

    names = ("generic", "specialized")
    results = {
        name: dict(gpu_seconds=[], synchronized_wall_seconds=[]) for name in names
    }
    for name, batch in zip(names, pair):
        trial(batch)  # First launch is never used as warm throughput.
        steps = 0
        elapsed = 0.0
        started = time.perf_counter()
        while steps < a.warmup_steps or elapsed < a.warmup_seconds:
            seconds, _ = trial(batch)
            elapsed += seconds
            steps += a.steps
        results[name].update(
            warmup_steps=steps,
            warmup_gpu_seconds=elapsed,
            warmup_wall_seconds=time.perf_counter() - started,
        )
    clocks_before = gpu_info()
    for repetition in range(a.repetitions):
        for which in (0, 1) if repetition % 2 == 0 else (1, 0):
            exclusive()
            seconds, wall = trial(pair[which])
            results[names[which]]["gpu_seconds"].append(seconds)
            results[names[which]]["synchronized_wall_seconds"].append(wall)
        monitor.states(pair, f"timing_{n}/trial_{repetition}")
        exclusive()
    for result in results.values():
        median = statistics.median(result["gpu_seconds"])
        result.update(
            median_gpu_seconds=median,
            env_steps_per_second=n * a.steps / median,
            timing_cv=statistics.pstdev(result["gpu_seconds"])
            / statistics.mean(result["gpu_seconds"]),
        )
    ratio = (
        results["specialized"]["env_steps_per_second"]
        / results["generic"]["env_steps_per_second"]
    )
    print(
        f"timing model={Path(spec['model']).parent.name} envs={n}: specialized/generic={ratio:.3f}",
        flush=True,
    )
    return dict(
        model=spec["model"],
        model_sha256=spec["model_sha256"],
        precision=a.timing_precision,
        num_envs=n,
        physics_steps_per_trial=a.steps,
        chunk_steps=a.chunk_steps,
        repetitions=a.repetitions,
        dt=a.dt,
        iterations=a.iterations,
        results=results,
        specialized_over_generic=ratio,
        parity=monitor.report(),
        timed_scope="Native step with fixed HOME targets; reset, diagnostics and host readback excluded",
        diagnostics_scope="Parity after each trial; accumulated diagnostics cover final chunk only",
        gpu_before=clocks_before,
        gpu_after=gpu_info(),
    )


def main():
    a = arguments()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        passed=False,
        validation=[],
        timing=[],
        arguments={
            k: str(v)
            if isinstance(v, Path)
            else [str(p) for p in v]
            if k == "model_dir"
            else v
            for k, v in vars(a).items()
        },
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version,
        command=sys.argv,
        seed=a.seed,
        model_specs=[],
    )
    path = a.output / "report.json"
    save(path, report)
    try:
        if platform.system() != "Linux":
            raise RuntimeError(
                "CUDA validation must run on the configured remote Linux host"
            )
        import torch
        from build_cuda import build
        from prepare_standing import prepare

        exclusive()
        ext = build()
        report.update(
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            native_binary_sha256=hashlib.sha256(
                Path(ext.__file__).read_bytes()
            ).hexdigest(),
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__).resolve(),
                    *sorted((ROOT / "src/cuda").glob("*.cu")),
                    *sorted((ROOT / "src/cuda").glob("*.cuh")),
                    *sorted((ROOT / "src/cuda").glob("*.hpp")),
                ]
            },
        )
        folder = a.output / "standing-model"
        specs = [model_spec(folder, prepare(folder))]
        specs.extend(model_spec(folder) for folder in a.model_dir)
        report["model_specs"] = specs
        precisions = ["fp32", "fp64"] if a.precision == "both" else [a.precision]
        for spec in specs:
            for precision in precisions:
                row = validate(ext, spec, precision, a)
                report["validation"].append(row)
                save(path, report)
                print(
                    f"validated {spec['model']} {precision}: max errors {row['parity']['maximum_absolute_errors']}",
                    flush=True,
                )
        report["validation_passed"] = True
        save(path, report)
        if not a.skip_timing:
            # All requested models/precisions must pass before any reported
            # performance result is collected.
            for spec in specs:
                for n in a.envs:
                    report["timing"].append(benchmark(ext, spec, n, a))
                    save(path, report)
        report.update(status="completed", passed=True)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(path, report)
    print(path, flush=True)


if __name__ == "__main__":
    main()
