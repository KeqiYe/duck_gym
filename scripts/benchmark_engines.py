"""Warm GPU physics-only comparison: native AVBD, MuJoCo Warp, and Newton.

Run through scripts/remote/run.py, with the isolated benchmark interpreter.
No policy, reward, observation construction, rendering, or timed host readback.
Each engine/env count runs in a fresh process on the same selected GPU.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument(
        "--engines", nargs="+", choices=["avbd", "mjwarp", "newton", "newton-avbd"], default=["avbd", "mjwarp"]
    )
    p.add_argument("--physics-profile", choices=("original", "avbd-common"), default="original",
                   help="avbd-common explicitly removes armature, joint dry friction and motor effort caps in both engines")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--chunk-steps", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--warmup-seconds", type=float, default=10.0)
    p.add_argument("--avbd-iterations", type=int, default=50)
    p.add_argument("--avbd-specialized", action="store_true", help="Use compact MicroDuck backend")
    p.add_argument("--avbd-shared-memory", action="store_true")
    p.add_argument("--newton-avbd-deterministic", action="store_true",
                   help="Diagnostic only: deterministic collision ordering and solver atomic reductions")
    p.add_argument("--newton-avbd-contact-history", action="store_true",
                   help="Newton contact warm start with latest contact matching")
    p.add_argument("--newton-avbd-verify-buffers", action="store_true",
                   help="Add the optional collision buffer diagnostic kernel inside every timed step")
    p.add_argument("--newton-avbd-replay-policy", choices=("strict", "report"), default="strict",
                   help="report preserves failed numerical repeatability checks for stock nondeterministic Newton; finite, overflow and exact reset checks still gate results")
    p.add_argument("--mjwarp-iterations", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--nconmax", type=int, default=32)
    p.add_argument("--njmax", type=int, default=128)
    p.add_argument("--wait-for-gpu-seconds", type=float, default=0)
    p.add_argument("--worker-engine", choices=["avbd", "mjwarp", "newton", "newton-avbd"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("DUCK_RUN_DIR", "runs")) / "engine-benchmark",
    )
    a = p.parse_args()
    selected = [a.worker_engine] if a.worker_engine else a.engines
    if "newton-avbd" in selected and a.physics_profile != "avbd-common":
        p.error("Newton AVBD requires the explicit --physics-profile avbd-common; original MicroDuck features are unsupported")
    if a.newton_avbd_deterministic and a.newton_avbd_replay_policy != "strict":
        p.error("Deterministic Newton diagnostics require strict replay checks")
    if a.physics_profile == "avbd-common" and any(e not in ("avbd", "newton-avbd") for e in selected):
        p.error("avbd-common is currently implemented only for avbd and newton-avbd")
    if any(n < 1 or n > 4096 for n in a.envs):
        p.error("Supported benchmark env range is 1..4096")
    if (
        min(
            a.steps,
            a.chunk_steps,
            a.repetitions,
            a.warmup_steps,
            a.avbd_iterations,
            a.mjwarp_iterations,
        )
        < 1
    ):
        p.error("Step, repetition, and iteration counts must be positive")
    if a.steps % a.chunk_steps or a.warmup_steps % a.chunk_steps:
        p.error("Steps and warmup steps must be divisible by chunk steps")
    if any(e in ("newton", "newton-avbd") for e in selected) and a.chunk_steps % 2:
        p.error("Newton ping-pong graph requires an even chunk size")
    if a.warmup_seconds < 0 or a.dt <= 0:
        p.error("Invalid duration/time step")
    return a


def dump(path, value):
    def finite_json(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite_json(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [finite_json(val) for val in item]
        return item
    path.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")


def gpu_info():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def assert_gpu_exclusive():
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    text = subprocess.check_output(
        ["nvidia-smi", "-i", physical, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    )
    others = [int(line.strip()) for line in text.splitlines() if line.strip().isdigit()]
    others = [pid for pid in others if pid != os.getpid()]
    if others:
        raise RuntimeError(f"Selected GPU has competing compute processes: {others}")


def worker(a):
    assert_gpu_exclusive()
    started = time.perf_counter()
    import numpy as np
    import torch
    import mujoco
    from prepare_standing import prepare

    phases = {"imports_seconds": time.perf_counter() - started}
    t = time.perf_counter()
    torch.set_num_threads(1)
    torch.cuda.init()
    # CUDA graph capture cannot use the legacy null stream. Keep both stream
    # wrappers alive, and place engine calls and timing events on this stream.
    torch_stream = torch.cuda.Stream()
    torch.cuda.set_stream(torch_stream)
    torch.cuda.synchronize()
    phases["cuda_context_seconds"] = time.perf_counter() - t
    n = a.envs[0]
    folder = a.output / f"{a.worker_engine}-{n}"
    folder.mkdir(parents=True, exist_ok=False)
    t = time.perf_counter()
    meta = prepare(folder / "model", physics_profile=a.physics_profile)
    phases["asset_compile_export_seconds"] = time.perf_counter() - t
    dtype = torch.float32
    settings = dict(
        dt=a.dt,
        precision="fp32",
        ground_only=True,
        self_collision=False,
        control="fixed upstream HOME targets; kp preserved; effort cap according to explicit physics profile; no PPO",
        kp=meta["kp"],
        torque_limit_nm=meta["torque_limit"] if meta["torque_limit_enabled"] else None,
        torque_limit_enabled=meta["torque_limit_enabled"],
        physics_profile=a.physics_profile,
        original_joint_physics=meta["original_joint_physics"],
        effective_joint_physics=meta["effective_joint_physics"],
    )
    graph = None
    t = time.perf_counter()
    if a.worker_engine == "avbd":
        from build_cuda import build

        # Start with a fresh per-run build directory, then reuse it for larger
        # batches. This exposes cold C++ compilation instead of comparing a
        # prebuilt AVBD binary against MJWarp's first uncached JIT compilation.
        build_dir = Path(os.environ["DUCK_CUDA_BUILD"]).parent / (
            "benchmark-" + a.output.parent.name
        )
        build_cache_files = (
            sum(p.is_file() for p in build_dir.rglob("*")) if build_dir.exists() else 0
        )
        os.environ["DUCK_CUDA_BUILD"] = str(build_dir)
        ext = build()
        phases["engine_import_build_load_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        sim = ext.CudaBatch(
            str(folder / "model/microduck.duck"), n, a.dt, a.avbd_iterations, 0, False,
            specialized=a.avbd_specialized, shared_memory=a.avbd_shared_memory,
        )
        q = torch.tensor(meta["home"], device="cuda", dtype=dtype).repeat(n, 1)
        f = torch.zeros(n, 3, device="cuda", dtype=dtype)
        mask = torch.ones(n, device="cuda", dtype=torch.bool)
        roots = torch.zeros(n, 6, device="cuda", dtype=dtype)

        def reset():
            sim.reset(mask, q, roots)

        def step_chunk():
            sim.step(q, f, a.chunk_steps, meta["kp"], meta["torque_limit"])

        def step_one():
            sim.step(q, f, 1, meta["kp"], meta["torque_limit"])

        def diagnostics():
            b, j, d = sim.state()
            return dict(
                finite=bool(torch.isfinite(b).all() and torch.isfinite(j).all()),
                numeric_failure_envs=int((d[:, 5] != 0).sum()),
                max_joint_anchor_error_m=float(d[:, 0].max()),
                max_penetration_m=float(d[:, 1].max()),
                max_contacts=int(d[:, 2].max()),
                all_last_chunk_substeps_converged=bool(d[:, 3].all()),
                max_axis_error_rad=float(d[:, 4].max()),
                axis_error_note="Legacy field name: this is the hinge axis orthogonality residual, dimensionless, not a measured angle",
                min_root_height_m=float(b[:, 1, 2].min()),
                max_root_height_m=float(b[:, 1, 2].max()),
                root_com_first_env=b[0, 1, :3].cpu().tolist(),
            )

        def snapshot():
            b, j, _ = sim.state()
            return {"body": b.cpu().numpy(), "joint": j.cpu().numpy()}

        binary_hash = hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()
        shutil.copy2(ext.__file__, folder / Path(ext.__file__).name)
        settings.update(
            solver="rigid AVBD maximal coordinates",
            microduck_specialized=a.avbd_specialized,
            shared_memory=a.avbd_shared_memory,
            iterations=a.avbd_iterations,
            launch="one native kernel per chunk; internal substep loop",
            implicit_pd=True,
            extension_build_directory=str(build_dir),
            extension_cache_files_before=build_cache_files,
            extension_cache=("Existing cache; builder checks incremental compilation" if build_cache_files
                             else "New per-run directory; first native compilation"),
        )
    else:
        import warp as wp
        import mujoco_warp as mjw

        cache = a.output / (a.worker_engine + "-warp-kernel-cache")
        cache_files = sum(p.is_file() for p in cache.rglob("*")) if cache.exists() else 0
        wp.config.kernel_cache_dir = str(cache)
        wp.init()
        warp_stream = wp.stream_from_torch(torch_stream)
        wp.set_stream(warp_stream)
        phases["engine_import_runtime_seconds"] = time.perf_counter() - t
        settings.update(warp_cache_files_before=cache_files, warp_cache=str(cache))
        t = time.perf_counter()
        import xml.etree.ElementTree as ET

        xml = ET.fromstring((folder / "model/visual.xml").read_text())
        if a.worker_engine == "newton-avbd":
            # Several upstream collision geoms are unnamed. Give the same
            # temporary names to the compiled reference and Newton import;
            # this changes neither the source assets nor their geometry.
            geoms = list(xml.find("worldbody").iter("geom"))
            names = {g.get("name") for g in geoms if g.get("name")}
            for index, geom in enumerate(geoms):
                if not geom.get("name"):
                    name = f"avbd_benchmark_geom_{index}"
                    if name in names:
                        raise ValueError("Benchmark geom name collides with an authored name")
                    geom.set("name", name)
                    names.add(name)
        actuator = ET.SubElement(xml, "actuator")
        for name, low, high in zip(meta["joint_names"], meta["lower"], meta["upper"]):
            ET.SubElement(
                actuator,
                "position",
                name="hold_" + name,
                joint=name,
                kp=str(meta["kp"]),
                **({"forcelimited": "true", "forcerange": f'-{meta["torque_limit"]} {meta["torque_limit"]}'}
                   if meta["torque_limit_enabled"] else {"forcelimited": "false"}),
                ctrllimited="true",
                ctrlrange=f"{low} {high}",
            )
        xml.find("option").set("timestep", str(a.dt))
        xml.find("option").set("iterations", str(a.mjwarp_iterations))
        xml.find("option").set("solver", "Newton")
        xml.find("option").set("integrator", "Euler")
        text = ET.tostring(xml, encoding="unicode")
        (folder / "model/benchmark.xml").write_text(text)
        mjm = mujoco.MjModel.from_xml_string(text)
        mjd = mujoco.MjData(mjm)
        mjd.qpos[:] = meta["qpos"]
        mjd.ctrl[:] = meta["home"]
        mujoco.mj_forward(mjm, mjd)
        if a.worker_engine == "newton-avbd":
            from benchmark_newton_avbd import setup_newton_avbd

            adapter = setup_newton_avbd(a, folder, meta, mjm)
            reset, step_one, diagnostics = adapter.reset, adapter.step, adapter.diagnostics
            def snapshot():
                return {"body_q": adapter.states[0].body_q.numpy(),
                        "body_qd": adapter.states[0].body_qd.numpy()}
            settings.update(adapter.settings)
            binary_hash = None

            def step_chunk():
                if graph is None:
                    for _ in range(a.chunk_steps):
                        step_one()
                else:
                    wp.capture_launch(graph)
        else:
            if a.worker_engine == "newton":
                from benchmark_newton import setup_newton

                adapter = setup_newton(a, folder, meta, mjm)
                m, d = adapter.solver.mjw_model, adapter.solver.mjw_data
                mjm, mjd = adapter.solver.mj_model, adapter.solver.mj_data
                settings.update(adapter.settings)
            else:
                m = mjw.put_model(mjm)
                d = mjw.put_data(mjm, mjd, nworld=n, nconmax=a.nconmax, njmax=a.njmax)
            initial_q = wp.array(np.tile(mjd.qpos, (n, 1)).astype(np.float32), dtype=float)
            initial_ctrl = wp.array(np.tile(mjd.ctrl, (n, 1)).astype(np.float32), dtype=float)

            def reset():
                if a.worker_engine == "newton":
                    adapter.reset()
                else:
                    mjw.reset_data(m, d)
                    wp.copy(d.qpos, initial_q)
                    wp.copy(d.ctrl, initial_ctrl)

            def step_one():
                if a.worker_engine == "newton":
                    adapter.step()
                else:
                    mjw.step(m, d)

            def step_chunk():
                if graph is None:
                    for _ in range(a.chunk_steps):
                        step_one()
                else:
                    wp.capture_launch(graph)

            def diagnostics():
                qpos, qvel = d.qpos.numpy(), d.qvel.numpy()
                nacon = int(d.nacon.numpy()[0])
                distances = d.contact.dist.numpy()[: min(nacon, n * a.nconmax)]
                mjd.qpos[:] = qpos[0]
                mjd.qvel[:] = qvel[0]
                mujoco.mj_forward(mjm, mjd)
                return dict(
                    finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
                    overflow_flags=d.overflow.numpy().tolist(),
                    max_penetration_m=float(max(0, -distances.min())) if len(distances) else 0.0,
                    max_joint_anchor_error_m=0.0,
                    anchor_error_note="Reduced coordinates enforce hinge anchors structurally; not an iterative residual",
                    contact_count_total=nacon,
                    max_constraints=int(d.nefc.numpy().max()),
                    max_newton_iterations=int(d.solver_niter.numpy().max()),
                    min_root_height_m=float(qpos[:, 2].min()),
                    max_root_height_m=float(qpos[:, 2].max()),
                    root_com_first_env=mjd.xipos[1].tolist(),
                )

            binary_hash = None
            settings.update(
                solver="MuJoCo Newton reduced coordinates",
                iterations=a.mjwarp_iterations,
                tolerance=float(mjm.opt.tolerance),
                ls_iterations=int(mjm.opt.ls_iterations),
                integrator="Euler",
                implicit_pd=False,
                nconmax=a.nconmax,
                njmax=a.njmax,
                launch=f"CUDA Graph containing {a.chunk_steps} full mjw.step calls",
                graph_conditional=bool(m.opt.graph_conditional),
                note="Same physical coefficients; implicit AVBD motor vs MuJoCo actuator discretization differs",
            )
    torch.cuda.synchronize()
    phases["model_data_control_allocation_seconds"] = time.perf_counter() - t
    phases["pre_first_step_total_seconds"] = time.perf_counter() - started

    t = time.perf_counter()
    reset()
    step_chunk()
    torch.cuda.synchronize()
    phases["first_reset_and_chunk_compile_seconds"] = time.perf_counter() - t
    dump(folder / "initialization.json", phases)
    t = time.perf_counter()
    if a.worker_engine in ("mjwarp", "newton", "newton-avbd"):
        with wp.ScopedCapture() as capture:
            for _ in range(a.chunk_steps):
                step_one()
        graph = capture.graph
        wp.capture_launch(graph)
        wp.synchronize()
    phases["graph_capture_and_first_launch_seconds"] = time.perf_counter() - t
    print(f"{a.worker_engine} n={n}: initialized, first use and graph ready", flush=True)
    preflight = None
    replay_check = None
    repeat_check = None
    if a.physics_profile == "avbd-common":
        # Inspect every physical step outside timing, not only a plausible
        # final pose after an invalid/overflowed intermediate state.
        t = time.perf_counter()
        reset()
        initial_state = snapshot()
        def verify_reset():
            actual = snapshot()
            if any(v.tobytes() != actual[k].tobytes() for k, v in initial_state.items()):
                raise RuntimeError("Reset did not restore identical physical state")

        def compare_states(reference, actual):
            result = {
                k: {"bitwise_equal": v.tobytes() == actual[k].tobytes(),
                    "allclose_rtol_1e_5_atol_1e_7": bool(np.allclose(v, actual[k], rtol=1e-5, atol=1e-7)),
                    "max_abs_difference": float(np.max(np.abs(v.astype(np.float64) - actual[k])))}
                for k, v in reference.items()
            }
            if a.worker_engine == "newton-avbd":
                dq = reference["body_q"].astype(np.float64) - actual["body_q"]
                dv = reference["body_qd"].astype(np.float64) - actual["body_qd"]
                result["body_q"].update(max_position_vector_difference_m=float(np.linalg.norm(dq[:, :3], axis=1).max()),
                                       max_quaternion_component_difference=float(np.abs(dq[:, 3:]).max()))
                result["body_qd"].update(max_linear_velocity_vector_difference_m_s=float(np.linalg.norm(dv[:, :3], axis=1).max()),
                                        max_angular_velocity_vector_difference_rad_s=float(np.linalg.norm(dv[:, 3:], axis=1).max()))
            return result

        def enforce_repeatability(check, label):
            passed = all(row["allclose_rtol_1e_5_atol_1e_7"] for row in check.values())
            if not passed:
                if a.worker_engine == "newton-avbd" and a.newton_avbd_replay_policy == "report":
                    print(f"WARNING: {label} exceeds strict repeatability tolerance; recorded, not accepted as exact replay", flush=True)
                else:
                    raise RuntimeError(f"{label} differs: {check}")

        preflight = []
        for physical_step in range(a.steps):
            step_one()
            torch.cuda.synchronize()
            check = diagnostics()
            if (not check["finite"] or check.get("numeric_failure_envs", 0)
                    or any(check.get("overflow_flags", []))):
                dump(folder / "failed-preflight.json", {"step": physical_step + 1, "check": check})
                raise RuntimeError(f"AVBD preflight failed at step {physical_step + 1}: {check}")
            preflight.append({"step": physical_step + 1, **check})
        direct_final = snapshot()
        np.savez_compressed(folder / "direct-final-state.npz", **direct_final)
        dump(folder / "preflight.json", preflight)
        reset()
        verify_reset()
        for _ in range(a.steps):
            step_one()
        torch.cuda.synchronize()
        direct_repeat = snapshot()
        repeat_diagnostics = diagnostics()
        dump(folder / "direct-repeat-diagnostics.json", repeat_diagnostics)
        if (not repeat_diagnostics["finite"] or repeat_diagnostics.get("numeric_failure_envs", 0)
                or any(repeat_diagnostics.get("overflow_flags", []))):
            raise RuntimeError("Invalid direct-repeat final state")
        np.savez_compressed(folder / "direct-repeat-final-state.npz", **direct_repeat)
        repeat_check = compare_states(direct_final, direct_repeat)
        dump(folder / "direct-repeat-comparison.json", repeat_check)
        enforce_repeatability(repeat_check, "Direct/direct repeat")
        reset()
        verify_reset()
        chunk_checks = []
        for _ in range(a.steps // a.chunk_steps):
            step_chunk()
            torch.cuda.synchronize()
            check = diagnostics()
            chunk_checks.append(check)
            if not check["finite"] or check.get("numeric_failure_envs", 0) or any(check.get("overflow_flags", [])):
                dump(folder / "failed-chunk-preflight.json", chunk_checks)
                raise RuntimeError("Invalid chunk preflight state")
        dump(folder / "chunk-preflight.json", chunk_checks)
        torch.cuda.synchronize()
        graph_final = snapshot()
        np.savez_compressed(folder / "chunk-final-state.npz", **graph_final)
        replay_check = compare_states(direct_final, graph_final)
        dump(folder / "direct-chunk-comparison.json", replay_check)
        # This is an execution-consistency bound, not a physical accuracy
        # acceptance criterion for comparing the two different AVBD solvers.
        enforce_repeatability(replay_check, "Direct/chunk rollout")
        phases["full_trajectory_preflight_seconds"] = time.perf_counter() - t
        print(f"{a.worker_engine} n={n}: {a.steps} physical-step preflight passed", flush=True)
    # Warm clocks, graph replay, allocator, and working set. Reset between
    # complete trajectories so warmup cannot benchmark a collapsed long rollout.
    t = time.perf_counter()
    warmed = 0
    warmup_gpu_seconds = 0.0
    while warmed < a.warmup_steps or warmup_gpu_seconds < a.warmup_seconds:
        reset()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(a.steps // a.chunk_steps):
            step_chunk()
        end.record()
        end.synchronize()
        warmup_gpu_seconds += begin.elapsed_time(end) / 1000
        warmed += a.steps
    phases["warmup_seconds"] = time.perf_counter() - t
    phases["warmup_gpu_seconds"] = warmup_gpu_seconds
    phases["ready_for_measurement_total_seconds"] = time.perf_counter() - started
    dump(folder / "initialization.json", phases)
    print(
        f"{a.worker_engine} n={n}: warmup {warmed} steps, {phases['warmup_seconds']:.3f}s",
        flush=True,
    )
    clock_before = gpu_info()
    times, wall_times, checks = [], [], []
    for repetition in range(a.repetitions):
        reset()
        if a.physics_profile == "avbd-common":
            verify_reset()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        begin.record()
        for _ in range(a.steps // a.chunk_steps):
            step_chunk()
        end.record()
        end.synchronize()
        wall_times.append(time.perf_counter() - t)
        times.append(begin.elapsed_time(end) / 1000)
        check = diagnostics()
        if a.physics_profile == "avbd-common":
            trial_state = snapshot()
            check["reset_replay_state"] = compare_states(graph_final, trial_state)
            dump(folder / f"trial-{repetition+1}-replay.json", check["reset_replay_state"])
            enforce_repeatability(check["reset_replay_state"], "Trial reset replay")
        checks.append(check)
        assert_gpu_exclusive()
        if (
            not check["finite"]
            or check.get("numeric_failure_envs", 0)
            or any(check.get("overflow_flags", []))
        ):
            raise RuntimeError(f"Invalid benchmark state: {check}")
        print(f"{a.worker_engine} n={n}: trial {repetition+1}, {times[-1]:.6f}s", flush=True)
    median = statistics.median(times)
    report = dict(
        engine=a.worker_engine,
        num_envs=n,
        physics_steps_per_trial=a.steps,
        chunk_steps=a.chunk_steps,
        repetitions=a.repetitions,
        initialization=phases,
        warmup_steps=warmed,
        requested_warmup_steps=a.warmup_steps,
        minimum_warmup_seconds=a.warmup_seconds,
        gpu_seconds=times,
        synchronized_wall_seconds=wall_times,
        median_gpu_seconds=median,
        env_steps_per_second=n * a.steps / median,
        per_trial_env_steps_per_second=[n * a.steps / seconds for seconds in times],
        physics_only=True,
        reset_in_timing=False,
        host_readback_in_timing=False,
        diagnostics_scope=("Per-step preflight outside timing; trial checks after timing. Native errors cover the last chunk, Newton AVBD errors the final state; see preflight.json for the entire trajectory"
                           if a.physics_profile == "avbd-common" else
                           "After each trial; AVBD maximum errors cover last chunk only, MJWarp contacts are from final forward pass"),
        checks=checks,
        physics_profile=a.physics_profile,
        preflight_physical_steps=len(preflight) if preflight is not None else 0,
        quality_status="Measured residuals; equal-accuracy throughput not established",
        execution_repeatability=dict(policy=(a.newton_avbd_replay_policy if a.worker_engine == "newton-avbd" else "strict"),
                                     reset_initial_state_bitwise_verified=a.physics_profile == "avbd-common",
                                     direct_repeat=repeat_check, direct_chunk=replay_check),
        settings=settings,
        native_binary_sha256=binary_hash,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ROOT / "scripts/benchmark_engines.py", ROOT / "scripts/prepare_standing.py"]
                       + ([ROOT / "scripts/benchmark_newton_avbd.py"] if a.worker_engine == "newton-avbd" else [])},
        asset_commit=meta["upstream_commit"],
        model_sha256=hashlib.sha256((folder / "model/microduck.duck").read_bytes()).hexdigest(),
        gpu_name=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "mujoco", "mujoco-warp", "warp-lang", "numpy"]
            + (["newton"] if a.worker_engine in ("newton", "newton-avbd") else [])
        },
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version,
        gpu_before_trials=clock_before,
        gpu_after_trials=gpu_info(),
    )
    dump(folder / "report.json", report)
    return report


def main():
    a = parse_args()
    if a.worker_engine:
        worker(a)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    dump(
        a.output / "config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
    )
    rows = []
    for n in a.envs:
        for engine in a.engines:
            wait_started = time.perf_counter()
            while True:
                try:
                    assert_gpu_exclusive()
                    break
                except RuntimeError:
                    if time.perf_counter() - wait_started >= a.wait_for_gpu_seconds:
                        raise
                    if time.perf_counter() - wait_started < 1:
                        print("Waiting for the selected GPU to become idle", flush=True)
                    time.sleep(2)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-engine",
                engine,
                "--envs",
                str(n),
                "--output",
                str(a.output),
            ]
            for flag in [
                "steps",
                "chunk_steps",
                "repetitions",
                "warmup_steps",
                "warmup_seconds",
                "avbd_iterations",
                "mjwarp_iterations",
                "dt",
                "physics_profile",
                "newton_avbd_replay_policy",
                "nconmax",
                "njmax",
            ]:
                command.extend(["--" + flag.replace("_", "-"), str(getattr(a, flag))])
            if a.avbd_shared_memory:
                command.append("--avbd-shared-memory")
            if a.avbd_specialized:
                command.append("--avbd-specialized")
            if a.newton_avbd_deterministic:
                command.append("--newton-avbd-deterministic")
            if a.newton_avbd_contact_history:
                command.append("--newton-avbd-contact-history")
            if a.newton_avbd_verify_buffers:
                command.append("--newton-avbd-verify-buffers")
            t = time.perf_counter()
            with (a.output / f"{engine}-{n}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            child_wall = time.perf_counter() - t
            if result.returncode:
                raise RuntimeError(f"{engine}/{n} failed; see {a.output / f'{engine}-{n}.log'}")
            path = a.output / f"{engine}-{n}/report.json"
            row = json.loads(path.read_text())
            row["child_process_wall_seconds"] = child_wall
            dump(path, row)
            rows.append(row)
            dump(a.output / "report.json", rows)
            print(engine, n, round(row["env_steps_per_second"]), "physics env-steps/s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
