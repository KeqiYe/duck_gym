"""Long HOME trajectory comparison of native and Newton rigid AVBD.

Diagnostic experiment, not a throughput benchmark. Run with remote/run.py and
venv-benchmark-newton. Every physical step is recorded; episodes never reset.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def quaternion_product(a, b):
    """Broadcast wxyz quaternion product, without changing physical state."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate((aw*bw-np.sum(av*bv, axis=-1, keepdims=True),
                           aw*bv+bw*av+np.cross(av, bv)), axis=-1)


def common_reference(folder, meta, dt):
    import mujoco
    xml = ET.fromstring((folder / "model/visual.xml").read_text())
    names = {g.get("name") for g in xml.find("worldbody").iter("geom") if g.get("name")}
    for index, geom in enumerate(xml.find("worldbody").iter("geom")):
        if not geom.get("name"):
            name = f"avbd_benchmark_geom_{index}"
            if name in names:
                raise ValueError("Temporary geom label collision")
            geom.set("name", name)
            names.add(name)
    actuator = ET.SubElement(xml, "actuator")
    for name, low, high in zip(meta["joint_names"], meta["lower"], meta["upper"]):
        ET.SubElement(actuator, "position", name="hold_"+name, joint=name,
                      kp=str(meta["kp"]), forcelimited="false",
                      ctrllimited="true", ctrlrange=f"{low} {high}")
    xml.find("option").set("timestep", str(dt))
    xml.find("option").set("solver", "Newton")
    xml.find("option").set("integrator", "Euler")
    xml.find("option").set("iterations", "50")
    raw = ET.tostring(xml, encoding="unicode")
    (folder / "model/benchmark.xml").write_text(raw)
    return mujoco.MjModel.from_xml_string(raw)


def newton_mapping(adapter, reference, meta, envs):
    """Map imported body origins into upstream COM/principal-frame convention."""
    import mujoco
    from benchmark_newton_avbd import _apply
    from model_io import mul

    model = adapter.model
    worlds = model.body_world.numpy()
    labels = [str(label).split("/")[-1] for label in model.body_label]
    names = [reference.body(i).name for i in range(1, reference.nbody)]
    indices = []
    for e in range(envs):
        row = []
        for name in names:
            matches = [i for i in np.flatnonzero(worlds == e) if labels[i] == name]
            if len(matches) != 1:
                raise ValueError(f"Body mapping is not a bijection: {e}/{name}")
            row.append(matches[0])
        indices.append(row)
    indices = np.asarray(indices)
    initial = model.body_q.numpy()[indices].astype(np.float64)
    local_com = model.body_com.numpy()[indices].astype(np.float64)
    data = mujoco.MjData(reference)
    data.qpos[:] = meta["qpos"]
    mujoco.mj_forward(reference, data)
    expected_q = np.asarray([mul(data.xquat[i], reference.body_iquat[i])
                             for i in range(1, reference.nbody)])
    # Newton preserves the imported MJCF link frame. Use the authored constant
    # link->principal frame, rather than fitting away initial orientation errors.
    correction = np.broadcast_to(reference.body_iquat[None, 1:], (envs, len(names), 4)).copy()
    mapped_initial = quaternion_product(initial[..., [6, 3, 4, 5]], correction)
    mapped_initial /= np.linalg.norm(mapped_initial, axis=-1, keepdims=True)
    expected_q /= np.linalg.norm(expected_q, axis=-1, keepdims=True)
    initial_angle = 2*np.arccos(np.clip(np.abs(np.sum(mapped_initial*expected_q[None], axis=-1)), 0, 1))
    com_error = _apply(initial, local_com)-data.xipos[None, 1:]
    return indices, local_com, correction, dict(
        native_convention="World COM position, upstream principal-frame wxyz, world COM linear velocity, world angular velocity",
        newton_velocity_layout="body_qd first3 world COM linear, last3 world angular",
        frame_correction="Constant authored reference.body_iquat right product; initial orientation errors are not fitted away",
        position_alignment="None: imported initial COM discrepancies remain in the recorded trajectories",
        max_initial_reference_com_component_difference_m=float(np.max(np.abs(com_error))),
        max_initial_reference_principal_orientation_difference_rad=float(initial_angle.max()),
        indices=indices.tolist(), frame_correction_wxyz=correction.tolist(),
    )


def newton_counters(adapter, wp):
    """Small per-step counters captured together with poses; host checks after chunks."""
    pipeline, solver, contacts = adapter.collision_pipeline, adapter.solver, adapter.contacts
    narrow = pipeline.narrow_phase
    reducer = narrow.global_contact_reducer
    rows = [
        ("final_contacts", contacts.rigid_contact_count, 0, contacts.rigid_contact_max, False),
        ("body_contact_overflow_max", solver.body_body_contact_overflow_max, 0,
         solver.body_body_contact_buffer_pre_alloc, False),
        ("broad_phase_pairs", pipeline.broad_phase_pair_count, 0, pipeline.broad_phase_shape_pairs.shape[0], False),
        ("mesh_plane_pairs", narrow.shape_pairs_mesh_plane_count, 0, narrow.shape_pairs_mesh_plane.shape[0], False),
        ("reducer_reserved_contacts", reducer.contact_count, 0, reducer.capacity, True),
        ("reducer_hash_active", reducer.hashtable.active_slots, reducer.hashtable.capacity,
         reducer.hashtable.capacity, False),
        ("reducer_hash_insert_failures", reducer.ht_insert_failures, 0, 0, False),
    ]
    tensors, specifications = [], []
    for name, array, index, capacity, saturated_is_unknown in rows:
        tensors.append(wp.to_torch(array).reshape(-1)[index:index+1])
        specifications.append(dict(name=name, capacity=int(capacity),
                                   saturated_is_unknown=saturated_is_unknown))
    return tensors, specifications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--newton-contact-history", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true",
                        help="Allow shared GPU for trajectory diagnostics only, never throughput measurements")
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("DUCK_RUN_DIR", "runs"))/"long-avbd")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("Require finite duration and positive finite dt")
    steps = round(args.duration/args.dt)
    if (args.duration <= 0 or args.dt <= 0 or args.num_envs < 1 or args.iterations < 1
            or args.chunk_steps < 2 or args.chunk_steps % 2 or steps % args.chunk_steps
            or not np.isclose(steps*args.dt, args.duration, rtol=0, atol=1e-10)):
        parser.error("Require positive duration/dt/envs/iterations and an even chunk dividing the physical steps")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    import mujoco
    from benchmark_engines import assert_gpu_exclusive, gpu_info
    from benchmark_newton_avbd import setup_newton_avbd, _apply
    from build_cuda import build
    from prepare_standing import prepare

    def gpu_snapshot(stage):
        if not args.allow_busy_gpu:
            assert_gpu_exclusive()
        dump(out/f"gpu-{stage}.json", dict(gpus=gpu_info(),
            sharing_allowed=args.allow_busy_gpu, timing_is_not_throughput=True,
            compute_processes=subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader"], text=True)))

    gpu_snapshot("before")
    torch.set_num_threads(1)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    wp.config.kernel_cache_dir = str(out/"warp-cache")
    wp.init()
    warp_stream = wp.stream_from_torch(stream)
    wp.set_stream(warp_stream)
    meta = prepare(out/"model", physics_profile="avbd-common")
    reference = common_reference(out, meta, args.dt)
    names = [reference.body(i).name for i in range(1, reference.nbody)]
    dump(out/"body_names.json", dict(body_names=names, root_body_name=reference.body(1).name))
    dump(out/"config.json", dict(
        **{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        physics_profile="avbd-common", control="Fixed HOME targets, no external force, no PPO, no rollout resets",
        sample_every_physical_step=True, seed=0, engine_precision="fp32",
        scope="Eight identical initial worlds by default; observed maxima are over this recorded run only",
        native_stop="Up to iteration budget with existing convergence stop",
        newton_stop="Fixed iteration budget", elapsed_physics_steps=steps,
        versions={name:importlib.metadata.version(name) for name in ("torch","warp-lang","newton","mujoco","numpy")},
        hostname=platform.node(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), ROOT/"scripts/benchmark_newton_avbd.py", ROOT/"scripts/prepare_standing.py"]},
        asset_commit=meta["upstream_commit"],
    ))

    ext = build()  # Target build check, even when compilation cache is warm.
    binary = Path(ext.__file__)
    shutil.copy2(binary, out/binary.name)
    dump(out/"native-binary.json", dict(source=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    n, bodies, chunk = args.num_envs, reference.nbody-1, args.chunk_steps
    sim = ext.CudaBatch(str(out/"model/microduck.duck"), n, args.dt, args.iterations, 0, False,
                        specialized=True, shared_memory=False)
    targets = torch.tensor(meta["home"], device="cuda", dtype=torch.float32).repeat(n, 1)
    forces = torch.zeros((n, 3), device="cuda")
    mask, roots = torch.ones(n, device="cuda", dtype=torch.bool), torch.zeros((n, 6), device="cuda")
    sim.reset(mask, targets, roots)
    initial_native = sim.state()[0].cpu().numpy().copy()
    native_trace = np.empty((steps+1, n, bodies, 13), dtype=np.float32)
    native_trace[0] = initial_native[:, 1:]
    native_diagnostics = np.empty((steps, n, 6), dtype=np.float32)
    device_body = torch.empty((chunk, n, reference.nbody, 13), device="cuda")
    device_diag = torch.empty((chunk, n, 6), device="cuda")
    start = time.perf_counter()
    for offset in range(0, steps, chunk):
        for i in range(chunk):
            sim.step(targets, forces, 1, meta["kp"], meta["torque_limit"])
            b, _, d = sim.state()
            device_body[i].copy_(b)
            device_diag[i].copy_(d)
        b = device_body.cpu().numpy()
        d = device_diag.cpu().numpy()
        native_trace[offset+1:offset+chunk+1] = b[:, :, 1:]
        native_diagnostics[offset:offset+chunk] = d
        if not np.isfinite(b).all() or np.any(d[..., 5] != 0):
            np.savez_compressed(out/"native-failure.npz", step=offset+np.arange(1,chunk+1), body=b, diagnostics=d)
            raise RuntimeError(f"Native numerical failure in physical steps {offset+1}..{offset+chunk}")
        if (offset+chunk) % 1000 == 0 or offset+chunk == steps:
            print(f"native recorded {offset+chunk}/{steps} physical steps", flush=True)
    native_seconds = time.perf_counter()-start
    t = np.arange(steps+1, dtype=np.float64)*args.dt
    np.savez_compressed(out/"native.npz", time=t, com_position=native_trace[..., :3],
                        orientation=native_trace[..., 3:7], linear_velocity=native_trace[..., 7:10],
                        angular_velocity=native_trace[..., 10:13])
    np.savez_compressed(out/"native-diagnostics.npz", time=t[1:], diagnostics=native_diagnostics)

    a = SimpleNamespace(physics_profile="avbd-common", envs=[n], dt=args.dt,
                        avbd_iterations=args.iterations, chunk_steps=chunk, nconmax=args.nconmax,
                        newton_avbd_deterministic=False,
                        newton_avbd_contact_history=args.newton_contact_history,
                        newton_avbd_verify_buffers=False)
    adapter = setup_newton_avbd(a, out, meta, reference)
    dump(out/"newton-settings.json", adapter.settings)
    indices, local_com, correction, mapping = newton_mapping(adapter, reference, meta, n)
    dump(out/"frame-mapping.json", mapping)
    counter_tensors, counter_specs = newton_counters(adapter, wp)
    dump(out/"counter-specifications.json", counter_specs)
    q_buffer = torch.empty((chunk, n*bodies, 7), device="cuda")
    v_buffer = torch.empty((chunk, n*bodies, 6), device="cuda")
    c_buffer = torch.empty((chunk, len(counter_tensors)), device="cuda", dtype=torch.int32)

    def record_chunk():
        for i in range(chunk):
            adapter.step()
            q_buffer[i].copy_(wp.to_torch(adapter.states[0].body_q))
            v_buffer[i].copy_(wp.to_torch(adapter.states[0].body_qd))
            for j, source in enumerate(counter_tensors):
                c_buffer[i, j:j+1].copy_(source)

    # Compile all physics/recording kernels and allocate before graph capture.
    adapter.reset()
    record_chunk()
    torch.cuda.synchronize()
    adapter.reset()
    with wp.ScopedCapture() as capture:
        record_chunk()
    graph = capture.graph
    adapter.reset()
    wp.capture_launch(graph)
    torch.cuda.synchronize()
    test_q, test_v = q_buffer.cpu().numpy(), v_buffer.cpu().numpy()
    if not np.isfinite(test_q).all() or not np.isfinite(test_v).all():
        raise RuntimeError("Captured recording preflight produced a nonfinite state")
    if not np.array_equal(test_q[-1], adapter.states[0].body_q.numpy()) or not np.array_equal(test_v[-1], adapter.states[0].body_qd.numpy()):
        raise RuntimeError("Captured recording does not end at the current Newton state")
    dump(out/"recording-preflight.json", dict(
        all_chunk_states_finite=True, final_record_equals_current_state_exactly=True,
        physical_steps_per_chunk=chunk, reset_before_measured_trajectory=True))
    adapter.reset()
    initial_q = adapter.states[0].body_q.numpy().copy()
    initial_v = adapter.states[0].body_qd.numpy().copy()
    raw_q = np.empty((steps+1, n*bodies, 7), dtype=np.float32)
    raw_v = np.empty((steps+1, n*bodies, 6), dtype=np.float32)
    raw_c = np.empty((steps, len(counter_tensors)), dtype=np.int32)
    raw_q[0], raw_v[0] = initial_q, initial_v
    start = time.perf_counter()
    for offset in range(0, steps, chunk):
        wp.capture_launch(graph)
        q = q_buffer.cpu().numpy()
        v = v_buffer.cpu().numpy()
        c = c_buffer.cpu().numpy()
        raw_q[offset+1:offset+chunk+1], raw_v[offset+1:offset+chunk+1] = q, v
        raw_c[offset:offset+chunk] = c
        invalid = not np.isfinite(q).all() or not np.isfinite(v).all()
        for j, spec in enumerate(counter_specs):
            invalid |= bool(np.any(c[:,j] < 0) or np.any(c[:,j] > spec["capacity"]))
            if spec["saturated_is_unknown"]:
                invalid |= bool(np.any(c[:,j] >= spec["capacity"]))
        if invalid:
            np.savez_compressed(out/"newton-failure.npz", step=offset+np.arange(1,chunk+1),
                                body_q=q, body_qd=v, counters=c)
            raise RuntimeError(f"Newton invalid state/capacity in physical steps {offset+1}..{offset+chunk}")
        if (offset+chunk) % 1000 == 0 or offset+chunk == steps:
            print(f"Newton recorded {offset+chunk}/{steps} physical steps", flush=True)
    newton_seconds = time.perf_counter()-start
    np.savez_compressed(out/"newton-raw.npz", time=t, body_q=raw_q, body_qd=raw_v)
    np.savez_compressed(out/"newton-counters.npz", time=t[1:], counters=raw_c)
    ordered_pose = raw_q[:, indices].astype(np.float64)
    ordered_velocity = raw_v[:, indices]
    com = _apply(ordered_pose, local_com[None])
    wxyz = ordered_pose[..., [6,3,4,5]]
    orientation = quaternion_product(wxyz, correction[None])
    np.savez_compressed(out/"newton.npz", time=t, com_position=com, orientation=orientation,
                        linear_velocity=ordered_velocity[..., :3], angular_velocity=ordered_velocity[..., 3:])
    dump(out/"run-summary.json", dict(
        completed_physical_steps_per_engine=steps, duration_seconds=steps*args.dt,
        envs=n, dynamic_bodies=bodies, native_record_wall_seconds=native_seconds,
        newton_record_wall_seconds=newton_seconds, timing_is_not_throughput=True,
        native_unconverged_env_steps=int(np.count_nonzero(native_diagnostics[..., 3] == 0)),
        native_numeric_failure_env_steps=int(np.count_nonzero(native_diagnostics[..., 5])),
        newton_final_diagnostics=adapter.diagnostics(),
        validation="Every recorded physical step checked finite and exposed capacity counters; no precision acceptance threshold imposed",
    ))
    gpu_snapshot("after")
    print(f"Completed both {steps*args.dt:g}s trajectories: {out}", flush=True)


if __name__ == "__main__":
    main()
