"""CUDA-event physics throughput, with explicit environment count and error gates."""

import argparse, json, os
from pathlib import Path
import torch
from build_cuda import build
from prepare_standing import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, nargs="+", default=[1, 32, 128, 512])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--fp64", action="store_true")
    a = p.parse_args()
    ext = build()
    model_dir = Path(os.environ["DUCK_CUDA_BUILD"]).parents[1] / "build/models/standing"
    meta = prepare(model_dir)
    dtype = torch.float64 if a.fp64 else torch.float32
    rows = []
    for n in a.envs:
        sim = ext.CudaBatch(str(model_dir / "microduck.duck"), n, 0.001, a.iterations, 0, a.fp64)
        q = torch.tensor(meta["home"], device="cuda", dtype=dtype).repeat(n, 1)
        f = torch.zeros(n, 3, device="cuda", dtype=dtype)
        mask = torch.ones(n, device="cuda", dtype=torch.bool)
        roots = torch.zeros(n, 6, device="cuda", dtype=dtype)
        sim.step(q, f, 20, 0.55, 0.96)  # Untimed kernel and clock warmup.
        times = []
        for _ in range(3):
            sim.reset(mask, q, roots)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            sim.step(q, f, a.steps, 0.55, 0.96)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000)
        b, j, d = sim.state()
        valid = bool(torch.isfinite(b).all() and not d[:, 5].any())
        row = dict(
            num_envs=n,
            physics_steps=a.steps,
            iterations=a.iterations,
            fp64=a.fp64,
            gpu=torch.cuda.get_device_name(),
            timing="CUDA events, identical reset state, 3 repetitions after warmup",
            seconds=times,
            env_steps_per_second=n * a.steps / sorted(times)[1],
            valid=valid,
            max_joint_error_m=float(d[:, 0].max()),
            max_penetration_m=float(d[:, 1].max()),
        )
        if not valid or row["max_joint_error_m"] > 0.005 or row["max_penetration_m"] > 0.005:
            raise RuntimeError(f"Invalid benchmark physics: {row}")
        rows.append(row)
        print(row, flush=True)
    (Path(os.environ["DUCK_RUN_DIR"]) / "benchmark.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
