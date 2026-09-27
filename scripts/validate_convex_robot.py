"""Replay independent maximal-coordinate body poses through MPR and MuJoCo."""

import bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from audit_bam_contacts import build_audit_model
from duck_gym.render import apply_body_poses
import _duck_reference as native


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model, pairs = build_audit_model(args.model_dir)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    record = np.load(args.trajectory)
    points = {}
    for g in set(x for pair in pairs for x in pair):
        mesh = model.geom_dataid[g]
        b = model.geom_bodyid[g]
        v = model.mesh_vert[
            model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
        ].astype(float)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        principal = np.empty(9)
        mujoco.mju_quat2Mat(principal, model.body_iquat[b])
        points[g] = (
            v @ rotation.reshape(3, 3).T + model.geom_pos[g] - model.body_ipos[b]
        ) @ principal.reshape(3, 3)
    counts = dict(
        samples=0,
        positive_pairs=0,
        fp32_failures=0,
        fp64_failures=0,
        fp32_mismatches=0,
        fp64_mismatches=0,
    )
    mismatches = []
    for k, state in enumerate(record["bodies"]):
        apply_body_poses(model, data, state)
        mujoco.mj_collision(model, data)
        penetration = {}
        for c in data.contact[: data.ncon]:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            if pair in pairs and c.dist < -2e-6:
                penetration[pair] = max(penetration.get(pair, 0), -float(c.dist))
        if not penetration and k % 10:
            continue
        for a, b in pairs:
            poses = state[model.geom_bodyid[[a, b]], :7]
            counts["samples"] += 1
            expected = (a, b) in penetration
            counts["positive_pairs"] += expected
            for precision, query, tol in [
                ("fp32", native.convex_query32, 1e-7),
                ("fp64", native.convex_query64, 1e-9),
            ]:
                c = query(points[a], points[b], poses, 0, tol)
                counts[precision + "_failures"] += c["status"] < 0
                # Independent depth <2um falls inside this validation's deadband.
                if (c["status"] == 1 and c["depth"] > 2e-6) != expected:
                    counts[precision + "_mismatches"] += 1
                    mismatches.append(
                        dict(
                            frame=k,
                            pair=[a, b],
                            precision=precision,
                            native=c,
                            reference_depth=penetration.get((a, b), 0),
                        )
                    )
    result = dict(**counts, mismatches=mismatches, trajectory=str(args.trajectory), pairs=pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}), flush=True)
    if mismatches or counts["fp32_failures"] or counts["fp64_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
