"""Offline collision audit of maximal-coordinate poses; never integrate physics."""

import bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco


def support_band_rms(body, points, band=0.0005):
    """Geometric support-band speed proxy, not a pressure-weighted contact speed."""
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, body[3:7])
    offset = points @ matrix.reshape(3, 3).T
    selected = offset[:, 2] <= offset[:, 2].min() + band
    velocity = body[7:10] + np.cross(body[10:13], offset[selected])
    return float(np.sqrt(np.square(velocity[:, :2]).sum(-1).mean()))


def build_audit_model(model_dir, *, explicit_pairs=True, xml_path=None):
    original = mujoco.MjSpec.from_file(str(model_dir / "scene.xml"))
    spec = mujoco.MjSpec.from_file(str(model_dir / "ground-only.xml"))
    # Upstream has unnamed geoms: preserve structural order, never name-alias.
    assert len(spec.geoms) == len(original.geoms)
    for i, (g, source) in enumerate(zip(spec.geoms, original.geoms, strict=True)):
        assert g.name == source.name and g.type == source.type
        np.testing.assert_allclose(g.pos, source.pos, atol=1e-12)
        np.testing.assert_allclose(g.quat, source.quat, atol=1e-12)
        g.contype, g.conaffinity = source.contype, source.conaffinity
        if not g.name:
            g.name = f"audit-geom-{i}"
    model = spec.compile()
    if model.npair or model.nexclude:
        raise ValueError(
            "Extend the audit pair filter before using a model with explicit pairs/exclusions"
        )
    # Maximal-coordinate poses bypass kinematics, so MuJoCo's dynamic BVH can
    # be stale. Explicit eligible pairs call narrow phase directly. No contact
    # response or artificial collision is added to the simulated trajectory.
    eligible = []
    for i in range(model.ngeom):
        a = int(model.body_weldid[model.geom_bodyid[i]])
        if a == 0:
            continue
        for j in range(i + 1, model.ngeom):
            b = int(model.body_weldid[model.geom_bodyid[j]])
            if b == 0 or a == b:
                continue
            if (
                a == model.body_weldid[model.body_parentid[b]]
                or b == model.body_weldid[model.body_parentid[a]]
            ):
                continue
            if not (
                (model.geom_contype[i] & model.geom_conaffinity[j])
                or (model.geom_contype[j] & model.geom_conaffinity[i])
            ):
                continue
            eligible.append((i, j))
            if explicit_pairs:
                spec.add_pair(geomname1=model.geom(i).name, geomname2=model.geom(j).name)
    if xml_path is not None:
        Path(xml_path).write_text(spec.to_xml())
    return spec.compile(), eligible


def main():
    from duck_gym.render import apply_body_poses

    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--physics", action="store_true", help="Audit all recorded physical substeps")
    args = p.parse_args()
    model, eligible = build_audit_model(args.model_dir)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    record = np.load(args.trajectory)
    prefix = "physics_" if args.physics else ""
    bodies = record[prefix + "bodies"]
    contact_flags = record[prefix + "contact"] if prefix + "contact" in record else None
    control_dt = float(record["physics_dt" if args.physics else "control_dt"])
    cfg = json.loads((args.model_dir / "config.json").read_text())
    support_speeds = [[], []]
    foot_points = [np.asarray(f["points"]) for f in cfg["feet"]]
    assert cfg["body_names"] == [model.body(i).name for i in range(model.nbody)]
    foot_bodies = {f["body"] for f in cfg["feet"]}
    visuals = []
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_bodyid[i] in foot_bodies:
            continue
        if model.geom_group[i] != 2:
            continue
        mesh = model.geom_dataid[i]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
        ]
        visuals.append((i, vertices))
    pairs = {}
    min_height = float("inf")
    body_contacts = np.zeros(len(bodies), dtype=int)
    deepest = np.zeros(len(bodies))
    for k, state in enumerate(bodies):
        if k * control_dt >= args.warmup and contact_flags is not None:
            for side, f in enumerate(cfg["feet"]):
                if contact_flags[k, side]:
                    support_speeds[side].append(
                        support_band_rms(state[f["body"]], foot_points[side])
                    )
        apply_body_poses(model, data, state)
        mujoco.mj_collision(model, data)
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if model.geom_bodyid[g1] == 0 or model.geom_bodyid[g2] == 0:
                continue
            if contact.dist < 0:
                body_contacts[k] += 1
                deepest[k] = max(deepest[k], -float(contact.dist))
                names = [
                    model.geom(g).name or f"geom#{g}:{model.body(model.geom_bodyid[g]).name}"
                    for g in (g1, g2)
                ]
                pair = "|".join(sorted(names))
                pairs[pair] = max(pairs.get(pair, 0.0), -float(contact.dist))
        for geom, vertices in visuals:
            z = vertices @ data.geom_xmat[geom].reshape(3, 3)[2] + data.geom_xpos[geom, 2]
            min_height = min(min_height, float(z.min()))
    result = dict(
        trajectory=str(args.trajectory),
        frames=len(bodies),
        sampling="every recorded physical substep" if args.physics else "control snapshots",
        sample_dt=control_dt,
        physics_steps=0,
        tested_self_collision_pairs=eligible,
        self_contact_frames=int(np.count_nonzero(body_contacts)),
        maximum_self_interpenetration=float(deepest.max()),
        support_band_velocity=dict(
            definition="RMS horizontal v_COM + omega cross r over collision vertices within 0.5 mm of the lowest point; contact-flag frames after warmup, no pressure weighting",
            warmup_seconds=args.warmup,
            samples=[len(v) for v in support_speeds],
            mean_rms_mps=[float(np.mean(v)) if v else None for v in support_speeds],
            p95_rms_mps=[float(np.quantile(v, 0.95)) if v else None for v in support_speeds],
        ),
        pairs_maximum_depth=pairs,
        minimum_nonfoot_visual_height=min_height if visuals else None,
        checked_nonfoot_mesh_geoms=len(visuals),
        accepted=False,
        scope="Independent offline check with original collision masks; response capabilities are recorded in model config, physical domain randomization is absent",
        simulated_model_scope=cfg["scope"],
    )
    if prefix + "root_quat" in record and prefix + "joints" in record:
        q = record[prefix + "root_quat"]
        tilt = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
        head = [i for i, name in enumerate(cfg["joint_names"]) if "head" in name or "neck" in name]
        delta = record[prefix + "joints"][:, head, 0] - np.asarray(cfg["default_qpos"][7:])[head]
        result["posture"] = dict(
            max_root_tilt_rad=float(tilt.max()),
            head_names=[cfg["joint_names"][i] for i in head],
            max_abs_head_offsets_rad=np.abs(delta).max(0).tolist(),
            p95_abs_head_offsets_rad=np.quantile(np.abs(delta), 0.95, axis=0).tolist(),
            note="Head offsets are relative to home; intended head/body commands are zero in this evaluation",
        )
    path = args.trajectory.parent / (
        "collision-audit-physics.json" if args.physics else "collision-audit.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
