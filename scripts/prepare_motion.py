"""Export exact support hulls for motion rewards; no dynamics or model changes."""

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from prepare_standing import prepare


def prepare_motion(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(folder / "visual.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = meta["qpos"]
    mujoco.mj_forward(m, d)
    clouds, bodies = [], []
    for body in range(1, m.nbody):
        cloud = []
        for geom in range(m.ngeom):
            if m.geom_bodyid[geom] != body or not m.geom_contype[geom]:
                continue
            if m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("Motion geometry exporter currently requires mesh robot assets")
            mesh = int(m.geom_dataid[geom])
            start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            world = m.mesh_vert[start : start + count] @ d.geom_xmat[geom].reshape(3, 3).T
            world += d.geom_xpos[geom]
            cloud.append((world - d.xipos[body]) @ d.ximat[body].reshape(3, 3))
        if cloud:
            points = np.unique(np.concatenate(cloud), axis=0)
            # A linear support function has the same extrema on the vertex
            # cloud and its hull. These points are sensors, not new colliders.
            points = points[ConvexHull(points).vertices]
            clouds.append(points)
            bodies.append(body)
    width = max(map(len, clouds))
    valid = np.arange(width)[None] < np.array([len(p) for p in clouds])[:, None]
    points = np.stack([np.pad(p, ((0, width - len(p)), (0, 0)), mode="edge") for p in clouds])
    feet = [int(m.site_bodyid[m.site(n).id]) for n in ("left_foot", "right_foot")]
    np.savez_compressed(
        folder / "motion.npz",
        points=points,
        valid=valid,
        bodies=bodies,
        foot_slots=[bodies.index(b) for b in feet],
        foot_bodies=feet,
        body_mass=m.body_mass,
        body_inertia=m.body_inertia,
        body_iquat=m.body_iquat,
        body_names=np.array([m.body(i).name for i in range(m.nbody)]),
    )
    report = dict(
        method="Convex support hull of original collision vertices in each COM/principal frame",
        physics_modified=False,
        bodies=bodies,
        hull_points=[len(p) for p in clouds],
        foot_bodies=feet,
        source_commit=meta["upstream_commit"],
    )
    (folder / "motion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    prepare(args.output)
    print(prepare_motion(args.output))
