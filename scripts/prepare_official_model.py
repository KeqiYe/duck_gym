"""Export a declared ground-contact-only nominal comparison model.

This is a controlled dynamics diagnostic, NOT an export of all upstream task
features. Self-collision, per-environment randomization and sensors remain gaps.
"""

import bootstrap
import json
import os
from pathlib import Path
from types import SimpleNamespace
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import mjlab.tasks
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.scene import Scene
from model_io import export_model, rot, inv
from importlib.resources import files
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML


def main():
    out = Path(os.environ["DUCK_RUN_DIR"]) / "official-model"
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, "cpu")
    # Persist the actual simulation options in both exported XML files.
    # Applying only to a compiled MjModel would leave Scene.write at defaults.
    cfg.sim.mujoco.apply(SimpleNamespace(opt=scene.spec.option))
    original = scene.compile()
    cfg.sim.mujoco.apply(original)
    scene.write(out)
    # mjlab's export_spec only writes in-memory assets. This upstream robot
    # also references disk meshes, so complete the portable export explicitly.
    for element in ET.parse(out / "scene.xml").getroot().iter():
        ref = element.get("file")
        if ref:
            dest = out / "assets" / ref
            if not dest.exists():
                matches = list((MICRODUCK_WALK_XML.parent / "assets").rglob(Path(ref).name))
                if len(matches) != 1:
                    raise RuntimeError(f"Cannot resolve unique pinned asset: {ref}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(matches[0], dest)
    spec = mujoco.MjSpec.from_file(str(out / "scene.xml"))
    for group in (list(spec.actuators), list(spec.sensors), list(spec.keys)):
        for element in group:
            spec.delete(element)
    spec.compiler.fusestatic = True
    # Both sides of this numerical comparison use the same declared subset.
    # The full exported scene.xml remains available for auditing what was removed.
    planes = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1:
        raise ValueError("Expected one flat ground plane")
    ground_type, ground_affinity = planes[0].contype, planes[0].conaffinity
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype, geom.conaffinity = 2, 1
        elif (geom.contype & ground_affinity) or (ground_type & geom.conaffinity):
            geom.contype, geom.conaffinity = 1, 2
        else:
            # In robot_walk, calf/trunk guards are self-collision-only. They
            # must not become new floor supports in this declared subset.
            geom.contype, geom.conaffinity = 0, 0
    xml = spec.to_xml()
    (out / "ground-only.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(out / "ground-only.xml"))
    cfg.sim.mujoco.apply(model)
    data = mujoco.MjData(model)
    # The scene home key contains the upstream named joint default positions.
    home_key = next(k for k in range(original.nkey) if original.key(k).name == "init_state")
    data.qpos[:] = original.key_qpos[home_key]
    data.qpos[2] = np.mean(cfg.events["reset_base"].params["pose_range"]["z"])
    bodies = export_model(model, data, out / "model.duck")
    feet = []
    for side in ("left", "right"):
        geom = model.geom(f"robot/{side}_foot_collision")
        b = int(geom.bodyid[0])
        mesh = int(geom.dataid[0])
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
        ]
        # Exact convex support points for minimum-height diagnostics, without
        # processing thousands of interior mesh vertices on every observation.
        vertices = vertices[ConvexHull(vertices).vertices]
        points = np.stack(
            [
                rot(
                    inv(bodies[b]["q"]),
                    data.geom_xpos[geom.id]
                    + data.geom_xmat[geom.id].reshape(3, 3) @ v
                    - bodies[b]["x"],
                )
                for v in vertices
            ]
        )
        site = model.site(f"robot/{side}_foot")
        feet.append(
            dict(
                body=b,
                points=points.tolist(),
                site=rot(inv(bodies[b]["q"]), data.site_xpos[site.id] - bodies[b]["x"]).tolist(),
            )
        )
    params = json.loads((files("bam") / "params/xl330/m6.json").read_text())
    actuator_cfg = cfg.scene.entities["robot"].articulation.actuators[0]
    metadata = dict(
        scope="Nominal ground-only controlled comparison; not full upstream task",
        omissions=["body-body contacts", "sensor observations", "per-env domain randomization"],
        physics_dt=model.opt.timestep,
        decimation=cfg.decimation,
        default_qpos=data.qpos.tolist(),
        root_com_offset=rot(inv(model.body_iquat[1]), model.body_ipos[1]).tolist(),
        root_principal_quat=model.body_iquat[1].tolist(),
        feet=feet,
        masses=model.body_mass.tolist(),
        ground_contact_geoms=[
            model.geom(i).name
            for i in range(model.ngeom)
            if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ],
        joint_names=[model.joint(i).name for i in range(1, model.njnt)],
        body_names=[model.body(i).name for i in range(model.nbody)],
        parameters=params,
        kp_fw=actuator_cfg.kp_fw,
        vin=float(np.mean(actuator_cfg.vin_range)),
        vin_drop_gain=float(np.mean(actuator_cfg.vin_drop_gain_range)),
        vin_min=actuator_cfg.vin_min,
        force_limit=max(actuator_cfg.vin_range) * params["kt"] / params["R"],
        delay_lag_steps=[actuator_cfg.delay_min_lag, actuator_cfg.delay_max_lag],
        initialization="Mean of official reset height range; official HOME joint positions",
        model_counts=dict(
            bodies=model.nbody, joints=model.njnt, geoms=model.ngeom, meshes=model.nmesh
        ),
    )
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
