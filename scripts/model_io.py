"""Offline MuJoCo model compilation -> a standalone, dependency-free CPU model.

MuJoCo is only called in this preparation tool and in reference experiments.
The duck_cpu library reads the exported constants and never links to MuJoCo.
"""

from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/microduck/robot_allcollisions.xml"


def mul(a, b):
    q = np.empty(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def inv(q):
    return q * np.array([1, -1, -1, -1])


def rot(q, x):
    y = np.empty(3)
    mujoco.mju_rotVecQuat(y, np.asarray(x, dtype=float), q)
    return y


def axis_quat(axis, angle):
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def prepare_microduck(ground=True, stiff_reference=True):
    """Use original visuals and inertias; explicit stage-1 ground-only baseline."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(ASSET)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ASSET.parent / "assets"))
    # Real-robot actuator extensions are deferred; baseline is unactuated release.
    for tag in ("actuator", "sensor", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.001")
    option.set("integrator", "Euler")
    option.set("iterations", "100")
    option.set("tolerance", "1e-10")
    world = root.find("worldbody")
    if ground:
        ET.SubElement(
            world,
            "geom",
            name="ground",
            type="plane",
            size="2 2 0.1",
            rgba="0.22 0.25 0.29 1",
            friction="0.8 0 0",
            contype="2",
            conaffinity="1",
        )
    # Reference softness is tightened for comparison with hard AVBD constraints.
    if stiff_reference:
        for joint in root.iter("joint"):
            joint.set("solreffriction", ".002 1")
            joint.set("solreflimit", ".002 1")
    for geom in world.iter("geom"):
        if stiff_reference:
            geom.set("solref", ".002 1")
            geom.set("solimp", ".99 .99 .001")
        if geom.get("name") == "ground":
            continue
        # Preserve disabled visual geoms; active robot geoms only collide with plane.
        if geom.get("class") != "visual" and geom.get("contype") != "0":
            geom.set("contype", "1")
            geom.set("conaffinity", "2" if ground else "0")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(world, "light", pos="1 -1 2", dir="-0.5 0.5 -1", diffuse="0.8 0.8 0.8")
    return ET.tostring(root, encoding="unicode")


def export_model(model, data, path, torques=None, *, self_contacts=False):
    """Export the supported subset; fail rather than silently changing physics."""
    m, d = model, data
    torques = torques or {}
    if m.nu or m.npair or m.nflex or m.nplugin:
        raise ValueError("Explicit actuators, contact pairs, flex and plugins require an adapter")
    if np.any(m.opt.wind) or m.opt.density or m.opt.viscosity:
        raise ValueError("Fluid forces are not supported")
    if np.any(m.geom_margin) or np.any(m.geom_gap):
        raise ValueError("Nonzero collision margins/gaps are not supported")
    if m.neq or m.ntendon:
        raise ValueError("Equality/tendon constraints are not supported")
    mujoco.mj_forward(m, d)
    B = []
    for i in range(m.nbody):
        q = mul(d.xquat[i], m.body_iquat[i])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jp, jr, i)
        if i and m.body_jntnum[i] == 0:
            raise ValueError(f"Fixed child body must be fused before export: {m.body(i).name}")
        B.append(
            dict(
                name=m.body(i).name or f"body_{i}",
                mass=m.body_mass[i],
                inertia=m.body_inertia[i],
                x=d.xipos[i].copy(),
                q=q,
                v=jp @ d.qvel,
                w=jr @ d.qvel,
            )
        )
    J = []
    for i in range(m.njnt):
        typ = m.jnt_type[i]
        if typ == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if typ != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("Only free and hinge joints are supported")
        b = int(m.jnt_bodyid[i])
        a = int(m.body_parentid[b])
        if m.body_jntnum[b] != 1:
            raise ValueError("Only one hinge per body is supported")
        axis = d.xaxis[i].copy()
        t = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.8 else [0, 1, 0])
        t /= np.linalg.norm(t)
        adr, vadr = m.jnt_qposadr[i], m.jnt_dofadr[i]
        angle = d.qpos[adr] - m.qpos0[adr]
        u = rot(axis_quat(axis, angle), t)
        name = m.joint(i).name or f"joint_{i}"
        J.append(
            [
                name,
                a,
                b,
                *rot(inv(B[a]["q"]), d.xanchor[i] - B[a]["x"]),
                *rot(inv(B[b]["q"]), d.xanchor[i] - B[b]["x"]),
                *rot(inv(B[a]["q"]), axis),
                *rot(inv(B[b]["q"]), axis),
                *rot(inv(B[a]["q"]), t),
                *rot(inv(B[b]["q"]), u),
                int(m.jnt_limited[i]),
                *(m.jnt_range[i] - m.qpos0[adr]),
                m.dof_damping[vadr],
                m.dof_armature[vadr],
                m.dof_frictionloss[vadr],
                torques.get(name, 0),
                0,
                0,
                0,
            ]
        )
    if m.nexclude:
        raise ValueError("Explicit body exclusions require an adapter")
    if m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT):
        raise ValueError("Disabled parent filtering requires an adapter")

    def compatible(a, b):
        return bool(
            (int(m.geom_contype[a]) & int(m.geom_conaffinity[b]))
            or (int(m.geom_contype[b]) & int(m.geom_conaffinity[a]))
        )

    pairs = []
    # Ground-only is a declared subset, not a silent omission of self-collision.
    for a in range(m.ngeom):
        for b in range(a + 1, m.ngeom):
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if (
                not ba
                or not bb
                or ba == bb
                or m.body_parentid[ba] == bb
                or m.body_parentid[bb] == ba
            ):
                continue
            if compatible(a, b):
                if not self_contacts:
                    raise ValueError("Body-body collision requires self_contacts=True")
                if max(m.geom_condim[a], m.geom_condim[b]) > 3:
                    raise ValueError("Body contacts support normal and sliding friction only")
                if m.geom_priority[a] != m.geom_priority[b]:
                    friction = m.geom_friction[
                        a if m.geom_priority[a] > m.geom_priority[b] else b, 0
                    ]
                else:
                    friction = max(m.geom_friction[a, 0], m.geom_friction[b, 0])
                pairs.append((a, b, float(friction)))
    S = []
    geom_to_shape = {}
    planes = np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    ground = False
    for i in range(m.ngeom):
        if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
            continue
        b = int(m.geom_bodyid[i])
        typ = m.geom_type[i]
        if typ == mujoco.mjtGeom.mjGEOM_PLANE:
            if (
                b
                or np.linalg.norm(d.geom_xpos[i]) > 1e-10
                or not np.allclose(d.geom_xmat[i].reshape(3, 3)[:, 2], [0, 0, 1])
            ):
                raise ValueError("Only the static z=0 ground plane is supported")
            ground = True
            continue
        if b == 0:
            raise ValueError("Only a plane is supported as a static collision obstacle")
        qlink = inv(m.body_iquat[b])
        center = rot(qlink, m.geom_pos[i] - m.body_ipos[b])
        q = mul(qlink, m.geom_quat[i])
        vertices = []
        if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            kind = 0
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
            kind = 1
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            kind = 2
        elif typ == mujoco.mjtGeom.mjGEOM_MESH:
            kind = 3
            mid = m.geom_dataid[i]
            adr = m.mesh_vertadr[mid]
            n = m.mesh_vertnum[mid]
            vertices = np.unique(m.mesh_vert[adr : adr + n].astype(float), axis=0).tolist()
        else:
            raise ValueError(f"Unsupported collision geom type {typ}: {m.geom(i).name}")
        floor_enabled = any(compatible(i, int(plane)) for plane in planes)
        if not self_contacts and len(planes) and not floor_enabled:
            raise ValueError("Per-shape collision filters require self_contacts=True")
        geom_to_shape[i] = len(S)
        row = [
            b,
            kind,
            *center,
            *q,
            *m.geom_size[i],
            m.geom_friction[i, 0],
            len(vertices),
            *np.asarray(vertices).ravel(),
        ]
        if self_contacts:
            row.append(int(floor_enabled))
        S.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def line(values):
        return " ".join(
            str(x) if isinstance(x, (str, int)) else format(float(x), ".17g") for x in values
        )

    lines = [
        f"DUCK_MODEL {2 if self_contacts else 1} {len(B)} {len(J)} {len(S)}",
        line([*m.opt.gravity, int(ground)]),
    ]
    lines += [
        line([b["name"], b["mass"], *b["inertia"], *b["x"], *b["q"], *b["v"], *b["w"]]) for b in B
    ]
    lines += [line(j) for j in J] + [line(s) for s in S]
    if self_contacts:
        lines += [str(len(pairs))] + [
            line([geom_to_shape[a], geom_to_shape[b], mu]) for a, b, mu in pairs
        ]
    path.write_text("\n".join(lines) + "\n")
    return B
