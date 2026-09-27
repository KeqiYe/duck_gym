"""Newton's maximal-coordinate rigid AVBD, with an explicit common-model audit.

No SolverMuJoCo stepping/export is used. MuJoCo supplies only the independent
compiled import reference. Diagnostics are read-only and outside GPU timing.
"""
from types import SimpleNamespace
import hashlib
import json


def _rotation(q):
    import numpy as np
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def _apply(pose, points):
    import numpy as np
    return pose[..., :3] + np.einsum("...ij,...j->...i", _rotation(pose[..., 3:]), points)


def _collision_buffer_diagnostics(pipeline, contacts, history):
    """Read Newton 1.6 mesh-plane counters; never launch collision or mutate history.

    The import audit excludes other pair types. Missing expected counters are
    reported explicitly, not interpreted as zero. Hashtable active count lives
    at active_slots[capacity], rather than active_slots[0].
    """
    import numpy as np

    counters, unsupported = {}, []

    def read(name, array, capacity, index=0, full_is_uncertain=False):
        if array is None or capacity is None or index >= array.shape[0]:
            counters[name] = dict(status="unsupported", count=None, capacity=capacity)
            unsupported.append(name)
            return
        value = int(array[index:index+1].numpy().reshape(-1)[0])
        counters[name] = dict(status="available", count=value, capacity=int(capacity),
                              overflow=bool(value < 0 or value > capacity))
        if full_is_uncertain:
            # export_contact_to_buffer undoes failed reservations, so the
            # published count saturates and cannot reveal the number dropped.
            counters[name]["capacity_exhausted_unproven"] = bool(value >= capacity)

    def size(array):
        return None if array is None else int(array.shape[0])

    narrow = getattr(pipeline, "narrow_phase", None)
    read("broad_phase_pairs", getattr(pipeline, "broad_phase_pair_count", None),
         size(getattr(pipeline, "broad_phase_shape_pairs", None)))
    read("mesh_plane_pairs", getattr(narrow, "shape_pairs_mesh_plane_count", None),
         size(getattr(narrow, "shape_pairs_mesh_plane", None)))
    read("final_contacts", contacts.rigid_contact_count, int(contacts.rigid_contact_max))
    reducer = getattr(narrow, "global_contact_reducer", None)
    if reducer is not None:
        read("reducer_reserved_contacts", getattr(reducer, "contact_count", None),
             getattr(reducer, "capacity", None), full_is_uncertain=True)
        table = getattr(reducer, "hashtable", None)
        table_capacity = getattr(table, "capacity", None)
        read("reducer_hash_active", getattr(table, "active_slots", None), table_capacity,
             index=0 if table_capacity is None else int(table_capacity))
        read("reducer_hash_insert_failures", getattr(reducer, "ht_insert_failures", None), 0)
        hash_count = counters["reducer_hash_active"].get("count")
        if hash_count is not None and table_capacity:
            counters["reducer_hash_active"]["load_fraction"] = hash_count / table_capacity
    elif getattr(narrow, "reduce_contacts", None) is False:
        counters["reducer"] = dict(status="disabled")
    else:
        counters["reducer"] = dict(status="unsupported")
        unsupported.append("reducer")

    matching = dict(enabled=history)
    if history:
        matcher = getattr(pipeline, "_contact_matcher", None)
        match_capacity = getattr(matcher, "_capacity", None)
        read("matching_saved_contacts", getattr(matcher, "prev_contact_count", None), match_capacity)
        indices = getattr(contacts, "rigid_contact_match_index", None)
        read("matching_input_contacts", contacts.rigid_contact_count, size(indices))
        if indices is not None and match_capacity is not None:
            count = int(contacts.rigid_contact_count.numpy()[0])
            # Warp 1.17 cannot convert a zero-length array view with numpy().
            # Copy the allocated buffer first, then slice on the host; early
            # airborne frames legitimately have zero contact matches.
            index = indices.numpy()[:min(max(count, 0), indices.shape[0])]
            invalid = int(np.count_nonzero((index < -2) | (index >= match_capacity)))
            matching.update(matched_contacts=int(np.count_nonzero(index >= 0)),
                            invalid_indices=invalid, capacity=int(match_capacity),
                            dedicated_overflow_counter="not_exposed; check input/saved counts and index bounds",
                            saved_count_scope="After collide save_sorted_state; this is the current frame count")
        else:
            matching.update(status="unsupported", invalid_indices=None)
            unsupported.append("matching_indices")
    issue = any(row.get("overflow", False) or row.get("capacity_exhausted_unproven", False)
                for row in counters.values()) or bool(matching.get("invalid_indices"))
    return dict(counters=counters, matching=matching, buffer_issue=bool(issue),
                unsupported=unsupported, exposed_counter_coverage_complete=not unsupported,
                reducer_dropped_contact_counter="not_exposed; saturated reservation buffer cannot certify no drops",
                scope="Last mesh-plane collision only; import audit excludes mesh-mesh, mesh-convex, SDF and generic GJK pairs")


def setup_newton_avbd(a, folder, meta, reference):
    import mujoco
    import newton
    import numpy as np
    import warp as wp
    from scipy.spatial import cKDTree

    if getattr(a, "physics_profile", None) != "avbd-common":
        raise ValueError("Newton rigid AVBD requires the explicit avbd-common profile")
    if getattr(a, "chunk_steps", 20) % 2:
        raise ValueError("Captured ping-pong replay requires an even chunk_steps")
    if not (a.dt > 0 and a.avbd_iterations > 0 and a.nconmax > 0):
        raise ValueError("Invalid dt, iterations, or contact capacity")
    if np.any(reference.dof_armature) or np.any(reference.dof_frictionloss):
        raise ValueError("Common reference must explicitly remove armature and joint dry friction")
    if np.any(reference.actuator_forcelimited):
        raise ValueError("Common reference must explicitly disable actuator effort caps")
    n = int(a.envs[0])
    template = newton.ModelBuilder(gravity=reference.opt.gravity.tolist())
    template.default_shape_cfg.gap = 0.0
    template.add_mjcf(str(folder / "model/benchmark.xml"), ctrl_direct=True,
                      enable_self_collisions=True, ignore_inertial_definitions=False,
                      parse_mujoco_options=True, mesh_maxhullvert=-1)
    template.gravity = reference.opt.gravity.tolist()
    if (template.joint_coord_count, template.joint_dof_count) != (21, 20):
        raise ValueError("Import changed the complete MicroDuck 21q/20v topology")

    def suffix(label):
        return str(label).split("/")[-1]

    def unique(labels, name):
        found = [i for i, label in enumerate(labels) if suffix(label) == name]
        if len(found) != 1:
            raise ValueError(f"Ambiguous/missing imported name {name}: {found}")
        return found[0]

    ref_data = mujoco.MjData(reference)
    ref_data.qpos[:] = meta["qpos"]
    mujoco.mj_forward(reference, ref_data)
    template.joint_q[:7] = list(meta["qpos"][:3])+list(meta["qpos"][4:7])+[meta["qpos"][3]]
    hinge_ids, joint_audit = [], []
    for name, target in zip(meta["joint_names"], meta["home"]):
        j = unique(template.joint_label, name)
        ref_j = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_JOINT, name)
        if ref_j < 0 or int(template.joint_type[j]) != int(newton.JointType.REVOLUTE):
            raise ValueError(f"Not the expected hinge: {name}")
        q, d = int(template.joint_q_start[j]), int(template.joint_qd_start[j])
        ref_d = int(reference.jnt_dofadr[ref_j])
        damping = float(reference.dof_damping[ref_d])
        if not np.isclose(damping, .053, rtol=0, atol=1e-8):
            raise ValueError(f"Unexpected upstream damping for {name}: {damping}")
        template.joint_q[q] = float(target)
        template.joint_target_q[q] = float(target)  # builder stores coord layout
        template.joint_target_qd[d] = 0.0
        template.joint_target_ke[d] = float(meta["kp"])
        template.joint_target_kd[d] = damping
        # VBD ignores joint_damping; its absolute drive/limit kd implement this
        # common viscous coefficient. Legacy mode selects drive OR limit.
        template.joint_damping[d] = damping
        template.joint_limit_ke[d] = 1e7
        template.joint_limit_kd[d] = damping
        template.joint_armature[d] = 0.0
        template.joint_friction[d] = 0.0
        template.joint_effort_limit[d] = float("inf")  # unsupported by VBD, uncapped
        actual_range = [template.joint_limit_lower[d], template.joint_limit_upper[d]]
        if not np.allclose(actual_range, reference.jnt_range[ref_j], atol=1e-6, rtol=1e-6):
            raise ValueError(f"Joint limit import mismatch: {name}")
        hinge_ids.append(j)
        joint_audit.append(dict(name=name, newton_joint=j, q_index=q, dof_index=d,
                                reference_joint=ref_j, target=float(target), kp=float(meta["kp"]),
                                absolute_viscous_kd=damping, lower_upper=list(map(float, actual_range))))
    if len(hinge_ids) != 14 or len(set(hinge_ids)) != 14:
        raise ValueError("All fourteen named hinges are required")
    if not np.isclose(meta["kp"], .55, rtol=0, atol=1e-8):
        raise ValueError("Expected the common HOME kp=.55")

    active = [i for i, f in enumerate(template.shape_flags)
              if int(f) & int(newton.ShapeFlags.COLLIDE_SHAPES)]
    ground = [i for i in active if int(template.shape_type[i]) == int(newton.GeoType.PLANE)]
    robot = [i for i in active if i not in ground]
    if len(ground) != 1 or len(robot) != 11:
        raise ValueError(f"Expected all 11 colliders and floor, got {len(robot)}, {len(ground)}")
    if not np.allclose([template.shape_material_mu[i] for i in robot], 1.0, rtol=0, atol=1e-7):
        raise ValueError("Native ground friction matching requires all robot collider mu=1")
    original_floor_mu = float(template.shape_material_mu[ground[0]])
    template.shape_material_mu[ground[0]] = 1.0
    material_before = [dict(shape=i, ke=float(template.shape_material_ke[i]),
                            kd=float(template.shape_material_kd[i])) for i in active]
    for i in active:
        template.shape_material_ke[i] = 1e9
        template.shape_material_kd[i] = 0.0
    builder = newton.ModelBuilder(gravity=reference.opt.gravity.tolist())
    builder.replicate(template, world_count=n)
    builder.color()
    model = builder.finalize(device=getattr(a, "newton_device", "cuda:0"))
    # This must happen before construction: VBD captures structural rest poses.
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)

    if model.particle_count != 0 or model.joint_count != 15*n:
        raise ValueError("Expected rigid-only model with fifteen joints per environment")
    bcount = len(template.body_mass)
    body_world = model.body_world.numpy()
    first_bodies = np.flatnonzero(body_world == 0)
    if len(first_bodies) != bcount or bcount != 15 or model.body_count != n*bcount:
        raise ValueError("Complete 15-body robot replication changed")
    body_pose = model.body_q.numpy()
    com_local = model.body_com.numpy()
    masses = model.body_mass.numpy()
    inertia = model.body_inertia.numpy().reshape(-1, 3, 3)
    ref_ids = []
    for b in first_bodies:
        name = suffix(model.body_label[b])
        ref_id = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, name)
        if ref_id <= 0:
            raise ValueError(f"Imported body has no reference match: {name}")
        ref_ids.append(ref_id)
    if sorted(ref_ids) != list(range(1, reference.nbody)):
        raise ValueError("Reference/import body bijection failed")
    ref_ids = np.asarray(ref_ids)
    initial_com = _apply(body_pose[first_bodies], com_local[first_bodies])
    r = _rotation(body_pose[first_bodies, 3:])
    wi = r @ inertia[first_bodies] @ r.transpose(0, 2, 1)
    rr = ref_data.ximat[ref_ids].reshape(-1, 3, 3)
    ref_wi = (rr * reference.body_inertia[ref_ids, None, :]) @ rr.transpose(0, 2, 1)
    audit = dict(reference_body_ids=ref_ids.tolist(), body_count_per_env=bcount,
                 joint_order=joint_audit, reference_mass=float(reference.body_mass.sum()),
                 newton_mass=float(masses[first_bodies].sum()),
                 max_body_mass_difference_kg=float(np.max(np.abs(masses[first_bodies]-reference.body_mass[ref_ids]))),
                 max_initial_com_difference_m=float(np.max(np.abs(initial_com-ref_data.xipos[ref_ids]))),
                 max_world_inertia_difference_kg_m2=float(np.max(np.abs(wi-ref_wi))),
                 mesh_maxhullvert=-1, full_vertices=[],
                 common_removed=["joint armature", "joint dry friction", "effort cap"],
                 friction=dict(original_floor_mu=original_floor_mu, newton_floor_mu=1.0,
                               robot_mu=1.0, newton_mixing="sqrt(mu0*mu1)", effective_mu=1.0,
                               reason="Match native ground use of robot mu; preserve native effective coefficient"),
                 contact_import_material_before_override=material_before,
                 contact_legacy_penalty_cap=1e9, contact_absolute_damping=0.0)
    if not (audit["max_body_mass_difference_kg"] < 1e-6 and
            audit["max_initial_com_difference_m"] < 1e-6 and
            audit["max_world_inertia_difference_kg_m2"] < 1e-9):
        raise ValueError(f"Mass/COM/world inertia import mismatch: {audit}")
    if not np.allclose(model.gravity.numpy(), reference.opt.gravity, rtol=1e-6, atol=1e-8):
        raise ValueError("Gravity import mismatch")

    shape_world = model.shape_world.numpy()
    first_shapes = np.flatnonzero(shape_world == 0)
    if len(first_shapes) != len(template.shape_type):
        raise ValueError("Shape replication order changed")
    shape_body = model.shape_body.numpy()
    shape_pose = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    flags = model.shape_flags.numpy()
    ref_active = np.flatnonzero(reference.geom_contype | reference.geom_conaffinity)
    if len(ref_active) != len(active):
        raise ValueError("Active collider count changed")
    rng = np.random.default_rng(0)
    directions = np.vstack((np.eye(3), -np.eye(3), rng.normal(size=(128, 3))))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for local in robot:
        s = int(first_shapes[local])
        name = suffix(model.shape_label[s])
        g = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0 or g not in ref_active or reference.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"Collider name/type mismatch: {name}")
        mesh = model.shape_source[s]
        if mesh is None or not hasattr(mesh, "vertices") or mesh.maxhullvert != -1:
            raise ValueError(f"Original uncapped collider mesh unavailable: {name}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)*shape_scale[s]
        local_points = vertices @ _rotation(shape_pose[s, 3:]).T+shape_pose[s, :3]
        b = int(shape_body[s])
        points = local_points @ _rotation(body_pose[b, 3:]).T+body_pose[b, :3]
        mid = int(reference.geom_dataid[g]); start = int(reference.mesh_vertadr[mid]); count = int(reference.mesh_vertnum[mid])
        rp = reference.mesh_vert[start:start+count] @ ref_data.geom_xmat[g].reshape(3, 3).T+ref_data.geom_xpos[g]
        # Bidirectional complete point-cloud check, allowing only duplicate
        # vertices introduced/removed by triangle mesh import, not simplification.
        error = max(float(cKDTree(points).query(rp)[0].max()), float(cKDTree(rp).query(points)[0].max()))
        support_error = float(np.max(np.abs((points @ directions.T).max(0)-(rp @ directions.T).max(0))))
        audit["full_vertices"].append(dict(name=name, reference_vertices=count, newton_vertices=len(points),
                                           bidirectional_vertex_error_m=error, support_error_m=support_error,
                                           newton_vertices_sha256=hashlib.sha256(np.asarray(mesh.vertices).tobytes()).hexdigest()))
        if error >= 1e-6 or support_error >= 1e-6:
            raise ValueError(f"Full mesh geometry import mismatch: {name}, {error}, {support_error}")
    pairs = model.shape_contact_pairs.numpy()
    if pairs.size:
        same_world = shape_world[pairs[:, 0]] == shape_world[pairs[:, 1]]
        plane_flags = model.shape_type.numpy() == int(newton.GeoType.PLANE)
        ground_only = plane_flags[pairs[:, 0]] ^ plane_flags[pairs[:, 1]]
        if not np.all(same_world & ground_only):
            raise ValueError("Collision mask contains cross-env or self-collision pairs")
    if len(shape_world) != n*len(template.shape_type) or not np.array_equal(
            shape_world, np.repeat(np.arange(n), len(template.shape_type))):
        raise ValueError("Expected contiguous complete per-world shape replication")
    starts = np.arange(n, dtype=np.int64)*len(template.shape_type)
    expected = np.stack((np.broadcast_to((starts+ground[0])[:, None], (n, 11)),
                         starts[:, None]+np.asarray(robot)[None, :]), axis=-1).reshape(-1, 2)
    expected = np.sort(expected, axis=1)
    actual = np.sort(pairs.reshape(-1, 2), axis=1)
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    actual = actual[np.lexsort((actual[:, 1], actual[:, 0]))]
    if not np.array_equal(actual, expected):
        raise ValueError("Ground-pair set differs from all 11 colliders in every world")
    audit["expected_contact_pair_count"] = 11*n
    audit["contact_pair_count"] = int(len(pairs))
    audit["active_colliders_per_env"] = len(active)
    audit["body_color_group_sizes"] = [int(len(group)) for group in model.body_color_groups]
    audit["reference_model_sha256"] = hashlib.sha256((folder / "model/benchmark.xml").read_bytes()).hexdigest()
    (folder / "model/newton-avbd-import-audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")

    contact_history = bool(getattr(a, "newton_avbd_contact_history", False))
    verify_buffers = bool(getattr(a, "newton_avbd_verify_buffers", False))
    matching_mode = "latest" if contact_history else "disabled"
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=n*int(a.nconmax),
                                        contact_matching=matching_mode, deterministic=a.newton_avbd_deterministic,
                                        verify_buffers=verify_buffers)
    contacts = pipeline.contacts()
    legacy = dict(iterations=int(a.avbd_iterations), rigid_compliant_alm=False,
                  rigid_avbd_alpha=.9, rigid_avbd_gamma=.99,
                  rigid_avbd_linear_beta=1e7, rigid_avbd_angular_beta=1e5,
                  rigid_contact_hard=True, rigid_contact_history=contact_history,
                  rigid_contact_k_start=1000.0, rigid_joint_linear_k_start=1000.0,
                  rigid_joint_angular_k_start=1.0, rigid_joint_linear_ke=1e9,
                  rigid_joint_angular_ke=1e7, rigid_joint_linear_kd=0.0,
                  rigid_joint_angular_kd=0.0, friction_epsilon=.01,
                  rigid_body_contact_buffer_size=max(64, int(a.nconmax)))
    mode = wp.DeterministicMode.RUN_TO_RUN if a.newton_avbd_deterministic else wp.DeterministicMode.NOT_GUARANTEED
    solver = newton.solvers.SolverVBD(model, deterministic=mode, **legacy)
    initial_states = (model.state(), model.state())
    states = list(initial_states)
    control = model.control()
    # Use the finalized target layout, which may be coord or legacy DOF layout.
    target = model.joint_target_q.numpy().copy()
    target_starts = model.joint_target_q_start.numpy()
    dof_starts = model.joint_qd_start.numpy()
    hinge_by_name = {row["name"]: row["target"] for row in joint_audit}
    hinge_all = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
    if len(hinge_all) != n*14:
        raise ValueError("Replicated hinge count mismatch")
    for j in hinge_all:
        target[int(target_starts[j])] = hinge_by_name[suffix(model.joint_label[j])]
    control.joint_target_q.assign(target)
    control.joint_target_qd.zero_()
    # PD targets remain active. There is no external generalized joint force;
    # None skips SolverVBD's otherwise unconditional copy + apply_joint_forces.
    control.joint_f = None

    def reset():
        # Stable identities ensure a captured even-length ping-pong chunk can
        # replay repeatedly after reset, without relying on Python graph replay.
        states[:] = initial_states
        for state in initial_states:
            solver.reset(state)
            wp.copy(state.joint_q, model.joint_q)
            wp.copy(state.joint_qd, model.joint_qd)
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            state.clear_forces()
        pipeline.reset_contact_matching()
        contacts.clear()

    def step():
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, a.dt)
        states.reverse()

    parent = model.joint_parent.numpy()[hinge_all]
    child = model.joint_child.numpy()[hinge_all]
    xp = model.joint_X_p.numpy()[hinge_all]
    xc = model.joint_X_c.numpy()[hinge_all]
    axis = model.joint_axis.numpy()[dof_starts[hinge_all]].astype(np.float64)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    # Orthonormal transverse basis, for the dimensionless axis dot residual.
    helper = np.eye(3)[np.argmin(np.abs(axis), axis=1)]
    tangent = np.cross(axis, helper); tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(axis, tangent)
    root_body_local = int(np.flatnonzero(ref_ids == 1)[0])
    root_ids = model.body_world_start.numpy()[:n]+root_body_local

    def diagnostics():
        # Read-only: never rerun collision detection, reset history or infer
        # body motion from VBD's stale generalized state.joint_q/qd arrays.
        pose = states[0].body_q.numpy().astype(np.float64)
        velocity = states[0].body_qd.numpy()
        rotation = _rotation(pose[:, 3:])
        invalid = ~(np.isfinite(pose).all(1) & np.isfinite(velocity).all(1))
        invalid |= np.linalg.norm(pose[:, 3:], axis=1) < .5
        bad_worlds = np.unique(body_world[invalid & (body_world >= 0)])
        parent_pose = np.zeros((len(parent), 7)); parent_pose[:, 6] = 1
        dynamic = parent >= 0; parent_pose[dynamic] = pose[parent[dynamic]]
        anchor_p, anchor_c = _apply(parent_pose, xp[:, :3]), _apply(pose[child], xc[:, :3])
        rp = _rotation(parent_pose[:, 3:]) @ _rotation(xp[:, 3:])
        rc = rotation[child] @ _rotation(xc[:, 3:])
        ap = np.einsum("nij,nj->ni", rp, axis); ac = np.einsum("nij,nj->ni", rc, axis)
        dot = np.sum(ap*ac, axis=1)
        angle = np.arctan2(np.linalg.norm(np.cross(ap, ac), axis=1), dot)
        transverse = np.stack((np.einsum("nij,nj->ni", rp, tangent), np.einsum("nij,nj->ni", rp, bitangent)), axis=1)
        axis_residual = np.max(np.abs(np.einsum("nkj,nj->nk", transverse, ac)))
        count = int(contacts.rigid_contact_count.numpy()[0]); capacity = int(contacts.rigid_contact_max)
        overflow_max = int(solver.body_body_contact_overflow_max.numpy()[0])
        overflow = count > capacity or overflow_max > int(solver.body_body_contact_buffer_pre_alloc)
        collision_buffers = _collision_buffer_diagnostics(pipeline, contacts, contact_history)
        used = min(max(count, 0), capacity)
        max_penetration = 0.0
        invalid_contact_ids = 0
        if used:
            s0, s1 = contacts.rigid_contact_shape0.numpy()[:used], contacts.rigid_contact_shape1.numpy()[:used]
            valid = (s0 >= 0) & (s1 >= 0) & (s0 < model.shape_count) & (s1 < model.shape_count)
            invalid_contact_ids = int((~valid).sum())
            b0, b1 = shape_body[s0[valid]], shape_body[s1[valid]]
            def point_world(body, point):
                out = point.astype(np.float64).copy(); moving = body >= 0
                out[moving] = _apply(pose[body[moving]], out[moving]); return out
            p0 = point_world(b0, contacts.rigid_contact_point0.numpy()[:used][valid])
            p1 = point_world(b1, contacts.rigid_contact_point1.numpy()[:used][valid])
            normal = contacts.rigid_contact_normal.numpy()[:used][valid]
            gap = np.sum((p1-p0)*normal, axis=1)-contacts.rigid_contact_margin0.numpy()[:used][valid]-contacts.rigid_contact_margin1.numpy()[:used][valid]
            if len(gap):
                max_penetration = float(max(0.0, -np.min(gap))) if np.isfinite(gap).all() else None
        root_com = _apply(pose[root_ids], com_local[root_ids])
        finite = not len(bad_worlds) and np.isfinite(root_com).all() and max_penetration is not None
        def finite_max(value):
            value = np.asarray(value)
            return float(np.max(value)) if np.isfinite(value).all() else None
        return dict(finite=bool(finite), numeric_failure_envs=int(len(bad_worlds)),
                    numeric_failure_scope="Observed invalid body pose/velocity; SolverVBD exposes no equivalent native failure flag",
                    max_joint_anchor_error_m=finite_max(np.abs(anchor_p-anchor_c)),
                    max_joint_anchor_separation_m=finite_max(np.linalg.norm(anchor_p-anchor_c, axis=1)),
                    anchor_error_note="Component L-infinity matches native key; separation is Euclidean norm",
                    max_axis_error_rad=finite_max(angle),
                    max_axis_orthogonality_residual=finite_max(axis_residual),
                    axis_error_note="Angle is atan2(|axisP cross axisC|,dot); orthogonality residual is dimensionless, native legacy max_axis_error_rad is two such dot residuals",
                    max_penetration_m=max_penetration,
                    penetration_scope="Last collision witnesses projected through final body poses, original normals and margins; no extra collision pass",
                    min_root_height_m=float(root_com[:, 2].min()) if finite else None,
                    max_root_height_m=float(root_com[:, 2].max()) if finite else None,
                    root_com_first_env=root_com[0].tolist() if finite else None,
                    contact_count_total=count, contact_capacity=capacity,
                    contact_overflow=bool(overflow),
                    collision_buffer_diagnostics=collision_buffers,
                    overflow_flags=[int(overflow), invalid_contact_ids, int(collision_buffers["buffer_issue"])],
                    max_body_contact_count=int(solver.body_body_contact_counts.numpy().max()),
                    overflow_scope="Last physical step; use per-step preflight to cover whole trajectory")

    reset()
    settings = dict(solver="Newton SolverVBD rigid legacy AVBD, maximal coordinates",
                    newton_solver="SolverVBD", physics_profile="avbd-common", legacy_parameters=legacy,
                    common_removed=audit["common_removed"], implicit_pd=True,
                    damping_mapping="joint_target_kd=.053, target_qd=0; joint_limit_kd=.053 retains damping when legacy limit takes priority",
                    effort_cap_enabled=False, effective_ground_friction=1.0,
                    model_difference_notes=["Legacy VBD chooses drive OR limit; native can differ at active limits",
                                            "Newton friction epsilon .01 m/s and contact manifold differ from native hard static friction",
                                            "Matched scalar ALM parameters do not prove identical constraint discretization or convergence"],
                    contact_history=contact_history, contact_matching=matching_mode,
                    contact_deterministic=bool(pipeline.deterministic),
                    contact_deterministic_requested=bool(a.newton_avbd_deterministic),
                    collision_verify_buffers=verify_buffers,
                    external_generalized_joint_force="None; PD targets and target damping retained",
                    contact_matching_thresholds=(dict(position_m=0.0005, normal_dot=0.995,
                                                       source="Newton 1.6 CollisionPipeline defaults")
                                                 if contact_history else None),
                    solver_deterministic_mode=mode.name,
                    mesh_maxhullvert=-1,
                    collision_broad_phase=pipeline.broad_phase_mode,
                    collision_gap_margin={name: getattr(model, name).numpy()[first_shapes].tolist()
                                          for name in ("shape_gap", "shape_margin") if hasattr(model, name) and getattr(model, name) is not None},
                    newton_import_audit="model/newton-avbd-import-audit.json",
                    timed_scope="Every step: clear external forces, CollisionPipeline.collide, SolverVBD.step, state ping-pong",
                    reset_scope="Both states, solver dual/penalty/history, generalized defaults + FK, forces, pipeline matching, contacts",
                    graph_even_chunk_required=True, diagnostics_read_only=True,
                    contact_capacity_total=n*int(a.nconmax), device=str(model.device))
    return SimpleNamespace(reset=reset, step=step, diagnostics=diagnostics, settings=settings,
                           model=model, solver=solver, states=states, control=control,
                           collision_pipeline=pipeline, contacts=contacts)
