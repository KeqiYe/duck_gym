"""Newton SolverMuJoCo adapter; full Newton state/control exchange is timed."""

from types import SimpleNamespace
import json


def setup_newton(a, folder, meta, reference):
    import mujoco
    import newton
    import numpy as np
    import warp as wp

    template = newton.ModelBuilder(gravity=reference.opt.gravity.tolist())
    template.default_shape_cfg.gap = 0.0
    template.add_mjcf(
        str(folder / "model/benchmark.xml"),
        ctrl_direct=True,
        # The imported MJCF masks already allow ground/robot pairs only.
        # False would filter *all* shapes in this import, including the floor.
        enable_self_collisions=True,
        ignore_inertial_definitions=False,
        parse_mujoco_options=True,
        mesh_maxhullvert=-1,
    )
    template.gravity = reference.opt.gravity.tolist()
    # MuJoCo free joint uses wxyz; Newton uses xyzw. Hinge order is checked below.
    assert template.joint_coord_count == 21
    template.joint_q[:7] = list(meta["qpos"][:3]) + list(meta["qpos"][4:7]) + [meta["qpos"][3]]
    for name, value in zip(meta["joint_names"], meta["home"]):
        indices = [
            i for i, label in enumerate(template.joint_label) if label.split("/")[-1] == name
        ]
        assert len(indices) == 1, (name, template.joint_label)
        template.joint_q[template.joint_q_start[indices[0]]] = value
    builder = newton.ModelBuilder(gravity=reference.opt.gravity.tolist())
    builder.replicate(template, world_count=a.envs[0])
    model = builder.finalize(device="cuda:0")
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        separate_worlds=True,
        iterations=a.mjwarp_iterations,
        ls_iterations=int(reference.opt.ls_iterations),
        tolerance=float(reference.opt.tolerance),
        integrator="euler",
        solver="newton",
        nconmax=a.nconmax,
        njmax=a.njmax,
        enable_sleeping=False,
        enable_multiccd=not bool(reference.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD),
        update_data_interval=1,
        skip_visual_only_geoms=False,
        save_to_mjcf=str(folder / "model/newton-export.xml"),
    )
    exported = solver.mj_model
    audit = dict(
        reference_gravity=reference.opt.gravity.tolist(),
        newton_gravity=exported.opt.gravity.tolist(),
        reference_counts={
            k: int(getattr(reference, k)) for k in ["nbody", "nq", "nv", "nu", "ngeom"]
        },
        newton_counts={k: int(getattr(exported, k)) for k in ["nbody", "nq", "nv", "nu", "ngeom"]},
        reference_mass=float(reference.body_mass.sum()),
        newton_mass=float(exported.body_mass.sum()),
        parameters={},
    )
    for field in [
        "body_mass",
        "body_inertia",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "jnt_range",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "geom_gap",
        "geom_margin",
    ]:
        x, y = getattr(reference, field), getattr(exported, field)
        audit["parameters"][field] = dict(
            reference=x.tolist(),
            newton=y.tolist(),
            max_abs_difference=float(np.max(np.abs(x - y))) if x.shape == y.shape else None,
        )
    # Newton may rotate link/COM frames and permute principal inertia axes.
    # Compare physical COM positions and inertia tensors in world coordinates.
    ref_data = mujoco.MjData(reference)
    ref_data.qpos[:] = meta["qpos"]
    mujoco.mj_forward(reference, ref_data)
    new_data = mujoco.MjData(exported)
    new_data.qpos[:] = solver.mj_data.qpos
    mujoco.mj_forward(exported, new_data)

    def world_inertias(m, d):
        r = d.ximat.reshape(-1, 3, 3)
        return (r * m.body_inertia[:, None, :]) @ r.transpose(0, 2, 1)

    audit["max_initial_com_difference_m"] = float(np.max(np.abs(ref_data.xipos - new_data.xipos)))
    audit["max_world_inertia_difference_kg_m2"] = float(
        np.max(np.abs(world_inertias(reference, ref_data) - world_inertias(exported, new_data)))
    )
    # Compare active collider counts and the broad-phase pair mask exactly.
    ref_active = np.flatnonzero(reference.geom_contype | reference.geom_conaffinity)
    new_active = np.flatnonzero(exported.geom_contype | exported.geom_conaffinity)
    audit["active_colliders"] = [len(ref_active), len(new_active)]
    audit["initial_joint_positions"] = new_data.qpos.tolist()
    rng = np.random.default_rng(0)
    directions = np.vstack([np.eye(3), -np.eye(3), rng.normal(size=(128, 3))])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    support_errors = []
    for ref_g, new_g in zip(ref_active, new_active):
        if reference.geom_type[ref_g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        def supports(m, d, g):
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = (
                m.mesh_vert[start : start + count] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            )
            return (vertices @ directions.T).max(axis=0)

        support_errors.append(
            float(
                np.max(
                    np.abs(
                        supports(reference, ref_data, ref_g) - supports(exported, new_data, new_g)
                    )
                )
            )
        )
    audit["max_collider_support_difference_m"] = max(support_errors)
    audit["mesh_maxhullvert"] = -1
    (folder / "model/import-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    assert audit["reference_counts"] == audit["newton_counts"], (
        audit["reference_counts"],
        audit["newton_counts"],
    )
    # Shape/ordering differences must be investigated rather than benchmarking a reduced robot.
    assert np.isclose(audit["reference_mass"], audit["newton_mass"], rtol=1e-6)
    assert np.allclose(exported.opt.gravity, reference.opt.gravity, rtol=1e-6, atol=1e-8)
    assert len(ref_active) == len(new_active) == 12, audit["active_colliders"]
    assert audit["max_initial_com_difference_m"] < 1e-6, audit["max_initial_com_difference_m"]
    assert audit["max_world_inertia_difference_kg_m2"] < 1e-9
    assert audit["max_collider_support_difference_m"] < 1e-6
    for field in [
        "body_mass",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "jnt_range",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    ]:
        assert audit["parameters"][field]["max_abs_difference"] < 1e-6, field
    for field in [
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_gap",
        "geom_margin",
        "geom_contype",
        "geom_conaffinity",
    ]:
        assert np.allclose(
            getattr(reference, field)[ref_active],
            getattr(exported, field)[new_active],
            rtol=1e-5,
            atol=1e-8,
        ), field
    states = [model.state(), model.state()]
    control = model.control()
    control.mujoco.ctrl.assign(np.tile(meta["home"], a.envs[0]).astype(np.float32))

    def reset():
        for state in states:
            solver.reset(state)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            state.clear_forces()
        solver.mjw_data.time.zero_()

    def step():
        solver.step(states[0], states[1], control, None, a.dt)
        states.reverse()

    return SimpleNamespace(
        solver=solver,
        model=model,
        states=states,
        control=control,
        reset=reset,
        step=step,
        settings=dict(
            newton_solver="SolverMuJoCo",
            newton_use_mujoco_contacts=True,
            newton_update_data_interval=1,
            newton_sleeping=False,
            newton_timed_scope="Full solver.step: control mapping, state synchronization, physics, output state mapping",
            newton_import_audit="model/import-audit.json",
        ),
    )
