"""Map maximal-coordinate poses to complete MuJoCo visual assets, without stepping."""

import numpy as np
import mujoco
from .env import quat_mul


def apply_body_poses(model, data, bodies):
    """Native poses are world COM + wxyz principal-frame orientation.

    Every link keeps its own simulated pose, including constraint residuals;
    reconstructing through joint qpos would incorrectly hide these residuals.
    """
    if bodies.shape != (model.nbody, 13) or not np.isfinite(bodies).all():
        raise ValueError("Expected finite [nbody,13] native snapshot")
    qnorm = np.linalg.norm(bodies[:, 3:7], axis=1)
    if np.any(np.abs(qnorm - 1) > 1e-6):
        raise ValueError("Invalid body quaternion")
    data.xipos[:] = bodies[:, :3]
    data.xquat[:] = quat_mul(bodies[:, 3:7], model.body_iquat * [1, -1, -1, -1])
    for b in range(model.nbody):
        mujoco.mju_quat2Mat(data.xmat[b], data.xquat[b])
        mujoco.mju_quat2Mat(data.ximat[b], bodies[b, 3:7])
    R = data.xmat.reshape(-1, 3, 3)
    data.xpos[:] = bodies[:, :3] - np.einsum("bij,bj->bi", R, model.body_ipos)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        data.geom_xpos[g] = data.xpos[b] + R[b] @ model.geom_pos[g]
        mujoco.mju_quat2Mat(data.geom_xmat[g], quat_mul(data.xquat[b], model.geom_quat[g]))
    for s in range(model.nsite):
        b = model.site_bodyid[s]
        data.site_xpos[s] = data.xpos[b] + R[b] @ model.site_pos[s]
        mujoco.mju_quat2Mat(data.site_xmat[s], quat_mul(data.xquat[b], model.site_quat[s]))
