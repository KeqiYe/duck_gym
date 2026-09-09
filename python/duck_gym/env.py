import json
from pathlib import Path
import numpy as np
import torch
from _duck_cpu import CpuBatch


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate(
        (aw * bw - (av * bv).sum(-1, keepdims=True), aw * bv + bw * av + np.cross(av, bv)), -1
    )


def inverse_rotate(q, v):
    qv = -q[..., 1:]
    t = 2 * np.cross(qv, v)
    return v + q[..., :1] * t + np.cross(qv, t)


class StandingEnv:
    """rsl-rl 2.3.3 CPU VecEnv contract. Native snapshots own their memory.

    Actions are bounded target offsets from the upstream standing pose. A call
    advances every environment by control_dt; done observations are post-reset.
    time_outs distinguishes the bootstrap mask from physical failure.
    """

    def __init__(
        self,
        model_dir,
        num_envs=8,
        threads=4,
        dt=0.001,
        substeps=20,
        iterations=200,
        episode_seconds=30,
        seed=42,
        randomize=True,
        action_scale=0.15,
        auto_reset=True,
    ):
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "metadata.json").read_text())
        if (
            not np.isfinite(episode_seconds)
            or episode_seconds < dt * substeps
            or substeps < 1
            or not np.isfinite(action_scale)
            or action_scale <= 0
        ):
            raise ValueError("Invalid episode duration/substeps")
        self.native = CpuBatch(
            str(self.model_dir / "microduck.duck"), num_envs, dt, iterations, threads
        )
        assert self.native.joint_names == self.meta["joint_names"]
        self.num_envs, self.num_actions, self.device = num_envs, len(self.meta["home"]), "cpu"
        self.dt, self.substeps, self.control_dt = dt, substeps, dt * substeps
        self.max_episode_length = round(episode_seconds / self.control_dt)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.rng = np.random.default_rng(seed)
        self.randomize, self.action_scale, self.auto_reset = randomize, action_scale, auto_reset
        self.home = np.array(self.meta["home"])
        self.actions = np.zeros((num_envs, self.num_actions))
        self.forces = np.zeros((num_envs, 3))
        self.returns = np.zeros(num_envs)
        self.reset(np.arange(num_envs))

    def reset(self, ids):
        ids = np.asarray(ids, dtype=np.int32)
        roots = np.zeros((len(ids), 6))
        angles = np.tile(self.home, (len(ids), 1))
        if self.randomize:
            angles += self.rng.uniform(-0.01, 0.01, angles.shape)
            roots[:, 3:5] = self.rng.uniform(-0.01, 0.01, (len(ids), 2))
            roots[:, 2] = 0.001
        self.native.reset(ids.tolist(), angles, roots)
        self.episode_length_buf[ids] = 0
        self.actions[ids] = 0
        self.forces[ids] = 0
        self.returns[ids] = 0

    def state(self):
        body = self.native.body_state()
        joints = self.native.joint_state()
        q = quat_mul(body[:, 1, 3:7], np.array(self.meta["root_iquat"]) * [1, -1, -1, -1])
        gravity = inverse_rotate(q, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        velocity = inverse_rotate(q, body[:, 1, 7:10])
        omega = inverse_rotate(q, body[:, 1, 10:13])
        return body, joints, gravity, velocity, omega

    def get_observations(self):
        b, j, g, v, w = self.state()
        obs = np.concatenate(
            (
                g,
                v,
                w,
                b[:, 1, 2:3] - self.meta["root_height"],
                j[:, :, 0] - self.home,
                0.1 * j[:, :, 1],
                self.actions,
            ),
            axis=1,
        )
        result = torch.from_numpy(obs.astype(np.float32))
        return result, {"observations": {}}

    def step(self, actions):
        a = actions.detach().cpu().numpy().astype(np.float64)
        if a.shape != self.actions.shape or not np.isfinite(a).all():
            raise ValueError("Actions must be finite [num_envs, num_actions]")
        a = np.clip(a, -1, 1)
        target = np.clip(self.home + self.action_scale * a, self.meta["lower"], self.meta["upper"])
        self.native.step(
            target, self.substeps, self.meta["kp"], self.meta["torque_limit"], self.forces
        )
        b, j, g, v, w = self.state()
        diag = self.native.diagnostics()
        height_error = b[:, 1, 2] - self.meta["root_height"]
        failed = (
            (g[:, 2] > -np.cos(0.5))
            | (np.abs(height_error) > 0.04)
            | (diag[:, 0] > 0.005)
            | (diag[:, 1] > 0.005)
            | ~np.isfinite(b).all(axis=(1, 2))
        )
        reward = (
            1.5 * np.exp(-20 * np.square(g[:, :2]).sum(1))
            + np.exp(-2000 * height_error**2)
            + 0.5 * np.exp(-10 * np.square(v).sum(1))
            - 0.02 * np.square(w).sum(1)
            - 0.05 * np.square(j[:, :, 0] - self.home).sum(1)
            - 0.01 * np.square(a).sum(1)
            - 0.02 * np.square(a - self.actions).sum(1)
        )
        reward = np.where(failed, -2.0, reward) * self.control_dt
        self.actions[:] = a
        self.episode_length_buf += 1
        self.returns += reward
        timed_out = (self.episode_length_buf.numpy() >= self.max_episode_length) & ~failed
        done = failed | timed_out
        infos = {
            "observations": {},
            "time_outs": torch.from_numpy(timed_out.copy()),
            "failed": torch.from_numpy(failed.copy()),
            "diagnostics": diag,
        }
        if done.any():
            infos["episode"] = {
                "return": float(self.returns[done].mean()),
                "length_seconds": float(
                    self.episode_length_buf.numpy()[done].mean() * self.control_dt
                ),
                "failure": float(failed[done].mean()),
            }
            infos["terminal_observation"] = self.get_observations()[0][done].clone()
            infos["terminal_env_ids"] = torch.from_numpy(np.flatnonzero(done))
            if self.auto_reset:
                self.reset(np.flatnonzero(done))
        obs, _ = self.get_observations()
        return (
            obs,
            torch.from_numpy(reward.astype(np.float32)),
            torch.from_numpy(done.copy()),
            infos,
        )
