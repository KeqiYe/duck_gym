"""Tensor VecEnv for rsl-rl; native CPU/CUDA advances all physics."""

import json
import math
from pathlib import Path
import torch


def multiply(a, b):
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
            a[..., :1] * b[..., 1:]
            + b[..., :1] * a[..., 1:]
            + torch.linalg.cross(a[..., 1:], b[..., 1:]),
        ),
        dim=-1,
    )


def inverse_rotate(q, v):
    t = 2 * torch.linalg.cross(-q[..., 1:], v)
    return v + q[..., :1] * t + torch.linalg.cross(-q[..., 1:], t)


class CpuAdapter:
    def __init__(self, model, n, dt, iterations, fp64, threads):
        from _duck_reference import CpuBatch32, CpuBatch64

        self.batch = (CpuBatch64 if fp64 else CpuBatch32)(model, n, dt, iterations, threads)
        self.joint_names = self.batch.joint_names

    def reset(self, mask, q, roots):
        self.batch.reset(mask.numpy(), q.numpy(), roots.numpy())

    def step(self, target, forces, substeps, kp, limit):
        self.batch.step(target.numpy(), forces.numpy(), substeps, kp, limit)

    def state(self):
        return tuple(torch.from_numpy(v) for v in self.batch.state())

    def contact_forces(self):
        return torch.from_numpy(self.batch.contact_forces())


class TensorEnv:
    def __init__(
        self,
        model_dir,
        num_envs=128,
        iterations=200,
        dt=0.001,
        substeps=20,
        seed=42,
        action_scale=0.3,
        episode_seconds=30,
        task="standing",
        randomize=True,
        auto_reset=True,
        fp64=False,
        command_speed=0.05,
        backend="cuda",
        cpu_threads=1,
        gait=False,
        foot_clearance=False,
        tracking_sigma=None,
        load_transfer=False,
        velocity_filter_seconds=0.0,
        tracking_weight=4.0,
        heading_weight=0.4,
    ):
        if substeps < 1 or episode_seconds <= 0 or action_scale <= 0:
            raise ValueError("Invalid control configuration")
        if backend not in ("cpu", "cuda"):
            raise ValueError(backend)
        self.device = "cuda:0" if backend == "cuda" else "cpu"
        self.num_envs = num_envs
        self.num_actions = 14
        self.dtype = torch.float64 if fp64 else torch.float32
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "metadata.json").read_text())
        if backend == "cuda":
            from build_cuda import build

            self.ext = build()
            self.native = self.ext.CudaBatch(
                str(self.model_dir / "microduck.duck"), num_envs, dt, iterations, 0, fp64
            )
        else:
            self.native = CpuAdapter(
                str(self.model_dir / "microduck.duck"), num_envs, dt, iterations, fp64, cpu_threads
            )
        if self.native.joint_names != self.meta["joint_names"]:
            raise ValueError("Joint order mismatch")
        if task not in ("standing", "locomotion"):
            raise ValueError(task)
        self.task = task
        if any(not math.isfinite(x) or x < 0 for x in (tracking_weight, heading_weight)):
            raise ValueError("Reward weights must be finite and nonnegative")
        self.tracking_weight = tracking_weight
        self.heading_weight = heading_weight
        self.randomize = randomize
        self.auto_reset = auto_reset
        self.velocity_filter_seconds = velocity_filter_seconds
        if not math.isfinite(velocity_filter_seconds) or velocity_filter_seconds < 0:
            raise ValueError("velocity filter must be nonnegative")
        self.filtered_velocity = torch.zeros(num_envs, 2, device=self.device, dtype=self.dtype)
        if velocity_filter_seconds > 0 and task != "locomotion":
            raise ValueError("Velocity filter is for locomotion")
        self.control_dt = dt * substeps
        self.velocity_alpha = (
            1 - math.exp(-self.control_dt / velocity_filter_seconds)
            if velocity_filter_seconds > 0
            else 1.0
        )
        self.substeps = substeps
        self.action_scale = action_scale
        self.max_episode_length = round(episode_seconds / self.control_dt)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.home = torch.tensor(self.meta["home"], device=self.device, dtype=self.dtype)
        self.lower = torch.tensor(self.meta["lower"], device=self.device, dtype=self.dtype)
        self.upper = torch.tensor(self.meta["upper"], device=self.device, dtype=self.dtype)
        self.iquat = torch.tensor(
            self.meta["root_iquat"], device=self.device, dtype=self.dtype
        ) * torch.tensor([1, -1, -1, -1], device=self.device)
        self.actions = torch.zeros(num_envs, 14, device=self.device, dtype=self.dtype)
        self.forces = torch.zeros(num_envs, 3, device=self.device, dtype=self.dtype)
        self.commands = torch.zeros(num_envs, 2, device=self.device, dtype=self.dtype)
        self.gait = gait
        self.foot_clearance = foot_clearance
        self.load_transfer = load_transfer
        self.tracking_sigma = (
            tracking_sigma if tracking_sigma is not None else (0.02 if gait else 0.04)
        )
        if self.tracking_sigma <= 0:
            raise ValueError("tracking sigma must be positive")
        if gait or foot_clearance or load_transfer:
            import numpy as np

            record = np.load(self.model_dir / "gait.npz")
            self.gait_table = torch.tensor(record["targets"], device=self.device, dtype=self.dtype)
            self.gait_period = float(record["period"])
            self.gait_lift = float(record.get("lift", 0.008))
            if foot_clearance or load_transfer:
                self.foot_points = torch.tensor(
                    record["foot_collision_points"], device=self.device, dtype=self.dtype
                )
                self.foot_bodies = torch.tensor(
                    record["foot_bodies"], device=self.device, dtype=torch.long
                )
            if abs(float(record["speed"]) - command_speed) > 1e-8:
                raise ValueError("Gait command speed mismatch")
            self.gait_directions = torch.tensor(
                [[1.0, 0], [-1, 0], [0, 1], [0, -1]], device=self.device, dtype=self.dtype
            )
        self.command_speed = command_speed
        self.returns = torch.zeros(num_envs, device=self.device)
        self.reset(torch.ones(num_envs, device=self.device, dtype=torch.bool))

    def reset(self, mask):
        q = self.home.expand(self.num_envs, -1).clone()
        roots = torch.zeros(self.num_envs, 6, device=self.device, dtype=self.dtype)
        if self.randomize:
            q += (
                torch.rand(q.shape, generator=self.generator, device=self.device, dtype=self.dtype)
                - 0.5
            ) * 0.02
            roots[:, 3:5] = (
                torch.rand(
                    (self.num_envs, 2),
                    generator=self.generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                - 0.5
            ) * 0.02
            roots[:, 2] = 0.001
        self.native.reset(mask.contiguous(), q.contiguous(), roots)
        self.filtered_velocity = torch.where(mask[:, None], 0, self.filtered_velocity)
        self.actions = torch.where(mask[:, None], 0, self.actions)
        self.forces = torch.where(mask[:, None], 0, self.forces)
        self.episode_length_buf = torch.where(mask, 0, self.episode_length_buf)
        self.returns = torch.where(mask, 0, self.returns)
        if self.task == "locomotion":
            directions = torch.tensor(
                [[1.0, 0], [-1, 0], [0, 1], [0, -1]], device=self.device, dtype=self.dtype
            )
            selected = torch.randint(
                4, (self.num_envs,), generator=self.generator, device=self.device
            )
            self.commands = torch.where(
                mask[:, None], directions[selected] * self.command_speed, self.commands
            )

    def state(self):
        b, j, d = [v.to(self.dtype) for v in self.native.state()]
        q = multiply(b[:, 1, 3:7], self.iquat.expand(self.num_envs, -1))
        gravity = inverse_rotate(
            q,
            torch.tensor([0.0, 0, -1], device=self.device, dtype=self.dtype).expand(
                self.num_envs, -1
            ),
        )
        v = inverse_rotate(q, b[:, 1, 7:10])
        w = inverse_rotate(q, b[:, 1, 10:13])
        yaw = torch.stack(
            (1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2), 2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])),
            dim=-1,
        )
        return b, j, d, gravity, v, w, yaw

    def observations(self, state):
        b, j, d, g, v, w, yaw = state
        pieces = [
            g,
            v,
            w,
            b[:, 1, 2:3] - self.meta["root_height"],
            j[:, :, 0] - self.home,
            0.1 * j[:, :, 1],
            self.actions,
        ]
        if self.task == "locomotion":
            phase = (
                self.episode_length_buf.to(self.dtype)
                * self.control_dt
                * (2 * torch.pi / getattr(self, "gait_period", 0.8))
            )
            pieces.extend([self.commands, yaw, torch.stack((phase.sin(), phase.cos()), dim=-1)])
        if self.velocity_filter_seconds > 0:
            pieces.append(self.filtered_velocity)
        return torch.cat(pieces, dim=-1).float()

    def get_observations(self):
        return self.observations(self.state()), {"observations": {}}

    def reference(self):
        if not self.gait:
            return self.home.expand(self.num_envs, -1)
        phase = self.episode_length_buf.to(self.dtype) * self.control_dt / self.gait_period
        count = self.gait_table.shape[1]
        index = (phase % 1) * count
        lo = index.long()
        fraction = (index - lo)[:, None]
        direction = (self.commands @ self.gait_directions.T).argmax(-1)
        ref = (
            self.gait_table[direction, lo] * (1 - fraction)
            + self.gait_table[direction, (lo + 1) % count] * fraction
        )
        blend = (self.episode_length_buf.to(self.dtype) * self.control_dt).clamp(0, 1)[:, None]
        return self.home + (ref - self.home) * blend

    def step(self, actions):
        a = actions.to(dtype=self.dtype).clamp(-1, 1).contiguous()
        target = (
            (self.reference() + self.action_scale * a).clamp(self.lower, self.upper).contiguous()
        )
        self.native.step(
            target,
            self.forces.contiguous(),
            self.substeps,
            self.meta["kp"],
            self.meta["torque_limit"],
        )
        state = self.state()
        b, j, diag, g, v, w, yaw = state
        height = b[:, 1, 2] - self.meta["root_height"]
        failed = (
            (g[:, 2] > -torch.cos(torch.tensor(0.5, device=self.device)))
            | (height.abs() > 0.04)
            | (diag[:, 0] > 0.005)
            | (diag[:, 1] > 0.005)
            | (diag[:, 5] > 0)
            | ~torch.isfinite(b).all(dim=(1, 2))
        )
        reward = (
            1.5 * torch.exp(-20 * g[:, :2].square().sum(-1))
            + torch.exp(-2000 * height.square())
            + 0.5 * torch.exp(-10 * v.square().sum(-1))
            - 0.02 * w.square().sum(-1)
            - 0.05 * (j[:, :, 0] - self.home).square().sum(-1)
            - 0.01 * a.square().sum(-1)
            - 0.02 * (a - self.actions).square().sum(-1)
        )
        if self.task == "locomotion":
            # Commands and velocity errors are in world coordinates; keep initial heading.
            velocity = b[:, 1, 7:9]
            if self.velocity_filter_seconds > 0:
                alpha = self.velocity_alpha
                self.filtered_velocity = (1 - alpha) * self.filtered_velocity + alpha * velocity
                velocity = self.filtered_velocity
            error = (velocity - self.commands).square().sum(-1)
            reward = (
                1.0 * torch.exp(-20 * g[:, :2].square().sum(-1))
                + torch.exp(-2000 * height.square())
                + self.tracking_weight * torch.exp(-error / self.tracking_sigma**2)
                - 0.1 * w.square().sum(-1)
                - self.heading_weight * (1 - yaw[:, 0])
                - 0.02 * a.square().sum(-1)
                - 0.03 * (a - self.actions).square().sum(-1)
            )
        if self.foot_clearance:
            poses = b[:, self.foot_bodies]
            up = torch.tensor([0.0, 0, 1.0], device=self.device, dtype=self.dtype).expand(
                self.num_envs, 2, -1
            )
            local_up = inverse_rotate(poses[:, :, 3:7], up)
            heights = (
                torch.einsum("fvi,efi->efv", self.foot_points, local_up) + poses[:, :, None, 2]
            )
            clearance = heights.amin(-1)
            phase = (
                (self.episode_length_buf[:, None] + 1) * self.control_dt / self.gait_period
                + torch.tensor([0.0, 0.5], device=self.device)
            ) % 1
            swing = ((phase - 0.6) / 0.4).clamp(0, 1)
            desired = self.gait_lift * torch.sin(torch.pi * swing).square()
            reward -= 20000 * (clearance.clamp_min(0) - desired).square().sum(-1)
        if self.load_transfer:
            forces = self.native.contact_forces()[:, self.foot_bodies, 2].to(self.dtype) / (
                self.meta["mass"] * 9.81
            )
            cycle = (
                (self.episode_length_buf.to(self.dtype) + 1) * self.control_dt / self.gait_period
            )
            left = 0.5 + 0.5 * torch.sin(2 * torch.pi * (cycle - 0.05))
            wanted = torch.stack((left, 1 - left), dim=-1)
            reward -= 3 * (forces.clamp(0, 3) - wanted).square().sum(-1)
        reward = torch.where(failed, -2.0, reward) * self.control_dt
        self.episode_length_buf += 1
        self.returns += reward.float()
        self.actions = a
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failed
        done = failed | timeout
        obs = self.observations(state)
        info = {
            "observations": {},
            "time_outs": timeout,
            "failed": failed,
            "diagnostics": diag,
            "terminal_observation": obs,
            "terminal_mask": done,
            "log": {
                "physics/failed_fraction": failed.float().mean(),
                "physics/max_joint_error_m": diag[:, 0].max(),
                "physics/max_penetration_m": diag[:, 1].max(),
            },
        }
        if self.task == "locomotion":
            direction = self.commands / max(self.command_speed, 1e-8)
            info["log"].update(
                {
                    "tracking/instant_error_mps": (b[:, 1, 7:9] - self.commands)
                    .norm(dim=-1)
                    .mean(),
                    "tracking/command_direction_mps": (b[:, 1, 7:9] * direction).sum(-1).mean(),
                    "tracking/reward_velocity_error_mps": error.sqrt().mean(),
                    "tracking/absolute_yaw_rad": torch.atan2(yaw[:, 1], yaw[:, 0]).abs().mean(),
                }
            )
        if self.auto_reset:
            self.reset(done)
            obs = self.get_observations()[0]
        return obs, reward.float(), done, info
