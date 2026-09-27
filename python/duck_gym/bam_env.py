"""Native CPU/CUDA nominal walking environment with the official actuator/actor contract.

Collision support follows the exported model (ground-only v1 or body contacts
v2). Physical domain randomization is still absent; this is not full task parity.
"""

import json
import math
from pathlib import Path
import torch
from .actuator import xl330_m6
from .tensor_env import CpuAdapter, multiply, inverse_rotate
from .walking_reward import WalkingReward


class BamEnv:
    def __init__(
        self,
        model_dir,
        num_envs=1,
        backend="cpu",
        fp64=False,
        iterations=200,
        auto_reset=False,
        seconds=40.0,
        seed=42,
        reward_enabled=False,
        source_iteration=0,
        training=None,
        heading_hold=False,
        physics_dt=None,
    ):
        self.cfg = json.loads((Path(model_dir) / "config.json").read_text())
        self.device = torch.device("cuda" if backend == "cuda" else "cpu")
        self.dtype = torch.float64 if fp64 else torch.float32
        self.num_envs = num_envs
        self.num_actions = 14
        original_dt = self.cfg["physics_dt"]
        self.dt = original_dt if physics_dt is None else float(physics_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Physics timestep must be positive and finite')
        control_dt = original_dt * self.cfg['decimation']
        self.decimation = round(control_dt / self.dt)
        if self.decimation < 1 or abs(self.decimation*self.dt-control_dt)>1e-9:
            raise ValueError('Physics timestep must divide the original control interval')
        self.step_dt = self.dt * self.decimation
        self.max_episode_length = round(seconds / self.step_dt)
        self.auto_reset = auto_reset
        self.reward_enabled = reward_enabled
        self.source_iteration = source_iteration
        self.training = training
        self.heading_hold = (training or {}).get("heading_hold", heading_hold)
        self.heading_reward_weight = (training or {}).get("heading_reward_weight", 0.0)
        if self.heading_reward_weight and not self.heading_hold:
            raise ValueError("Heading reward requires the initial-heading command controller")
        self.mean_velocity_tau = (training or {}).get("mean_velocity_tau", 0.0)
        if self.mean_velocity_tau < 0:
            raise ValueError("Mean velocity time constant must be nonnegative")
        self.common_step_counter = 0
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        t = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)
        self.home = t(self.cfg["default_qpos"][7:]).expand(num_envs, -1).clone()
        self.root_offset = t(self.cfg["root_com_offset"]).expand(num_envs, -1).contiguous()
        self.root_rotation_inv = t(self.cfg["root_principal_quat"]) * t([1, -1, -1, -1])
        self.foot_ids = [f["body"] for f in self.cfg["feet"]]
        self.foot_points = [t(f["points"]) for f in self.cfg["feet"]]
        self.foot_sites = t([f["site"] for f in self.cfg["feet"]])
        self.zeros_force = torch.zeros(num_envs, 3, device=self.device, dtype=self.dtype)
        model = str(Path(model_dir) / "model.duck")
        lines = Path(model).read_text().splitlines()
        body_count = int(lines[0].split()[2])
        body_rows = [line.split() for line in lines[2 : 2 + body_count]]
        self.mass = t([float(row[1]) for row in body_rows])
        self.inertia = t([[float(v) for v in row[2:5]] for row in body_rows])
        joint_rows = [line.split() for line in lines[2 + body_count : 2 + body_count + 14]]
        limits = t([[float(row[22]), float(row[23])] for row in joint_rows])
        center = limits.mean(-1, keepdim=True)
        self.soft_limits = (center + 0.9 * (limits - center))[None, :, :].expand(num_envs, -1, -1)
        if backend == "cpu":
            self.native = CpuAdapter(model, num_envs, self.dt, iterations, fp64, 1)
        else:
            from build_cuda import build

            self.native = build().CudaBatch(model, num_envs, self.dt, iterations, 0, fp64)
        self.action = torch.zeros_like(self.home)
        self.old_action = torch.zeros_like(self.home)
        self.previous_motor = torch.zeros_like(self.home)
        self.previous_applied = torch.zeros_like(self.home)
        self.previous_velocity = torch.zeros_like(self.home)
        self.observed_velocity = torch.zeros_like(self.home)
        self.command = torch.zeros(num_envs, 3, device=self.device, dtype=self.dtype)
        self.mean_velocity_world = torch.zeros_like(self.command)
        self.target_yaw = torch.zeros(num_envs, device=self.device, dtype=self.dtype)
        self.air_time = torch.zeros(num_envs, 2, device=self.device, dtype=self.dtype)
        self.contact_time = torch.zeros_like(self.air_time)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.pending = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        lag_seconds = round(sum(self.cfg["delay_lag_steps"]) / 2)*original_dt
        self.lag = round(lag_seconds/self.dt)
        if abs(self.lag*self.dt-lag_seconds)>1e-9:
            raise ValueError('Physics timestep must preserve the original actuator lag')
        self.target_history = self.home[None, :, :].expand(self.lag + 1, -1, -1).clone()
        self.history_initialized = torch.zeros_like(self.pending)
        self.history_cursor = 0
        self.head_ids = [
            i for i, n in enumerate(self.cfg["joint_names"]) if "head" in n or "neck" in n
        ]
        self.neutral_head = (training or {}).get("neutral_head", False)
        self.leg_ids = [i for i in range(14) if i not in self.head_ids]
        self.head_average = torch.zeros(num_envs, 4, device=self.device, dtype=self.dtype)
        self.reward_model = WalkingReward(
            self.home,
            self.cfg["joint_names"],
            self.soft_limits,
            linear_variance=(training or {}).get("horizontal_velocity_variance", 0.1),
            vertical_variance=(training or {}).get("vertical_velocity_variance", 0.1),
        )
        self.self_contacts = self.home.new_zeros(self.num_envs)
        self.self_penetration = self.home.new_zeros(self.num_envs)
        self.reward_weight_overrides = (training or {}).get("reward_weight_overrides", {})
        unknown = set(self.reward_weight_overrides) - set(self.reward_model.weights(0))
        if unknown or any(not math.isfinite(v) for v in self.reward_weight_overrides.values()):
            raise ValueError("Unknown or nonfinite reward weight override")
        self.refresh()
        if training is not None:
            self.reset()

    def refresh(self):
        self.bodies, self.joints, self.diagnostics = self.native.state()
        self.root_quat = multiply(
            self.bodies[:, 1, 3:7], self.root_rotation_inv.expand(self.num_envs, -1)
        )
        self.gravity = inverse_rotate(
            self.root_quat,
            torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=self.dtype).expand(
                self.num_envs, -1
            ),
        )
        self.ang_vel = inverse_rotate(self.root_quat, self.bodies[:, 1, 10:13])
        # Convert principal COM velocity to root-link-origin velocity first.
        root_r = -inverse_rotate(
            self.bodies[:, 1, 3:7]
            * torch.tensor([1, -1, -1, -1], device=self.device, dtype=self.dtype),
            self.root_offset,
        )
        origin_vel = self.bodies[:, 1, 7:10] + torch.linalg.cross(self.bodies[:, 1, 10:13], root_r)
        self.origin_world_velocity = origin_vel
        self.lin_vel = inverse_rotate(self.root_quat, origin_vel)
        self.contact_force = self.native.contact_forces(ground_only=True)[:, self.foot_ids]
        self.contact = self.contact_force.norm(dim=-1) > 1e-5
        foot = self.bodies[:, self.foot_ids]
        conj = foot[:, :, 3:7] * torch.tensor([1, -1, -1, -1], device=self.device, dtype=self.dtype)
        r = inverse_rotate(conj, self.foot_sites[None, :, :].expand(self.num_envs, -1, -1))
        self.site_pos = foot[:, :, :3] + r
        self.site_vel = foot[:, :, 7:10] + torch.linalg.cross(foot[:, :, 10:13], r)

    @property
    def sole_height(self):
        # Full convex support is needed for validation, not for the actor's
        # observations. Avoid thousands of vertices per foot in every substep.
        foot = self.bodies[:, self.foot_ids]
        conj = foot[:, :, 3:7] * torch.tensor([1, -1, -1, -1], device=self.device, dtype=self.dtype)
        heights = []
        for i, points in enumerate(self.foot_points):
            world = inverse_rotate(
                conj[:, i, None, :].expand(-1, len(points), -1),
                points[None, :, :].expand(self.num_envs, -1, -1),
            )
            heights.append((world[:, :, 2] + foot[:, i, 2, None]).min(-1).values)
        return torch.stack(heights, -1)

    def get_observations(self):
        obs = torch.cat(
            (
                self.ang_vel,
                self.gravity,
                self.joints[:, :, 0] - self.home,
                self.observed_velocity,
                self.action,
                self.command,
                torch.zeros(self.num_envs, 10, device=self.device, dtype=self.dtype),
            ),
            dim=-1,
        ).float()
        forces = self.contact_force.flatten(1)
        critic = torch.cat(
            (
                self.lin_vel,
                self.ang_vel,
                self.gravity,
                self.joints[:, :, 0] - self.home,
                self.joints[:, :, 1],
                self.action,
                self.command,
                self.site_pos[:, :, 2],
                self.air_time,
                self.contact.to(self.dtype),
                forces.sign() * forces.abs().log1p(),
                torch.zeros(self.num_envs, 10, device=self.device, dtype=self.dtype),
            ),
            dim=-1,
        ).float()
        if self.mean_velocity_tau:
            critic = torch.cat(
                (critic, inverse_rotate(self.root_quat, self.mean_velocity_world).float()), dim=-1
            )
        if self.heading_reward_weight:
            critic = torch.cat((critic, self.heading_error()[:, None].float()), dim=-1)
        return {"actor": obs, "critic": critic}

    def resample_commands(self, mask):
        commands = self.home.new_tensor(self.training["commands"])
        indices = torch.randint(
            len(commands), (self.num_envs,), generator=self.generator, device=self.device
        )
        self.command[mask] = commands[indices[mask]]

    def yaw(self):
        q = self.root_quat
        return torch.atan2(
            2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1 - 2 * (q[:, 2].square() + q[:, 3].square()),
        )

    def heading_error(self):
        error = self.target_yaw - self.yaw()
        return torch.atan2(error.sin(), error.cos())

    def update_heading_command(self):
        if self.heading_hold:
            self.command[:, 2] = (0.5 * self.heading_error()).clamp(-1.0, 1.0)

    def reset(self, mask=None, *, joint_noise=None, tilt_noise=None):
        if mask is None:
            mask = torch.ones_like(self.pending)
        self.self_contacts[mask] = 0
        self.self_penetration[mask] = 0
        initial = self.home.clone()
        root = torch.zeros(self.num_envs, 6, device=self.device, dtype=self.dtype)
        joint_noise = (
            (self.training or {}).get("reset_joint_noise", 0.0)
            if joint_noise is None
            else joint_noise
        )
        tilt_noise = (
            (self.training or {}).get("reset_tilt_noise", 0.0) if tilt_noise is None else tilt_noise
        )
        if joint_noise < 0 or tilt_noise < 0:
            raise ValueError("Reset perturbation amplitudes must be nonnegative")
        if joint_noise or tilt_noise:
            initial += (
                torch.rand(
                    initial.shape, generator=self.generator, device=self.device, dtype=self.dtype
                )
                * 2
                - 1
            ) * joint_noise
            root[:, 3:5] = (
                torch.rand(
                    (self.num_envs, 2),
                    generator=self.generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                * 2
                - 1
            ) * tilt_noise
        if self.training is not None:
            self.resample_commands(mask)
        self.native.reset(mask, initial, root)
        for state in (
            self.action,
            self.old_action,
            self.previous_motor,
            self.previous_applied,
            self.previous_velocity,
            self.observed_velocity,
            self.air_time,
            self.contact_time,
            self.head_average,
            self.mean_velocity_world,
        ):
            state[mask] = 0
        self.episode_length_buf[mask] = 0
        self.pending[mask] = False
        self.history_initialized[mask] = False
        self.target_history[:, mask] = self.home[mask]
        self.reward_model.reset(mask)
        self.refresh()
        self.target_yaw[mask] = self.yaw()[mask]
        self.update_heading_command()
        return self.get_observations()

    def step(self, actions, *, substep_callback=None):
        if not self.auto_reset and bool(self.pending.any()):
            raise RuntimeError("Reset terminated environments before stepping")
        self.old_action.copy_(self.action)
        self.action.copy_(actions.to(self.dtype))
        if self.neutral_head:
            # Position targets remain at HOME through the real BAM actuator.
            # These joints remain dynamic and can deflect under load.
            self.action[:, self.head_ids] = 0
        self.observed_velocity.copy_(self.joints[:, :, 1])
        target = self.home + self.action
        self.self_contacts.zero_()
        self.self_penetration.zero_()
        for _ in range(self.decimation):
            self.target_history[self.history_cursor] = target
            new = ~self.history_initialized
            self.target_history[:, new] = target[new]
            self.history_initialized[:] = True
            delayed = self.target_history[(self.history_cursor - self.lag) % (self.lag + 1)]
            self.history_cursor = (self.history_cursor + 1) % (self.lag + 1)
            loads = self.native.generalized_loads(self.root_offset)
            motor, raw, _ = xl330_m6(
                delayed,
                self.joints[:, :, 0],
                self.joints[:, :, 1],
                self.previous_motor,
                self.previous_applied,
                -loads[:, :, 0] + loads[:, :, 1],
                parameters=self.cfg["parameters"],
                vin=torch.full(
                    (self.num_envs, 1), self.cfg["vin"], device=self.device, dtype=self.dtype
                ),
                vin_drop_gain=self.cfg["vin_drop_gain"],
                vin_min=self.cfg["vin_min"],
                kp_fw=self.cfg["kp_fw"],
                kp_scale=1.0,
                kd_scale=1.0,
                friction_scale=1.0,
                force_limit=self.cfg["force_limit"],
            )
            self.previous_motor.copy_(raw)
            self.previous_applied.copy_(motor[:, :, 0])
            self.native.step_motor(motor.contiguous(), self.zeros_force)
            self.refresh()
            if substep_callback is not None:
                substep_callback(self)
            collision = self.native.self_contact_stats()
            self.self_contacts = torch.maximum(self.self_contacts, collision[:, 0])
            self.self_penetration = torch.maximum(self.self_penetration, collision[:, 1])
            if self.mean_velocity_tau:
                alpha = -math.expm1(-self.dt / self.mean_velocity_tau)
                self.mean_velocity_world.lerp_(self.origin_world_velocity, alpha)
            self.air_time = torch.where(self.contact, 0.0, self.air_time + self.dt)
            self.contact_time = torch.where(self.contact, self.contact_time + self.dt, 0.0)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # Reward equations are separately checked against pinned upstream.
        # Native state timing/contact definitions still require migration checks.
        reward = torch.zeros(self.num_envs, device=self.device)
        terms = {}
        if self.reward_enabled:
            terms = self.reward_model.terms(self.reward_features(), self.step_dt)
            weights = self.reward_model.weights(
                self.source_iteration + self.common_step_counter // 24
            )
            weights.update(self.reward_weight_overrides)
            # Optional native walking correction. Instantaneous squared errors
            # cannot cancel when the head oscillates around a zero mean.
            head_stability = (self.training or {}).get("head_stability", {})
            if head_stability:
                terms["head_position_l2"] = (
                    self.joints[:, self.head_ids, 0] - self.home[:, self.head_ids]
                ).square().sum(-1)
                terms["head_velocity_l2"] = self.joints[:, self.head_ids, 1].square().sum(-1)
                weights["head_position_l2"] = -head_stability["position_weight"]
                weights["head_velocity_l2"] = -head_stability["velocity_weight"]
            if self.mean_velocity_tau:
                local_mean = inverse_rotate(self.root_quat, self.mean_velocity_world)
                terms["track_mean_velocity"] = torch.exp(
                    -((local_mean[:, :2] - self.command[:, :2]).square().sum(-1))
                    / self.training["mean_velocity_variance"]
                )
                weights["track_mean_velocity"] = self.training["mean_velocity_weight"]
            if self.heading_reward_weight:
                terms["track_heading"] = torch.exp(
                    -self.heading_error().square() / self.training["heading_reward_variance"]
                )
                weights["track_heading"] = self.heading_reward_weight
            reward = sum(weights[name] * value for name, value in terms.items()) * self.step_dt
        timeout = self.episode_length_buf >= self.max_episode_length
        failed = self.failure_mask()
        done = failed | timeout
        self.pending.copy_(done)
        self.update_heading_command()
        observations = self.get_observations()
        info = {
            "time_outs": timeout,
            "terminal_observation": observations,
            "reward_implemented": self.reward_enabled,
            "log": {
                **{f"Reward/{name}": value.mean() for name, value in terms.items()},
                "Physics/self_contact_pairs": self.self_contacts.mean(),
                "Physics/self_penetration": self.self_penetration.max(),
                "Physics/numerical_failure": (self.diagnostics[:, 5] > 0).float().mean(),
            },
        }
        if self.auto_reset:
            self.reset(done)
            if self.training is not None:
                period = round(self.training["command_resample_seconds"] / self.step_dt)
                self.resample_commands((self.episode_length_buf % period == 0) & ~done)
                self.update_heading_command()
            observations = self.get_observations()
        return observations, reward.float(), done.long(), info

    def failure_mask(self):
        """Walking failure rule; aerial tasks supply a phase-aware rule."""
        return (
            (self.gravity[:, 2] > -math.cos(math.radians(70.0)))
            | (self.diagnostics[:, 5] > 0)
            | (self.diagnostics[:, 0] > 0.005)
            | (self.diagnostics[:, 1] > 0.005)
            | (self.self_penetration > 0.005)
            | (~torch.isfinite(self.joints).all(dim=(1, 2)))
        )
    def reward_features(self):
        q = self.bodies[:, :, 3:7]
        sign = self.home.new_tensor([1, -1, -1, -1])
        local_w = inverse_rotate(q, self.bodies[:, :, 10:13])
        spin = inverse_rotate(q * sign, self.inertia[None, :, :] * local_w)
        com = (self.bodies[:, :, :3] * self.mass[None, :, None]).sum(1) / self.mass.sum()
        orbital = torch.linalg.cross(
            self.bodies[:, :, :3] - com[:, None, :],
            self.bodies[:, :, 7:10] * self.mass[None, :, None],
        )
        return dict(
            command=self.command,
            lin_vel=self.lin_vel,
            ang_vel=self.ang_vel,
            gravity=self.gravity,
            world_ang_vel=self.bodies[:, 1, 10:13],
            angular_momentum=(spin + orbital).sum(1),
            joint_pos=self.joints[:, :, 0],
            action=self.action,
            previous_action=self.old_action,
            contact=self.contact,
            contact_time=self.contact_time,
            air_time=self.air_time,
            foot_height=self.site_pos[:, :, 2],
            foot_vel=self.site_vel,
            episode_steps=self.episode_length_buf,
            head_command=torch.zeros_like(self.head_average),
            self_collisions=self.self_contacts,
        )
