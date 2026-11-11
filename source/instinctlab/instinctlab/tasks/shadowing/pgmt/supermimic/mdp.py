"""Closed-loop observations, exogenous motion windows and analytic soft failure."""

import math
import torch
from torch.nn import functional as F

from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat
from isaaclab.utils.noise import UniformNoiseCfg

from instinctlab.tasks.shadowing.pgmt.mdp.terrain_math import corrected_planar_velocity, tolerance_error
from instinctlab.tasks.shadowing.pgmt.models.cross_attention import unpack_terms
from instinctlab.tasks.shadowing.pgmt.tensor_math import quat_apply, quat_apply_inverse
from instinctlab.tasks.shadowing.pgmt.wm_dynamics import yaw_angle

from .layouts import TensorLayout
from .terrain import TERRAIN_FAMILIES


def lambda_returns(rewards, continuation, values, done, gamma=0.99, lam=0.95):
    """Bootstrap timeouts from terminal V, but never recurse into a reset episode."""
    result, future = [], values[:, -1]
    for t in range(rewards.shape[1] - 1, -1, -1):
        future = rewards[:, t] + gamma * continuation[:, t] * (
            (1 - lam) * values[:, t + 1] + lam * torch.where(done[:, t], values[:, t + 1], future)
        )
        result.append(future)
    return torch.stack(result[::-1], 1)


class PGMTMDP:
    def __init__(self, rewards, normalizer, dynamics, history_length, reference_frames, softness):
        self.rewards, self.normalizer, self.dynamics = rewards, normalizer, dynamics
        self.command = c = rewards.command
        self.layout = dynamics.representation.layout
        self.state_layout = self.layout.state
        self.history_length, self.reference_frames = history_length, reference_frames
        self.softness = softness
        self.proprio_widths = dict(
            projected_gravity=self.layout.spatial,
            base_ang_vel=self.layout.spatial,
            joint_pos=self.layout.joints,
            joint_vel=self.layout.joints,
            last_action=self.layout.joints,
        )
        self.reference_joint_width = self.layout.joints * reference_frames
        self.proprio_size = sum(self.proprio_widths.values()) * history_length
        self.max_level = c.env.scene.terrain.cfg.terrain_generator.num_rows - 1
        self.family_count = len(TERRAIN_FAMILIES)
        self.bank = c.motion._all_motion_sequences
        self.heightfield = rewards.heightfield
        self.default_q = c.robot.data.default_joint_pos.torch.clone()
        self.default_qd = c.robot.data.default_joint_vel.torch.clone()
        self.gravity = self.default_q.new_tensor([0, 0, -1])
        commands = c.env.command_manager
        self.reference_q = commands.get_term("joint_pos_ref_command")._default_joint_pos
        self.reference_qd = commands.get_term("joint_vel_ref_command")._default_joint_vel
        self.thresholds = {
            name: c.env.termination_manager.get_term_cfg(name).params["threshold"]
            for name in ("root_height", "root_tilt", "body_error")
        }
        manager = c.env.observation_manager
        self.observation_terms = dict(
            zip(manager._group_obs_term_names["policy"], manager._group_obs_term_cfgs["policy"])
        )
        self.observation_widths = dict(
            joint_pos_ref=self.reference_joint_width,
            joint_vel_ref=self.reference_joint_width,
            motion_orientation=reference_frames * self.layout.spatial * 2,
            motion_height=reference_frames,
            corrected_velocity=reference_frames * self.layout.spatial,
            terrain=len(self.heightfield.local_xy),
        )
        self.observation_layout = TensorLayout((("history", self.proprio_size), *self.observation_widths.items()))
        self.critic_context_width = 1 + 1 + self.family_count + 1
        for name, cfg in self.observation_terms.items():
            if cfg.noise is not None and (not isinstance(cfg.noise, UniformNoiseCfg) or cfg.noise.operation != "add"):
                raise ValueError(f"Imagined observation noise requires additive uniform noise: {name}")
            if cfg.modifiers or cfg.delay_max_lag > 0:
                raise ValueError(f"Imagined observations do not implement modifiers or delay: {name}")

    def process_observation(self, value, name, generator=None):
        cfg = self.observation_terms[name]
        if generator is not None and cfg.noise is not None:
            noise = (
                generator[name]
                if isinstance(generator, dict)
                else torch.rand(value.shape, device=value.device, dtype=value.dtype, generator=generator)
            )
            value = value + cfg.noise.n_min + noise * (cfg.noise.n_max - cfg.noise.n_min)
        if cfg.clip:
            value = value.clamp(*cfg.clip)
        if cfg.scale is not None:
            value = value * cfg.scale
        return value

    def draw_noise(self, batch, widths, generator):
        # Same individual shapes and draw order as process_observation. Keeping
        # RNG outside compile/checkpoint must not change the stochastic inputs.
        return {
            name: torch.rand(
                (batch, width), device=self.default_q.device, dtype=self.default_q.dtype, generator=generator
            )
            for name, width in widths.items()
            if self.observation_terms[name].noise is not None
        }

    def placement(self):
        c = self.command
        return dict(
            motion=c.motion_ids.clone(),
            frame=(c.frame_ids - c.label_starts[c.motion_ids]).clone(),
            placement_position=c.offset.clone(),
            placement_rotation=c.rotation.clone(),
            level=c.terrain.level.clone(),
            family=c.terrain.family.clone(),
        )

    def context(self, start, step=0):
        ids, motion = start["env_id"], start["motion"]
        frames = start["frame"][:, None] + step + torch.arange(self.reference_frames, device=ids.device)
        lengths = self.command.label_lengths[motion, None]
        valid = ((frames >= 0) & (frames < lengths)).all(-1)
        # Real terminal windows may reach the clip end. Match the reference's final-frame padding.
        frames = torch.minimum(frames.clamp_min(0), lengths - 1)
        grid = motion[:, None].expand_as(frames).flatten()

        def read(name):
            value = getattr(self.bank, name)[grid, frames.flatten()]
            return value.reshape(len(ids), self.reference_frames, *value.shape[1:]).to(ids.device)

        p, q, v = read("base_pos_w"), read("base_quat_w"), read("base_lin_vel_w")
        joint, joint_velocity = read("joint_pos"), read("joint_vel")
        placement = start["placement_rotation"]
        future_rotation = placement[:, None].expand_as(q)
        body_q = read("link_quat_w")[:, 0]
        body_rotation = placement[:, None].expand_as(body_q)
        offset = torch.cat((start["placement_position"][:, :2], p.new_zeros(len(ids), 1)), -1)
        labels = self.command.label_starts[motion] + frames[:, 0]
        return dict(
            ref_position=quat_apply(placement, p[:, 0]) + offset,
            ref_rotation=quat_mul(placement, q[:, 0]),
            ref_body_position=quat_apply(body_rotation, read("link_pos_w")[:, 0] - p[:, :1]),
            ref_body_rotation=quat_mul(body_rotation, body_q),
            ref_body_velocity=quat_apply(body_rotation, read("link_lin_vel_w")[:, 0]),
            ref_body_angular=quat_apply(body_rotation, read("link_ang_vel_w")[:, 0]),
            ref_velocity=quat_apply(placement, v[:, 0]),
            ref_q=joint[:, 0],
            ref_qd=joint_velocity[:, 0],
            ref_contact=self.command.labels[labels].float(),
            ref_valid=self.command.label_valid[labels].float(),
            ref_joints=torch.cat(
                (
                    (joint - self.reference_q[ids, None]).flatten(1),
                    (joint_velocity - self.reference_qd[ids, None]).flatten(1),
                ),
                -1,
            ),
            window_rotation=quat_mul(future_rotation, q),
            window_velocity=quat_apply(future_rotation, v),
            ref_clearance=p[..., 2],
            reference_valid=valid,
            level=start["level"],
            family=start["family"],
            phase=frames[:, 0].float() / lengths[:, 0],
        )

    def proprioception(self, state, previous_action, ids, generator=None):
        gravity = self.gravity.expand(len(state), -1)
        terms = (
            quat_apply_inverse(state[:, self.state_layout["rotation"]], gravity),
            quat_apply_inverse(
                state[:, self.state_layout["rotation"]], state[:, self.state_layout["angular_velocity"]]
            ),
            state[:, self.state_layout["joint_position"]] - self.default_q[ids],
            state[:, self.state_layout["joint_velocity"]] - self.default_qd[ids],
            previous_action,
        )
        names = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action")
        return torch.cat([self.process_observation(v, k, generator) for k, v in zip(names, terms)], -1)

    def observations(self, state, history, support, context, generator=None):
        rotation = context["window_rotation"]
        inverse = quat_inv(state[:, self.state_layout["rotation"]])[:, None].expand_as(rotation)
        orientation = matrix_from_quat(quat_mul(inverse, rotation))[..., :2].flatten(1)
        heading = yaw_quat(context["ref_rotation"])[:, None].expand_as(rotation)
        target = torch.cat((context["ref_position"][:, :2], context["ref_position"][:, 2:3] + support[:, None]), -1)
        error = (target - state[:, self.state_layout["position"]])[:, None].expand_as(context["window_velocity"])
        correction = corrected_planar_velocity(
            quat_apply_inverse(heading, context["window_velocity"]), quat_apply_inverse(heading, error)
        )
        heights, valid = self.heightfield.local(
            state[:, self.state_layout["position"]], yaw_angle(state[:, self.state_layout["rotation"]])
        )
        # History is already processed once per frame; never add noise to it again.
        values = (
            context["ref_joints"][:, : self.reference_joint_width],
            context["ref_joints"][:, self.reference_joint_width :],
            orientation,
            context["ref_clearance"],
            correction.flatten(1),
            heights,
        )
        names = (
            "joint_pos_ref",
            "joint_vel_ref",
            "motion_orientation",
            "motion_height",
            "corrected_velocity",
            "terrain",
        )
        obs = torch.cat(
            (self.pack_history(history), *[self.process_observation(v, k, generator) for k, v in zip(names, values)]),
            -1,
        )
        return obs, valid

    def value_features(self, state, observation, support, context, xi):
        ground = self.heightfield.query(state[:, self.state_layout["position"]][:, :-1])[0]
        features = self.dynamics.representation.represent(state, ground)
        norm = self.dynamics.input_norm
        features = (features - norm.mean[: self.layout.feature_width]) / norm.std()[: self.layout.feature_width]
        xi = (xi - norm.mean[-xi.shape[-1] :]) / norm.std()[-xi.shape[-1] :]
        return torch.cat(
            (
                self.normalizer(observation),
                features,
                xi,
                (support - state[:, 2])[:, None],
                context["level"][:, None].float() / self.max_level,
                F.one_hot(context["family"].long(), self.family_count).float(),
                context["phase"][:, None],
            ),
            -1,
        )

    def soft_survival(self, state, support, context, bodies):
        c = self.command
        position = bodies[0]
        delta = yaw_quat(quat_mul(state[:, self.state_layout["rotation"]], quat_inv(context["ref_rotation"])))
        target_root = torch.cat((state[:, :2], context["ref_position"][:, 2:3] + support[:, None]), -1)
        target = target_root[:, None] + quat_apply(
            delta[:, None].expand(-1, position.shape[1], -1), context["ref_body_position"]
        )
        errors = (target - position).norm(dim=-1)
        lower = tolerance_error(
            errors[:, c.lower_ids],
            context["level"],
            context["family"],
            c.cfg.tolerance_position_m,
            c.cfg.tolerance_alpha if c.cfg.enable_tolerance else 0,
        )
        errors = errors.clone()
        errors[:, c.lower_ids] = lower
        root, reference = matrix_from_quat(state[:, self.state_layout["rotation"]]), matrix_from_quat(
            context["ref_rotation"]
        )
        tilt_cos = (root[:, 2] * reference[:, 2]).sum(-1)
        height_error = (target_root[:, 2] - state[:, 2]).abs()
        alive = torch.sigmoid((self.thresholds["root_height"] - height_error) / self.softness["height_m"])
        alive = alive * torch.sigmoid((tilt_cos - math.cos(self.thresholds["root_tilt"])) / self.softness["tilt_cos"])
        return alive * torch.sigmoid((self.thresholds["body_error"] - errors.amax(-1)) / self.softness["body_m"])

    def pack_history(self, history):
        return torch.cat([term.flatten(1) for term in history.split(tuple(self.proprio_widths.values()), -1)], -1)

    def raw_history(self, observation):
        return unpack_terms(
            observation[:, : self.proprio_size], tuple(self.proprio_widths.values()), self.history_length
        )
