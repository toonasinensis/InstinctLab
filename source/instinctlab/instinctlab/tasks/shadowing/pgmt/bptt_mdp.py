"""Closed-loop observations, exogenous motion windows and analytic soft failure."""

import math
import torch
from torch.nn import functional as F

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat

from .mdp.terrain_math import corrected_planar_velocity, tolerance_error
from .models.cross_attention import unpack_terms
from .wm_dynamics import state_features, yaw_angle


def pack_history(history):
    return torch.cat([term.flatten(1) for term in history.split((3, 3, 29, 29, 29), -1)], -1)


def raw_history(observation):
    return unpack_terms(observation[:, :930], (3, 3, 29, 29, 29), 10)


def lambda_returns(rewards, continuation, values, done, gamma=0.99, lam=0.95):
    """Bootstrap timeouts from terminal V, but never recurse into a reset episode."""
    result, future = [], values[:, -1]
    for t in range(rewards.shape[1] - 1, -1, -1):
        future = rewards[:, t] + gamma * continuation[:, t] * (
            (1 - lam) * values[:, t + 1] + lam * torch.where(done[:, t], values[:, t + 1], future)
        )
        result.append(future)
    return torch.stack(result[::-1], 1)


class BPTTMDP:
    def __init__(self, rewards, normalizer, dynamics):
        self.rewards, self.normalizer, self.dynamics = rewards, normalizer, dynamics
        self.command = c = rewards.command
        self.bank = c.motion._all_motion_sequences
        self.heightfield = rewards.heightfield
        self.default_q = c.robot.data.default_joint_pos.torch.clone()
        self.default_qd = c.robot.data.default_joint_vel.torch.clone()
        commands = c.env.command_manager
        self.reference_q = commands.get_term("joint_pos_ref_command")._default_joint_pos
        self.reference_qd = commands.get_term("joint_vel_ref_command")._default_joint_vel
        self.thresholds = {
            name: c.env.termination_manager.get_term_cfg(name).params["threshold"]
            for name in ("root_height", "root_tilt", "body_error")
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
        frames = start["frame"][:, None] + step + torch.arange(11, device=ids.device)
        lengths = self.command.label_lengths[motion, None]
        valid = ((frames >= 0) & (frames < lengths)).all(-1)
        # Real terminal windows may reach the clip end. Match the reference's final-frame padding.
        frames = torch.minimum(frames.clamp_min(0), lengths - 1)
        grid = motion[:, None].expand_as(frames).flatten()

        def read(name):
            value = getattr(self.bank, name)[grid, frames.flatten()]
            return value.reshape(len(ids), 11, *value.shape[1:]).to(ids.device)

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

    def proprioception(self, state, previous_action, ids):
        gravity = state.new_tensor([0, 0, -1]).expand(len(state), -1)
        return torch.cat(
            (
                quat_apply_inverse(state[:, 3:7], gravity),
                quat_apply_inverse(state[:, 3:7], state[:, 10:13]),
                state[:, 13:42] - self.default_q[ids],
                state[:, 42:71] - self.default_qd[ids],
                previous_action,
            ),
            -1,
        )

    def observations(self, state, history, support, context):
        rotation = context["window_rotation"]
        inverse = quat_inv(state[:, 3:7])[:, None].expand_as(rotation)
        orientation = matrix_from_quat(quat_mul(inverse, rotation))[..., :2].flatten(1)
        heading = yaw_quat(context["ref_rotation"])[:, None].expand_as(rotation)
        target = torch.cat((context["ref_position"][:, :2], context["ref_position"][:, 2:3] + support[:, None]), -1)
        error = (target - state[:, :3])[:, None].expand_as(context["window_velocity"])
        correction = corrected_planar_velocity(
            quat_apply_inverse(heading, context["window_velocity"]), quat_apply_inverse(heading, error)
        )
        heights, valid = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
        obs = torch.cat(
            (
                pack_history(history),
                context["ref_joints"],
                orientation,
                context["ref_clearance"],
                correction.flatten(1),
                heights,
            ),
            -1,
        )
        return obs, valid

    def value_features(self, state, observation, support, context, xi):
        features = state_features(state, observation[:, 1678:])
        norm = self.dynamics.input_norm
        features = (features - norm.mean[:68]) / norm.std()[:68]
        xi = (xi - norm.mean[-xi.shape[-1] :]) / norm.std()[-xi.shape[-1] :]
        return torch.cat(
            (
                self.normalizer(observation),
                features,
                xi,
                (support - state[:, 2])[:, None],
                context["level"][:, None].float() / 9,
                F.one_hot(context["family"].long(), 5).float(),
                context["phase"][:, None],
            ),
            -1,
        )

    def soft_survival(self, state, support, context, bodies):
        c = self.command
        position = bodies[0]
        delta = yaw_quat(quat_mul(state[:, 3:7], quat_inv(context["ref_rotation"])))
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
        root, reference = matrix_from_quat(state[:, 3:7]), matrix_from_quat(context["ref_rotation"])
        tilt_cos = (root[:, 2] * reference[:, 2]).sum(-1)
        height_error = (target_root[:, 2] - state[:, 2]).abs()
        alive = torch.sigmoid((self.thresholds["root_height"] - height_error) / 0.03)
        alive = alive * torch.sigmoid((tilt_cos - math.cos(self.thresholds["root_tilt"])) / 0.04)
        return alive * torch.sigmoid((self.thresholds["body_error"] - errors.amax(-1)) / 0.04)
