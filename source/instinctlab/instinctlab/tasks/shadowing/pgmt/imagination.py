"""Differentiable PGMT observation reconstruction and short-horizon objectives.

References are exogenous motion windows, never future simulated observations.
The support-height filter, contact belief, previous action and proprioceptive
history are carried through imagination. Discrete contact rewards remain in PPO.
"""

import math
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat

from .mdp.terrain_math import corrected_planar_velocity, tolerance_error
from .mdp.terrain_observations import critic_state
from .models.cross_attention import unpack_terms
from .world_model import yaw_angle


def pack_history(history):
    return torch.cat([term.flatten(1) for term in history.split((3, 3, 29, 29, 29), -1)], -1)


def stance_slip(previous_feet, feet, contact, dt):
    # Stop the contact-weight gradient: the actor must reduce stance motion,
    # rather than make the model predict that a sliding foot lost contact.
    velocity_xy = (feet[..., :2] - previous_feet[..., :2]) / dt
    return (velocity_xy.square().sum(-1) * contact.detach()).sum(-1)


class ImaginationMDP:
    def __init__(self, env, heightfield):
        self.env = env
        self.command = env.command_manager.get_term("motion")
        self.reference = self.command.reference
        self.heightfield = heightfield
        self.limits = env.scene["robot"].data.soft_joint_pos_limits.torch[0].clone()
        self.dt = env.step_dt

    def state(self):
        robot = self.env.scene["robot"].data
        return torch.cat(
            (
                robot.root_pos_w.torch,
                robot.root_quat_w.torch,
                robot.root_lin_vel_w.torch,
                robot.root_ang_vel_w.torch,
                robot.joint_pos.torch,
                robot.joint_vel.torch,
            ),
            -1,
        ).clone()

    def context(self):
        c, ref = self.command, self.reference.data
        origin = self.env.scene.env_origins
        return {
            "ref_joints": torch.cat(
                (
                    self.env.command_manager.get_command("joint_pos_ref_command").flatten(1),
                    self.env.command_manager.get_command("joint_vel_ref_command").flatten(1),
                ),
                -1,
            ).clone(),
            "ref_rotation": c.anchor_quat_w_future.clone(),
            "ref_clearance": (ref.base_pos_w.torch[..., 2] - origin[:, None, 2]).clone(),
            "ref_position": c.anchor_pos_w.clone(),
            "ref_velocity": c._velocity(ref.base_lin_vel_w.torch).clone(),
            "ref_body_position": (c.body_pos_w - c.anchor_pos_w[:, None]).clone(),
            "ref_body_rotation": c.body_quat_w.clone(),
            "ref_q": c.joint_pos.clone(),
            "ref_qd": c.joint_vel.clone(),
            "ref_contact": c.reference_contacts.float().clone(),
            "ref_valid": c.reference_contact_valid.float().clone(),
            "level": c.terrain.level.clone(),
            "family": c.terrain.family.clone(),
            "default_q": self.env.scene["robot"].data.default_joint_pos.torch.clone(),
            "default_qd": self.env.scene["robot"].data.default_joint_vel.torch.clone(),
            # Only the exogenous reference command is retained. Imagined robot
            # features below are reconstructed from the predicted state.
            "critic_command": critic_state(self.env)[:, :68].clone(),
        }

    def bodies(self, state):
        ref = self.reference
        transforms = ref._robot_kinematics_chain.forward_kinematics(
            state[:, 13:42][:, ref._joint_order_isaac_to_pk],
            frame_indices=ref._link_indices_pk,
        )
        matrices = torch.stack([transforms[n].get_matrix() for n in ref.cfg.link_of_interests], 1)
        root_R = matrix_from_quat(state[:, 3:7])
        position = state[:, None, :3] + (root_R[:, None] @ matrices[..., :3, 3:4]).squeeze(-1)
        rotation = root_R[:, None] @ matrices[..., :3, :3]
        return position, rotation

    def update_support(self, state, support, context):
        feet = self.bodies(state)[0][:, self.command.foot_ids]
        foot_height, valid = self.heightfield.query(feet[..., :2])
        root_height, root_valid = self.heightfield.query(state[:, :2])
        labels = (context["ref_contact"] * context["ref_valid"])[:, self.command.foot_ids]
        target = torch.where(
            labels.sum(-1) > 0, (foot_height * labels).sum(-1) / labels.sum(-1).clamp_min(1), root_height
        )
        blend = 1 - math.exp(-self.dt / self.command.cfg.support_filter_seconds)
        return support + blend * (target - support), valid.all(-1) & root_valid

    def proprioception(self, state, previous_action, context):
        gravity = state.new_tensor([0, 0, -1]).expand(state.shape[0], -1)
        return torch.cat(
            (
                quat_apply_inverse(state[:, 3:7], gravity),
                quat_apply_inverse(state[:, 3:7], state[:, 10:13]),
                state[:, 13:42] - context["default_q"],
                state[:, 42:71] - context["default_qd"],
                previous_action,
            ),
            -1,
        )

    def observations(self, state, history, context):
        rotation = context["ref_rotation"]
        inverse = quat_inv(state[:, 3:7])[:, None].expand_as(rotation)
        orientation = matrix_from_quat(quat_mul(inverse, rotation))[..., :2].flatten(1)
        heading = yaw_quat(rotation[:, 0])[:, None].expand_as(rotation)
        velocity = quat_apply_inverse(heading, context["ref_velocity"])
        error = (context["ref_position"] - state[:, :3])[:, None].expand_as(velocity)
        correction = corrected_planar_velocity(velocity, quat_apply_inverse(heading, error))
        heights, valid = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
        return (
            torch.cat(
                (
                    pack_history(history),
                    context["ref_joints"],
                    orientation,
                    context["ref_clearance"],
                    correction.flatten(1),
                    heights,
                ),
                -1,
            ),
            valid,
        )

    def critic_observations(self, state, action, support, context):
        """Native 296-value critic layout, with counterfactual robot/support state."""
        position, rotation = self.bodies(state)
        root_R = matrix_from_quat(state[:, 3:7])
        inverse_R = root_R.transpose(-1, -2)
        target = torch.cat((context["ref_position"][:, :2], context["ref_clearance"][:, :1] + support[:, None]), -1)
        anchor_pos = quat_apply_inverse(state[:, 3:7], target - state[:, :3])
        anchor_ori = matrix_from_quat(quat_mul(quat_inv(state[:, 3:7]), context["ref_rotation"][:, 0]))
        body_pos = (inverse_R[:, None] @ (position - state[:, None, :3])[..., None]).squeeze(-1)
        body_ori = inverse_R[:, None] @ rotation
        return torch.cat(
            (
                context["critic_command"],
                anchor_pos,
                anchor_ori[..., :2].flatten(1),
                body_pos.flatten(1),
                body_ori[..., :2].flatten(1),
                quat_apply_inverse(state[:, 3:7], state[:, 7:10]),
                quat_apply_inverse(state[:, 3:7], state[:, 10:13]),
                state[:, 13:42] - context["default_q"],
                state[:, 42:71] - context["default_qd"],
                action,
            ),
            -1,
        )

    def reward(self, state, action, previous_action, support, context, previous_state, contact):
        """Analytic tracking surrogate; real PPO retains all 27 original reward terms."""
        c = self.command
        position, rotation = self.bodies(state)
        delta = yaw_quat(quat_mul(state[:, 3:7], quat_inv(context["ref_rotation"][:, 0])))
        target_root = torch.cat((state[:, :2], context["ref_clearance"][:, :1] + support[:, None]), -1)
        target_body = target_root[:, None] + quat_apply(
            delta[:, None].expand(-1, position.shape[1], -1),
            context["ref_body_position"],
        )
        target_R = matrix_from_quat(
            quat_mul(delta[:, None].expand(-1, position.shape[1], -1), context["ref_body_rotation"])
        )
        pos_error = (target_body - position).norm(dim=-1)
        reward = state.new_zeros(state.shape[0])
        for body_ids, joint_ids, lower in (
            (c.upper_ids, c.upper_joint_ids, False),
            (c.lower_ids, c.lower_joint_ids, True),
        ):
            ep = pos_error[:, body_ids]
            eq = (context["ref_q"][:, joint_ids] - state[:, 13:42][:, joint_ids]).abs()
            # Smooth chordal orientation error avoids acos singularities at identity.
            er = (target_R[:, body_ids] - rotation[:, body_ids]).square().sum((-1, -2)) / 2
            if lower and c.cfg.enable_tolerance:
                ep = tolerance_error(ep, context["level"], context["family"], c.cfg.tolerance_position_m)
                eq = tolerance_error(eq, context["level"], context["family"], c.cfg.tolerance_joint_rad)
            reward = reward + (0.5 if lower else 1) * torch.exp(-ep.square().mean(-1) / 0.3**2)
            reward = reward + (2 if lower else 1) * torch.exp(-er.mean(-1) / 0.4**2)
            reward = reward + (0.5 if lower else 1) * torch.exp(-eq.square().mean(-1) / 0.5**2)
            qd_error = context["ref_qd"][:, joint_ids] - state[:, 42:71][:, joint_ids]
            reward = reward + 0.5 * torch.exp(-qd_error.square().mean(-1) / 25)
        heading = yaw_quat(context["ref_rotation"][:, 0])
        velocity = quat_apply_inverse(heading, context["ref_velocity"][:, 0])
        error = quat_apply_inverse(heading, context["ref_position"] - state[:, :3])
        desired = quat_apply(heading, corrected_planar_velocity(velocity, error))
        reward = reward + 2 * torch.exp(-(desired[:, :2] - state[:, 7:9]).square().sum(-1))
        height_error = target_root[:, 2] - state[:, 2]
        reward = reward + torch.exp(-height_error.square() / 0.3**2)
        root_R = matrix_from_quat(state[:, 3:7])
        ref_R = matrix_from_quat(context["ref_rotation"][:, 0])
        reward = reward + 0.5 * torch.exp(-(root_R - ref_R).square().sum((-1, -2)) / (2 * 0.4**2))
        reward = reward - 0.05 * (action - previous_action).square().sum(-1)
        previous_feet = self.bodies(previous_state)[0][:, c.foot_ids]
        reward = reward - stance_slip(previous_feet, position[:, c.foot_ids], contact[:, c.foot_ids], self.dt)
        q = state[:, 13:42]
        reward = reward - 15 * ((self.limits[:, 0] - q).clamp_min(0) + (q - self.limits[:, 1]).clamp_min(0)).sum(-1)
        # Smooth survival penalizes model trajectories leaving the true task's viable region.
        tilt_cos = (root_R[:, 2] * ref_R[:, 2]).sum(-1)
        alive = torch.sigmoid((0.4 - height_error.abs()) / 0.03)
        alive = alive * torch.sigmoid((tilt_cos - math.cos(1.0)) / 0.04)
        alive = alive * torch.sigmoid((0.65 - pos_error.amax(-1)) / 0.04)
        return reward * self.dt, alive


def raw_history(observation):
    return unpack_terms(observation[:, :930], (3, 3, 29, 29, 29), 10)
