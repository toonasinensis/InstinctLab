"""PGMT Table I weights with explicit, auditable local kernel definitions."""

import torch

from isaaclab.utils.math import quat_error_magnitude


def tracking(env, group, quantity, std):
    c = env.command_manager.get_term("motion")
    bodies = c.upper_ids if group == "upper" else c.lower_ids
    joints = c.upper_joint_ids if group == "upper" else c.lower_joint_ids
    if quantity == "position":
        raw = (c.body_pos_relative_w[:, bodies] - c.robot_body_pos_w[:, bodies]).norm(dim=-1)
        maximum = c.cfg.tolerance_position_m
    elif quantity == "orientation":
        raw = quat_error_magnitude(c.body_quat_relative_w[:, bodies], c.robot_body_quat_w[:, bodies])
        maximum = c.cfg.tolerance_orientation_rad
    elif quantity == "linear_velocity":
        raw = (c.body_lin_vel_w[:, bodies] - c.robot_body_lin_vel_w[:, bodies]).norm(dim=-1)
        maximum = 0
    elif quantity == "angular_velocity":
        raw = (c.body_ang_vel_w[:, bodies] - c.robot_body_ang_vel_w[:, bodies]).norm(dim=-1)
        maximum = 0
    elif quantity == "joint_position":
        raw = (c.joint_pos[:, joints] - c.robot_joint_pos[:, joints]).abs()
        maximum = c.cfg.tolerance_joint_rad
    elif quantity == "joint_velocity":
        raw = (c.joint_vel[:, joints] - c.robot_joint_vel[:, joints]).abs()
        maximum = 0
    else:
        raise KeyError(quantity)
    effective = c.terrain.tolerance(raw, maximum) if group == "lower" else raw
    if not hasattr(c, "reward_errors"):
        c.reward_errors = {}
    c.reward_errors[group + "/" + quantity] = (raw.mean(-1), effective.mean(-1))
    return torch.exp(-effective.square().mean(-1) / (std * std))


def auxiliary(env, quantity, std=1.0):
    c = env.command_manager.get_term("motion")
    if quantity == "root_orientation":
        e = quat_error_magnitude(c.anchor_quat_w, c.robot_anchor_quat_w).square()
    elif quantity == "root_velocity":
        e = (c.corrected_velocity_w[:, :2] - c.robot_anchor_lin_vel_w[:, :2]).square().sum(-1)
    elif quantity == "floating_anchor":
        # XY is handled by observed velocity correction; only terrain-relative clearance here.
        e = (c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]).square()
    elif quantity == "vertical_acceleration":
        return c.contacts.acceleration[:, c.motion_anchor_body_index, 2].square() * c.contacts.valid_derivative
    elif quantity == "ee_acceleration":
        ids = c.foot_ids + c.hand_ids
        return (c.contacts.acceleration[:, ids] - c.contacts.ref_acceleration[:, ids]).square().sum(-1).mean(
            -1
        ) * c.contacts.valid_derivative
    elif quantity == "action_rate":
        return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)
    elif quantity == "joint_limits":
        limits = c.robot.data.soft_joint_pos_limits.torch
        return (
            (limits[..., 0] - c.robot_joint_pos).clamp_min(0) + (c.robot_joint_pos - limits[..., 1]).clamp_min(0)
        ).sum(-1)
    elif quantity == "undesired_contact":
        allowed = c.reference_contacts | ~c.reference_contact_valid
        allowed[:, c.foot_ids] = True
        return (c.contacts.contact & ~allowed).float().sum(-1)
    elif quantity == "impact":
        ids = c.impact_ids
        # Configured trunk bodies use reference-aware impact excess.
        unexpected = ~c.reference_contacts[:, ids] & c.reference_contact_valid[:, ids]
        return (c.contacts.force[:, ids].norm(dim=-1).sub(100).clamp_min(0).square() * unexpected).sum(-1)
    else:
        raise KeyError(quantity)
    return torch.exp(-e / (std * std))


def terrain(env, quantity):
    c = env.command_manager.get_term("motion")
    if not c.cfg.enable_contact_objectives:
        return torch.zeros(env.num_envs, device=env.device)
    return c.contacts.terrain_term(quantity)
