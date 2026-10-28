"""Stage-two observation terms: original layout plus velocity correction and raw terrain."""

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul

from .observations import privileged_state


def orientation(env):
    c = env.command_manager.get_term("motion")
    q = quat_inv(c.robot_anchor_quat_w)[:, None].expand_as(c.anchor_quat_w_future)
    return matrix_from_quat(quat_mul(q, c.anchor_quat_w_future))[..., :2].flatten(1)


def clearance(env):
    reference = env.scene["motion_reference"]
    return reference.data.base_pos_w.torch[..., 2] - env.scene.env_origins[:, None, 2]


def velocity(env):
    return env.command_manager.get_term("motion").corrected_velocity_future_b.flatten(1)


def elevation(env):
    robot = env.scene["robot"]
    hits = env.scene["height_scanner"].data.ray_hits_w.torch
    if not torch.isfinite(hits).all():
        raise FloatingPointError("PGMT elevation scan missed terrain")
    return (hits[..., 2] - robot.data.root_pos_w.torch[:, None, 2]).clamp(-5, 5)


def critic_state(env):
    state = privileged_state(env)
    c = env.command_manager.get_term("motion")
    state[:, 67:68] = c.anchor_pos_w[:, 2:3] - c.terrain.support_z[:, None]
    state[:, 68:71] = quat_apply_inverse(c.robot_anchor_quat_w, c.anchor_pos_w - c.robot_anchor_pos_w)
    state[:, 71:77] = matrix_from_quat(quat_mul(quat_inv(c.robot_anchor_quat_w), c.anchor_quat_w))[..., :2].flatten(1)
    return state
