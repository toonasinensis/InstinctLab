"""Stage-two observation terms: original layout plus velocity correction and raw terrain."""

import torch

from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul

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


def foothold_joint_reference(env, command_name, derivative=False):
    term = env.command_manager.get_term(command_name)
    reference = term.command
    adaptation = env.command_manager.get_term("motion").foothold_reference()
    if adaptation is not None:
        delta = adaptation[2 if derivative else 1]
        reference = reference + delta[..., term._asset_joint_ids] * term.mask
    return reference


def elevation(env):
    robot = env.scene["robot"]
    hits = env.scene["height_scanner"].data.ray_hits_w.torch
    if not torch.isfinite(hits).all():
        raise FloatingPointError("PGMT elevation scan missed terrain")
    return (hits[..., 2] - robot.data.root_pos_w.torch[:, None, 2]).clamp(-5, 5)


def critic_state(env):
    c = env.command_manager.get_term("motion")
    anchor_pos_w = c.anchor_pos_w
    return privileged_state(
        env,
        reference_height=anchor_pos_w[:, 2:3] - c.terrain.support_z[:, None],
        anchor_pos_w=anchor_pos_w,
        anchor_quat_w=c.anchor_quat_w,
    )


def bilinear_elevation(env):
    """Query the same immutable 5 cm terrain used by imagined observations."""
    from instinctlab.tasks.shadowing.pgmt.tensor_math import yaw_angle

    from ..heightfield import HeightField

    if not hasattr(env, "pgmt_heightfield"):
        env.pgmt_heightfield = HeightField.from_context(
            env.command_manager.get_term("motion").terrain, getattr(env.cfg, "heightfield_resolution", 0.05)
        )
    robot = env.scene["robot"].data
    heights, valid = env.pgmt_heightfield.local(robot.root_pos_w.torch, yaw_angle(robot.root_quat_w.torch))
    if not valid.all():
        raise ValueError("Real heightscan left the terrain height field")
    return heights
