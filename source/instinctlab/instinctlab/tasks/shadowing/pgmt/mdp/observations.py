"""The stage-one observation contract over native motion-reference data."""

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul


def reference_orientation(env):
    ref = env.scene["motion_reference"].data
    robot = env.scene["robot"]
    rotation = ref.base_quat_w.torch
    q = quat_inv(robot.data.root_quat_w.torch)[:, None].expand_as(rotation)
    return matrix_from_quat(quat_mul(q, rotation))[..., :2].flatten(1)


def reference_height(env):
    return env.scene["motion_reference"].data.base_pos_w[..., 2]


def privileged_state(env): 
    motion = env.scene["motion_reference"]
    ref = motion.reference_frame
    robot = env.scene["robot"]
    q = ref.base_quat_w[:, 0]
    gravity = q.new_tensor([0.0, 0.0, -1.0]).expand(env.num_envs, -1)
    command = torch.cat(
        (
            ref.joint_pos[:, 0],
            ref.joint_vel[:, 0],
            quat_apply_inverse(q, ref.base_lin_vel_w[:, 0]),
            quat_apply_inverse(q, ref.base_ang_vel_w[:, 0]),
            quat_apply_inverse(q, gravity),
            ref.base_pos_w[:, 0, 2:3],
        ),
        -1,
    )
    ids = robot.find_bodies(motion.cfg.link_of_interests, preserve_order=True)[0]
    pos = robot.data.body_pos_w.torch[:, ids]
    rot = robot.data.body_quat_w.torch[:, ids]
    root_q = robot.data.root_quat_w.torch
    anchor_pos = quat_apply_inverse(root_q, ref.base_pos_w[:, 0] - robot.data.root_pos_w.torch)
    anchor_ori = matrix_from_quat(quat_mul(quat_inv(root_q), q))[..., :2].flatten(1)
    inverse = quat_inv(root_q[:, None].expand_as(rot))
    body_pos = quat_apply_inverse(root_q[:, None].expand_as(rot), pos - robot.data.root_pos_w.torch[:, None]).flatten(1)
    body_ori = matrix_from_quat(quat_mul(inverse, rot))[..., :2].flatten(1)
    return torch.cat(
        (
            command,
            anchor_pos,
            anchor_ori,
            body_pos,
            body_ori,
            robot.data.root_lin_vel_b.torch,
            robot.data.root_ang_vel_b.torch,
            robot.data.joint_pos.torch - robot.data.default_joint_pos.torch,
            robot.data.joint_vel.torch - robot.data.default_joint_vel.torch,
            env.action_manager.action,
        ),
        -1,
    )


def anchor_height_reward(env, std=0.3): # 这个不应该出现
    ref = env.scene["motion_reference"].reference_frame
    error = ref.base_pos_w[:, 0, 2] - env.scene["robot"].data.root_pos_w.torch[:, 2]
    return torch.exp(-error.square() / std**2)
