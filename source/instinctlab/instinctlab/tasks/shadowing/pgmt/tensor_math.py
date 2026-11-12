"""Quaternion vector operations without dynamic permutation indices in compiled kernels."""

import torch


def cross(left, right, dim=-1):
    if dim != -1:
        raise ValueError("PGMT cross products use the last coordinate dimension")
    return torch.stack(
        (
            left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
            left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
            left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
        ),
        -1,
    )


def quat_apply(quaternion, vector):
    t = cross(quaternion[..., :3], vector) * 2
    return vector + quaternion[..., 3:4] * t + cross(quaternion[..., :3], t)


def quat_apply_inverse(quaternion, vector):
    t = cross(quaternion[..., :3], vector) * 2
    return vector - quaternion[..., 3:4] * t + cross(quaternion[..., :3], t)


def yaw_angle(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def rotation_increment(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat((0.5 * torch.sinc(angle / (2 * torch.pi)) * vector, (angle / 2).cos()), -1)


def rotation_vector(quaternion):
    # Avoid an unselected 0/0 branch in the upstream axis-angle conversion: its
    # forward result at identity is finite, but its backward result contains NaNs.
    quaternion = quaternion * torch.where(quaternion[..., 3:4] < 0, -1.0, 1.0)
    angle = 2 * torch.atan2(quaternion[..., :3].norm(dim=-1, keepdim=True), quaternion[..., 3:4])
    return 2 * quaternion[..., :3] / torch.sinc(angle / (2 * torch.pi))
