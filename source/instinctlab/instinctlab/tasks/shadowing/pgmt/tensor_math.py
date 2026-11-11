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
