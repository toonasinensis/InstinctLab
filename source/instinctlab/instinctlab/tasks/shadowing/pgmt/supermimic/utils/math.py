"""Keep upstream WXYZ math explicit at the IsaacLab XYZW boundary.

Copied SuperMimic helpers use WXYZ. PGMT task extensions use IsaacLab math
directly and retain XYZW; no state is silently reinterpreted between them.
"""

from isaaclab.utils import math as lab


def matrix_from_quat(q):
    return lab.matrix_from_quat(q.roll(-1, -1))


def quat_apply(q, vector):
    return lab.quat_apply(q.roll(-1, -1), vector)


def quat_apply_inverse(q, vector):
    return lab.quat_apply_inverse(q.roll(-1, -1), vector)


def quat_inv(q):
    return lab.quat_inv(q.roll(-1, -1)).roll(1, -1)


def quat_mul(left, right):
    return lab.quat_mul(left.roll(-1, -1), right.roll(-1, -1)).roll(1, -1)


def yaw_quat(q):
    return lab.yaw_quat(q.roll(-1, -1)).roll(1, -1)


def subtract_frame_transforms(position, rotation, target_position, target_rotation=None):
    p, q = lab.subtract_frame_transforms(
        position,
        rotation.roll(-1, -1),
        target_position,
        None if target_rotation is None else target_rotation.roll(-1, -1),
    )
    return p, q.roll(1, -1)
