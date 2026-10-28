from __future__ import annotations

import numpy as np


def as_vector(meta: dict, key: str, size: int, default: float = 0.0) -> np.ndarray:
    value = meta.get(key, default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size != size:
        raise ValueError(f"Metadata '{key}' has {arr.size} values, expected {size}.")
    return arr


def resize_or_zero(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(1, -1)
    if values.shape[1] == dim:
        return values
    out = np.zeros((1, dim), dtype=np.float32)
    width = min(dim, values.shape[1])
    out[:, :width] = values[:, :width]
    return out


def quat_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    norm_sq = np.sum(q * q, axis=-1, keepdims=True)
    return out / np.clip(norm_sq, 1.0e-9, None)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v - q[..., :1] * t + np.cross(xyz, t)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    r, i, j, k = np.moveaxis(q, -1, 0)
    two_s = 2.0 / np.clip(np.sum(q * q, axis=-1), 1.0e-9, None)
    mat = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return mat.reshape(q.shape[:-1] + (3, 3))


def shape_dim(shape: list[int] | tuple[int, ...]) -> int:
    return int(np.prod(shape)) if shape else 0
