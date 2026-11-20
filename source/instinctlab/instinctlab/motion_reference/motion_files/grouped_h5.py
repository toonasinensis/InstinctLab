import fcntl
import h5py
import hashlib
import json
import math
import numpy as np
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

_FORMAT = "instinctlab_grouped_motion"
_VERSION = 2
_POSE_FIELDS = ("root_trans", "root_wxyz", "dof_pos")
_UNITS = {
    "root_position_unit": "meters",
    "joint_angle_unit": "radians",
    "quaternion_order": "wxyz",
    "root_quaternion_order": "wxyz",
}


def _dataset(owner, name):
    dataset = owner.get(name)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{owner.name} requires a {name!r} dataset")
    return dataset


def _names(owner, name):
    dataset = _dataset(owner, name)
    if dataset.ndim != 1 or h5py.check_string_dtype(dataset.dtype) is None:
        raise ValueError(f"{owner.name}/{name} must be a vector of strings")
    names = tuple(dataset.asstr(encoding="utf-8")[:])
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError(f"{owner.name}/{name} must contain unique, nonempty names")
    return names


def _check_metadata(owner, fps):
    for field, expected in _UNITS.items():
        if field in owner.attrs:
            value = owner.attrs[field]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if not isinstance(value, str) or value != expected:
                raise ValueError(f"{owner.name}: expected {field}={expected}")
    rates = []
    if "fps" in owner.attrs:
        rates.append(owner.attrs["fps"])
    if "fps" in owner:
        dataset = _dataset(owner, "fps")
        if dataset.size != 1 or dataset.dtype.kind not in "iuf":
            raise ValueError(f"{owner.name}: fps must be a numeric scalar")
        rates.append(dataset[()])
    for value in rates:
        rate = np.asarray(value)
        if rate.size != 1 or rate.dtype.kind not in "iuf" or float(rate.item()) != fps:
            raise ValueError(f"{owner.name}: expected source fps {fps:g}, got {value!r}")
    return bool(rates)


def _clip_headers(file, fps, joint_names, motion_names):
    file_has_fps = _check_metadata(file, fps)
    clips = []

    def visit(name, group):
        if not isinstance(group, h5py.Group) or not any(field in group for field in _POSE_FIELDS):
            return
        if not _check_metadata(group, fps) and not file_has_fps and fps != 50:
            raise ValueError(f"{group.name}: expected an explicit source fps of {fps:g}")
        names = _names(group, "joint_names")
        if set(names) != set(joint_names) or len(names) != len(joint_names):
            raise ValueError(f"{group.name}: motion joints do not match the robot joint names")
        order_by_name = {name: index for index, name in enumerate(names)}
        order = tuple(order_by_name[name] for name in joint_names)
        joints = _dataset(group, "dof_pos")
        if joints.ndim != 2 or joints.shape[0] < 2:
            raise ValueError(f"{group.name}: dof_pos must be a matrix with at least two frames")
        frames = joints.shape[0]
        for field, width in zip(_POSE_FIELDS, (3, 4, len(joint_names)), strict=True):
            dataset = _dataset(group, field)
            if dataset.shape != (frames, width) or dataset.dtype.kind not in "iuf":
                raise ValueError(f"{group.name}/{field} must be numeric with shape {(frames, width)}")
        clips.append((name, frames, order))

    if motion_names is None:
        file.visititems(visit)
    else:
        for name in motion_names:
            group = file.get(name)
            if not isinstance(group, h5py.Group) or group.name != f"/{name}":
                raise ValueError(f"{file.filename}: motion group {name!r} was not found")
            visit(name, group)
            if not clips or clips[-1][0] != name:
                raise ValueError(f"{file.filename}: {name!r} is not a motion group")
    if not clips:
        raise ValueError(f"{file.filename}: no grouped H5 motion clips found")
    return clips


def _expected_inputs(fps, joint_names):
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Expected fps must be finite and positive")
    joint_names = tuple(joint_names)
    if not joint_names or not all(joint_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError("Robot joint names must be nonempty and unique")
    return fps, joint_names


def _selected_names(motion_names):
    if motion_names is None:
        return None
    if isinstance(motion_names, str):
        raise ValueError("motion_names must be a sequence of group names")
    names = tuple(motion_names)
    if not names or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError("motion_names must contain unique, nonempty group names")
    return tuple(sorted(names))


def read_grouped_motion_index(source, fps, joint_names, *, motion_names=None):
    fps, joint_names = _expected_inputs(fps, joint_names)
    with h5py.File(Path(source).expanduser(), "r") as file:
        clips = _clip_headers(file, fps, joint_names, _selected_names(motion_names))
    return tuple(clip[0] for clip in clips), tuple(clip[1] for clip in clips)


def _source_signature(source):
    stat = source.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _data_key(contract, source_digest):
    return hashlib.sha256(_json({"contract": contract, "source_sha256": source_digest}).encode("utf-8")).hexdigest()


def _frame_fields(joints, bodies):
    return {
        "joint_pos": ((joints,), np.float32),
        "joint_vel": ((joints,), np.float32),
        "body_pos_w": ((bodies, 3), np.float32),
        "body_quat_w": ((bodies, 4), np.float32),
        "body_lin_vel_w": ((bodies, 3), np.float32),
        "body_ang_vel_w": ((bodies, 3), np.float32),
        "contact": ((bodies,), np.bool_),
        "valid": ((bodies,), np.bool_),
        "bottom": ((bodies,), np.float32),
    }


def _cache_index(file, settings, key, contract, fields):
    if (
        file.attrs.get("format") != _FORMAT
        or file.attrs.get("version") != _VERSION
        or not file.attrs.get("complete", False)
        or file.attrs.get("cache_key") != key
    ):
        raise ValueError("Incomplete or incompatible grouped motion cache")
    cached_settings = _dataset(file, "settings")
    if cached_settings.shape != () or cached_settings.asstr()[()] != settings:
        raise ValueError("Grouped motion cache settings do not match")
    digest = file.attrs.get("source_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or file.attrs.get("data_key") != _data_key(contract, digest)
    ):
        raise ValueError("Grouped motion cache has an invalid source digest or data key")
    if _names(file, "joint_names") != contract["joint_names"] or _names(file, "body_names") != contract["body_names"]:
        raise ValueError("Grouped motion cache robot names do not match")
    rate = _dataset(file, "fps")
    if rate.shape != () or rate.dtype != np.dtype(np.float64) or rate[()] != contract["fps"]:
        raise ValueError("Grouped motion cache frame rate does not match")
    names = _names(file, "motion_names")
    if contract["motion_names"] is not None and names != contract["motion_names"]:
        raise ValueError("Grouped motion cache selection does not match")
    index = [_dataset(file, field) for field in ("starts", "lengths")]
    if any(dataset.shape != (len(names),) or dataset.dtype != np.dtype(np.int64) for dataset in index):
        raise ValueError("Grouped motion cache has an invalid clip index")
    starts, lengths = (dataset[:] for dataset in index)
    total = sum(int(length) for length in lengths)
    if (
        np.any(lengths < 2)
        or total > np.iinfo(np.int64).max
        or starts[0] != 0
        or not np.array_equal(starts[1:], lengths[:-1].cumsum())
    ):
        raise ValueError("Grouped motion cache has invalid clip boundaries")
    for field, (shape, dtype) in fields.items():
        dataset = _dataset(file, field)
        if dataset.shape != (total, *shape) or dataset.dtype != np.dtype(dtype):
            raise ValueError(f"Grouped motion cache has invalid {field!r} data")


def _open_cache(target, settings, key, contract, fields):
    file = None
    try:
        file = h5py.File(target, "r")
        _cache_index(file, settings, key, contract, fields)
    except (OSError, KeyError, TypeError, ValueError):
        if file is not None:
            file.close()
        return None
    return file


def _read_pose(group, order):
    positions, quaternions, joints = (np.asarray(group[field], dtype=np.float64) for field in _POSE_FIELDS)
    if not all(np.isfinite(values).all() for values in (positions, quaternions, joints)):
        raise ValueError(f"{group.name}: motion contains nonfinite pose data")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-8):
        raise ValueError(f"{group.name}: motion contains invalid root quaternion norms")
    with np.errstate(over="ignore"):
        positions = positions.astype(np.float32)
        joints = joints[:, order].astype(np.float32)
    if not np.isfinite(positions).all() or not np.isfinite(joints).all():
        raise ValueError(f"{group.name}: motion poses exceed float32 range")
    return positions, (quaternions / norms).astype(np.float32), joints


def _cache_clip(file, name, frames, start, fields, payload):
    for field, (shape, dtype) in fields.items():
        values = np.asarray(payload[field])
        numeric = values.dtype.kind in "iuf" if dtype is np.float32 else values.dtype.kind == "b"
        if values.shape != (frames, *shape) or not numeric:
            raise ValueError(f"{name}: preprocessed {field} has invalid shape or dtype")
        values = values.astype(dtype, copy=False)
        invalid = np.isinf(values) if field == "bottom" else ~np.isfinite(values)
        if invalid.any():
            raise ValueError(f"{name}: preprocessed {field} contains invalid values")
        file[field][start : start + frames] = values


def _write_cache(source, signature, target, settings, key, contract, fields, preprocess):
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    os.close(descriptor)
    try:
        with h5py.File(source, "r") as raw, h5py.File(temporary, "w") as file:
            fps = contract["fps"]
            clips = _clip_headers(raw, fps, contract["joint_names"], contract["motion_names"])
            total = sum(clip[1] for clip in clips)
            if total > np.iinfo(np.int64).max:
                raise ValueError("Grouped H5 motion has too many frames")
            lengths = np.asarray([clip[1] for clip in clips], dtype=np.int64)
            starts = np.concatenate((np.zeros(1, dtype=np.int64), lengths[:-1].cumsum()))
            file.attrs.update(format=_FORMAT, version=_VERSION, cache_key=key, complete=False, fps=fps)
            file.create_dataset("settings", data=settings, dtype=h5py.string_dtype(encoding="utf-8"))
            for field, values in (
                ("joint_names", contract["joint_names"]),
                ("body_names", contract["body_names"]),
                ("motion_names", [clip[0] for clip in clips]),
            ):
                file.create_dataset(field, data=values, dtype=h5py.string_dtype(encoding="utf-8"))
            file["starts"] = starts
            file["lengths"] = lengths
            file["fps"] = fps
            for field, (shape, dtype) in fields.items():
                file.create_dataset(
                    field, (total, *shape), dtype=dtype, chunks=(min(4096, total), *shape), fletcher32=True
                )
            for (name, frames, order), start in zip(clips, starts, strict=True):
                positions, quaternions, joints = _read_pose(raw[name], order)
                payload = preprocess(positions, quaternions, joints, fps)
                _cache_clip(file, name, frames, int(start), fields, payload)
                del payload
            with source.open("rb") as raw_bytes:
                digest = hashlib.file_digest(raw_bytes, "sha256").hexdigest()
            file.attrs.update(complete=True, source_sha256=digest, data_key=_data_key(contract, digest))
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        if _source_signature(source) != signature:
            raise RuntimeError(f"Grouped H5 motion source changed during preprocessing: {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def open_grouped_motion_cache(
    source, cache_dir, joint_names, body_names, fps, identity, preprocess, *, motion_names=None
):
    fps, joint_names = _expected_inputs(fps, joint_names)
    motion_names = _selected_names(motion_names)
    body_names = tuple(body_names)
    if not body_names or not all(body_names) or len(set(body_names)) != len(body_names):
        raise ValueError("Robot body names must be nonempty and unique")
    source = Path(source).expanduser().resolve(strict=True)
    signature = _source_signature(source)
    contract = {
        "format": _FORMAT,
        "version": _VERSION,
        "identity": identity,
        "joint_names": joint_names,
        "body_names": body_names,
        "fps": fps,
        "motion_names": motion_names,
    }
    settings = _json({"source": str(source), "signature": signature, "contract": contract})
    key = hashlib.sha256(settings.encode("utf-8")).hexdigest()
    target = Path(cache_dir).expanduser() / f"{source.stem[:80]}_{key}.h5"
    fields = _frame_fields(len(joint_names), len(body_names))
    cached = _open_cache(target, settings, key, contract, fields)
    if cached is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".h5.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            cached = _open_cache(target, settings, key, contract, fields)
            if cached is None:
                _write_cache(source, signature, target, settings, key, contract, fields, preprocess)
                cached = _open_cache(target, settings, key, contract, fields)
                if cached is None:
                    raise RuntimeError(f"Could not reopen the processed motion cache: {target}")
    with cached:
        if _source_signature(source) != signature:
            raise RuntimeError(f"Grouped H5 motion source changed while opening the cache: {source}")
        yield cached
