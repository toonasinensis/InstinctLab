from __future__ import annotations

import json
import numpy as np
from collections.abc import Sequence
from pathlib import Path

from .math_utils import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, resize_or_zero

JOINT_NAME_KEYS = ("joint_names", "motion_joint_names", "robot_joint_names", "action_joint_names")
BODY_NAME_KEYS = ("body_names", "motion_body_names", "robot_body_names")
BODY_ARRAY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_name_list_from_arrays(arrays: dict[str, np.ndarray], keys: Sequence[str]) -> list[str] | None:
    for key in keys:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key])
        if arr.shape == ():
            value = arr.item()
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
            else:
                return [_decode_name(value)]
        return [_decode_name(value) for value in arr.reshape(-1).tolist()]
    return None


def _name_indexes(source_names: Sequence[str], target_names: Sequence[str], path: str, kind: str) -> list[int]:
    source_to_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_to_index]
    if missing:
        raise ValueError(
            f"Motion {path} {kind}_names is missing required names {missing}. "
            f"Available {kind}_names: {list(source_names)}"
        )
    return [source_to_index[name] for name in target_names]


class MotionData:
    def __init__(self, path: str):
        self.path = str(path)
        with np.load(path, allow_pickle=True) as data:
            self.arrays = {name: self._load_value(data[name]) for name in data.files}
        self.files = list(self.arrays.keys())
        self.num_frames = int(self.arrays["joint_pos"].shape[0])
        self.total_bytes = sum(value.nbytes for value in self.arrays.values() if isinstance(value, np.ndarray))

    @staticmethod
    def _load_value(value):
        if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "U", "O") and value.shape == ():
            return value.item()
        return np.asarray(value)

    def __getitem__(self, key: str):
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    @property
    def joint_names(self) -> list[str] | None:
        return _read_name_list_from_arrays(self.arrays, JOINT_NAME_KEYS)

    @property
    def body_names(self) -> list[str] | None:
        return _read_name_list_from_arrays(self.arrays, BODY_NAME_KEYS)

    def print_config(self) -> None:
        print(
            f"[INFO] Preloaded motion arrays: frames={self.num_frames}, "
            f"fields={len(self.files)}, size={self.total_bytes / (1024 * 1024):.2f} MiB"
        )

    def aligned_to(
        self,
        *,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        model_nbody: int | None = None,
        body_ids: np.ndarray | None = None,
    ) -> tuple[MotionData, dict[str, np.ndarray | str]]:
        arrays = dict(self.arrays)
        info: dict[str, np.ndarray | str] = {}

        source_joint_names = _read_name_list_from_arrays(arrays, JOINT_NAME_KEYS)
        joint_dim = int(np.asarray(arrays["joint_pos"]).shape[1])
        if source_joint_names is not None:
            if len(source_joint_names) != joint_dim:
                raise ValueError(
                    f"Motion {self.path} joint_names length {len(source_joint_names)} "
                    f"does not match joint_pos dim {joint_dim}."
                )
            joint_indexes = np.asarray(
                _name_indexes(source_joint_names, joint_names, self.path, "joint"), dtype=np.int64
            )
            for key in ("joint_pos", "joint_vel"):
                if key in arrays:
                    arrays[key] = np.asarray(arrays[key])[:, joint_indexes]
            arrays["joint_names"] = np.asarray(list(joint_names))
            info["joint_order_source"] = "metadata"
        elif joint_dim == len(joint_names):
            info["joint_order_source"] = "legacy_dim"
        else:
            raise ValueError(
                f"Motion {self.path} joint_pos dim {joint_dim} does not match expected joint dim {len(joint_names)}, "
                "and no joint_names metadata was found. Run scripts/attach_npz_names.py or regenerate the npz."
            )

        body_indexes = resolve_body_indices_from_names(
            arrays=arrays,
            target_body_names=body_names,
            model_nbody=model_nbody,
            body_ids=body_ids,
            path=self.path,
        )
        body_indexes = np.asarray(body_indexes, dtype=np.int64)
        for key in BODY_ARRAY_KEYS:
            if key in arrays:
                arrays[key] = np.asarray(arrays[key])[:, body_indexes]
        arrays["body_names"] = np.asarray(list(body_names))
        info["motion_body_indices"] = body_indexes
        info["body_order_source"] = "metadata" if self.body_names is not None else "legacy_dim"
        return MotionData.from_arrays(self.path, arrays), info

    @classmethod
    def from_arrays(cls, path: str, arrays: dict[str, np.ndarray]):
        obj = cls.__new__(cls)
        obj.path = str(path)
        obj.arrays = {name: obj._load_value(value) for name, value in arrays.items()}
        obj.files = list(obj.arrays.keys())
        obj.num_frames = int(obj.arrays["joint_pos"].shape[0])
        obj.total_bytes = sum(value.nbytes for value in obj.arrays.values() if isinstance(value, np.ndarray))
        return obj


def resolve_body_indices_from_names(
    *,
    arrays: dict[str, np.ndarray],
    target_body_names: Sequence[str],
    model_nbody: int | None,
    body_ids: np.ndarray | None,
    path: str,
) -> np.ndarray:
    body_dim = int(np.asarray(arrays["body_pos_w"]).shape[1])
    for key in BODY_ARRAY_KEYS:
        if key in arrays and int(np.asarray(arrays[key]).shape[1]) != body_dim:
            raise ValueError(
                f"Motion {path} body dimension mismatch: body_pos_w has {body_dim}, "
                f"but {key} has {int(np.asarray(arrays[key]).shape[1])}."
            )

    source_body_names = _read_name_list_from_arrays(arrays, BODY_NAME_KEYS)
    if source_body_names is not None:
        if len(source_body_names) != body_dim:
            raise ValueError(
                f"Motion {path} body_names length {len(source_body_names)} does not match body array dim {body_dim}."
            )
        return np.asarray(_name_indexes(source_body_names, target_body_names, path, "body"), dtype=np.int64)

    selected_body_count = len(target_body_names)
    if body_dim == selected_body_count:
        return np.arange(selected_body_count, dtype=np.int64)
    if body_ids is not None and model_nbody is not None:
        body_ids = np.asarray(body_ids, dtype=np.int64)
        if body_dim == int(model_nbody) - 1:
            return body_ids - 1
        if body_dim == int(model_nbody):
            return body_ids
        body_indexes = body_ids - 1
        if body_indexes.size > 0 and int(body_indexes.max()) < body_dim:
            return body_indexes
    raise ValueError(
        f"Motion {path} cannot map body dim {body_dim} to selected body dim {selected_body_count}. "
        "Add body_names metadata with scripts/attach_npz_names.py or regenerate the npz."
    )


def motion_files(motion_file: str, dataset_txt: str | None) -> list[str]:
    root = Path(motion_file)
    dataset_txt = dataset_txt.strip() if dataset_txt is not None else None
    if dataset_txt:
        files: list[str] = []
        for line in Path(dataset_txt).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            p = Path(line)
            files.append(str(p if p.is_absolute() else root / p))
        if files:
            return files
        raise ValueError(f"No motion entries in dataset txt: {dataset_txt}")
    if root.is_file():
        if root.suffix == ".jsonl":
            files = []
            for line_no, line in enumerate(root.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "tracking_motion" not in record:
                    raise KeyError(f"{root}:{line_no}: missing required 'tracking_motion' field.")
                p = Path(str(record["tracking_motion"])).expanduser()
                files.append(str(p if p.is_absolute() else root.parent / p))
            if files:
                return files
            raise ValueError(f"No motion entries in manifest: {root}")
        return [str(root)]
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {motion_file}")
    return [str(path) for path in files]


def first_motion_file(motion_file: str, dataset_txt: str | None) -> str:
    return motion_files(motion_file, dataset_txt)[0]


def future_indices(t: int, offsets: list[int], total: int) -> np.ndarray:
    return np.clip(t + np.asarray(offsets, dtype=np.int64), 0, total - 1)


def motion_body_index(meta: dict, body_name: str | None = None) -> tuple[str, int]:
    body_names = list(meta["motion_body_names"])
    name = body_name or meta.get("root_body_name") or body_names[0]
    if name not in body_names:
        raise ValueError(f"Motion body '{name}' is not in motion_body_names: {body_names}")
    body_index = body_names.index(name)
    motion_body_indices = meta.get("motion_body_indices")
    if motion_body_indices is not None:
        body_index = int(motion_body_indices[body_index])
    return name, body_index


def motion_frame_root_state(
    motion: MotionData,
    meta: dict,
    frame: int,
    root_body_name: str | None = None,
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = int(np.clip(frame, 0, motion.num_frames - 1))
    name, body_idx = motion_body_index(meta, root_body_name)
    pos = np.asarray(motion["body_pos_w"][frame, body_idx], dtype=np.float64)
    quat = np.asarray(motion["body_quat_w"][frame, body_idx], dtype=np.float64)
    lin_vel = (
        np.asarray(motion["body_lin_vel_w"][frame, body_idx], dtype=np.float64)
        if "body_lin_vel_w" in motion
        else np.zeros(3, dtype=np.float64)
    )
    ang_vel = (
        np.asarray(motion["body_ang_vel_w"][frame, body_idx], dtype=np.float64)
        if "body_ang_vel_w" in motion
        else np.zeros(3, dtype=np.float64)
    )
    return name, body_idx, pos, quat, lin_vel, ang_vel


def motion_local_step_summary(
    motion: MotionData,
    meta: dict,
    root_body_name: str | None = None,
) -> dict[str, np.ndarray]:
    name, body_idx = motion_body_index(meta, root_body_name)
    pos = np.asarray(motion["body_pos_w"][:, body_idx], dtype=np.float64)
    quat = np.asarray(motion["body_quat_w"][:, body_idx], dtype=np.float64)
    if pos.shape[0] < 2:
        zeros = np.zeros(3, dtype=np.float64)
        return {"root_body": np.asarray(name), "mean": zeros, "min": zeros, "max": zeros, "total": zeros}
    local_step = quat_apply_inverse(quat[:-1], pos[1:] - pos[:-1])
    return {
        "root_body": np.asarray(name),
        "mean": local_step.mean(axis=0),
        "min": local_step.min(axis=0),
        "max": local_step.max(axis=0),
        "total": pos[-1] - pos[0],
    }


def motion_groups(
    motion: MotionData,
    t: int,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = meta["future_step_num"]
    future = future_indices(t, offsets, motion["joint_pos"].shape[0])
    rbt_dim = int(np.prod(meta.get("observation_shapes", {}).get("rbt_cmd_mf", [0])))
    rbt_cmd_mf = build_rbt_cmd_mf(motion, future, meta, robot_anchor_quat_w, rbt_dim)
    if rbt_dim <= 0:
        rbt_dim = rbt_cmd_mf.shape[1]
    if rbt_cmd_mf.shape[1] != rbt_dim:
        if "rbt_cmd_mf" not in meta.get("observation_terms", {}):
            raise ValueError(
                f"Built rbt_cmd_mf with dim {rbt_cmd_mf.shape[1]}, but ONNX metadata expects {rbt_dim}. "
                "ONNX metadata has no observation_terms for rbt_cmd_mf, so sim2sim cannot know the exported term "
                "layout. Re-export the policy with the updated exporter to save explicit observation_terms."
            )
        raise ValueError(
            f"Built rbt_cmd_mf with dim {rbt_cmd_mf.shape[1]}, but ONNX metadata expects {rbt_dim}. "
            "Check motion joint order/future_step_num and the exported task config."
        )
    smpl_dim = int(np.prod(meta["observation_shapes"].get("smpl_cmd_mf", [0])))
    smpl_cmd_mf = build_smpl_cmd_mf(motion, future, meta, robot_anchor_quat_w, smpl_dim)
    return rbt_cmd_mf, smpl_cmd_mf


def build_rbt_cmd_mf(
    motion: MotionData,
    future: np.ndarray,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
    rbt_dim: int,
) -> np.ndarray:
    builders = {
        "motion_joint_pos_multi_future": lambda: motion["joint_pos"][future].reshape(1, -1),
        "motion_joint_vel_multi_future": lambda: motion["joint_vel"][future].reshape(1, -1),
        "motion_anchor_ori_b_multi_future": lambda: motion_anchor_ori_b_mf(motion, future, meta, robot_anchor_quat_w),
        "motion_anchor_z_multi_future": lambda: motion_anchor_z_mf(motion, future, meta),
    }
    terms = meta.get("observation_terms", {}).get("rbt_cmd_mf", {}).get("terms", [])
    pieces = []
    for term in terms:
        name = term.get("name")
        if name not in builders:
            raise KeyError(f"Unsupported rbt_cmd_mf observation term in ONNX metadata: {name!r}")
        pieces.append(builders[name]())

    if not pieces:
        pieces = [
            builders["motion_joint_pos_multi_future"](),
            builders["motion_joint_vel_multi_future"](),
            builders["motion_anchor_ori_b_multi_future"](),
        ]
        built_dim = sum(piece.shape[1] for piece in pieces)
        if rbt_dim == built_dim + len(future):
            pieces.append(builders["motion_anchor_z_multi_future"]())

    return np.concatenate(pieces, axis=-1).astype(np.float32)


def motion_anchor_ori_mf(motion: MotionData, future: np.ndarray, meta: dict) -> np.ndarray:
    _, anchor_idx = motion_body_index(meta, meta["anchor_body_name"])
    return motion["body_quat_w"][future, anchor_idx, :]


def motion_anchor_ori_b_mf(
    motion: MotionData,
    future: np.ndarray,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
) -> np.ndarray:
    target_quat = motion_anchor_ori_mf(motion, future, meta).reshape(1, len(future), 4)
    robot_anchor = np.repeat(robot_anchor_quat_w[:, None, :], len(future), axis=1)
    rel_quat = quat_mul(quat_inv(robot_anchor), target_quat)
    return matrix_from_quat(rel_quat)[..., :2].reshape(1, -1).astype(np.float32)


def motion_anchor_z_mf(motion: MotionData, future: np.ndarray, meta: dict) -> np.ndarray:
    _, anchor_idx = motion_body_index(meta, meta["anchor_body_name"])
    return np.asarray(motion["body_pos_w"][future, anchor_idx, 2], dtype=np.float32).reshape(1, -1)


def build_smpl_cmd_mf(
    motion: MotionData,
    future: np.ndarray,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
    smpl_dim: int,
) -> np.ndarray:
    if smpl_dim == 0:
        return np.zeros((1, 0), dtype=np.float32)
    if "smpl_joints" not in motion:
        return np.zeros((1, smpl_dim), dtype=np.float32)

    joints = np.asarray(motion["smpl_joints"][future], dtype=np.float32)
    joints = joints.reshape(1, joints.shape[0], 24, 3)
    root_quat_dim = 6 * len(future)
    if "smpl_poses" in motion.files:
        root_6d = np.zeros((1, root_quat_dim), dtype=np.float32)
    else:
        root_6d = np.zeros((1, root_quat_dim), dtype=np.float32)
    local_joints = joints.reshape(1, -1)
    return resize_or_zero(np.concatenate([local_joints, root_6d], axis=-1), smpl_dim)
