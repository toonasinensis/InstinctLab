from __future__ import annotations

import numpy as np

from .math_utils import as_vector, matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, shape_dim
from .motion import MotionData, motion_body_index, motion_groups
from .mujoco_robot import validate_sensor

G1_PROP_TERM_ORDER = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")


class ImuReader:
    def __init__(
        self,
        model,
        quat_sensor_name: str = "base_quat",
        gyro_sensor_name: str = "base_gyro",
    ):
        self.quat_sensor_name = quat_sensor_name
        self.gyro_sensor_name = gyro_sensor_name
        validate_sensor(model, quat_sensor_name, expected_dim=4)
        validate_sensor(model, gyro_sensor_name, expected_dim=3)

    def read(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root_quat = np.asarray(data.sensor(self.quat_sensor_name).data, dtype=np.float64)[None, :]
        root_ang_vel_b = np.asarray(data.sensor(self.gyro_sensor_name).data, dtype=np.float64)[None, :]
        gravity_b = quat_apply_inverse(root_quat, np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64))
        return root_quat, root_ang_vel_b, gravity_b

    def print_config(self) -> None:
        print(f"[INFO] IMU source: MuJoCo XML sensors quat={self.quat_sensor_name} gyro={self.gyro_sensor_name}")


class HistoryBuffer:
    def __init__(self, dim: int, length: int):
        self.dim = int(dim)
        self.length = max(1, int(length))
        self._values = np.zeros((self.length, self.dim), dtype=np.float32)
        self._filled = False

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != self.dim:
            raise ValueError(f"History value dim {value.shape[0]} does not match expected dim {self.dim}.")
        if not self._filled:
            self._values[:] = value
            self._filled = True
        else:
            self._values[:-1] = self._values[1:]
            self._values[-1] = value
        return self.value()

    def value(self) -> np.ndarray:
        return self._values.reshape(1, -1).astype(np.float32)


class TermMajorHistory:
    def __init__(self, term_dims: list[tuple[str, int, int]]):
        self.term_dims = term_dims
        self.buffers = {name: HistoryBuffer(dim, history_len) for name, dim, history_len in term_dims}

    @property
    def dim(self) -> int:
        return sum(dim * history_len for _, dim, history_len in self.term_dims)

    @property
    def max_history_length(self) -> int:
        return max((history_len for _, _, history_len in self.term_dims), default=1)

    def update(self, values: dict[str, np.ndarray]) -> np.ndarray:
        pieces = []
        for name, _, _ in self.term_dims:
            if name not in values:
                raise KeyError(f"Missing prop observation term '{name}'.")
            pieces.append(self.buffers[name].update(values[name]))
        return np.concatenate(pieces, axis=-1).astype(np.float32)


def fallback_prop_terms(prop_dim: int, num_joints: int) -> list[tuple[str, int, int]]:
    base_dims = {
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": num_joints,
        "joint_vel": num_joints,
        "actions": num_joints,
    }
    single_dim = sum(base_dims.values())
    if prop_dim % single_dim != 0:
        raise ValueError(f"prop dim {prop_dim} is not divisible by fallback single-frame dim {single_dim}.")
    history_len = prop_dim // single_dim
    return [(name, base_dims[name], history_len) for name in G1_PROP_TERM_ORDER]


def prop_terms_from_metadata(meta: dict, num_joints: int) -> list[tuple[str, int, int]]:
    prop_dim = int(np.prod(meta["observation_shapes"].get("prop", [0])))
    observation_terms = meta.get("observation_terms", {})
    prop_terms = observation_terms.get("prop", {}).get("terms", [])
    if not prop_terms:
        return fallback_prop_terms(prop_dim, num_joints)

    out = []
    for term in prop_terms:
        name = term["name"]
        shape = term.get("shape", [])
        base_shape = term.get("base_shape", shape)
        history_len = max(1, int(term.get("history_length", 0)))
        term_dim = shape_dim(base_shape)
        if term_dim <= 0:
            term_dim = shape_dim(shape)
            if history_len > 1 and term_dim % history_len == 0:
                term_dim //= history_len
        out.append((name, term_dim, history_len))

    total = sum(dim * history_len for _, dim, history_len in out)
    if total != prop_dim:
        raise ValueError(f"prop metadata describes dim {total}, but ONNX metadata expects {prop_dim}: {out}")
    return out


def print_obs_layout(meta: dict, prop_history: TermMajorHistory, input_names: list[str]) -> None:
    shapes = meta.get("observation_shapes", {})
    print("[INFO] ONNX inputs:", input_names)
    print(f"[INFO] prop dim: {prop_history.dim}, history max length: {prop_history.max_history_length}")
    for name, dim, history_len in prop_history.term_dims:
        print(f"[INFO]   prop/{name}: base_dim={dim}, history={history_len}, flat_dim={dim * history_len}")
    for name in input_names:
        print(f"[INFO]   input/{name}: expected_shape={tuple(shapes.get(name, []))}")


def print_imu_debug(data, imu_reader: ImuReader) -> None:
    sensor_quat = np.asarray(data.sensor(imu_reader.quat_sensor_name).data, dtype=np.float64)
    sensor_gyro = np.asarray(data.sensor(imu_reader.gyro_sensor_name).data, dtype=np.float64)
    sensor_gravity_b = quat_apply_inverse(sensor_quat[None, :], np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64))[0]
    print("========== MUJOCO IMU DEBUG ==========")
    print("[IMUDBG] active source: MuJoCo XML sensors")
    print(f"[IMUDBG] sensor {imu_reader.quat_sensor_name} quat wxyz: {sensor_quat.tolist()}")
    print(f"[IMUDBG] sensor {imu_reader.gyro_sensor_name} gyro: {sensor_gyro.tolist()}")
    print(f"[IMUDBG] sensor projected gravity: {sensor_gravity_b.tolist()}")
    print("======================================")


def build_obs(
    data,
    motion: MotionData,
    t: int,
    meta: dict,
    imu_reader: ImuReader,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    prop_history: TermMajorHistory,
    body_ids: np.ndarray | None = None,
    terrain_scanner=None,
    velcommand_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    root_quat, root_ang_vel_b, gravity_b = imu_reader.read(data)
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_qpos), 0.0)
    default_joint_vel = as_vector(meta, "default_joint_vel", len(joint_qvel), 0.0)
    qpos = data.qpos[joint_qpos][None, :]
    qvel = data.qvel[joint_qvel][None, :]
    prop_terms = {
        "projected_gravity": gravity_b,
        "base_ang_vel": root_ang_vel_b,
        "joint_pos": qpos - default_joint_pos[None, :],
        "joint_vel": qvel - default_joint_vel[None, :],
        "actions": last_action,
    }

    prop = prop_history.update(prop_terms)
    rbt_cmd_mf, smpl_cmd_mf = motion_groups(motion, t, meta, root_quat)
    obs = {
        "prop": prop.astype(np.float32),
        "rbt_cmd_mf": rbt_cmd_mf,
        "smpl_cmd_mf": smpl_cmd_mf,
    }
    shapes = meta.get("observation_shapes", {})
    if "terrain" in shapes:
        terrain_dim = shape_dim(shapes["terrain"])
        if terrain_scanner is None:
            obs["terrain"] = np.zeros((1, terrain_dim), dtype=np.float32)
        else:
            terrain = terrain_scanner.scan(data)
            if terrain.shape[1] != terrain_dim:
                raise ValueError(f"Height scan dim {terrain.shape[1]} does not match ONNX terrain dim {terrain_dim}.")
            obs["terrain"] = terrain
    if "task" in shapes:
        obs["task"] = build_task_obs(data, motion, t, meta, body_ids)
    if "velcommand" in shapes:
        velcommand_dim = shape_dim(shapes["velcommand"])
        if velcommand_override is None:
            velcommand = build_velcommand_obs(motion, t, meta)
        else:
            velcommand = np.asarray(velcommand_override, dtype=np.float32).reshape(1, -1)
        if velcommand.shape[1] != velcommand_dim:
            raise ValueError(f"velcommand dim {velcommand.shape[1]} does not match ONNX dim {velcommand_dim}.")
        obs["velcommand"] = velcommand
    if "wbc_cmd" in shapes:
        obs["wbc_cmd"] = build_wbc_cmd_obs(data, motion, t, meta, body_ids)
    if "vel_task_mask" in shapes:
        vel_task_mask_dim = shape_dim(shapes["vel_task_mask"])
        vel_task_value = 0.0 if velcommand_override is None else 1.0
        obs["vel_task_mask"] = np.full((1, vel_task_mask_dim), vel_task_value, dtype=np.float32)
    if "aux_mask" in shapes:
        aux_mask_dim = shape_dim(shapes["aux_mask"])
        obs["aux_mask"] = np.ones((1, aux_mask_dim), dtype=np.float32)
    return obs


def build_velcommand_obs(motion: MotionData, t: int, meta: dict) -> np.ndarray:
    _, anchor_idx = motion_body_index(meta, meta["anchor_body_name"])
    frame = int(np.clip(t, 0, motion.num_frames - 1))
    quat_w = np.asarray(motion["body_quat_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 4)
    lin_vel_w = (
        np.asarray(motion["body_lin_vel_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
        if "body_lin_vel_w" in motion
        else np.zeros((1, 3), dtype=np.float64)
    )
    ang_vel_w = (
        np.asarray(motion["body_ang_vel_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
        if "body_ang_vel_w" in motion
        else np.zeros((1, 3), dtype=np.float64)
    )
    yaw_quat = _yaw_quat(quat_w)
    lin_vel_yaw_b = quat_apply_inverse(yaw_quat, lin_vel_w)
    ang_vel_yaw_b = quat_apply_inverse(yaw_quat, ang_vel_w)
    return np.concatenate((lin_vel_yaw_b[:, :2], ang_vel_yaw_b[:, 2:3]), axis=-1).astype(np.float32)


def build_wbc_cmd_obs(
    data,
    motion: MotionData,
    t: int,
    meta: dict,
    body_ids: np.ndarray | None,
) -> np.ndarray:
    terms = meta.get("observation_terms", {}).get("wbc_cmd", {}).get("terms", [])
    builders = {
        "command": lambda: motion_generated_command(motion, t, meta),
        "generated_commands": lambda: motion_generated_command(motion, t, meta),
        "motion_anchor_pos_b": lambda: motion_anchor_pos_b(data, motion, t, meta, body_ids),
        "motion_anchor_ori_b": lambda: motion_anchor_ori_b(data, motion, t, meta, body_ids),
    }
    if not terms:
        terms = [{"name": "command"}, {"name": "motion_anchor_pos_b"}, {"name": "motion_anchor_ori_b"}]

    pieces = []
    for term in terms:
        name = term.get("name")
        if name not in builders:
            raise KeyError(f"Unsupported wbc_cmd observation term in ONNX metadata: {name!r}")
        pieces.append(builders[name]())
    wbc_cmd = np.concatenate(pieces, axis=-1).astype(np.float32)

    expected_dim = shape_dim(meta.get("observation_shapes", {}).get("wbc_cmd", []))
    if expected_dim > 0 and wbc_cmd.shape[1] != expected_dim:
        raise ValueError(f"wbc_cmd dim {wbc_cmd.shape[1]} does not match ONNX dim {expected_dim}.")
    return wbc_cmd


def motion_generated_command(motion: MotionData, t: int, meta: dict) -> np.ndarray:
    _, anchor_idx = motion_body_index(meta, meta["anchor_body_name"])
    frame = int(np.clip(t, 0, motion.num_frames - 1))
    anchor_quat_w = np.asarray(motion["body_quat_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 4)
    anchor_lin_vel_w = (
        np.asarray(motion["body_lin_vel_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
        if "body_lin_vel_w" in motion
        else np.zeros((1, 3), dtype=np.float64)
    )
    anchor_ang_vel_w = (
        np.asarray(motion["body_ang_vel_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
        if "body_ang_vel_w" in motion
        else np.zeros((1, 3), dtype=np.float64)
    )
    anchor_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
    anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    anchor_project_gravity = quat_apply_inverse(anchor_quat_w, np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64))
    anchor_pos_z = np.asarray(motion["body_pos_w"][frame, anchor_idx, 2], dtype=np.float64).reshape(1, 1)
    return np.concatenate(
        (
            np.asarray(motion["joint_pos"][frame], dtype=np.float64).reshape(1, -1),
            np.asarray(motion["joint_vel"][frame], dtype=np.float64).reshape(1, -1),
            anchor_lin_vel_b,
            anchor_ang_vel_b,
            anchor_project_gravity,
            anchor_pos_z,
        ),
        axis=-1,
    ).astype(np.float32)


def motion_anchor_ori_b(data, motion: MotionData, t: int, meta: dict, body_ids: np.ndarray | None) -> np.ndarray:
    _, anchor_idx = motion_body_index(meta, meta["anchor_body_name"])
    frame = int(np.clip(t, 0, motion.num_frames - 1))
    target_quat_w = np.asarray(motion["body_quat_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 4)
    robot_anchor_quat_w = robot_anchor_pose_w(data, meta, body_ids)[1]
    rel_quat = quat_mul(quat_inv(robot_anchor_quat_w), target_quat_w)
    return matrix_from_quat(rel_quat)[..., :2].reshape(1, -1).astype(np.float32)


def robot_anchor_pose_w(data, meta: dict, body_ids: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    anchor_local_idx = int(list(meta["motion_body_names"]).index(meta["anchor_body_name"]))
    if body_ids is None:
        raise ValueError("body_ids is required to build wbc_cmd anchor observations.")
    body_id = int(np.asarray(body_ids, dtype=np.int64)[anchor_local_idx])
    pos_w = np.asarray(data.xpos[body_id], dtype=np.float64).reshape(1, 3)
    quat_w = np.asarray(data.xquat[body_id], dtype=np.float64).reshape(1, 4)
    return pos_w, quat_w


def _yaw_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat_wxyz, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return np.stack((np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)), axis=-1)


def build_task_obs(data, motion: MotionData, t: int, meta: dict, body_ids: np.ndarray | None) -> np.ndarray:
    terms = meta.get("observation_terms", {}).get("task", {}).get("terms", [])
    if not terms:
        return np.zeros((1, shape_dim(meta.get("observation_shapes", {}).get("task", []))), dtype=np.float32)
    pieces = []
    for term in terms:
        name = term.get("name")
        if name == "motion_anchor_pos_b":
            pieces.append(motion_anchor_pos_b(data, motion, t, meta, body_ids))
        else:
            raise KeyError(f"Unsupported task observation term in ONNX metadata: {name!r}")
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def motion_anchor_pos_b(data, motion: MotionData, t: int, meta: dict, body_ids: np.ndarray | None) -> np.ndarray:
    _, anchor_idx = motion_body_index(meta, meta["anchor_body_name"])
    frame = int(np.clip(t, 0, motion.num_frames - 1))
    target_pos_w = np.asarray(motion["body_pos_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
    robot_anchor_pos_w, robot_anchor_quat_w = robot_anchor_pose_w(data, meta, body_ids)
    return quat_apply_inverse(robot_anchor_quat_w, target_pos_w - robot_anchor_pos_w).astype(np.float32)


def validate_inputs(obs: dict[str, np.ndarray], input_names: list[str], meta: dict) -> None:
    shapes = meta.get("observation_shapes", {})
    missing = [name for name in input_names if name not in obs]
    if missing:
        raise KeyError(f"ONNX expects unsupported input groups: {missing}")
    for name in input_names:
        expected = tuple(shapes.get(name, obs[name].shape[1:]))
        got = tuple(obs[name].shape[1:])
        if got != expected:
            raise ValueError(f"Observation group '{name}' has shape {got}, expected {expected}.")
