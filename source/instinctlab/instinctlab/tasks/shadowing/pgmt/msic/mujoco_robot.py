from __future__ import annotations

import numpy as np
from pathlib import Path

from .math_utils import as_vector

G1_MJCF = Path(__file__).with_name("assets") / "g1.xml"


def name_to_joint_qvel_addrs(model, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    qpos_addrs = []
    qvel_addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' from ONNX metadata not found in MuJoCo model.")
        qpos_addrs.append(model.jnt_qposadr[jid])
        qvel_addrs.append(model.jnt_dofadr[jid])
    return np.asarray(qpos_addrs, dtype=np.int32), np.asarray(qvel_addrs, dtype=np.int32)


def name_to_joint_ids(model, joint_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' from ONNX metadata not found in MuJoCo model.")
        ids.append(jid)
    return np.asarray(ids, dtype=np.int32)


def name_to_body_ids(model, body_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' from ONNX metadata not found in MuJoCo model.")
        ids.append(bid)
    return np.asarray(ids, dtype=np.int32)


def validate_sensor(model, sensor_name: str, expected_dim: int | None = None) -> None:
    import mujoco

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise ValueError(f"Sensor '{sensor_name}' not found in MuJoCo model.")
    dim = int(model.sensor_dim[sid])
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"Sensor '{sensor_name}' has dim {dim}, expected {expected_dim}.")


def name_to_actuator_ids(model, joint_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in joint_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(
                f"Actuator '{name}' from ONNX action order not found in MuJoCo model. "
                "Use actuator names that match action_joint_names."
            )
        ids.append(aid)
    return np.asarray(ids, dtype=np.int32)


def print_joint_map(
    joint_names: list[str],
    joint_ids: np.ndarray,
    actuator_ids: np.ndarray,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    action_offset: np.ndarray,
    torque_limits: np.ndarray,
) -> None:
    print("[INFO] Joint map: ONNX/action order -> MuJoCo joint id/qpos/qvel")
    for idx, name in enumerate(joint_names):
        print(
            f"[INFO]   action[{idx:02d}] {name}: "
            f"mj_joint_id={int(joint_ids[idx])}, "
            f"actuator_id={int(actuator_ids[idx])}, "
            f"qpos_addr={int(joint_qpos[idx])}, "
            f"qvel_addr={int(joint_qvel[idx])}, "
            f"default={float(default_joint_pos[idx]): .6f}, "
            f"scale={float(action_scale[idx]): .6f}, "
            f"offset={float(action_offset[idx]): .6f}, "
            f"tau_limit=({float(torque_limits[idx, 0]): .3f}, {float(torque_limits[idx, 1]): .3f})"
        )


def initialize_from_motion(
    data,
    motion,
    meta: dict,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    root_body_name: str | None = None,
    frame: int = 0,
    set_root_velocity: bool = True,
) -> str:
    from .motion import motion_frame_root_state

    root_body, _, root_pos, root_quat, root_lin_vel, root_ang_vel = motion_frame_root_state(
        motion, meta, frame, root_body_name
    )
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[joint_qpos] = np.asarray(motion["joint_pos"][frame], dtype=np.float64)
    data.qvel[:] = 0.0
    if set_root_velocity:
        data.qvel[:3] = root_lin_vel
        data.qvel[3:6] = root_ang_vel
    if "joint_vel" in motion:
        data.qvel[joint_qvel] = np.asarray(motion["joint_vel"][frame], dtype=np.float64)
    return root_body


def initialize_default_pose(data, meta: dict, joint_names: list[str], joint_qpos: np.ndarray) -> None:
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_names), 0.0)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = 0.793
    data.qpos[3] = 1.0
    data.qpos[joint_qpos] = default_joint_pos


def action_to_target(raw_action: np.ndarray, action_scale: np.ndarray, action_offset: np.ndarray) -> np.ndarray:
    return raw_action[0].astype(np.float64) * action_scale + action_offset


def gains_from_metadata(
    meta: dict, size: int, kp_override: float | None, kd_override: float | None
) -> tuple[np.ndarray, np.ndarray]:
    kp = as_vector(meta, "joint_stiffness", size, 60.0)
    kd = as_vector(meta, "joint_damping", size, 2.0)
    if kp_override is not None:
        kp = np.full(size, float(kp_override), dtype=np.float64)
    if kd_override is not None:
        kd = np.full(size, float(kd_override), dtype=np.float64)
    return kp, kd


def apply_pd_control(
    data,
    actuator_ids: np.ndarray,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    target: np.ndarray,
    kp: float | np.ndarray,
    kd: float | np.ndarray,
    torque_limits: np.ndarray,
) -> None:
    q = data.qpos[joint_qpos]
    qd = data.qvel[joint_qvel]
    tau = kp * (target - q) - kd * qd
    data.ctrl[actuator_ids] = np.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
