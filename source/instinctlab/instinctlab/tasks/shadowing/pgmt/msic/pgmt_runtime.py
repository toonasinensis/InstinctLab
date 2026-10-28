"""PGMT observation/control contract in MuJoCo (no Isaac Sim imports)."""

from __future__ import annotations

import numpy as np
import re
from pathlib import Path

import mujoco

from .math_utils import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul
from .motion import MotionData
from .mujoco_robot import name_to_actuator_ids, name_to_body_ids, name_to_joint_qvel_addrs
from .observations import ImuReader, TermMajorHistory, fallback_prop_terms
from .pgmt_terrain import TerrainScanner, yaw_rotation

REFERENCE_DISPLAYS = {"terrain": "地形适配参考", "source": "原始 NPZ（不抬高）"}

# Isaac articulation order, verified against the exported G1 action_joint_names.
# MuJoCo's qpos tree order is different: always gather/scatter by joint name.
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def resolve(value, name, default=None):
    if not isinstance(value, dict):
        return float(value)
    matches = [v for k, v in value.items() if re.fullmatch(k, name)]
    if len(matches) > 1 or (not matches and default is None):
        raise ValueError(f"Ambiguous/missing joint setting for {name}: {value}")
    return float(matches[0] if matches else default)


def rotate(q, v):
    return quat_apply_inverse(quat_inv(q), v)


def yaw_quat(q):
    rotation = yaw_rotation(q)
    angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def validate_reset_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("RESET 时间必须是非负秒数，0 表示关闭定时重置") from None
    if isinstance(value, bool) or not np.isfinite(seconds) or seconds < 0:
        raise ValueError("RESET 时间必须是有限的非负秒数，0 表示关闭定时重置")
    return seconds


class PGMTMuJoCo:
    def __init__(
        self,
        model,
        cfg,
        motion_file: Path,
        contact_dir: Path,
        terrain_info,
        start_frame=300,
        yaw=0.0,
        start_xy=(0.0, 0.0),
        spawn_clearance=0.075,
        absolute_heading=False,
        surface_placement=False,
        reset_seconds=None,
    ):
        self.model, self.cfg = model, cfg
        self.reset_seconds = validate_reset_seconds(cfg["episode_length_s"] if reset_seconds is None else reset_seconds)
        self.data = mujoco.MjData(model)
        self.command_cfg = cfg["commands"]["motion"]
        self.body_names = self.command_cfg["body_names"]
        self.body_ids = name_to_body_ids(model, self.body_names)
        self.pelvis = self.body_names.index("pelvis")
        self.pelvis_id = int(self.body_ids[self.pelvis])
        self.feet = [self.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        self.lower = [i for i, name in enumerate(self.body_names) if any(k in name for k in ("hip", "knee", "ankle"))]
        self.qpos_ids, self.qvel_ids = name_to_joint_qvel_addrs(model, JOINT_NAMES)
        self.actuator_ids = name_to_actuator_ids(model, JOINT_NAMES)
        motion = MotionData(str(motion_file))
        if motion.joint_names is None or motion.body_names is None:
            raise ValueError("Named NPZ with explicit joint/body mapping is required")
        self.motion, self.mapping = motion.aligned_to(joint_names=JOINT_NAMES, body_names=self.body_names)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        self.decimation = int(cfg["decimation"])
        self.dt = float(cfg["sim"]["dt"]) * self.decimation
        if not np.isclose(fps, 1 / self.dt):
            raise ValueError(f"Motion is {fps} Hz; resample to {1/self.dt} Hz before playback.")
        self.future_offsets = np.asarray(self.command_cfg["future_step_num"])
        assert np.array_equal(self.future_offsets, np.arange(11))
        self.start_frame = int(start_frame)
        if not 0 <= self.start_frame < self.motion.num_frames - 11:
            raise ValueError("Start frame is outside motion/future horizon")
        self.start_xy = np.asarray(start_xy, dtype=float)
        if self.start_xy.shape != (2,) or not np.isfinite(self.start_xy).all() or not np.isfinite(yaw):
            raise ValueError("Placement coordinates and heading must be finite")
        if absolute_heading:
            rotation = yaw_rotation(self.motion["body_quat_w"][start_frame, self.pelvis])
            yaw -= np.arctan2(rotation[1, 0], rotation[0, 0])
        self.placement_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.placement_xy = (
            self.start_xy - rotate(self.placement_quat, self.motion["body_pos_w"][start_frame, self.pelvis])[:2]
        )
        self.surface_placement = surface_placement
        self.spawn_clearance = spawn_clearance
        self.terrain_info = terrain_info
        self.scanner = TerrainScanner(model)
        self.imu = ImuReader(model)
        self.init_joints = np.array(
            [resolve(cfg["scene"]["robot"]["init_state"]["joint_pos"], name, 0) for name in JOINT_NAMES]
        )
        self.action_scale = np.array([resolve(cfg["actions"]["joint_pos"]["scale"], name) for name in JOINT_NAMES])
        self.kp, self.kd, self.limits, self.velocity_limits = (np.zeros(29) for _ in range(4))
        for j, name in enumerate(JOINT_NAMES):
            actuators = [
                a
                for a in cfg["scene"]["robot"]["actuators"].values()
                if any(re.fullmatch(k, name) for k in a["joint_names_expr"])
            ]
            if len(actuators) != 1:
                raise ValueError(f"Joint {name} matched {len(actuators)} actuators")
            a = actuators[0]
            self.kp[j] = resolve(a["stiffness"], name)
            self.kd[j] = resolve(a["damping"], name)
            self.limits[j] = resolve(a["effort_limit_sim"], name)
            self.velocity_limits[j] = resolve(a["velocity_limit_sim"], name)
            model.dof_armature[self.qvel_ids[j]] = resolve(a["armature"], name)
        self._load_contacts(contact_dir, motion_file)
        self.episodes = 0
        self.reset()

    def _load_contacts(self, contact_dir, motion_file):
        path = contact_dir / (motion_file.name + ".contacts.npz")
        with np.load(path, allow_pickle=False) as labels:
            ids = [labels["body_names"].tolist().index(n) for n in self.body_names]
            self.contacts = labels["contact"][:, ids] & labels["valid"][:, ids]
            self.bottoms = labels["bottom"][:, ids]
            if len(self.contacts) != self.motion.num_frames:
                raise ValueError("Contact labels need motion-rate resampling")

    @property
    def root_pos(self):
        return self.data.xpos[self.pelvis_id].copy()

    @property
    def root_quat(self):
        return self.data.xquat[self.pelvis_id].copy()

    def reference_pose(self, frames=None):
        frames = self.frame if frames is None else frames
        pos = rotate(self.placement_quat, self.motion["body_pos_w"][frames]).copy()
        pos[..., :2] += self.placement_xy
        pos[..., 2] += self.support_z
        quat = quat_mul(self.placement_quat, self.motion["body_quat_w"][frames])
        return pos, quat

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.frame = self.start_frame
        self.steps = 0
        self.distance_xy = 0.0
        self.off_platform_steps = 0
        self.support_z = 0.0
        self.last_action = np.zeros((1, 29), dtype=np.float32)
        self.history = TermMajorHistory(fallback_prop_terms(930, 29))
        # Raise the complete reference pose above the actual terrain mesh.
        mujoco.mj_forward(self.model, self.data)
        pos, _ = self.reference_pose()
        heights = self.scanner.heights(self.data, pos[:, :2])
        bottoms = self.bottoms[self.frame]
        root_ground = float(self.scanner.heights(self.data, self.start_xy[None])[0])
        if self.surface_placement:
            # A high step underneath an already-high arm/torso must not raise the
            # feet to that step. Compare terrain against each actual body bottom.
            # Keep the chosen motion's flight clearance when selecting a jump frame.
            self.support_z = max(root_ground, float(np.max((heights - bottoms)[np.isfinite(bottoms)])))
        else:
            self.support_z = float(np.max((heights - np.minimum(bottoms, 0))[np.isfinite(bottoms)]))
        pos, quat = self.reference_pose()
        root_q = quat[self.pelvis]
        self.data.qpos[:3] = pos[self.pelvis] + [0, 0, self.spawn_clearance]
        self.data.qpos[3:7] = root_q
        # Training uses 0.9 soft joint limits on initialization.
        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        bounds = self.model.jnt_range[joint_ids]
        mid, half = bounds.mean(-1), (bounds[:, 1] - bounds[:, 0]) * 0.45
        self.data.qpos[self.qpos_ids] = np.clip(self.motion["joint_pos"][self.frame], mid - half, mid + half)
        self.data.qvel[self.qvel_ids] = np.clip(
            self.motion["joint_vel"][self.frame], -self.velocity_limits, self.velocity_limits
        )
        self.data.qvel[:3] = rotate(self.placement_quat, self.motion["body_lin_vel_w"][self.frame, self.pelvis])
        # MuJoCo free-joint angular qvel is local; NPZ angular velocities are world-frame.
        world_omega = rotate(self.placement_quat, self.motion["body_ang_vel_w"][self.frame, self.pelvis])
        self.data.qvel[3:6] = quat_apply_inverse(root_q, world_omega)
        mujoco.mj_forward(self.model, self.data)
        lift = 0.0
        if self.surface_placement:
            # Resolve any residual MuJoCo collision-geometry penetration before
            # committing a placement, without advancing time or the reference.
            for _ in range(60):
                distances = [
                    c.dist
                    for c in self.data.contact
                    if ((self.model.geom_bodyid[c.geom1] == 0) != (self.model.geom_bodyid[c.geom2] == 0))
                ]
                penetration = max(0.0, -min(distances, default=0.0))
                if penetration <= 1e-6:
                    break
                rise = penetration + 0.002
                lift += rise
                self.data.qpos[2] += rise
                mujoco.mj_forward(self.model, self.data)
            else:
                raise ValueError("所选位置无法消除身体与地形穿插，请稍微移动放置点")
        self.update_support(reset=True, reference_feet=pos[self.feet])
        self.initial_root_xy = self.root_pos[:2]
        heading = yaw_rotation(self.root_quat)
        self.spawn_info = dict(
            x=float(self.initial_root_xy[0]),
            y=float(self.initial_root_xy[1]),
            heading_deg=float(np.rad2deg(np.arctan2(heading[1, 0], heading[0, 0]))),
            robot_z_m=float(self.root_pos[2]),
            ground_z_m=root_ground,
            clearance_m=self.spawn_clearance,
            collision_lift_m=lift,
            placement_method="surface" if self.surface_placement else "training",
        )
        self.episodes += 1

    def update_support(self, reset=False, reference_feet=None):
        feet = self.data.xpos[self.body_ids[self.feet]] if reference_feet is None else reference_feet
        ground = self.scanner.heights(self.data, feet[:, :2])
        contact = self.contacts[self.frame, self.feet]
        target = (
            float(ground[contact].mean())
            if contact.any()
            else float(self.scanner.heights(self.data, self.root_pos[None, :2])[0])
        )
        blend = 1.0 if reset else 1 - np.exp(-self.dt / self.command_cfg["support_filter_seconds"])
        self.support_z += blend * (target - self.support_z)

    def observations(self):
        quat, omega, gravity = self.imu.read(self.data)
        prop = self.history.update(
            dict(
                projected_gravity=gravity,
                base_ang_vel=omega,
                joint_pos=self.data.qpos[self.qpos_ids] - self.init_joints,
                joint_vel=self.data.qvel[self.qvel_ids],
                actions=self.last_action,
            )
        )
        future = np.minimum(self.frame + self.future_offsets, self.motion.num_frames - 1)
        pos, ref_quat = self.reference_pose(future)
        orientation = matrix_from_quat(quat_mul(quat_inv(quat[0]), ref_quat[:, self.pelvis]))[:, :, :2]
        ref_yaw = yaw_quat(ref_quat[0, self.pelvis])
        vel = rotate(self.placement_quat, self.motion["body_lin_vel_w"][future, self.pelvis])
        vel = quat_apply_inverse(ref_yaw, vel)
        error = quat_apply_inverse(ref_yaw, pos[0, self.pelvis] - self.root_pos)
        gate = np.clip(
            np.linalg.norm(vel[:, :2], axis=-1, keepdims=True) / self.command_cfg["velocity_gate_speed"], 0, 1
        )
        corrected = np.zeros_like(vel)
        corrected[:, :2] = vel[:, :2] + np.clip(
            gate * self.command_cfg["velocity_gain"] * error[:2],
            -self.command_cfg["velocity_cap"],
            self.command_cfg["velocity_cap"],
        )
        reference = np.concatenate(
            [
                self.motion["joint_pos"][future].reshape(-1),
                self.motion["joint_vel"][future].reshape(-1),
                orientation.reshape(-1),
                (pos[:, self.pelvis, 2] - self.support_z).reshape(-1),
                corrected.reshape(-1),
            ]
        )
        terrain, self.scan_points = self.scanner.scan(self.data, self.root_pos, self.root_quat)
        self.flat_height = float(self.scanner.heights(self.data, self.root_pos[None, :2])[0] - self.root_pos[2])
        return dict(prop=prop, rbt_cmd_mf=reference[None].astype(np.float32), terrain=terrain)

    def step(self, action):
        action = np.asarray(action).reshape(29)
        previous_xy = self.root_pos[:2]
        target = action * self.action_scale + self.init_joints
        maximum_torque_fraction = 0.0
        for _ in range(self.decimation):
            tau = self.kp * (target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
            maximum_torque_fraction = max(maximum_torque_fraction, float(np.max(np.abs(tau) / self.limits)))
            self.data.ctrl[self.actuator_ids] = np.clip(tau, -self.limits, self.limits)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite MuJoCo state")
        self.last_action[:] = action
        self.frame = min(self.frame + 1, self.motion.num_frames - 1)
        self.steps += 1
        self.distance_xy += float(np.linalg.norm(self.root_pos[:2] - previous_xy))
        self.off_platform_steps += int(np.max(np.abs(self.root_pos[:2])) > 1.1)
        self.update_support()
        return self.metrics(maximum_torque_fraction)

    def aligned_reference(self):
        pos, quat = self.reference_pose()
        anchor = self.root_pos
        anchor[2] = pos[self.pelvis, 2]
        delta = yaw_quat(quat_mul(self.root_quat, quat_inv(quat[self.pelvis])))
        return anchor + rotate(delta, pos - pos[self.pelvis]), quat_mul(delta, quat)

    def display_reference(self, mode="terrain"):
        """Ghost only: retain XY/yaw alignment, optionally restore NPZ world Z.

        Never use this display choice for policy observations or tracking metrics.
        Neither mode includes the physical robot's spawn clearance.
        """
        if mode not in REFERENCE_DISPLAYS:
            raise ValueError(f"Unknown reference display: {mode}")
        pos, quat = self.aligned_reference()
        if mode == "source":
            pos[..., 2] -= self.support_z
        return pos, quat

    def metrics(self, torque_fraction=0.0):
        pos, quat = self.reference_pose()
        actual_gravity = quat_apply_inverse(self.root_quat, np.array([0.0, 0.0, -1.0]))
        ref_gravity = quat_apply_inverse(quat[self.pelvis], np.array([0.0, 0.0, -1.0]))
        tilt = float(np.arccos(np.clip(np.dot(actual_gravity, ref_gravity), -1, 1)))
        height = float(abs(pos[self.pelvis, 2] - self.root_pos[2]))
        aligned, _ = self.aligned_reference()
        body_error = np.linalg.norm(aligned - self.data.xpos[self.body_ids], axis=-1)
        tolerance = self.command_cfg["tolerance_position_m"] * self.terrain_info["level"] / 9
        if self.terrain_info["kind"] != "flat":
            body_error[self.lower] = np.maximum(0, body_error[self.lower] - tolerance)
        failures = []
        if height > 0.4:
            failures.append("root_height")
        if tilt > 1.0:
            failures.append("root_tilt")
        if body_error.max() > 0.65:
            failures.append("body_error")
        timeouts = []
        if self.reset_seconds > 0 and self.steps * self.dt + 1e-12 >= self.reset_seconds:
            timeouts.append("time_out")
        if self.frame >= self.motion.num_frames - 11:
            timeouts.append("motion_end")
        if np.max(np.abs(self.root_pos[:2])) >= 4:
            timeouts.append("tile_exit")
        return dict(
            frame=self.frame,
            episode=self.episodes,
            episode_time_s=self.steps * self.dt,
            joint_rmse_rad=float(
                np.sqrt(np.mean((self.data.qpos[self.qpos_ids] - self.motion["joint_pos"][self.frame]) ** 2))
            ),
            root_height_error_m=height,
            root_tilt_error_rad=tilt,
            body_error_max_m=float(body_error.max()),
            root_z_m=float(self.root_pos[2]),
            support_z_m=self.support_z,
            root_x_m=float(self.root_pos[0]),
            root_y_m=float(self.root_pos[1]),
            traveled_xy_m=self.distance_xy,
            off_platform_s=self.off_platform_steps * self.dt,
            displacement_xy_m=float(np.linalg.norm(self.root_pos[:2] - self.initial_root_xy)),
            reference_xy_error_m=float(np.linalg.norm(pos[self.pelvis, :2] - self.root_pos[:2])),
            raw_torque_limit_ratio=torque_fraction,
            failures=failures,
            timeouts=timeouts,
        )

    def contract(self):
        return dict(
            joint_names=JOINT_NAMES,
            body_names=self.body_names,
            kp=self.kp.tolist(),
            kd=self.kd.tolist(),
            action_scale=self.action_scale.tolist(),
            default_joint_pos=self.init_joints.tolist(),
            torque_limits=self.limits.tolist(),
            sim_dt=self.model.opt.timestep,
            control_dt=self.dt,
            observation_dims=dict(prop=930, rbt_cmd_mf=748, terrain=441),
            obs_noise=False,
            domain_randomization=False,
            recovery=False,
            curriculum=False,
            terrain_level="fixed for repeatable sim2sim",
            pd="explicit clipped torque at each physics step",
            reset_seconds=self.reset_seconds,
            reset_clock="simulation episode time; zero disables time_out only",
            reference="episode placement + floating support + corrected reference-yaw velocity",
            reset="same selected frame and placement on each episode; nominal velocities; 0.075 m clearance by default",
            limitations=[
                "MuJoCo contact solver/collision shapes and explicit PD differ from Isaac PhysX implicit drives",
                "MuJoCo has no Isaac articulation joint-velocity solver limit",
                "Single fixed terrain tile, no curriculum updates or adaptive motion sampling",
            ],
        )
