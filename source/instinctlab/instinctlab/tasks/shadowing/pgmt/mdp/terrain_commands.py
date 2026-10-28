"""将原始 LAFAN 参考动作放置到非配对地形上，并按支撑高度整体调整参考位置；恢复功能保持关闭。"""

import numpy as np
import torch
from pathlib import Path

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul, yaw_quat

from .terrain_contacts import TerrainContacts
from .terrain_context import TerrainContext
from .terrain_math import corrected_planar_velocity


class TerrainMotion(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self.robot = env.scene["robot"]
        self.reference = env.scene["motion_reference"]
        self.motion = self.reference.motion_buffers["LAFAN"]
        # 身体状态、接触标签及身体分组均按 link_of_interests 的顺序索引。
        self.body_ids = self.robot.find_bodies(self.reference.cfg.link_of_interests, preserve_order=True)[0]
        names = self.reference.cfg.link_of_interests
        self.foot_ids = [names.index(s + "_ankle_roll_link") for s in ("left", "right")]
        self.hand_ids = [names.index(s + "_wrist_yaw_link") for s in ("left", "right")]
        self.impact_ids = [names.index(n) for n in ("pelvis", "torso_link")]
        self.motion_anchor_body_index = names.index("pelvis")
        self.lower_ids = [i for i, n in enumerate(names) if any(s in n for s in ("hip", "knee", "ankle"))]
        self.upper_ids = [
            i for i, n in enumerate(names) if any(s in n for s in ("torso", "shoulder", "elbow", "wrist"))
        ]
        self.lower_joint_ids = [
            i for i, n in enumerate(self.robot.joint_names) if any(s in n for s in ("hip", "knee", "ankle"))
        ]
        self.upper_joint_ids = [i for i in range(self.robot.num_joints) if i not in self.lower_joint_ids]
        # 每个环境保存独立的参考动作放置变换：出生时确定 XY 平移和 yaw，运行时动态更新支撑高度。
        self.offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.rotation = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotation[:, 3] = 1
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 按「5 类地形 × 10 级难度 × 40 段动作」统计失败，用于后续动作采样。
        self.stratum_failure = torch.zeros(5, 10, 40, device=self.device)
        self.stratum_visits = torch.zeros_like(self.stratum_failure)
        self._load_contacts()
        self.terrain = TerrainContext(self)
        self.contacts = TerrainContacts(self)
        for name in ("error_body_pos", "error_joint_pos", "terrain_level", "terrain_gate"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _load_contacts(self):
        """按动作文件名加载离线接触标签，并统一身体顺序。"""
        directory = Path(self.cfg.contact_labels_dir)
        contacts, validity, bottom = [], [], []
        for path in self.motion._all_motion_files:
            label = directory / (Path(path).name + ".contacts.npz")
            with np.load(label, allow_pickle=False) as data:
                # 统一身体顺序；bottom 是各部位碰撞几何在源动作平地坐标中的最低高度。
                order = [data["body_names"].tolist().index(n) for n in self.reference.cfg.link_of_interests]
                for target, key in ((contacts, "contact"), (validity, "valid"), (bottom, "bottom")):
                    target.append(torch.as_tensor(data[key][:, order], device=self.device))
        # 所有动作沿帧维拼接，通过各段起始索引定位当前帧的标签。
        self.labels = torch.cat(contacts)
        self.label_valid = torch.cat(validity)
        self.label_bottom = torch.cat(bottom)
        lengths = self.motion._all_motion_sequences.buffer_length.long().to(self.device)
        self.label_starts = lengths.cumsum(0) - lengths
        self.label_lengths = lengths

    @property
    def frame_ids(self):
        # 源动作按 50 Hz 取帧：片段起始时间 + 当前播放时间，再映射到拼接后的标签索引。
        sample = ((self.motion._motion_buffer_start_time_s + self.reference._timestamp_torch) * 50).round().long()
        return self.label_starts[self.motion_ids] + torch.minimum(sample, self.label_lengths[self.motion_ids] - 1)

    @property
    def reference_contacts(self):
        return self.labels[self.frame_ids]

    @property
    def reference_contact_valid(self):
        return self.label_valid[self.frame_ids]

    @property
    def command(self):
        return self.corrected_velocity_future_b.flatten(1)

    def _resample_command(self, env_ids):
        pass  # 由 reset 事件在观测历史清空前完成机器人放置。

    def _update_command(self):
        pass  # 由 advance() 在终止判断和奖励计算前统一更新，不在这里隐式重置动作。

    def _update_metrics(self):
        self.metrics["error_body_pos"][:] = (self.body_pos_relative_w - self.robot_body_pos_w).norm(dim=-1).mean(-1)
        self.metrics["error_joint_pos"][:] = (self.joint_pos - self.robot_joint_pos).abs().mean(-1)
        self.metrics["terrain_level"][:] = self.terrain.level

    def _position(self, raw):
        """将当前或未来的源参考位置变换到地形世界坐标，兼容身体维和时间维。"""
        q = self.rotation.reshape(self.num_envs, *([1] * (raw.ndim - 2)), 4).expand(*raw.shape[:-1], 4)
        offset = self.offset.clone()
        # 整套参考动作共用一个动态支撑高度；不逐脚修正位置，也不修改参考关节姿态。
        offset[:, 2] = self.terrain.support_z
        # 先去除源参考中的环境原点，再施加出生 yaw，最后加上 XY 放置偏移和支撑高度。
        return quat_apply(
            q, raw - self.env.scene.env_origins.reshape(self.num_envs, *([1] * (raw.ndim - 2)), 3)
        ) + offset.reshape(self.num_envs, *([1] * (raw.ndim - 2)), 3)

    def _rotate(self, raw):
        q = self.rotation.reshape(self.num_envs, *([1] * (raw.ndim - 2)), 4).expand_as(raw)
        return quat_mul(q, raw)

    def _velocity(self, raw):
        # 速度只随出生朝向旋转；这里不加入 support_z 变化对应的竖直速度。
        q = self.rotation.reshape(self.num_envs, *([1] * (raw.ndim - 2)), 4).expand(*raw.shape[:-1], 4)
        return quat_apply(q, raw)

    @property
    def body_pos_w(self):
        return self._position(self.reference.reference_frame.link_pos_w[:, 0])

    @property
    def body_quat_w(self):
        return self._rotate(self.reference.reference_frame.link_quat_w[:, 0])

    @property
    def body_lin_vel_w(self):
        return self._velocity(self.reference.reference_frame.link_lin_vel_w[:, 0])

    @property
    def body_ang_vel_w(self):
        return self._velocity(self.reference.reference_frame.link_ang_vel_w[:, 0])

    @property
    def anchor_pos_w(self):
        return self._position(self.reference.reference_frame.base_pos_w[:, 0])

    @property
    def anchor_quat_w(self):
        return self._rotate(self.reference.reference_frame.base_quat_w[:, 0])

    @property
    def anchor_pos_w_future(self):
        return self._position(self.reference.data.base_pos_w.torch)

    @property
    def anchor_quat_w_future(self):
        return self._rotate(self.reference.data.base_quat_w.torch)

    @property
    def joint_pos(self):
        return self.reference.reference_frame.joint_pos[:, 0]

    @property
    def joint_vel(self):
        return self.reference.reference_frame.joint_vel[:, 0]

    @property
    def robot_body_pos_w(self):
        return self.robot.data.body_pos_w.torch[:, self.body_ids]

    @property
    def robot_body_quat_w(self):
        return self.robot.data.body_quat_w.torch[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self):
        return self.robot.data.body_lin_vel_w.torch[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self):
        return self.robot.data.body_ang_vel_w.torch[:, self.body_ids]

    @property
    def robot_joint_pos(self):
        return self.robot.data.joint_pos.torch

    @property
    def robot_joint_vel(self):
        return self.robot.data.joint_vel.torch

    @property
    def robot_anchor_pos_w(self):
        return self.robot.data.root_pos_w.torch

    @property
    def robot_anchor_quat_w(self):
        return self.robot.data.root_quat_w.torch

    @property
    def robot_anchor_lin_vel_w(self):
        return self.robot.data.root_lin_vel_w.torch

    @property
    def body_pos_relative_w(self):
        # 跟踪目标的 XY 和 yaw 对齐实际机器人，但保留地形调整后的参考根部高度。
        delta = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        root = self.robot_anchor_pos_w.clone()
        root[:, 2] = self.anchor_pos_w[:, 2]
        return root[:, None] + quat_apply(
            delta[:, None].expand_as(self.body_quat_w), self.body_pos_w - self.anchor_pos_w[:, None]
        )

    @property
    def body_quat_relative_w(self):
        delta = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        return quat_mul(delta[:, None].expand_as(self.body_quat_w), self.body_quat_w)

    @property
    def corrected_velocity_future_b(self):
        # 在参考 yaw 坐标系中，为 11 帧未来平面速度加入位置误差修正。
        # 修正随参考平面速度门控并限幅，竖直速度输出为零，具体规则见 corrected_planar_velocity。
        q = yaw_quat(self.anchor_quat_w)[:, None].expand(-1, 11, -1)
        velocity = self._velocity(self.reference.data.base_lin_vel_w.torch)
        error = (self.anchor_pos_w - self.robot_anchor_pos_w)[:, None].expand_as(velocity)
        return corrected_planar_velocity(quat_apply_inverse(q, velocity), quat_apply_inverse(q, error))

    @property
    def corrected_velocity_w(self):
        return quat_apply(yaw_quat(self.anchor_quat_w), self.corrected_velocity_future_b[:, 0])

    def advance(self):
        # 每个控制步先更新地形支撑高度，再采样接触事件及实际/参考加速度，供终止和奖励使用。
        self.terrain.update()
        self.contacts.update()

    def reset_placement(self, ids):
        """结算上一轮、采样参考片段，并将机器人放置到所分配地形块的出生位置。"""
        if isinstance(ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[ids]
        old = ids[self.started[ids]]
        t = self.env.scene.terrain
        if len(old):
            # 首次出生不结算；后续按上一轮实际所在的地形类别、难度和动作累积失败统计。
            family, level = self.terrain.family[old], self.terrain.level[old]
            failed = self.env.termination_manager.terminated[old]
            index = (family * 10 + level) * 40 + self.motion_ids[old]
            self.stratum_visits.view(-1).scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
            counts = torch.zeros_like(self.stratum_failure).view(-1)
            counts.scatter_add_(0, index, failed.float())
            self.stratum_failure.mul_(0.99).add_(counts.reshape_as(self.stratum_failure), alpha=0.01)
            if self.cfg.enable_terrain_curriculum and t.cfg.terrain_type == "generator":
                # 无失败且未越界地跑满时长才算成功；非平地还需累计超过 50 步离开中心平台。
                # 成功升一级、失败降一级，并同步更新地形出生原点。
                success = self.env.termination_manager.get_term("time_out")[old] & ~failed & ~self.terrain.outside[old]
                success &= (self.terrain.off_platform_steps[old] > 50) | (family == 0)
                t.terrain_levels[old] = (t.terrain_levels[old] + success.long() - failed.long()).clamp(0, 9)
                t.env_origins[old] = t.terrain_origins[t.terrain_levels[old], t.terrain_types[old]]
        if t.cfg.terrain_type == "generator":
            family, level = t.terrain_types[ids] // 2, t.terrain_levels[ids]
        else:
            family = level = torch.zeros_like(ids)
        # 当前地形下失败较多的动作获得更高采样权重，同时保留 20% 均匀采样。
        weights = 1 + self.stratum_failure[family, level]
        weights = 0.8 * weights / weights.sum(-1, keepdim=True) + 0.2 / 40
        selected = torch.multinomial(weights, 1).squeeze(-1)
        self.motion._assigned_env_motion_selection[ids] = selected
        # 均匀采样源动作起始帧，预留完整的 11 帧未来窗口；50 Hz 帧索引换算为秒。
        available = self.label_lengths[selected] - 11
        self.motion._motion_buffer_start_time_s[ids] = (
            torch.rand(len(ids), device=self.device) * available
        ).long() / 50
        self.motion_ids[ids] = selected
        self.reference._reset_data(ids)
        raw = self.reference.get_init_reference_state(ids)
        # 仅随机 yaw；设置 XY 偏移，使选中起始帧的根部落在地形块原点。
        yaw = torch.empty(len(ids), device=self.device).uniform_(-torch.pi, torch.pi)
        self.rotation[ids] = quat_from_euler_xyz(yaw * 0, yaw * 0, yaw)
        local = raw.base_pos_w - self.env.scene.env_origins[ids]
        self.offset[ids] = self.env.scene.env_origins[ids]
        self.offset[ids, :2] -= quat_apply(self.rotation[ids], local)[:, :2]
        # 出生先以地形原点高度为基准，之后由 TerrainContext 查询局部地面并更新参考支撑高度。
        self.terrain.support_z[ids] = self.env.scene.env_origins[ids, 2]
        bottom = self.label_bottom[self.frame_ids[ids]].nan_to_num(nan=0).amin(-1)
        state_pos = self.anchor_pos_w[ids].clone()
        # 按源帧所有部位的最低碰撞点，仅向上抬高实际机器人，争取相对基准面 7.5 cm 的净空。
        # 此额外出生抬升不写入参考轨迹，也不是对局部地形逐个碰撞点做避障。
        state_pos[:, 2] += (0.075 - bottom).clamp_min(0)
        self.robot.write_root_pose_to_sim(torch.cat((state_pos, self.anchor_quat_w[ids]), -1), env_ids=ids)
        # 根部速度随出生 yaw 旋转；关节位置和速度直接使用所选参考帧。
        self.robot.write_root_velocity_to_sim(
            torch.cat(
                (
                    self._velocity(self.reference.reference_frame.base_lin_vel_w[:, 0])[ids],
                    self._velocity(self.reference.reference_frame.base_ang_vel_w[:, 0])[ids],
                ),
                -1,
            ),
            env_ids=ids,
        )
        self.robot.write_joint_state_to_sim(self.joint_pos[ids], self.joint_vel[ids], env_ids=ids)
        # 重置时用已知参考脚位置查询地形，避免读取尚未刷新的旧身体缓存；同时清空接触事件历史。
        self.terrain.update(reset_ids=ids)
        self.contacts.reset(ids)
        self.started[ids] = True


@configclass
class TerrainMotionCfg(CommandTermCfg):
    class_type: type | str = "{DIR}.terrain_commands:TerrainMotion"
    resampling_time_range: tuple = (1e9, 1e9)
    contact_labels_dir: str = ""
    enable_terrain_curriculum: bool = True
    enable_contact_objectives: bool = True
    enable_tolerance: bool = True
    tolerance_alpha: float = 1.0
    tolerance_position_m: float = 0.15
    tolerance_orientation_rad: float = 0.35
    tolerance_joint_rad: float = 0.25
    support_filter_seconds: float = 0.10  # 支撑高度的一阶平滑时间常数，单位为秒。


def reset_on_terrain(env, env_ids):
    env.command_manager.get_term("motion").reset_placement(env_ids)
