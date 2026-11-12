"""从 MuJoCo 模型构造可微 FK，供公开版 mjlab 的 SM 训练使用。

与项目既有 G1Kinematics 使用相同的 hinge-anchor 和 link-origin 速度公式，
按树深度并行计算各分支。输入根角速度和输出刚体速度均为世界坐标系。
"""

import numpy as np
import torch
from torch import nn

import mujoco

from instinctlab.tasks.shadowing.pgmt.supermimic.utils.math import quat_apply, quat_mul


class ForwardKinematics(nn.Module):
    def __init__(self, model, body_ids, joint_q_adr, tracked, device):
        super().__init__()
        ids = body_ids.tolist()
        row = {b: i for i, b in enumerate(ids)}
        qcol = {q: i for i, q in enumerate(joint_q_adr.tolist())}
        self.num_bodies, self.num_joints = len(ids), len(qcol)
        parents, depths, cols = [], [], []
        anchors, axes, zeros = [], [], []
        for i, b in enumerate(ids):
            parent = row.get(int(model.body_parentid[b]), -1)
            parents.append(max(parent, 0))
            depths.append(0 if parent < 0 else depths[parent] + 1)
            assert parent >= 0 or i == 0, "FK expects one rooted robot"
            count = int(model.body_jntnum[b])
            j = int(model.body_jntadr[b])
            if i == 0:
                assert count == 1 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            else:
                assert count <= 1, "FK supports fixed bodies and one hinge per body"
                assert count == 0 or model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            hinge = i > 0 and count == 1
            cols.append(qcol[int(model.jnt_qposadr[j])] if hinge else self.num_joints)
            anchors.append(model.jnt_pos[j].copy() if hinge else [0.0, 0.0, 0.0])
            axes.append(model.jnt_axis[j].copy() if hinge else [0.0, 0.0, 0.0])
            zeros.append(float(model.qpos0[model.jnt_qposadr[j]]) if hinge else 0.0)
        for name, data, dtype in (
            ("offset", model.body_pos[ids], torch.float32),
            ("rotation", model.body_quat[ids], torch.float32),
            ("anchor", anchors, torch.float32),
            ("axis", axes, torch.float32),
            ("q0", zeros, torch.float32),
            ("parents", parents, torch.long),
            ("cols", cols, torch.long),
            ("tracked", tracked, torch.long),
        ):
            values = data if isinstance(data, torch.Tensor) else np.asarray(data)
            self.register_buffer(name, torch.as_tensor(values, dtype=dtype, device=device))
        self.levels = []
        for depth in range(1, max(depths) + 1):
            name = f"level_{depth}"
            self.register_buffer(
                name,
                torch.tensor([i for i, d in enumerate(depths) if d == depth], device=device),
            )
            self.levels.append(name)

    def forward(self, root_pos, root_quat, qj, root_vel, root_avel, vj):
        batch = root_pos.shape[:-1]
        empty3 = root_pos.new_zeros(*batch, self.num_bodies - 1, 3)
        empty4 = root_pos.new_zeros(*batch, self.num_bodies - 1, 4)
        pos = torch.cat([root_pos.unsqueeze(-2), empty3], -2)
        quat = torch.cat([root_quat.unsqueeze(-2), empty4], -2)
        vel = torch.cat([root_vel.unsqueeze(-2), empty3], -2)
        avel = torch.cat([root_avel.unsqueeze(-2), empty3], -2)
        q = torch.cat([qj, qj.new_zeros(*batch, 1)], -1)
        v = torch.cat([vj, vj.new_zeros(*batch, 1)], -1)
        for name in self.levels:
            idx = getattr(self, name)
            parent = self.parents[idx]
            pp, pq = pos[..., parent, :], quat[..., parent, :]
            pv, pw = vel[..., parent, :], avel[..., parent, :]
            bq = quat_mul(pq, self.rotation[idx].expand_as(pq))
            bp = pp + quat_apply(pq, self.offset[idx].expand_as(pp))
            anchor = bp + quat_apply(bq, self.anchor[idx].expand_as(bp))
            axis = quat_apply(bq, self.axis[idx].expand_as(bp))
            half = (q[..., self.cols[idx]] - self.q0[idx]).unsqueeze(-1) * 0.5
            delta = torch.cat([half.cos(), self.axis[idx] * half.sin()], -1)
            bq = quat_mul(bq, delta)
            bp = anchor - quat_apply(bq, self.anchor[idx].expand_as(bp))
            jw = axis * v[..., self.cols[idx]].unsqueeze(-1)
            bv = pv + torch.cross(pw, bp - pp, dim=-1)
            bv = bv + torch.cross(jw, bp - anchor, dim=-1)
            pos = pos.index_copy(-2, idx, bp)
            quat = quat.index_copy(-2, idx, bq)
            vel = vel.index_copy(-2, idx, bv)
            avel = avel.index_copy(-2, idx, pw + jw)
        return tuple(x[..., self.tracked, :] for x in (pos, quat, vel, avel))
