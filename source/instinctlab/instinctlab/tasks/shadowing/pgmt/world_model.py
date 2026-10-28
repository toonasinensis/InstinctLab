"""SuperMimic-style frame-local delta dynamics, conditioned on PGMT terrain.

State: pelvis position, XYZW quaternion, world linear/angular velocity, then
29 joint positions and velocities in PGMT order. No SuperMimic checkpoint is
silently reinterpreted in this different robot/state contract.
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat


def average_gradients(module):
    """Average auxiliary gradients before clipping, including locally unused parameters."""
    if not dist.is_initialized():
        return
    parameters = [p for p in module.parameters() if p.requires_grad]
    gradients = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in parameters])
    dist.all_reduce(gradients)
    gradients.div_(dist.get_world_size())
    offset = 0
    for parameter in parameters:
        parameter.grad = gradients[offset : offset + parameter.numel()].view_as(parameter)
        offset += parameter.numel()


def yaw_angle(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def rotation_increment(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat((0.5 * torch.sinc(angle / (2 * torch.pi)) * vector, (angle / 2).cos()), -1)


def state_features(state, heights):
    p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
    gravity = torch.zeros_like(p)
    gravity[:, 2] = -1
    return torch.cat(
        (
            -heights[:, 220:221],
            quat_apply_inverse(rot, gravity),
            quat_apply_inverse(rot, v),
            quat_apply_inverse(rot, w),
            q,
            qd,
        ),
        -1,
    )


class Moments(nn.Module):
    def __init__(self, size, floor=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.register_buffer("count", torch.zeros(()))
        self.floor = floor

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        mean, variance = x.mean(0), x.var(0, unbiased=False)
        if dist.is_initialized():
            sums = torch.cat((mean * n, mean.new_tensor([n])))
            dist.all_reduce(sums)
            global_mean, count = sums[:-1] / sums[-1], sums[-1]
            variance = (variance + (mean - global_mean).square()) * n
            dist.all_reduce(variance)
            mean, variance, n = global_mean, variance / count, count
        total = self.count + n
        delta = mean - self.mean
        self.variance.copy_(
            (self.variance * self.count + variance * n + delta.square() * self.count * n / total) / total
        )
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def std(self):
        return self.variance.clamp_min(self.floor**2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


class TerrainWorldModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.terrain_encoder = nn.Sequential(
            nn.Linear(441, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.input_norm = Moments(68 + 29 + 29 + 14)
        self.delta_norm = Moments(70, floor=1e-3)
        layers = []
        width = 68 + 29 + 29 + 14 + 64
        for _ in range(3):
            layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.dynamics = nn.Linear(hidden, 70)
        self.contact_head = nn.Linear(hidden, 14)
        nn.init.zeros_(self.dynamics.weight)
        nn.init.zeros_(self.dynamics.bias)

    def inputs(self, state, action, previous_action, contact, heights):
        return torch.cat((state_features(state, heights), action, previous_action, contact), -1)

    def forward(self, state, action, previous_action, contact, heights):
        x = self.input_norm(self.inputs(state, action, previous_action, contact, heights))
        hidden = self.trunk(torch.cat((x, self.terrain_encoder(heights)), -1))
        return self.dynamics(hidden), self.contact_head(hidden)

    @staticmethod
    def target(state, next_state):
        p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
        p1, rot1, v1, w1, q1, qd1 = next_state.split((3, 4, 3, 3, 29, 29), -1)
        return torch.cat(
            (
                quat_apply_inverse(yaw_quat(rot), p1 - p),
                axis_angle_from_quat(quat_mul(quat_inv(rot), rot1)),
                quat_apply_inverse(rot, v1 - v),
                quat_apply_inverse(rot, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, state, action, previous_action, contact, heights):
        delta, logits = self(state, action, previous_action, contact, heights)
        dp, dr, dv, dw, dq, dqd = (delta * self.delta_norm.std() + self.delta_norm.mean).split((3, 3, 3, 3, 29, 29), -1)
        p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
        rot1 = F.normalize(quat_mul(rot, rotation_increment(dr)), dim=-1)
        return (
            torch.cat(
                (
                    p + quat_apply(yaw_quat(rot), dp),
                    rot1,
                    v + quat_apply(rot, dv),
                    w + quat_apply(rot, dw),
                    q + dq,
                    qd + dqd,
                ),
                -1,
            ),
            logits.sigmoid(),
        )
