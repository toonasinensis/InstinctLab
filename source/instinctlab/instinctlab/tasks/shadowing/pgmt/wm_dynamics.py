"""Scheme A dynamics for auxiliary supervised learning; owns no policy parameters."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat


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


def rotation_vector(quaternion):
    # Avoid an unselected 0/0 branch in the upstream axis-angle conversion: its
    # forward result at identity is finite, but its backward result contains NaNs.
    quaternion = quaternion * torch.where(quaternion[..., 3:4] < 0, -1.0, 1.0)
    angle = 2 * torch.atan2(quaternion[..., :3].norm(dim=-1, keepdim=True), quaternion[..., 3:4])
    return 2 * quaternion[..., :3] / torch.sinc(angle / (2 * torch.pi))


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


def state_delta(state, next_state):
    p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
    p1, rot1, v1, w1, q1, qd1 = next_state.split((3, 4, 3, 3, 29, 29), -1)
    return torch.cat(
        (
            quat_apply_inverse(yaw_quat(rot), p1 - p),
            rotation_vector(quat_mul(quat_inv(rot), rot1)),
            quat_apply_inverse(rot, v1 - v),
            quat_apply_inverse(rot, w1 - w),
            q1 - q,
            qd1 - qd,
        ),
        -1,
    )


class RunningMoments(nn.Module):
    """SuperMimic's online global moments, collected independently of forwards."""

    def __init__(self, width, epsilon=1e-2, max_count=2e8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.epsilon = epsilon
        self.max_count = max_count

    @torch.no_grad()
    def update(self, values):
        if self.count >= self.max_count:
            return
        values = values.reshape(-1, self.mean.numel()).float()
        shifted = values - self.mean
        stats = torch.cat((shifted.sum(0), shifted.square().sum(0), values.new_tensor([len(values)])))
        if dist.is_initialized():
            dist.all_reduce(stats)
        total_batch = stats[-1]
        if total_batch == 0:
            return
        if not torch.isfinite(stats).all():
            raise FloatingPointError("Nonfinite world-model normalization data")
        summed, squared = stats[:-1].chunk(2)
        delta = summed / total_batch
        variance = (squared / total_batch - delta.square()).clamp_min(0)
        total = self.count + total_batch
        self.variance.copy_(
            (self.variance * self.count + variance * total_batch + delta.square() * self.count * total_batch / total)
            / total
        )
        self.mean.add_(delta * (total_batch / total))
        self.count.copy_(total)

    def std(self):
        return (self.variance + self.epsilon**2).sqrt()

    def forward(self, values):
        return (values - self.mean) / self.std()


class TerrainDynamics(nn.Module):
    """Frame-local delta dynamics with applied simulator context and contact heads."""

    def __init__(self, context_width, hidden=2048, amp=True):
        super().__init__()
        self.amp = amp
        self.context_width = context_width
        self.input_norm = RunningMoments(68 + 29 + 29 + 14 + context_width)
        self.delta_norm = RunningMoments(70, epsilon=1e-4)
        self.force_norm = RunningMoments(42)
        self.terrain_encoder = nn.Sequential(nn.Linear(441, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        layers, width = [], self.input_norm.mean.numel() + 64
        for _ in range(3):
            layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden, 70)
        self.contact_head = nn.Linear(hidden, 14)
        self.force_head = nn.Linear(hidden, 42)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.register_buffer(
            "error_floor",
            torch.cat(
                (
                    torch.full((3,), 3e-3),
                    torch.full((3,), 5e-3),
                    torch.full((3,), 2e-2),
                    torch.full((3,), 5e-2),
                    torch.full((29,), 3e-3),
                    torch.full((29,), 5e-2),
                )
            ),
        )

    def inputs(self, state, action, previous_action, contact, heights, context):
        return torch.cat((state_features(state, heights), action, previous_action, contact, context), -1)

    @staticmethod
    def target(state, next_state):
        return state_delta(state, next_state)

    @staticmethod
    def state_error(predicted, reference):
        # Use the ground-truth orientation for error coordinates, as in the base.
        return state_delta(reference, predicted)

    def forward(self, state, action, previous_action, contact, heights, context):
        normalized = self.input_norm(self.inputs(state, action, previous_action, contact, heights, context))
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.amp and state.is_cuda):
            hidden = self.trunk(torch.cat((normalized, self.terrain_encoder(heights)), -1))
            delta = self.delta_head(hidden).float()
            logits = self.contact_head(hidden).float()
            forces = self.force_head(hidden).float()
        delta = delta * self.delta_norm.std() + self.delta_norm.mean
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
        p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
        predicted = torch.cat(
            (
                p + quat_apply(yaw_quat(rot), dp),
                F.normalize(quat_mul(rot, rotation_increment(dr)), dim=-1),
                v + quat_apply(rot, dv),
                w + quat_apply(rot, dw),
                q + dq,
                qd + dqd,
            ),
            -1,
        )
        # Local force coordinates use the resulting body's root frame. Decode in
        # the reward adapter; the raw normalized output is used for supervision.
        return predicted, logits, forces
