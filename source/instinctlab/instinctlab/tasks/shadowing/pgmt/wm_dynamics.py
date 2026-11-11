"""Scheme A dynamics for auxiliary supervised learning; owns no policy parameters."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from isaaclab.utils.math import quat_inv, quat_mul, yaw_quat

from .tensor_math import cross, quat_apply, quat_apply_inverse


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


def state_features(state, heights, ground_height=None):
    p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
    gravity = torch.zeros_like(p)
    gravity[:, 2] = -1
    return torch.cat(
        (
            -heights[:, 220:221] if ground_height is None else (p[:, 2:3] - ground_height[:, None]).clamp(-5, 5),
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

    def __init__(
        self,
        context_width,
        hidden=512,
        amp=True,
        predict_contact=True,
        predict_force=True,
        contact_indices=None,
        scan_points=441,
        ground_query=None,
        structured=False,
        root_com=None,
        dt=0.02,
        prediction_mode="delta",
    ):
        super().__init__()
        self.amp = amp
        self.amp_heads = True
        self.context_width = context_width
        self.predict_contact, self.predict_force = predict_contact, predict_force
        self.register_buffer(
            "contact_indices",
            None if contact_indices is None else torch.tensor(contact_indices, dtype=torch.long),
            persistent=False,
        )
        self.contact_input = predict_contact and contact_indices is None
        self.ground_query = ground_query
        self.structured, self.dt = structured, dt
        if prediction_mode not in ("delta", "velocity"):
            raise ValueError(f"Unknown dynamics prediction: {prediction_mode}")
        self.prediction_mode = prediction_mode
        self.register_buffer(
            "root_com",
            torch.zeros(3) if root_com is None else root_com.detach().clone(),
            persistent=structured or prediction_mode == "velocity",
        )
        self.input_norm = RunningMoments(68 + 29 + 29 + (14 if self.contact_input else 0) + context_width)
        self.delta_norm = RunningMoments(35 if prediction_mode == "velocity" else 70, epsilon=1e-4)
        self.error_norm = RunningMoments(70, epsilon=1e-4) if structured or prediction_mode == "velocity" else None
        self.force_norm = RunningMoments(42) if predict_force else None
        self.terrain_encoder = nn.Sequential(nn.Linear(scan_points, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        layers, width = [], self.input_norm.mean.numel() + 64
        for _ in range(3):
            layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden, self.delta_norm.mean.numel())
        self.contact_head = (
            nn.Linear(hidden, len(contact_indices) if contact_indices is not None else 14) if predict_contact else None
        )
        self.force_head = nn.Linear(hidden, 42) if predict_force else None
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.register_buffer(
            "nominal_error",
            torch.tensor([0.01] * 3 + [0.02] * 3 + [0.1] * 3 + [0.3] * 3 + [0.1] * 29 + [3.14] * 29),
            persistent=False,
        )
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
        ground = self.ground_query(state[:, :2])[0] if self.ground_query is not None else None
        features = [state_features(state, heights, ground), action, previous_action]
        if self.contact_input:
            features.append(contact)
        return torch.cat((*features, context), -1)

    def drift(self, state, next_state):
        """Semi-implicit drift in link-origin coordinates; stored velocities are COM velocities."""
        rot, rot1 = state[:, 3:7], next_state[:, 3:7]
        offset = quat_apply(rot, self.root_com.expand(len(state), -1))
        offset1 = quat_apply(rot1, self.root_com.expand(len(state), -1))
        v = state[:, 7:10] - cross(state[:, 10:13], offset, dim=-1)
        v1 = next_state[:, 7:10] - cross(next_state[:, 10:13], offset1, dim=-1)
        blend = lambda a, b: self.dt * (0.375 * a + 0.625 * b)
        return (
            quat_apply_inverse(yaw_quat(rot), blend(v, v1)),
            quat_apply_inverse(rot, blend(state[:, 10:13], next_state[:, 10:13])),
            blend(state[:, 42:71], next_state[:, 42:71]),
        )

    def target(self, state, next_state):
        delta = state_delta(state, next_state)
        if self.prediction_mode == "velocity":
            return torch.cat((delta[:, 6:12], delta[:, 41:]), -1)
        if self.structured:
            dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
            gp, gr, gq = self.drift(state, next_state)
            delta = torch.cat((dp - gp, dr - gr, dv, dw, dq - gq, dqd), -1)
        return delta

    @staticmethod
    def state_error(predicted, reference):
        # Use the ground-truth orientation for error coordinates, as in the base.
        return state_delta(reference, predicted)

    def predict_delta(self, normalized, heights):
        with torch.autocast(
            device_type=heights.device.type, dtype=torch.bfloat16, enabled=self.amp and heights.is_cuda
        ):
            hidden = self.trunk(torch.cat((normalized, self.terrain_encoder(heights)), -1))
            if self.amp and not self.amp_heads:
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    return self.predict_heads(hidden.float())
            return self.predict_heads(hidden)

    def predict_heads(self, hidden):
        delta = self.delta_head(hidden).float()
        logits = self.contact_head(hidden).float() if self.contact_head is not None else None
        forces = self.force_head(hidden).float() if self.force_head is not None else None
        return delta, logits, forces

    def forward(self, state, action, previous_action, contact, heights, context, checkpoint_network=False):
        normalized = self.input_norm(self.inputs(state, action, previous_action, contact, heights, context))
        if checkpoint_network and torch.is_grad_enabled():
            delta, logits, forces = checkpoint(
                self.predict_delta, normalized, heights, use_reentrant=False, preserve_rng_state=False, early_stop=False
            )
        else:
            delta, logits, forces = self.predict_delta(normalized, heights)
        delta = delta * self.delta_norm.std() + self.delta_norm.mean
        if self.prediction_mode == "velocity":
            return self.integrate_velocity(state, delta), logits, forces
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
        p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
        if self.structured:
            v1, w1, qd1 = v + quat_apply(rot, dv), w + quat_apply(rot, dw), qd + dqd
            # Rotation drift does not depend on the next orientation.
            gr = quat_apply_inverse(rot, self.dt * (0.375 * w + 0.625 * w1))
            rot1 = F.normalize(quat_mul(rot, rotation_increment(dr + gr)), dim=-1)
            next_state = torch.cat((p, rot1, v1, w1, q, qd1), -1)
            gp, _, gq = self.drift(state, next_state)
            dp, dr, dq = dp + gp, dr + gr, dq + gq
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

    def integrate_velocity(self, state, delta):
        """Integrate COM translation and root rotation, then recover the link origin.

        A constant acceleration over four semi-implicit substeps gives the new
        velocity increment weight 5/8. No position/orientation residual is learned.
        """
        p, rot, v, w, q, qd = state.split((3, 4, 3, 3, 29, 29), -1)
        dv, dw, dqd = delta.split((3, 3, 29), -1)
        dv, dw = quat_apply(rot, dv), quat_apply(rot, dw)
        mean_w = w + 0.625 * dw
        rot1 = F.normalize(quat_mul(rot, rotation_increment(quat_apply_inverse(rot, mean_w) * self.dt)), dim=-1)
        com = self.root_com.expand(len(state), -1)
        p1 = p + quat_apply(rot, com) + self.dt * (v + 0.625 * dv) - quat_apply(rot1, com)
        return torch.cat((p1, rot1, v + dv, w + dw, q + self.dt * (qd + 0.625 * dqd), qd + dqd), -1)
