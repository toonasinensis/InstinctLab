"""Differentiable two-foot contact rewards shared by real and imagined states."""

import torch

from .mdp.contact_labels import collision_vertices
from .tensor_math import cross, quat_apply, quat_apply_inverse


class FootContactRewards:
    names = ("match", "support", "penetration", "slip")

    def __init__(self, rewards, cfg):
        self.rewards, self.cfg = rewards, cfg
        self.feet = rewards.command.foot_ids
        vertices = collision_vertices(rewards.command.reference.cfg.robot_model_path)
        names = rewards.command.reference.cfg.link_of_interests
        self.points = torch.stack(
            [
                torch.as_tensor(vertices[names[index]], device=rewards.env.device, dtype=torch.float32)
                for index in self.feet
            ]
        )
        self.weights = self.points.new_tensor([cfg["match"], -cfg["support"], -cfg["penetration"], -cfg["slip"]])
        self.scale = self.points.new_zeros(())

    def set_updates(self, updates):
        age = updates - self.cfg["warmup_updates"]
        self.scale.fill_(self.cfg["enabled"] * max(0.0, min(1.0, age / self.cfg["ramp_updates"])))

    def __call__(self, state, probability, reference, valid, ids, bodies=None):
        if not self.cfg["enabled"]:
            return state.new_zeros(len(state), 4)
        positions, rotations, velocities, angular = self.rewards.bodies(state, ids) if bodies is None else bodies
        rotation = rotations[:, self.feet]
        offsets = torch.einsum("bfij,fvj->bfvi", rotation, self.points)
        points = positions[:, self.feet, None] + offsets
        height, surface_valid = self.rewards.heightfield.query(points[..., :2])
        clearance, index = (points[..., 2] - height).min(-1)
        gather = index[..., None, None].expand(-1, -1, 1, 3)
        offset = offsets.gather(2, gather).squeeze(2)
        closest = points.gather(2, gather).squeeze(2)
        # Body velocities are COM velocities; evaluate slip at the selected sole point.
        com = (rotation @ self.rewards.body_com[ids][:, self.feet, :, None]).squeeze(-1)
        velocity = velocities[:, self.feet] + cross(angular[:, self.feet], offset - com, dim=-1)
        normal, _, _ = self.rewards.surface(closest[..., :2])
        tangent = velocity - (velocity * normal).sum(-1, keepdim=True) * normal
        reference, valid = reference[:, self.feet], valid[:, self.feet]
        valid = valid * surface_valid.all(-1)
        match = ((reference * probability + (1 - reference) * (1 - probability)) * valid).sum(-1)
        match = match / valid.sum(-1).clamp_min(1)
        support = ((clearance / self.cfg["height_scale"]).square() * reference * valid).mean(-1)
        penetration = (
            (torch.relu(-clearance - self.cfg["penetration_tolerance"]) / self.cfg["penetration_scale"])
            .square()
            .mean(-1)
        )
        slip = (probability * tangent.square().sum(-1)).mean(-1)
        return torch.stack((match, support, penetration, slip), -1) * self.weights * self.scale * self.rewards.dt
