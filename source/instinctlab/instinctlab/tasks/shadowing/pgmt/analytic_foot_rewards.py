"""LightLP footstep and impact penalties gated by learned foot contact."""

import math
import torch

from .mdp.contact_labels import collision_vertices


class AnalyticFootRewards:
    names = ("footstep/illegal", "footstep/acceleration")

    def __init__(self, rewards, cfg):
        self.rewards, self.cfg = rewards, cfg
        self.feet = rewards.command.foot_ids
        vertices = collision_vertices(rewards.command.reference.cfg.robot_model_path)
        names = rewards.command.reference.cfg.link_of_interests
        points = torch.stack(
            [torch.as_tensor(vertices[names[i]], device=rewards.env.device, dtype=torch.float32) for i in self.feet]
        )
        lo, hi = points.amin(1), points.amax(1)
        u, v = torch.meshgrid(
            torch.linspace(0, 1, cfg["grid_x"], device=points.device),
            torch.linspace(0, 1, cfg["grid_y"], device=points.device),
            indexing="ij",
        )
        self.grid = lo[:, None].expand(-1, u.numel(), -1).clone()
        self.grid[..., 0] += u.flatten() * (hi[:, 0] - lo[:, 0])[:, None]
        self.grid[..., 1] += v.flatten() * (hi[:, 1] - lo[:, 1])[:, None]
        self.alpha = math.exp(-rewards.dt / cfg["tau"])
        self.memory = points.new_zeros(rewards.env.num_envs)

    def filter(self, excess, initial):
        """Exact recurrence in batch: no truncation or extra (1-alpha) factor."""
        steps = torch.arange(excess.shape[1], device=excess.device)
        lag = steps[:, None] - steps[None, :]
        kernel = self.alpha ** lag.clamp_min(0) * (lag >= 0)
        return excess @ kernel.T + initial[:, None] * self.alpha ** (steps + 1)

    def evaluate(self, bodies, acceleration, valid_derivative, ids, initial, horizon, contact, hard=False):
        """Use true contact and hard drops in real data, predicted contact and smooth drops in imagination."""
        position, rotation = bodies[:2]
        rotation = rotation[:, self.feet]
        grid = position[:, self.feet, None] + torch.einsum("bfij,fkj->bfki", rotation, self.grid)
        height, valid = self.rewards.heightfield.query(grid[..., :2])
        clearance = grid[..., 2] - height - self.cfg["terrain_margin"]
        overhang = (
            (clearance > self.cfg["drop"]).to(clearance.dtype)
            if hard
            else torch.sigmoid((clearance - self.cfg["drop"]) / self.cfg["drop_temperature"])
        )
        fraction = (overhang * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        illegal = (contact * fraction).sum(-1)
        excess = torch.relu(acceleration[:, self.feet].norm(dim=-1) - self.cfg["acceleration_threshold"]).sum(-1)
        excess = excess * valid_derivative
        filtered = self.filter(excess.reshape(-1, horizon), initial)
        terms = torch.stack(
            (illegal * self.cfg["illegal_weight"], filtered.flatten() * self.cfg["acceleration_weight"]), -1
        )
        return terms * self.rewards.dt, filtered[:, -1]
