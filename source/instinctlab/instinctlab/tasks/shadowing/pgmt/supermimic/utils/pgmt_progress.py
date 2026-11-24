import torch

from instinctlab.tasks.shadowing.pgmt.selection import TERRAIN_FAMILIES


class StairProgressReward:
    def __init__(self, dt, window_s, walk_speed, deadband_speed, full_fraction, weight):
        if window_s < dt or not 0 <= deadband_speed < walk_speed * full_fraction <= walk_speed or weight <= 0:
            raise ValueError("Invalid stair progress reward parameters")
        self.steps = round(window_s / dt)
        self.seconds = self.steps * dt
        self.walk_speed = walk_speed
        self.deadband_speed = deadband_speed
        self.full_fraction = full_fraction
        self.scale = weight * dt

    @staticmethod
    def positions(state, reference):
        return torch.cat((state[..., :2], reference[..., :2]), dim=-1)

    def record(self, history, position, reset):
        start = position[:, None].expand(-1, self.steps, -1)
        if history is None:
            return start.clone()
        return torch.where(reset[:, None, None], start, torch.cat((history[:, 1:], position[:, None]), dim=1))

    def apply(self, rewards, history, positions, family):
        timeline = torch.cat((history, positions), dim=1)
        actual, reference = (timeline[:, self.steps :] - timeline[:, : -self.steps]).split(2, dim=-1)
        distance = reference.norm(dim=-1)
        direction = reference / distance[..., None].clamp_min(1e-6)
        speed = (actual * direction).sum(-1) / self.seconds
        desired_speed = distance / self.seconds
        walking = (family == TERRAIN_FAMILIES["stairs_up"]) & (desired_speed >= self.walk_speed)
        target = desired_speed * self.full_fraction
        width = (target - self.deadband_speed).clamp_min(1e-6)
        gate = ((speed - self.deadband_speed) / width).clamp(0, 1)
        gate = torch.where(walking, gate, 1)
        signed = (speed / desired_speed.clamp_min(self.walk_speed)).clamp(-1, 1)
        progress = torch.where(walking, signed * self.scale, 0)
        gated = torch.where(rewards > 0, rewards * gate[..., None], rewards)
        return torch.cat((gated, progress[..., None]), dim=-1)
