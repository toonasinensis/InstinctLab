"""Stateful contact events sampled exactly once per control step, reset by episode."""
## 辅助计算reward
import torch

from .terrain_math import tangent_velocity


class TerrainContacts:
    def __init__(self, command):
        self.c = command
        sensor = command.env.scene["contact_forces"]
        self.sensor_ids = [sensor.body_names.index(name) for name in command.reference.cfg.link_of_interests]
        shape = (command.num_envs, len(self.sensor_ids))
        self.contact = torch.zeros(shape, dtype=torch.bool, device=command.device)
        self.touchdown = self.contact.clone()
        self.switch = self.contact.clone()
        self.previous_velocity = command.robot_body_lin_vel_w.clone()
        self.previous_ref_velocity = command.body_lin_vel_w.clone()
        self.acceleration = torch.zeros_like(self.previous_velocity)
        self.ref_acceleration = torch.zeros_like(self.previous_velocity)
        self.valid_derivative = torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
        self.just_reset = ~self.valid_derivative
        self.force = torch.zeros(*shape, 3, device=command.device)

    def reset(self, ids):
        self.contact[ids] = False
        self.touchdown[ids] = False
        self.switch[ids] = False
        self.acceleration[ids] = 0
        self.ref_acceleration[ids] = 0
        self.previous_velocity[ids] = self.c.robot_body_lin_vel_w[ids]
        self.previous_ref_velocity[ids] = self.c.body_lin_vel_w[ids]
        self.valid_derivative[ids] = False
        self.just_reset[ids] = True

    def update(self):
        c = self.c
        self.force = c.env.scene["contact_forces"].data.net_forces_w.torch[:, self.sensor_ids]
        magnitude = self.force.norm(dim=-1)
        now = torch.where(self.contact, magnitude > 5.0, magnitude > 10.0)
        self.touchdown[:] = now & ~self.contact & ~self.just_reset[:, None]
        self.switch[:] = (now != self.contact) & ~self.just_reset[:, None]
        self.contact.copy_(now)
        self.valid_derivative[:] = ~self.just_reset
        self.acceleration[:] = (c.robot_body_lin_vel_w - self.previous_velocity) / c.env.step_dt
        self.ref_acceleration[:] = (c.body_lin_vel_w - self.previous_ref_velocity) / c.env.step_dt
        self.acceleration[~self.valid_derivative] = 0
        self.ref_acceleration[~self.valid_derivative] = 0
        self.previous_velocity.copy_(c.robot_body_lin_vel_w)
        self.previous_ref_velocity.copy_(c.body_lin_vel_w)
        self.just_reset[:] = False

    def terrain_term(self, name):
        c, feet = self.c, self.c.foot_ids
        contact = self.contact[:, feet]
        normal = c.terrain.foot_normal
        velocity = c.robot_body_lin_vel_w[:, feet]
        force = self.force[:, feet]
        if name == "touchdown":
            # Local 5-point relief at landing, in metres; event reward only.
            xy = c.robot_body_pos_w[:, feet, :2]
            offsets = xy.new_tensor([[0, 0], [-0.08, 0], [0.08, 0], [0, -0.04], [0, 0.04]])
            h, _, valid = c.terrain.query(xy[:, :, None] + offsets)
            relief = h.amax(-1) - h.amin(-1)
            quality = torch.exp(-(relief / 0.04).square()) * normal[..., 2].clamp(0, 1)
            return (quality * self.touchdown[:, feet] * valid.all(-1)).mean(-1)
        if name == "match":
            valid = c.reference_contact_valid
            match = (self.contact == c.reference_contacts).float() * valid
            return match.sum(-1) / valid.sum(-1).clamp_min(1)
        if name == "slip":
            return (tangent_velocity(velocity, normal).square().sum(-1) * contact).sum(-1)
        if name == "stumble":
            normal_force = (force * normal).sum(-1).abs()
            return (
                ((tangent_velocity(force, normal).norm(dim=-1) > 5 * normal_force.clamp_min(1)) * contact)
                .float()
                .sum(-1)
            )
        if name == "switch":
            # Penalize unexpected changes only; ordinary reference contact changes are allowed.
            valid = c.reference_contact_valid[:, feet]
            unexpected = self.contact[:, feet] != c.reference_contacts[:, feet]
            return (self.switch[:, feet] * unexpected * valid).float().mean(-1)
        if name == "force":
            return force.square().sum((-1, -2))
        raise KeyError(name)
