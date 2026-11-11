"""Generalized-coordinate representations following SuperMimic's representation boundary.

PGMT stores root COM velocity and link-origin position. Integration explicitly
converts between them; quaternion convention is xyzw, as in the Newton task.
"""

import torch
from torch.nn import functional as F

from isaaclab.utils.math import quat_inv, quat_mul, yaw_quat

from instinctlab.tasks.shadowing.pgmt.tensor_math import cross, quat_apply, quat_apply_inverse
from instinctlab.tasks.shadowing.pgmt.wm_dynamics import rotation_increment, rotation_vector

from .config import RepresentationCfg
from .interfaces import StateRepresentation
from .layouts import RobotLayout


class GeneralizedCoordinates(StateRepresentation):
    def __init__(self, layout: RobotLayout, cfg: RepresentationCfg, dt: float, substeps: int, root_com):
        super().__init__()
        self.layout, self.cfg, self.dt = layout, cfg, dt
        # Exact constant-acceleration weight for semi-implicit Euler substeps.
        self.substep_coeff = (substeps + 1) / (2 * substeps)
        self.register_buffer("root_com", root_com.detach().clone())
        widths = [width for _, width in layout.delta.fields]
        self.register_buffer(
            "nominal_error", torch.cat([torch.full((w,), v) for w, v in zip(widths, cfg.nominal_error, strict=True)])
        )
        self.register_buffer(
            "error_floor", torch.cat([torch.full((w,), v) for w, v in zip(widths, cfg.error_floor, strict=True)])
        )

    def represent(self, state, ground):
        x = self.layout.state.unpack(state)
        gravity = torch.zeros_like(x["position"])
        gravity[..., -1] = -1
        height = (x["position"][..., -1:] - ground[..., None]).clamp(-self.cfg.height_clip_m, self.cfg.height_clip_m)
        return torch.cat(
            (
                height,
                quat_apply_inverse(x["rotation"], gravity),
                quat_apply_inverse(x["rotation"], x["linear_velocity"]),
                quat_apply_inverse(x["rotation"], x["angular_velocity"]),
                x["joint_position"],
                x["joint_velocity"],
            ),
            -1,
        )

    def state_error(self, predicted, reference):
        x, y = self.layout.state.unpack(reference), self.layout.state.unpack(predicted)
        return torch.cat(
            (
                quat_apply_inverse(yaw_quat(x["rotation"]), y["position"] - x["position"]),
                rotation_vector(quat_mul(quat_inv(x["rotation"]), y["rotation"])),
                quat_apply_inverse(x["rotation"], y["linear_velocity"] - x["linear_velocity"]),
                quat_apply_inverse(x["rotation"], y["angular_velocity"] - x["angular_velocity"]),
                y["joint_position"] - x["joint_position"],
                y["joint_velocity"] - x["joint_velocity"],
            ),
            -1,
        )

    def _blend(self, before, after):
        return self.dt * ((1 - self.substep_coeff) * before + self.substep_coeff * after)

    def drift(self, state, successor):
        x, y = self.layout.state.unpack(state), self.layout.state.unpack(successor)
        offset = quat_apply(x["rotation"], self.root_com.expand_as(x["position"]))
        offset_next = quat_apply(y["rotation"], self.root_com.expand_as(y["position"]))
        v = x["linear_velocity"] - cross(x["angular_velocity"], offset)
        vn = y["linear_velocity"] - cross(y["angular_velocity"], offset_next)
        return (
            quat_apply_inverse(yaw_quat(x["rotation"]), self._blend(v, vn)),
            quat_apply_inverse(x["rotation"], self._blend(x["angular_velocity"], y["angular_velocity"])),
            self._blend(x["joint_velocity"], y["joint_velocity"]),
        )


class DeltaRepresentation(GeneralizedCoordinates):
    @property
    def prediction_dim(self):
        return self.layout.delta.width

    def target(self, state, successor):
        return self.state_error(successor, state)

    def apply_prediction(self, state, prediction):
        x, d = self.layout.state.unpack(state), self.layout.delta.unpack(prediction)
        return torch.cat(
            (
                x["position"] + quat_apply(yaw_quat(x["rotation"]), d["position"]),
                F.normalize(quat_mul(x["rotation"], rotation_increment(d["rotation"])), dim=-1),
                x["linear_velocity"] + quat_apply(x["rotation"], d["linear_velocity"]),
                x["angular_velocity"] + quat_apply(x["rotation"], d["angular_velocity"]),
                x["joint_position"] + d["joint_position"],
                x["joint_velocity"] + d["joint_velocity"],
            ),
            -1,
        )


class VelocityRepresentation(GeneralizedCoordinates):
    @property
    def prediction_dim(self):
        return self.layout.velocity.width

    def target(self, state, successor):
        error = self.layout.delta.unpack(self.state_error(successor, state))
        return torch.cat([error[name] for name, _ in self.layout.velocity.fields], -1)

    def apply_prediction(self, state, prediction):
        x, d = self.layout.state.unpack(state), self.layout.velocity.unpack(prediction)
        dv = quat_apply(x["rotation"], d["linear_velocity"])
        dw = quat_apply(x["rotation"], d["angular_velocity"])
        w = x["angular_velocity"] + self.substep_coeff * dw
        rotation = F.normalize(
            quat_mul(x["rotation"], rotation_increment(quat_apply_inverse(x["rotation"], w) * self.dt)), dim=-1
        )
        com = self.root_com.expand_as(x["position"])
        position = (
            x["position"]
            + quat_apply(x["rotation"], com)
            + self.dt * (x["linear_velocity"] + self.substep_coeff * dv)
            - quat_apply(rotation, com)
        )
        return torch.cat(
            (
                position,
                rotation,
                x["linear_velocity"] + dv,
                x["angular_velocity"] + dw,
                x["joint_position"] + self.dt * (x["joint_velocity"] + self.substep_coeff * d["joint_velocity"]),
                x["joint_velocity"] + d["joint_velocity"],
            ),
            -1,
        )


class StructuredRepresentation(DeltaRepresentation):
    def target(self, state, successor):
        delta = self.layout.delta.unpack(super().target(state, successor))
        for name, drift in zip(("position", "rotation", "joint_position"), self.drift(state, successor), strict=True):
            delta[name] = delta[name] - drift
        return torch.cat(tuple(delta.values()), -1)

    def apply_prediction(self, state, prediction):
        x, d = self.layout.state.unpack(state), self.layout.delta.unpack(prediction)
        v = x["linear_velocity"] + quat_apply(x["rotation"], d["linear_velocity"])
        w = x["angular_velocity"] + quat_apply(x["rotation"], d["angular_velocity"])
        qd = x["joint_velocity"] + d["joint_velocity"]
        dr = quat_apply_inverse(x["rotation"], self._blend(x["angular_velocity"], w))
        rotation = F.normalize(quat_mul(x["rotation"], rotation_increment(d["rotation"] + dr)), dim=-1)
        successor = torch.cat((x["position"], rotation, v, w, x["joint_position"], qd), -1)
        dp, _, dq = self.drift(state, successor)
        for name, drift in zip(("position", "rotation", "joint_position"), (dp, dr, dq), strict=True):
            d[name] = d[name] + drift
        return super().apply_prediction(state, torch.cat(tuple(d.values()), -1))


REPRESENTATIONS = {
    "delta": DeltaRepresentation,
    "velocity": VelocityRepresentation,
    "structured": StructuredRepresentation,
}
