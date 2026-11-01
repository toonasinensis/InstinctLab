"""Stateless native reward reconstruction, separate from learned dynamics.

Hard contact events deliberately retain native semantics for diagnostic parity.
No reward here is an actor training objective.
"""

import torch
import xml.etree.ElementTree as ET

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    yaw_quat,
)

from .mdp.terrain_math import corrected_planar_velocity, tangent_velocity, tolerance_error
from .wm_dynamics import rotation_increment


class RobotKinematics:
    """Link poses and COM twists matching native PGMT articulation data."""

    def __init__(self, urdf_path, joint_names, tracked_bodies, device):
        root = ET.parse(urdf_path).getroot()
        joints = root.findall("joint")
        child_names = {joint.find("child").get("link") for joint in joints}
        roots = [link.get("name") for link in root.findall("link") if link.get("name") not in child_names]
        if roots != ["pelvis"]:
            raise ValueError(f"Expected PGMT pelvis root, found {roots}")
        self.names, self.parents, self.joint_ids = ["pelvis"], [-1], [-1]
        positions, rotations, axes = [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]
        pending = list(joints)
        while pending:
            ready = [joint for joint in pending if joint.find("parent").get("link") in self.names]
            if not ready:
                raise ValueError("Disconnected PGMT kinematics tree")
            for joint in ready:
                kind = joint.get("type")
                if kind not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"Unsupported PGMT joint type: {kind}")
                origin, axis = joint.find("origin"), joint.find("axis")
                self.parents.append(self.names.index(joint.find("parent").get("link")))
                self.names.append(joint.find("child").get("link"))
                self.joint_ids.append(-1 if kind == "fixed" else joint_names.index(joint.get("name")))
                positions.append(
                    [float(x) for x in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
                )
                rotations.append(
                    [float(x) for x in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
                )
                axes.append([float(x) for x in (axis.get("xyz") if axis is not None else "0 0 1").split()])
                pending.remove(joint)
        self.offset = torch.tensor(positions, device=device)
        rpy = torch.tensor(rotations, device=device)
        self.rotation = matrix_from_quat(quat_from_euler_xyz(*rpy.unbind(-1)))
        self.axis = torch.nn.functional.normalize(torch.tensor(axes, device=device), dim=-1)
        self.tracked = [self.names.index(name) for name in tracked_bodies]

    def forward(self, state, root_com, tracked_com):
        positions, rotations = [state[:, :3]], [matrix_from_quat(state[:, 3:7])]
        # Native state pairs a link-origin position with a COM linear velocity.
        root_offset = (rotations[0] @ root_com[..., None]).squeeze(-1)
        velocities = [state[:, 7:10] - torch.cross(state[:, 10:13], root_offset, dim=-1)]
        angular = [state[:, 10:13]]
        q, qd = state[:, 13:42], state[:, 42:71]
        for index in range(1, len(self.names)):
            parent, joint = self.parents[index], self.joint_ids[index]
            offset = (rotations[parent] @ self.offset[index, :, None]).squeeze(-1)
            position = positions[parent] + offset
            rotation = rotations[parent] @ self.rotation[index]
            velocity = velocities[parent] + torch.cross(angular[parent], offset, dim=-1)
            spin = angular[parent]
            if joint >= 0:
                axis = (rotation @ self.axis[index, :, None]).squeeze(-1)
                spin = spin + axis * qd[:, joint : joint + 1]
                rotation = rotation @ matrix_from_quat(rotation_increment(self.axis[index] * q[:, joint : joint + 1]))
            positions.append(position)
            rotations.append(rotation)
            velocities.append(velocity)
            angular.append(spin)
        position, rotation, velocity, spin = (
            torch.stack([values[index] for index in self.tracked], 1)
            for values in (positions, rotations, velocities, angular)
        )
        com_offset = (rotation @ tracked_com[..., None]).squeeze(-1)
        return position, rotation, velocity + torch.cross(spin, com_offset, dim=-1), spin


def rotation_error(left, right):
    """SO(3) angle, with finite identity derivatives (no acos singularity)."""
    relative = left.transpose(-1, -2) @ right
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    sine = 0.5 * skew.norm(dim=-1)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    return torch.atan2(sine, cosine.clamp(-1, 1))


class AnalyticRewards:
    contact_terms = {
        "auxiliary/undesired_contact",
        "terrain/slip",
        "terrain/match",
        "terrain/switch",
        "terrain/touchdown",
        "terrain/stumble",
    }
    force_terms = {"auxiliary/impact", "terrain/stumble", "terrain/force"}

    def supported_keys(self, contact=True, force=True):
        omitted = (self.contact_terms if not contact else set()) | (self.force_terms if not force else set())
        return [key for key in self.terms if key not in omitted]

    def __init__(self, env, heightfield):
        self.env, self.heightfield = env, heightfield
        self.command = c = env.command_manager.get_term("motion")
        self.dt = env.step_dt
        robot = env.scene["robot"]
        self.kinematics = RobotKinematics(
            c.reference.cfg.robot_model_path, robot.joint_names, c.reference.cfg.link_of_interests, env.device
        )
        self.root_com = robot.data.body_com_pos_b.torch[:, 0].clone()
        self.body_com = robot.data.body_com_pos_b.torch[:, c.body_ids].clone()
        self.limits = robot.data.soft_joint_pos_limits.torch.clone()
        self.terms = {
            group + "/" + name: env.reward_manager.get_term_cfg(name, group)
            for group, names in env.reward_manager.active_terms.items()
            for name in names
        }
        self.contract = {
            key: dict(
                weight=cfg.weight,
                params=cfg.params,
                mode=(
                    "hard_event"
                    if key
                    in {
                        "terrain/stumble",
                        "terrain/switch",
                        "terrain/touchdown",
                        "terrain/match",
                        "auxiliary/undesired_contact",
                    }
                    else "analytic"
                ),
            )
            for key, cfg in self.terms.items()
        }

    def state(self):
        c = self.command
        return torch.cat(
            (
                c.robot_anchor_pos_w,
                c.robot_anchor_quat_w,
                c.robot_anchor_lin_vel_w,
                c.robot.data.root_ang_vel_w.torch,
                c.robot_joint_pos,
                c.robot_joint_vel,
            ),
            -1,
        )

    def context(self):
        """Exogenous reference at reward time, before resets; no future simulated state."""
        c = self.command
        return dict(
            ref_position=c.anchor_pos_w
            - torch.stack(
                (torch.zeros_like(c.terrain.support_z), torch.zeros_like(c.terrain.support_z), c.terrain.support_z), -1
            ),
            ref_rotation=c.anchor_quat_w,
            ref_body_position=c.body_pos_w - c.anchor_pos_w[:, None],
            ref_body_rotation=c.body_quat_w,
            ref_body_velocity=c.body_lin_vel_w,
            ref_body_angular=c.body_ang_vel_w,
            ref_velocity=c._velocity(c.reference.data.base_lin_vel_w.torch)[:, 0],
            ref_q=c.joint_pos,
            ref_qd=c.joint_vel,
            ref_contact=c.reference_contacts.float(),
            ref_valid=c.reference_contact_valid.float(),
            level=c.terrain.level,
            family=c.terrain.family,
        )

    def bodies(self, state, ids):
        return self.kinematics.forward(state, self.root_com[ids], self.body_com[ids])

    def support(self, state, previous, context, bodies):
        c = self.command
        h, valid = self.heightfield.query(bodies[0][:, c.foot_ids, :2])
        root, root_valid = self.heightfield.query(state[:, :2])
        labels = context["ref_contact"][:, c.foot_ids] * context["ref_valid"][:, c.foot_ids]
        target = torch.where(labels.sum(-1) > 0, (h * labels).sum(-1) / labels.sum(-1).clamp_min(1), root)
        blend = 1 - torch.exp(state.new_tensor(-self.dt / c.cfg.support_filter_seconds))
        return previous.lerp(target, blend), valid.all(-1) & root_valid

    def surface(self, xy, exact=False):
        offsets = xy.new_tensor([[0, 0], [-0.08, 0], [0.08, 0], [0, -0.04], [0, 0.04]])
        if exact:
            _, normal, valid = self.command.terrain.query(xy)
            h, _, valid_h = self.command.terrain.query(xy[..., None, :] + offsets)
        else:
            eps = self.heightfield.resolution
            d = xy.new_tensor([[eps, 0], [-eps, 0], [0, eps], [0, -eps]])
            hn, vn = self.heightfield.query(xy[..., None, :] + d)
            normal = torch.stack(
                (
                    -(hn[..., 0] - hn[..., 1]) / (2 * eps),
                    -(hn[..., 2] - hn[..., 3]) / (2 * eps),
                    torch.ones_like(hn[..., 0]),
                ),
                -1,
            )
            normal = torch.nn.functional.normalize(normal, dim=-1)
            valid = vn.all(-1)
            h, valid_h = self.heightfield.query(xy[..., None, :] + offsets)
        quality = torch.exp(-((h.amax(-1) - h.amin(-1)) / 0.04).square()) * normal[..., 2].clamp(0, 1)
        return normal, quality, valid & valid_h.all(-1)

    def evaluate(
        self,
        state,
        previous_state,
        action,
        previous_action,
        contact,
        previous_contact,
        force,
        support,
        context,
        previous_ref_velocity,
        derivative_valid,
        ids,
        exact_surface=False,
    ):
        c = self.command
        position, rotation, velocity, spin = self.bodies(state, ids)
        previous_velocity = self.bodies(previous_state, ids)[2]
        delta = yaw_quat(quat_mul(state[:, 3:7], quat_inv(context["ref_rotation"])))
        target_root = torch.cat((state[:, :2], context["ref_position"][:, 2:3] + support[:, None]), -1)
        aligned = delta[:, None].expand(-1, position.shape[1], -1)
        target = target_root[:, None] + quat_apply(aligned, context["ref_body_position"])
        target_rotation = matrix_from_quat(quat_mul(aligned, context["ref_body_rotation"]))
        errors = dict(
            position=(target - position).norm(dim=-1),
            orientation=rotation_error(target_rotation, rotation),
            linear_velocity=(context["ref_body_velocity"] - velocity).norm(dim=-1),
            angular_velocity=(context["ref_body_angular"] - spin).norm(dim=-1),
            joint_position=(context["ref_q"] - state[:, 13:42]).abs(),
            joint_velocity=(context["ref_qd"] - state[:, 42:71]).abs(),
        )
        tolerances = dict(
            position=c.cfg.tolerance_position_m,
            orientation=c.cfg.tolerance_orientation_rad,
            joint_position=c.cfg.tolerance_joint_rad,
        )
        terms = {}
        for group, bodies, joints in (
            ("upper", c.upper_ids, c.upper_joint_ids),
            ("lower", c.lower_ids, c.lower_joint_ids),
        ):
            for name, error in errors.items():
                error = error[:, joints if name.startswith("joint_") else bodies]
                if group == "lower":
                    error = tolerance_error(
                        error,
                        context["level"],
                        context["family"],
                        tolerances.get(name, 0),
                        c.cfg.tolerance_alpha if c.cfg.enable_tolerance else 0,
                    )
                key = group + "/" + name
                terms[key] = torch.exp(-error.square().mean(-1) / self.terms[key].params["std"] ** 2)
        root_R, ref_R = matrix_from_quat(state[:, 3:7]), matrix_from_quat(context["ref_rotation"])
        heading = yaw_quat(context["ref_rotation"])
        desired = quat_apply(
            heading,
            corrected_planar_velocity(
                quat_apply_inverse(heading, context["ref_velocity"]),
                quat_apply_inverse(heading, context["ref_position"] - state[:, :3]),
            ),
        )
        for name, error in dict(
            root_orientation=rotation_error(root_R, ref_R).square(),
            root_velocity=(desired[:, :2] - state[:, 7:9]).square().sum(-1),
            floating_anchor=(target_root[:, 2] - state[:, 2]).square(),
        ).items():
            key = "auxiliary/" + name
            terms[key] = torch.exp(-error / self.terms[key].params.get("std", 1) ** 2)
        acceleration = (velocity - previous_velocity) / self.dt
        ref_acceleration = (context["ref_body_velocity"] - previous_ref_velocity) / self.dt
        terms["auxiliary/vertical_acceleration"] = (
            acceleration[:, c.motion_anchor_body_index, 2].square() * derivative_valid
        )
        ee = c.foot_ids + c.hand_ids
        terms["auxiliary/ee_acceleration"] = (acceleration[:, ee] - ref_acceleration[:, ee]).square().sum(-1).mean(
            -1
        ) * derivative_valid
        terms["auxiliary/action_rate"] = (action - previous_action).square().sum(-1)
        limits = self.limits[ids]
        joints = state[:, 13:42]
        terms["auxiliary/joint_limits"] = (
            (limits[..., 0] - joints).clamp_min(0) + (joints - limits[..., 1]).clamp_min(0)
        ).sum(-1)
        reference, valid = context["ref_contact"], context["ref_valid"]
        feet = c.foot_ids
        if contact is not None:
            unexpected = ((1 - reference) * valid).clone()
            unexpected[:, feet] = 0
            terms["auxiliary/undesired_contact"] = (contact * unexpected).sum(-1)
            normal, quality, surface_valid = self.surface(position[:, feet, :2], exact_surface)
            foot_contact = contact[:, feet]
            terms["terrain/slip"] = (tangent_velocity(velocity[:, feet], normal).square().sum(-1) * foot_contact).sum(
                -1
            )
            terms["terrain/match"] = ((reference * contact + (1 - reference) * (1 - contact)) * valid).sum(
                -1
            ) / valid.sum(-1).clamp_min(1)
            changed = (contact - previous_contact).abs() * derivative_valid[:, None]
            mismatch = (contact - reference).abs()
            terms["terrain/switch"] = (changed[:, feet] * mismatch[:, feet] * valid[:, feet]).mean(-1)
            touchdown = contact[:, feet] * (1 - previous_contact[:, feet]) * derivative_valid[:, None]
            terms["terrain/touchdown"] = (quality * touchdown * surface_valid).mean(-1)
        if force is not None:
            impact_unexpected = (1 - reference[:, c.impact_ids]) * valid[:, c.impact_ids]
            terms["auxiliary/impact"] = (
                force[:, c.impact_ids].norm(dim=-1).sub(100).clamp_min(0).square() * impact_unexpected
            ).sum(-1)
            foot_force = force[:, feet]
            terms["terrain/force"] = foot_force.square().sum((-1, -2))
            if contact is not None:
                normal_force = (foot_force * normal).sum(-1).abs()
                terms["terrain/stumble"] = (
                    (tangent_velocity(foot_force, normal).norm(dim=-1) > 5 * normal_force.clamp_min(1)).float()
                    * foot_contact
                ).sum(-1)
        if not c.cfg.enable_contact_objectives:
            for key in terms:
                if key.startswith("terrain/"):
                    terms[key] = torch.zeros_like(terms[key])
        keys = self.supported_keys(contact is not None, force is not None)
        terms = {key: value for key, value in terms.items() if key in self.terms}
        if set(terms) != set(keys):
            raise ValueError("Analytic reward implementation does not cover the active native reward contract")
        weighted = torch.stack([terms[key] * self.terms[key].weight * self.dt for key in keys], -1)
        return weighted, terms
