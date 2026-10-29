"""Stateless reference windows and kinematics for pure world-model imagination."""

import math
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

from .imagination import ImaginationMDP
from .mdp.terrain_math import corrected_planar_velocity, tangent_velocity, tolerance_error
from .world_model import rotation_increment


class MotionWindows:
    """Advance an episode's motion/placement without consulting future replay rows."""

    def __init__(self, command):
        self.command = command
        self.bank = command.motion._all_motion_sequences
        self.lengths = command.label_lengths
        # Reference commands cache their defaults before startup randomization;
        # proprioception instead uses the randomized actuator defaults.
        commands = command.env.command_manager
        self.reference_default_q = commands.get_term("joint_pos_ref_command")._default_joint_pos
        self.reference_default_qd = commands.get_term("joint_vel_ref_command")._default_joint_vel

    def placement(self):
        command = self.command
        return dict(
            env_id=torch.arange(command.num_envs, device=command.device),
            motion=command.motion_ids.clone(),
            frame=(command.frame_ids - command.label_starts[command.motion_ids]).clone(),
            placement_position=command.offset.clone(),
            placement_rotation=command.rotation.clone(),
            level=command.terrain.level.clone(),
            family=command.terrain.family.clone(),
            tile_origin=command.env.scene.env_origins.clone(),
        )

    def context(self, start, step, default_q, default_qd):
        frames = start["frame"][:, None] + step + torch.arange(11, device=start["frame"].device)
        motion = start["motion"]
        if (frames >= self.lengths[motion, None]).any() or (frames < 0).any():
            raise ValueError("Imagined reference window crosses a motion boundary")
        motion_grid = motion[:, None].expand_as(frames).flatten()

        def read(name):
            value = getattr(self.bank, name)[motion_grid, frames.flatten()]
            return value.reshape(len(motion), 11, *value.shape[1:]).to(default_q.device)

        root_position, root_rotation = read("base_pos_w"), read("base_quat_w")
        root_velocity, root_angular = read("base_lin_vel_w"), read("base_ang_vel_w")
        joints, joint_velocity = read("joint_pos"), read("joint_vel")
        body_position, body_rotation = read("link_pos_w")[:, 0], read("link_quat_w")[:, 0]
        body_velocity, body_angular = read("link_lin_vel_w")[:, 0], read("link_ang_vel_w")[:, 0]
        placement = start["placement_rotation"]
        future_rotation = placement[:, None].expand_as(root_rotation)
        body_placement = placement[:, None].expand_as(body_rotation)
        root_q = root_rotation[:, 0]
        gravity = default_q.new_tensor([0, 0, -1]).expand(len(motion), -1)
        label = self.command.label_starts[motion] + frames[:, 0]
        return dict(
            ref_joints=torch.cat(
                (
                    (joints - self.reference_default_q[start["env_id"], None]).flatten(1),
                    (joint_velocity - self.reference_default_qd[start["env_id"], None]).flatten(1),
                ),
                -1,
            ),
            ref_rotation=quat_mul(future_rotation, root_rotation),
            ref_clearance=root_position[..., 2],
            ref_position=quat_apply(placement, root_position[:, 0]) + start["placement_position"],
            ref_velocity=quat_apply(future_rotation, root_velocity),
            ref_body_position=quat_apply(body_placement, body_position - root_position[:, :1]),
            ref_body_rotation=quat_mul(body_placement, body_rotation),
            ref_body_velocity=quat_apply(body_placement, body_velocity),
            ref_body_angular=quat_apply(body_placement, body_angular),
            ref_q=joints[:, 0],
            ref_qd=joint_velocity[:, 0],
            ref_contact=self.command.labels[label].float(),
            ref_valid=self.command.label_valid[label].float(),
            level=start["level"],
            family=start["family"],
            tile_origin=start["tile_origin"],
            default_q=default_q,
            default_qd=default_qd,
            critic_command=torch.cat(
                (
                    joints[:, 0],
                    joint_velocity[:, 0],
                    quat_apply_inverse(root_q, root_velocity[:, 0]),
                    quat_apply_inverse(root_q, root_angular[:, 0]),
                    quat_apply_inverse(root_q, gravity),
                    root_position[:, 0, 2:3],
                ),
                -1,
            ),
        )


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


class PureImaginationMDP(ImaginationMDP):
    """PGMT analytic rewards with explicit terrain/contact approximations.

    Force-based impact/stumble/force costs remain measured in simulation but
    are omitted from the first actor objective. Contact probabilities only
    weight geometric costs; they cannot directly suppress a slip penalty.
    """

    omitted = {"auxiliary/impact", "terrain/stumble", "terrain/force"}

    def __init__(self, env, heightfield):
        super().__init__(env, heightfield)
        self.windows = MotionWindows(self.command)
        robot = env.scene["robot"]
        self.kinematics = RobotKinematics(
            self.reference.cfg.robot_model_path, robot.joint_names, self.reference.cfg.link_of_interests, env.device
        )
        self.root_com = robot.data.body_com_pos_b.torch[:, 0].clone()
        self.body_com = robot.data.body_com_pos_b.torch[:, self.command.body_ids].clone()
        self.terms = {}
        self.contract = {}
        for group, names in env.reward_manager.active_terms.items():
            for name in names:
                cfg = env.reward_manager.get_term_cfg(name, group)
                key = group + "/" + name
                self.terms[key] = cfg
                mode = (
                    "omitted"
                    if key in self.omitted
                    else ("surrogate" if group == "terrain" or name == "undesired_contact" else "analytic")
                )
                self.contract[key] = dict(weight=cfg.weight, params=cfg.params, mode=mode)
        self.thresholds = {
            name: env.termination_manager.get_term_cfg(name).params["threshold"]
            for name in ("root_height", "root_tilt", "body_error")
        }
        state_half = (
            env.scene.terrain.cfg.terrain_generator.size
            if env.scene.terrain.cfg.terrain_type == "generator"
            else (200, 200)
        )
        self.tile_half = torch.tensor(state_half, device=env.device) / 2

    def reference_context(self, start, step):
        robot = self.env.scene["robot"].data
        ids = start["env_id"]
        context = self.windows.context(
            start, step, robot.default_joint_pos.torch[ids], robot.default_joint_vel.torch[ids]
        )
        context["root_com"], context["body_com"] = self.root_com[ids], self.body_com[ids]
        return context

    def surface_normal(self, xy):
        eps = self.heightfield.resolution
        offsets = xy.new_tensor([[eps, 0], [-eps, 0], [0, eps], [0, -eps]])
        h, valid = self.heightfield.query(xy[..., None, :] + offsets)
        normal = torch.stack(
            (-(h[..., 0] - h[..., 1]) / (2 * eps), -(h[..., 2] - h[..., 3]) / (2 * eps), torch.ones_like(h[..., 0])), -1
        )
        return torch.nn.functional.normalize(normal, dim=-1), valid.all(-1)

    def objective(
        self,
        state,
        action,
        previous_action,
        support,
        context,
        previous_state,
        contact,
        previous_contact,
        previous_context,
        derivative_valid,
    ):
        c = self.command
        position, rotation, velocity, spin = self.kinematics.forward(state, context["root_com"], context["body_com"])
        previous_velocity = self.kinematics.forward(previous_state, context["root_com"], context["body_com"])[2]
        delta = yaw_quat(quat_mul(state[:, 3:7], quat_inv(context["ref_rotation"][:, 0])))
        target_root = torch.cat((state[:, :2], context["ref_clearance"][:, :1] + support[:, None]), -1)
        aligned = delta[:, None].expand(-1, position.shape[1], -1)
        target_body = target_root[:, None] + quat_apply(aligned, context["ref_body_position"])
        target_rotation = matrix_from_quat(quat_mul(aligned, context["ref_body_rotation"]))
        errors = dict(
            position=(target_body - position).norm(dim=-1),
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
                if group == "lower" and c.cfg.enable_tolerance:
                    error = tolerance_error(error, context["level"], context["family"], tolerances.get(name, 0))
                key = group + "/" + name
                terms[key] = torch.exp(-error.square().mean(-1) / self.terms[key].params["std"] ** 2)
        root_R, ref_R = matrix_from_quat(state[:, 3:7]), matrix_from_quat(context["ref_rotation"][:, 0])
        heading = yaw_quat(context["ref_rotation"][:, 0])
        reference_velocity = quat_apply_inverse(heading, context["ref_velocity"][:, 0])
        reference_error = quat_apply_inverse(heading, context["ref_position"] - state[:, :3])
        desired = quat_apply(heading, corrected_planar_velocity(reference_velocity, reference_error))
        height_error = target_root[:, 2] - state[:, 2]
        for name, error in dict(
            root_orientation=rotation_error(root_R, ref_R).square(),
            root_velocity=(desired[:, :2] - state[:, 7:9]).square().sum(-1),
            floating_anchor=height_error.square(),
        ).items():
            key = "auxiliary/" + name
            terms[key] = torch.exp(-error / self.terms[key].params.get("std", 1.0) ** 2)
        acceleration = (velocity - previous_velocity) / self.dt
        reference_acceleration = (context["ref_body_velocity"] - previous_context["ref_body_velocity"]) / self.dt
        terms["auxiliary/vertical_acceleration"] = (
            acceleration[:, c.motion_anchor_body_index, 2].square() * derivative_valid
        )
        ee = c.foot_ids + c.hand_ids
        terms["auxiliary/ee_acceleration"] = (acceleration[:, ee] - reference_acceleration[:, ee]).square().sum(
            -1
        ).mean(-1) * derivative_valid
        terms["auxiliary/action_rate"] = (action - previous_action).square().sum(-1)
        joints = state[:, 13:42]
        terms["auxiliary/joint_limits"] = (
            (self.limits[:, 0] - joints).clamp_min(0) + (joints - self.limits[:, 1]).clamp_min(0)
        ).sum(-1)
        contact, previous_contact = contact.detach(), previous_contact.detach()
        reference, valid = context["ref_contact"], context["ref_valid"]
        unexpected = (1 - reference) * valid
        unexpected = unexpected.clone()
        unexpected[:, c.foot_ids] = 0
        terms["auxiliary/undesired_contact"] = (contact * unexpected).sum(-1)
        normal, normal_valid = self.surface_normal(position[:, c.foot_ids, :2])
        terms["terrain/slip"] = (
            tangent_velocity(velocity[:, c.foot_ids], normal).square().sum(-1) * contact[:, c.foot_ids]
        ).sum(-1)
        terms["terrain/match"] = ((reference * contact + (1 - reference) * (1 - contact)) * valid).sum(-1) / valid.sum(
            -1
        ).clamp_min(1)
        changed = contact * (1 - previous_contact) + (1 - contact) * previous_contact
        mismatch = reference * (1 - contact) + (1 - reference) * contact
        terms["terrain/switch"] = (changed[:, c.foot_ids] * mismatch[:, c.foot_ids] * valid[:, c.foot_ids]).mean(
            -1
        ) * derivative_valid
        offsets = position.new_tensor([[0, 0], [-0.08, 0], [0.08, 0], [0, -0.04], [0, 0.04]])
        h, quality_valid = self.heightfield.query(position[:, c.foot_ids, None, :2] + offsets)
        quality = torch.exp(-((h.amax(-1) - h.amin(-1)) / 0.04).square()) * normal[..., 2].clamp(0, 1)
        touchdown = contact[:, c.foot_ids] * (1 - previous_contact[:, c.foot_ids]) * derivative_valid[:, None]
        terms["terrain/touchdown"] = (quality * touchdown * quality_valid.all(-1)).mean(-1)
        reward = sum(value * self.terms[key].weight for key, value in terms.items()) * self.dt
        body_error = errors["position"].clone()
        if c.cfg.enable_tolerance:
            body_error[:, c.lower_ids] = tolerance_error(
                body_error[:, c.lower_ids], context["level"], context["family"], c.cfg.tolerance_position_m
            )
        tilt_cos = (root_R[:, 2] * ref_R[:, 2]).sum(-1)
        survival = torch.sigmoid((self.thresholds["root_height"] - height_error.abs()) / 0.03)
        survival = survival * torch.sigmoid((tilt_cos - math.cos(self.thresholds["root_tilt"])) / 0.04)
        survival = survival * torch.sigmoid((self.thresholds["body_error"] - body_error.amax(-1)) / 0.04)
        inside = ((state[:, :2] - context["tile_origin"][:, :2]).abs() < self.tile_half).all(-1)
        return reward, survival * inside * normal_valid.all(-1), terms
