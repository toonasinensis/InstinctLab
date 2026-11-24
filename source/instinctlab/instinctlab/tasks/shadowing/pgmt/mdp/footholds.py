import torch
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from instinctlab.tasks.shadowing.pgmt.tensor_math import quat_apply, quat_apply_inverse

from ..heightfield import HeightField
from .contact_labels import collision_vertices


class FootholdReference:
    def __init__(self, command):
        c = command
        self.bank = c.motion._all_motion_sequences
        self.starts, self.lengths = c.label_starts, c.label_lengths
        self.feet = c.foot_ids
        self.foot_index = torch.tensor(self.feet, device=c.device)
        self.device, self.dt = c.device, c.env.step_dt
        self.heightfield = getattr(c.env, "pgmt_heightfield", None)
        if self.heightfield is None:
            self.heightfield = c.env.pgmt_heightfield = HeightField.from_context(
                c.terrain, c.env.cfg.heightfield_resolution
            )

        contact = c.labels[:, self.feet] & c.label_valid[:, self.feet]
        index = torch.arange(len(contact), device=self.device)[:, None]
        touchdown = contact & ~torch.cat((torch.zeros_like(contact[:1]), contact[:-1]))
        touchdown[self.starts] = contact[self.starts]
        self.last_touchdown = torch.where(touchdown, index, -1).cummax(0).values
        self.last_contact = torch.where(contact, index, -1).cummax(0).values
        self.next_contact = torch.where(contact, index, len(contact)).flip((0,)).cummin(0).values.flip((0,))
        self.stance = contact
        self.offsets = torch.arange(-4, 5, device=self.device) * 0.03
        vertices = collision_vertices(c.reference.cfg.robot_model_path)
        feet = [
            torch.as_tensor(vertices[name][:, :2], device=self.device, dtype=torch.float32)
            for name in c.cfg.foot_body_names
        ]
        lo, hi = torch.stack([foot.amin(0) for foot in feet]), torch.stack([foot.amax(0) for foot in feet])
        self.sole = torch.stack(
            (
                torch.zeros_like(lo),
                lo,
                hi,
                torch.stack((lo[:, 0], hi[:, 1]), -1),
                torch.stack((hi[:, 0], lo[:, 1]), -1),
            ),
            dim=1,
        )
        self.local_x = self.sole.new_tensor([1, 0, 0])
        self.local_y = self.sole.new_tensor([0, 1, 0])
        body_names = c.reference.cfg.link_of_interests
        names = c.robot.joint_names
        side = ("left", "right")
        joints = ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
        self.joints = torch.tensor(
            [[names.index(f"{s}_{name}_joint") for name in joints] for s in side], device=self.device
        )
        self.hip_roll = [body_names.index(f"{s}_hip_roll_link") for s in side]
        self.knees = [body_names.index(f"{s}_knee_link") for s in side]
        root = ET.parse(Path(c.reference.cfg.robot_model_path)).getroot()
        hip_joints = [root.find(f"./joint[@name='{s}_hip_pitch_joint']") for s in side]
        origins = [j.find("origin") for j in hip_joints]
        self.hip_offset = self.sole.new_tensor([[float(x) for x in o.attrib["xyz"].split()] for o in origins])
        hip_rotation = quat_from_euler_xyz(
            *self.sole.new_tensor([[float(x) for x in o.attrib["rpy"].split()] for o in origins]).unbind(-1)
        )
        axis = self.sole.new_tensor([[float(x) for x in j.find("axis").get("xyz").split()] for j in hip_joints])
        self.pitch_axis = quat_apply(hip_rotation, axis)
        self.damping = torch.eye(3, device=self.device) * 0.045**2
        self.limits = c.robot.data.soft_joint_pos_limits.torch[0, self.joints].clone()

    def landing(self, motion, frames, placement, offset):
        motion = motion[:, None, None].expand_as(frames)
        pos = (
            self.bank.link_pos_w[motion, frames]
            .gather(-2, self.foot_index[None, None, :, None, None].expand(*frames.shape, 1, 3))
            .squeeze(-2)
        )
        orientation = (
            self.bank.link_quat_w[motion, frames]
            .gather(-2, self.foot_index[None, None, :, None, None].expand(*frames.shape, 1, 4))
            .squeeze(-2)
        )
        rotation = quat_mul(placement[:, None, None].expand_as(orientation), orientation)
        xy = quat_apply(placement[:, None, None].expand_as(orientation), pos)[..., :2] + offset[:, None, None, :2]
        forward = quat_apply(rotation, self.local_x.expand_as(pos))[..., :2]
        forward = torch.nn.functional.normalize(forward, dim=-1)
        lateral = torch.stack((-forward[..., 1], forward[..., 0]), -1)
        shift = forward[..., None, :] * self.offsets[:, None]
        sole = forward[..., None, :] * self.sole[..., :1] + lateral[..., None, :] * self.sole[..., 1:]
        points = xy[..., None, None, :] + shift[..., :, None, :] + sole[..., None, :, :]
        heights, valid = self.heightfield.query(points)
        safe = valid.all(-1) & ((heights.amax(-1) - heights.amin(-1)) <= 0.025)
        score = torch.where(safe, self.offsets.abs(), float("inf"))
        choice = score.argmin(-1, keepdim=True)
        patch_height = heights[..., 0].gather(-1, choice).squeeze(-1)
        shift = shift.gather(-2, choice[..., None].expand(*choice.shape, 2)).squeeze(-2)
        return shift, patch_height, safe.any(-1)

    def solve(self, delta, q, root_q, root_pos, link_pos, link_q):
        pelvis = root_q[..., None, :].expand(*root_q.shape[:-1], len(self.feet), 4)
        hip_pitch = root_pos[..., None, :] + quat_apply(pelvis, self.hip_offset.expand(*pelvis.shape[:-1], 3))
        centers = torch.stack((hip_pitch, link_pos[..., self.hip_roll, :], link_pos[..., self.knees, :]), -2)
        axes = torch.stack(
            (
                quat_apply(pelvis, self.pitch_axis.expand(*pelvis.shape[:-1], 3)),
                quat_apply(link_q[..., self.hip_roll, :], self.local_x.expand(*pelvis.shape[:-1], 3)),
                quat_apply(link_q[..., self.knees, :], self.local_y.expand(*pelvis.shape[:-1], 3)),
            ),
            -2,
        )
        jacobian = torch.linalg.cross(axes, link_pos[..., self.feet, None, :] - centers, dim=-1).transpose(-1, -2)
        transpose = jacobian.transpose(-1, -2)
        target = torch.linalg.solve_ex(jacobian @ transpose + self.damping, delta[..., None]).result
        angles = (transpose @ target).squeeze(-1).clamp(-0.5, 0.5)
        angular = (axes * angles[..., None]).sum(-2)
        foot = link_q[..., self.feet, :]
        ankle_pitch = -(quat_apply(foot, self.local_y.expand_as(angular)) * angular).sum(-1)
        ankle_roll = -(quat_apply(foot, self.local_x.expand_as(angular)) * angular).sum(-1)
        angles = torch.cat((angles, torch.stack((ankle_pitch, ankle_roll), -1).clamp(-0.45, 0.45)), -1)
        base = q[..., self.joints]
        angles = angles.clamp((self.limits[..., 0] - base).clamp_max(0), (self.limits[..., 1] - base).clamp_min(0))
        joint_delta = q.new_zeros(q.shape)
        joint_delta[..., self.joints.flatten()] = angles.flatten(-2)
        return joint_delta

    @torch.no_grad()
    def window(self, motion, frames, placement, offset, family):
        lengths = self.lengths[motion, None]
        frames = torch.minimum(frames.clamp_min(0), lengths - 1)
        full = self.starts[motion, None]
        index = full + frames
        first, last = full[..., None], (full + lengths - 1)[..., None]
        stance = self.stance[index]
        takeoff = self.last_contact[index]
        touchdown = self.next_contact[index]
        previous = self.last_touchdown[index]
        phases_valid = (takeoff >= first) & (touchdown <= last) & (touchdown - takeoff <= round(0.65 / self.dt))
        previous = previous.minimum(last).maximum(first) - first
        touchdown = touchdown.minimum(last).maximum(first) - first
        phase = ((index[..., None] - takeoff) / (full[..., None] + touchdown - takeoff).clamp_min(1)).clamp(0, 1)
        shift0, h0, safe0 = self.landing(motion, previous, placement, offset)
        shift1, h1, safe1 = self.landing(motion, touchdown, placement, offset)
        smooth = phase.square() * (3 - 2 * phase)
        zphase = torch.where(h1 >= h0, phase / 0.60, (phase - 0.40) / 0.60).clamp(0, 1)
        rise = h1 - h0
        vertical = zphase.square() * (3 - 2 * zphase)
        lift = smooth * (1 - smooth) * (rise.abs() * 0.8).clamp_max(0.16)
        xy = torch.lerp(shift0, shift1, smooth[..., None])
        height = torch.lerp(h0, h1, vertical) + lift
        root_pos = self.bank.base_pos_w[motion[:, None], frames]
        root_quat = self.bank.base_quat_w[motion[:, None], frames]
        link_pos = self.bank.link_pos_w[motion[:, None], frames]
        link_quat = self.bank.link_quat_w[motion[:, None], frames]
        rotation = placement[:, None].expand_as(root_quat)
        p = quat_apply(rotation, root_pos)[..., :2] + offset[:, None, :2]
        heading = quat_apply(quat_mul(rotation, root_quat), self.local_x.expand_as(root_pos))[..., :2]
        points = p[..., None, :] + heading[..., None, :] * self.sole.new_tensor([-0.12, 0, 0.12])[:, None]
        base_height, base_valid = self.heightfield.query(points)
        baseline = base_height.mean(-1)
        height = torch.where(stance, h0, height)
        xy = torch.where(stance[..., None], shift0, xy)
        use = base_valid.all(-1)[..., None] & safe0 & (stance | (phases_valid & safe1))
        use &= (family == 2)[:, None, None] & (self.last_touchdown[index] >= first)
        delta = torch.cat((xy, (height - baseline[..., None]).clamp(-0.28, 0.28)[..., None]), -1)
        delta = torch.where(use[..., None], delta, 0)
        delta *= ((height - baseline[..., None]).abs() > 0.012)[..., None] | (xy.norm(dim=-1) > 0.005)[..., None]
        local = quat_apply_inverse(rotation[..., None, :], delta)
        q = self.bank.joint_pos[motion[:, None], frames]
        correction = self.solve(local, q, root_quat, root_pos, link_pos, link_quat)
        rate = ((correction[:, 1:] - correction[:, :-1]) / self.dt).clamp(-8, 8)
        return delta[:, :-1], correction[:, :-1], rate
