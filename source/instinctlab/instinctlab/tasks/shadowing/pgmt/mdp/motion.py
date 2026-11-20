"""LAFAN named-NPZ input for the existing motion-reference pipeline."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, cast

from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, convert_quat, quat_apply_inverse, quat_inv, quat_mul

from instinctlab.motion_reference import MotionSequence
from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.utils.math import quat_angular_velocity
from instinctlab.utils.torch import ConcatBatchTensor

from ..selection import select_motion_files
from .contact_labels import collision_vertices, labels_from_pose

if TYPE_CHECKING:
    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def _named_motion_sequence(joint_pos, joint_vel, bodies, base, rate):
    position, rotation, linear, angular = bodies
    base_position, base_rotation, base_linear, base_angular = base
    root_q = base_rotation[:, None].expand_as(rotation)
    relative_position = position - base_position[:, None]
    return MotionSequence(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_pos_w=base_position,
        base_quat_w=base_rotation,
        base_lin_vel_w=base_linear,
        base_ang_vel_w=base_angular,
        link_pos_w=position,
        link_quat_w=rotation,
        link_lin_vel_w=linear,
        link_ang_vel_w=angular,
        link_pos_b=quat_apply_inverse(root_q, relative_position),
        link_quat_b=quat_mul(quat_inv(root_q), rotation),
        link_lin_vel_b=quat_apply_inverse(
            root_q,
            linear
            - base_linear[:, None]
            - torch.cross(base_angular[:, None].expand_as(relative_position), relative_position, dim=-1),
        ),
        link_ang_vel_b=quat_apply_inverse(root_q, angular - base_angular[:, None]),
        framerate=torch.tensor(rate),
        buffer_length=torch.tensor(len(position)),
    )


class LafanNamedMotion(AmassMotion):
    """Preserve supplied velocities and link poses instead of differentiating them again."""

    def _refresh_motion_file_list(self) -> None:
        manifest_path = Path(self.cfg.manifest).expanduser()
        manifest = json.loads(manifest_path.read_text())
        files = select_motion_files([f["name"] for f in manifest["files"]], self.cfg.motion_names)
        self._all_motion_files = [str(Path(self.cfg.path).expanduser() / name) for name in files]
        self._init_motion_weights()

    def _read_motion_file(self, motion_file_idx: int) -> MotionSequence:
        path = Path(self._all_motion_files[motion_file_idx])
        with np.load(path, allow_pickle=False) as source:
            joints = [source["joint_names"].tolist().index(n) for n in self.isaac_joint_names]
            links = [source["body_names"].tolist().index(n) for n in self.link_of_interests]
            root = source["body_names"].tolist().index("pelvis")

            def tensor(name, ids):
                return torch.as_tensor(source[name][:, ids], dtype=torch.float32, device=self.buffer_device)

            position = tensor("body_pos_w", links)
            rotation = convert_quat(tensor("body_quat_w", links), to="xyzw")
            base_position = tensor("body_pos_w", root)
            base_rotation = convert_quat(tensor("body_quat_w", root), to="xyzw")
            linear = tensor("body_lin_vel_w", links)
            angular = tensor("body_ang_vel_w", links)
            base_linear = tensor("body_lin_vel_w", root)
            base_angular = tensor("body_ang_vel_w", root)
            rate = float(source["fps"].item())
            if rate != 50.0:
                raise ValueError(f"Expected 50 Hz LAFAN: {path}")
            return _named_motion_sequence(
                tensor("joint_pos", joints),
                tensor("joint_vel", joints),
                (position, rotation, linear, angular),
                (base_position, base_rotation, base_linear, base_angular),
                rate,
            )

    def _init_motion_bin_weights(self) -> None:
        # Include the final partial bin, reserving the entire 11-frame reference window.
        lengths = self._all_motion_sequences.buffer_length.to(self.buffer_device)
        counts = ((lengths - 11) // 50 + 1).long()
        self._motion_bin_weights = ConcatBatchTensor(batch_sizes=counts, data_shape=tuple(), device=self.buffer_device)
        self._motion_bin_weights._concatenated_tensor.fill_(1.0)

    def _sample_as_concat_motion_bins(self, assigned_ids: torch.Tensor) -> None:
        weights = self._motion_bin_weights
        selected = torch.multinomial(weights._concatenated_tensor, len(assigned_ids), replacement=True)
        motion_ids = torch.searchsorted(weights._batch_starts[1:], selected, right=True)
        bins = selected - weights._batch_starts[motion_ids]
        lengths = self._all_motion_sequences.buffer_length[motion_ids]
        upper = torch.minimum((bins + 1) * 50 - 1, lengths - 11)
        lower = torch.maximum(bins * 50, torch.full_like(bins, 5)).minimum(upper)
        frames = (
            lower + torch.floor(torch.rand(len(assigned_ids), device=self.buffer_device) * (upper - lower + 1)).long()
        )
        self._assigned_env_motion_selection[assigned_ids] = motion_ids
        self._motion_buffer_start_time_s[assigned_ids] = frames.float() / 50.0


@configclass
class LafanNamedMotionCfg(AmassMotionCfg):
    class_type: type | str = "{DIR}.motion:LafanNamedMotion"
    manifest: str = ""
    motion_names: list[str] | None = None
    motion_bin_length_s: float = 1.0
    env_starting_stub_sampling_strategy: str = "concat_motion_bins"
    ensure_link_below_zero_ground: bool = False
    motion_start_height_offset: float = 0.0
    motion_interpolate_func = None
    velocity_estimation_method = None


class PGMTGroupedH5Motion(LafanNamedMotion):
    cfg: PGMTGroupedH5MotionCfg

    def _refresh_motion_file_list(self):
        from instinctlab.motion_reference.motion_files.grouped_h5 import read_grouped_motion_index

        self._source = Path(self.cfg.path).expanduser().resolve()
        names, lengths = read_grouped_motion_index(
            self._source, 50.0, self.isaac_joint_names, motion_names=self.cfg.motion_names
        )
        for name, length in zip(names, lengths, strict=True):
            if length < 11:
                raise ValueError(f"PGMT requires at least 11 frames in H5 group {name!r}")
        self._all_motion_files = list(names)
        self._init_motion_weights()

    @torch.no_grad()
    def _load_motion_sequences(self):
        from instinctlab.motion_reference.motion_files.grouped_h5 import open_grouped_motion_cache

        if self.buffer_device != self.output_device:
            raise ValueError("PGMT grouped H5 requires the motion buffer on the environment device")
        manager = cast("MotionReferenceManagerBase", self.articulation_access)
        urdf = Path(manager.cfg.robot_model_path).expanduser().resolve()
        body_names = list(self.link_of_interests)
        root_index = body_names.index("pelvis")
        cache_dir = (
            Path(self.cfg.cache_dir).expanduser()
            if self.cfg.cache_dir is not None
            else Path(__file__).resolve().parents[8] / "InstinctLab-local" / "motion_cache"
        )
        vertices = None

        def preprocess(root_pos, root_wxyz, joint_pos, fps):
            nonlocal vertices
            if vertices is None:
                vertices = collision_vertices(urdf)
            return self._preprocess_clip(root_pos, root_wxyz, joint_pos, fps, root_index, vertices)

        with open_grouped_motion_cache(
            self._source,
            cache_dir,
            self.isaac_joint_names,
            body_names,
            50.0,
            _asset_identity(urdf),
            preprocess,
            motion_names=self._all_motion_files if self.cfg.motion_names is not None else None,
        ) as cache:
            self.motion_data_key = str(cache.attrs["data_key"])
            slices = zip(cache["starts"][:], cache["lengths"][:], strict=True)
            offsets = dict(zip(cache["motion_names"].asstr()[:], slices, strict=True))
            lengths = [int(offsets[name][1]) for name in self._all_motion_files]
            packed = MotionSequence.make_emtpy_concat_batch(
                lengths, self.articulation_access.max_dofs, len(body_names), self.buffer_device
            )
            packed.framerate.fill_(50.0)
            shape = (sum(lengths), len(body_names))
            contact = torch.empty(shape, device=self.buffer_device, dtype=torch.bool)
            valid = torch.empty_like(contact)
            bottom = torch.empty(shape, device=self.buffer_device, dtype=torch.float32)
            target = 0
            for index, name in enumerate(self._all_motion_files):
                start, count = offsets[name]
                rows = slice(start, start + count)

                def tensor(key):
                    return torch.as_tensor(cache[key][rows], device=self.buffer_device)

                bodies = (
                    tensor("body_pos_w"),
                    convert_quat(tensor("body_quat_w"), to="xyzw"),
                    tensor("body_lin_vel_w"),
                    tensor("body_ang_vel_w"),
                )
                motion = _named_motion_sequence(
                    tensor("joint_pos"),
                    tensor("joint_vel"),
                    bodies,
                    tuple(value[:, root_index] for value in bodies),
                    50.0,
                )
                for attr in MotionSequence.attrs_with_frame_dim:
                    getattr(packed, attr)[index] = getattr(motion, attr)
                destination = slice(target, target + count)
                for store, key in ((contact, "contact"), (valid, "valid"), (bottom, "bottom")):
                    store[destination] = tensor(key)
                target += count
        self._all_motion_sequences = packed
        self.reference_contact_labels = contact, valid, bottom

    def _preprocess_clip(self, root_pos, root_wxyz, joint_pos, fps, root_index, vertices):
        root_pos = torch.as_tensor(root_pos, device=self.buffer_device)
        root_quat = convert_quat(torch.as_tensor(root_wxyz, device=self.buffer_device), to="xyzw")
        joint_pos = torch.as_tensor(joint_pos, device=self.buffer_device)
        local = self.forward_kinematics_func(joint_pos)
        position, rotation = combine_frame_transforms(
            root_pos[:, None].expand_as(local[..., :3]),
            root_quat[:, None].expand_as(local[..., 3:]),
            local[..., :3],
            local[..., 3:],
        )
        position[:, root_index] = root_pos
        rotation[:, root_index] = root_quat
        dt = 1.0 / fps
        angular = torch.cat(
            (
                quat_angular_velocity(rotation[:1], rotation[1:2], dt),
                quat_angular_velocity(rotation[:-2], rotation[2:], 2 * dt),
                quat_angular_velocity(rotation[-2:-1], rotation[-1:], dt),
            )
        )
        arrays = {
            "joint_pos": joint_pos.cpu().numpy(),
            "joint_vel": torch.gradient(joint_pos, spacing=(dt,), dim=(0,))[0].cpu().numpy(),
            "body_pos_w": position.cpu().numpy(),
            "body_quat_w": convert_quat(rotation, to="wxyz").cpu().numpy(),
            "body_lin_vel_w": torch.gradient(position, spacing=(dt,), dim=(0,))[0].cpu().numpy(),
            "body_ang_vel_w": angular.cpu().numpy(),
        }
        arrays.update(
            labels_from_pose(
                arrays["body_pos_w"],
                arrays["body_quat_w"],
                arrays["body_lin_vel_w"],
                self.link_of_interests,
                vertices,
            )
        )
        return arrays


def _asset_identity(urdf):
    meshes = {mesh.attrib["filename"] for mesh in ET.parse(urdf).findall(".//collision/geometry/mesh")}
    hashes = {}
    for name in sorted(meshes):
        with (urdf.parent / name).open("rb") as file:
            hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
    with urdf.open("rb") as file:
        return {"preprocess_version": 1, "urdf": hashlib.file_digest(file, "sha256").hexdigest(), "meshes": hashes}


@configclass
class PGMTGroupedH5MotionCfg(LafanNamedMotionCfg):
    class_type: type | str = "{DIR}.motion:PGMTGroupedH5Motion"
    cache_dir: str | None = None
