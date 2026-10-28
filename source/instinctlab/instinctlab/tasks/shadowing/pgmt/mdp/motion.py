"""LAFAN named-NPZ input for the existing motion-reference pipeline."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.utils.math import convert_quat, quat_apply_inverse, quat_inv, quat_mul

from instinctlab.motion_reference import MotionSequence
from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.utils.torch import ConcatBatchTensor


class LafanNamedMotion(AmassMotion):
    """Preserve supplied velocities and link poses instead of differentiating them again."""

    def _refresh_motion_file_list(self) -> None:
        manifest_path = Path(self.cfg.manifest).expanduser()
        manifest = json.loads(manifest_path.read_text())
        self._all_motion_files = [str(Path(self.cfg.path).expanduser() / f["name"]) for f in manifest["files"]]
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
            root_q = base_rotation[:, None].expand_as(rotation)
            relative_position = position - base_position[:, None]
            rate = float(source["fps"].item())
            if rate != 50.0:
                raise ValueError(f"Expected 50 Hz LAFAN: {path}")
            return MotionSequence(
                joint_pos=tensor("joint_pos", joints),
                joint_vel=tensor("joint_vel", joints),
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
        lower = torch.maximum(bins * 50, torch.full_like(bins, 5))
        upper = torch.minimum((bins + 1) * 50 - 1, lengths - 11)
        frames = (
            lower + torch.floor(torch.rand(len(assigned_ids), device=self.buffer_device) * (upper - lower + 1)).long()
        )
        self._assigned_env_motion_selection[assigned_ids] = motion_ids
        self._motion_buffer_start_time_s[assigned_ids] = frames.float() / 50.0


@configclass
class LafanNamedMotionCfg(AmassMotionCfg):
    class_type: type | str = "{DIR}.motion:LafanNamedMotion"
    manifest: str = ""
    motion_bin_length_s: float = 1.0
    env_starting_stub_sampling_strategy: str = "concat_motion_bins"
    ensure_link_below_zero_ground: bool = False
    motion_start_height_offset: float = 0.0
    motion_interpolate_func = None
    velocity_estimation_method = None
