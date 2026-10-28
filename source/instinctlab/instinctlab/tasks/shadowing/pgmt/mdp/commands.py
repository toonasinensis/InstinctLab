"""Preserve continuous-episode clip resampling for the flat LAFAN baseline."""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from instinctlab.envs.mdp.events.motion_reference import reset_robot_state_by_reference

RESET_VELOCITY = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


class LafanMotionLifecycle(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reference = env.scene["motion_reference"]
        self.robot = env.scene["robot"]
        self.body_ids = self.robot.find_bodies(self.reference.cfg.link_of_interests, preserve_order=True)[0]
        for name in ("error_body_pos", "error_joint_pos"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return torch.zeros(self.num_envs, 1, device=self.device)

    def _resample_command(self, env_ids):
        pass  # Scene reset and the reference reset event own initialization.

    def _update_command(self):
        remaining = self.reference.assigned_motion_lengths - self.reference._timestamp_torch
        ids = torch.where(remaining <= 0.2 + 1e-6)[0]
        if len(ids):
            self.reference.reset(ids)
            reset_robot_state_by_reference(
                self._env,
                ids,
                position_offset=[0.0, 0.0, 0.0],
                dof_vel_ratio=1.0,
                base_lin_vel_ratio=1.0,
                base_ang_vel_ratio=1.0,
                randomize_velocity_range=RESET_VELOCITY,
                randomize_joint_pos_range=(0.0, 0.0),
            )

    def _update_metrics(self):
        ref = self.reference.reference_frame
        self.metrics["error_body_pos"][:] = (
            (self.reference.reference_link_pos_relative_w - self.robot.data.body_pos_w.torch[:, self.body_ids])
            .norm(dim=-1)
            .mean(-1)
        )
        self.metrics["error_joint_pos"][:] = (ref.joint_pos[:, 0] - self.robot.data.joint_pos.torch).abs().mean(-1)


@configclass
class LafanMotionLifecycleCfg(CommandTermCfg):
    class_type: type | str = "{DIR}.commands:LafanMotionLifecycle"
    resampling_time_range: tuple = (1e9, 1e9)
