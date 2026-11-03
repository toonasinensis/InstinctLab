"""Reference-aware failure rules. No recovery window or environment exemptions."""

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse


class InitialTimeOut(ManagerTermBase):
    """Stagger the first episode without changing its real age or later time limits."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        generator = torch.Generator(device=self.device).manual_seed(env.cfg.seed + 15013)
        self.limits = torch.randint(
            1, env.max_episode_length + 1, (self.num_envs,), device=self.device, generator=generator
        )

    def __call__(self, env):
        return (self.limits < env.max_episode_length) & (env.episode_length_buf >= self.limits)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        # The wrapper and runner both reset before stepping. Preserve the first
        # deadline across those calls; any actual episode ending consumes it.
        ids = ids[self._env.episode_length_buf[ids] > 0]
        self.limits[ids] = self._env.max_episode_length


def motion_end(env):
    c = env.command_manager.get_term("motion")
    return c.reference.assigned_motion_lengths - c.reference._timestamp_torch <= 0.2 + 1e-6


def tile_exit(env):
    return env.command_manager.get_term("motion").terrain.outside


def height_failure(env, threshold=0.4):
    c = env.command_manager.get_term("motion")
    return (c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]).abs() > threshold


def tilt_failure(env, threshold=1.0):
    c = env.command_manager.get_term("motion")
    gravity = c.robot.data.GRAVITY_VEC_W.torch
    ref = quat_apply_inverse(c.anchor_quat_w, gravity)
    actual = quat_apply_inverse(c.robot_anchor_quat_w, gravity)
    angle = torch.acos(torch.nn.functional.cosine_similarity(ref, actual).clamp(-1, 1))
    return angle > threshold


def body_failure(env, threshold=0.65):
    c = env.command_manager.get_term("motion")
    # Same aligned target and lower-body tolerance as reward; upper-body remains strict.
    error = (c.body_pos_relative_w - c.robot_body_pos_w).norm(dim=-1)
    error[:, c.lower_ids] = c.terrain.tolerance(error[:, c.lower_ids], c.cfg.tolerance_position_m)
    return error.amax(-1) > threshold
