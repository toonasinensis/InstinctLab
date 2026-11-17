"""Read-only first-episode metrics; targets and failure flags come from the training task."""

import torch

ERROR_KEYS = ("error_upper_cm", "error_lower_cm", "error_all_cm", "error_pelvis_cm", "root_xy_cm")


class EpisodeMetrics:
    """Read post-physics state and native termination flags before the training task resets."""

    def __init__(self, env, record_trace=True):
        self.env = env
        self.command = env.command_manager.get_term("motion")
        self.finished = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.sums = torch.zeros(env.num_envs, 10, device=env.device, dtype=torch.float64)
        self.terms = tuple(env.termination_manager.active_terms)
        self.flags = torch.zeros(env.num_envs, len(self.terms), device=env.device, dtype=torch.bool)
        self.failed = self.finished.clone()
        self.reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float64)
        self.record_trace = record_trace
        self.trace, self.active_trace = [], []

    def __call__(self, dt):
        env, c = self.env, self.command
        aligned = (c.body_pos_relative_w - c.robot_body_pos_w).norm(dim=-1)
        world = (c.body_pos_w - c.robot_body_pos_w).norm(dim=-1)
        root = (c.anchor_pos_w - c.robot_anchor_pos_w).norm(dim=-1)
        xy = (c.anchor_pos_w[:, :2] - c.robot_anchor_pos_w[:, :2]).norm(dim=-1)
        values = 100 * torch.stack(
            [
                error[:, ids].mean(-1)
                for error in (aligned, world)
                for ids in (c.upper_ids, c.lower_ids, list(range(aligned.shape[1])))
            ]
            + [root, xy, aligned[:, c.motion_anchor_body_index], world[:, c.motion_anchor_body_index]],
            -1,
        )
        active = ~self.finished
        if not torch.isfinite(values[active]).all():
            raise FloatingPointError("Nonfinite physics state; abort instead of publishing a score")
        self.sums[active] += values[active].double()
        self.steps[active] += 1
        native_reward = torch.stack(tuple(env.reward_manager._reward_buf.values()), -1).sum(-1)
        self.reward[active] += native_reward[active].double()
        self.flags[active] |= torch.stack([env.termination_manager.get_term(n) for n in self.terms], -1)[active]
        self.failed[active] |= env.termination_manager.terminated[active]
        if self.record_trace:
            self.trace.append(values.detach().cpu())
            self.active_trace.append(active.detach().cpu())
        self.finished |= env.reset_buf

    def rows(self, cases):
        means = (self.sums / self.steps.clamp_min(1)[:, None]).cpu().numpy()
        result = []
        for i, case in enumerate(cases):
            flags = [name for j, name in enumerate(self.terms) if self.flags[i, j]]
            values = dict(zip(ERROR_KEYS, means[i, [0, 1, 2, 6, 7]].tolist()))
            values.update({"world_" + key: float(means[i, 3 + j]) for j, key in enumerate(ERROR_KEYS[:3])})
            result.append(
                dict(
                    **case,
                    **values,
                    fail=bool(self.failed[i]),
                    complete=bool(self.finished[i]),
                    duration_s=int(self.steps[i]) * self.env.step_dt,
                    steps=int(self.steps[i]),
                    termination_flags=flags,
                    native_env_return=float(self.reward[i]),
                )
            )
        return result
