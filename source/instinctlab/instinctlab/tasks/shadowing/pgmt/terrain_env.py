"""Task-local pre-termination reference update and pre-reset terminal observations."""

import torch

from instinctlab.envs.manager_based_rl_env import InstinctRlEnv


class PGMTTerrainEnv(InstinctRlEnv):
    def load_managers(self):
        # Place every initial articulation on a valid tile before observation shape probes raycast.
        robot = self.scene["robot"]
        state = robot.data.default_root_state.torch.clone()
        state[:, :3] += self.scene.env_origins
        robot.write_root_state_to_sim(state)
        self.sim.forward()
        self.scene.update(self.physics_dt)
        super().load_managers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        termination = self.termination_manager.compute
        reward = self.reward_manager.compute

        def compute_termination():
            self.command_manager.get_term("motion").advance()
            return termination()

        def compute_reward(dt):
            values = reward(dt)
            c = self.command_manager.get_term("motion")
            self.extras["step"] = {
                "PGMT/body_position_error_m": (c.body_pos_relative_w - c.robot_body_pos_w).norm(dim=-1).mean(),
                "PGMT/joint_position_error_rad": (c.joint_pos - c.robot_joint_pos).abs().mean(),
                "PGMT/terrain_level": c.terrain.level.float().mean(),
                "PGMT/terrain_gate": c.metrics["terrain_gate"].mean(),
                "PGMT/failures": self.reset_terminated.float().sum(),
                "PGMT/completions": (
                    (self.termination_manager.get_term("time_out") & ~self.reset_terminated).float().sum()
                ),
                "PGMT/episodes": self.reset_buf.float().sum(),
            }
            for key, (raw, effective) in c.reward_errors.items():
                self.extras["step"]["PGMT/raw/" + key] = raw.mean()
                self.extras["step"]["PGMT/effective/" + key] = effective.mean()
            self.extras["bootstrap_time_outs"] = (self.reset_time_outs & ~self.reset_terminated).clone()
            self.extras.pop("terminal_observations", None)
            if self.extras["bootstrap_time_outs"].any():
                # Take one terminal history frame and restore the buffers; normal compute appends once.
                saved = []
                for group in self.observation_manager._group_obs_term_history_buffer.values():
                    for buffer in group.values():
                        saved.append(
                            (
                                buffer,
                                buffer._need_reset,
                                buffer._num_pushes.clone(),
                                buffer._buffer.clone() if buffer._buffer is not None else None,
                            )
                        )
                try:
                    terminal = self.observation_manager.compute(update_history=True)
                    self.extras["terminal_observations"] = {
                        key: (
                            torch.cat([v.reshape(self.num_envs, -1) for v in value.values()], -1).clone()
                            if isinstance(value, dict)
                            else value.clone()
                        )
                        for key, value in terminal.items()
                    }
                finally:
                    for buffer, need_reset, pushes, data in saved:
                        buffer._need_reset = need_reset
                        buffer._num_pushes.copy_(pushes)
                        if data is not None:
                            buffer._buffer.copy_(data)
            return values

        self.termination_manager.compute = compute_termination
        self.reward_manager.compute = compute_reward
