"""Checkpoint native motion sampling with the policy and optimizer."""

import torch
import torch.distributed as dist

from instinct_rl.runners import OnPolicyRunner


class PGMTRunner(OnPolicyRunner):
    def _task_state(self):
        motion = self.env.unwrapped.scene["motion_reference"].motion_buffers["LAFAN"]
        curriculum = self.env.unwrapped.curriculum_manager._term_cfgs[
            self.env.unwrapped.curriculum_manager.active_terms.index("beyond_adaptive_sampling")
        ].func
        return {
            "num_envs": self.env.num_envs,
            "weights": motion._motion_bin_weights._concatenated_tensor.cpu(),
            "failures": curriculum.motion_bin_fail_counter._concatenated_tensor.cpu(),
            "pending_failures": curriculum.current_motion_bin_fail_counter._concatenated_tensor.cpu(),
            "learning_rate": self.alg.learning_rate,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device),
        }

    def save(self, path, infos=None):
        state = self._task_state()
        states = [None] * dist.get_world_size() if dist.is_initialized() else [state]
        if dist.is_initialized():
            dist.all_gather_object(states, state)
        super().save(path, infos={"schema": "pgmt-lafan-native-v1", "rank_states": states, "user": infos})

    def load(self, path):
        info = torch.load(path, map_location="cpu", weights_only=True)["infos"]
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        motion = self.env.unwrapped.scene["motion_reference"].motion_buffers["LAFAN"]
        if info["schema"] != "pgmt-lafan-native-v1" or len(info["rank_states"]) != world:
            raise ValueError("PGMT resume changed schema or rank count")
        state = info["rank_states"][rank]
        if state["num_envs"] != self.env.num_envs:
            raise ValueError("PGMT resume changed environment count")
        super().load(path)
        curriculum = self.env.unwrapped.curriculum_manager._term_cfgs[
            self.env.unwrapped.curriculum_manager.active_terms.index("beyond_adaptive_sampling")
        ].func
        motion._motion_bin_weights._concatenated_tensor.copy_(state["weights"])
        curriculum.motion_bin_fail_counter._concatenated_tensor.copy_(state["failures"])
        curriculum.current_motion_bin_fail_counter._concatenated_tensor.copy_(state["pending_failures"])
        self.alg.learning_rate = state["learning_rate"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        self.env.unwrapped.termination_manager.terminated.zero_()
        self.env.reset()
        return info["user"]
