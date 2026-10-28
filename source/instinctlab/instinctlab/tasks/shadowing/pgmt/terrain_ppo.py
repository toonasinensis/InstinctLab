"""Four-return PPO with summed, globally normalized advantages and terminal bootstrap."""

import torch
import torch.distributed as dist

from instinct_rl.algorithms import PPO
from instinct_rl.storage.rollout_storage import RolloutStorage
from instinct_rl.utils.utils import get_subobs_size

from .mdp.terrain_math import grouped_gae


class TerrainStorage(RolloutStorage):
    def compute_returns(self, last_values, gamma, lam):
        returns, advantage = grouped_gae(self.rewards, self.values, self.dones, last_values, gamma, lam)
        self.returns.copy_(returns)
        moments = torch.stack((advantage.sum(), advantage.square().sum(), advantage.new_tensor(advantage.numel())))
        if dist.is_initialized():
            dist.all_reduce(moments)
        mean = moments[0] / moments[2]
        variance = (moments[1] / moments[2] - mean.square()).clamp_min(0)
        normalized = (advantage - mean) / (variance.sqrt() + 1e-8)
        # Base PPO averages the reward axis; repeating the summed advantage preserves its magnitude.
        self.advantages.copy_(normalized.expand_as(self.advantages))


class TerrainPPO(PPO):
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=4):
        self.transition = RolloutStorage.Transition()
        self.storage = TerrainStorage(
            num_envs,
            num_transitions_per_env,
            [get_subobs_size(obs_format["policy"])],
            [get_subobs_size(obs_format["critic"])],
            [num_actions],
            num_rewards=num_rewards,
            device=self.device,
        )

    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
        rewards = rewards.clone()
        timeout = infos["bootstrap_time_outs"].to(self.device)
        if timeout.any():
            terminal = infos["terminal_observations"]
            normalized = {}
            for key in ("policy", "critic"):
                normalizer = self.normalizers[key]
                was_training = normalizer.training
                normalizer.eval()
                normalized[key] = normalizer(terminal[key].to(self.device))
                normalizer.train(was_training)
            self.actor_critic.act_inference(normalized["policy"])
            values = self.actor_critic.evaluate(normalized["critic"]).detach()
            rewards += self.gamma * values * timeout[:, None]
        # Skip the parent's pre-action-value timeout approximation; rewards already bootstrap terminal V.
        info = {k: v for k, v in infos.items() if k != "time_outs"}
        super().process_env_step(rewards, dones, info, next_obs, next_extra_obs)

    def compute_returns(self, last_extra_obs):
        self.actor_critic.act_inference(last_extra_obs["policy"])
        super().compute_returns(last_extra_obs)
