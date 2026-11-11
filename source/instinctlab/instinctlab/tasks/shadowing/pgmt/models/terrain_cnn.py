"""Full heightscan CNN shared by the PGMT actor and four-value critic."""

import torch
from torch import nn

from instinct_rl.modules.actor_critic import ActorCritic


def mlp(widths):
    layers = []
    for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class TerrainCNNActor(nn.Module):
    def __init__(self, num_actions, scan_shape=(21, 21)):
        super().__init__()
        self.scan_shape = tuple(scan_shape)
        pooled = ((scan_shape[0] + 3) // 4) * ((scan_shape[1] + 3) // 4) * 64
        self.amp = False
        self.amp_heads = True
        self.register_buffer("terrain_gate", torch.tensor(1.0))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(pooled, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
        )
        self.action_head = mlp((1710, 1024, 512, 256, num_actions))

    def encode(self, heights):
        return self.encoder(heights.reshape(-1, 1, *self.scan_shape))

    def inputs(self, obs):
        return obs[:, :1678], obs[:, 1678:]

    def action_from_inputs(self, history_reference, heights):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and heights.is_cuda):
            return self.action_head(torch.cat((history_reference, self.encode(heights)), -1)).float()

    def forward(self, obs):
        self.intent = self.encode(obs[:, 1678:])
        return self.action_head(torch.cat((obs[:, :1678], self.intent), -1))

    def action(self, obs):
        return self.action_from_inputs(*self.inputs(obs))


class TerrainCNNActorCritic(ActorCritic):
    def __init__(self, *args, num_rewards=4, scan_shape=(21, 21), **kwargs):
        self.scan_shape = tuple(scan_shape)
        if num_rewards not in (3, 4):
            raise ValueError("PGMT CNN requires three or four reward groups")
        self.num_value_groups = num_rewards
        super().__init__(*args, num_rewards=1, **kwargs)

    def _build_actor(self, num_actions):
        if self.mlp_input_dim_a != 1678 + self.scan_shape[0] * self.scan_shape[1] or num_actions != 29:
            raise ValueError("PGMT CNN policy inputs must match its configured scan shape and 29 actions")
        return TerrainCNNActor(num_actions, self.scan_shape)

    def _build_critic(self, num_values=1):
        if self.mlp_input_dim_c != 296:
            raise ValueError(f"Expected 296 privileged inputs, got {self.mlp_input_dim_c}")
        return mlp((328, *self.critic_hidden_dims, self.num_value_groups))

    def evaluate(self, critic_observations, **kwargs):
        # The PPO adapter stores privileged state + the SAME sample's raw policy scan.
        # Re-encoding here keeps minibatch gradients current and avoids actor-side caches.
        scan_width = self.scan_shape[0] * self.scan_shape[1]
        if critic_observations.shape[-1] != 296 + scan_width:
            raise ValueError("Critic requires paired privileged state (296) and policy heightscan (441)")
        state, heights = critic_observations.split((296, scan_width), dim=-1)
        return self.critic(torch.cat((state, self.actor.encode(heights)), -1))
