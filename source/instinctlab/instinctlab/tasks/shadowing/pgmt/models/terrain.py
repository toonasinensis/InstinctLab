"""Terrain glimpses and shared-intent four-value PGMT model."""

import torch
from torch import nn
from torch.nn import functional as F

from .cross_attention import CrossAttentionActor, CrossAttentionActorCritic, unpack_terms


class TerrainActor(CrossAttentionActor):
    def __init__(self, num_joints=29):
        super().__init__(num_joints)
        self.reference_widths = (*self.reference_widths, 3)
        self.command_embedding = nn.Linear(68, 256)
        self.terrain_selector = nn.Sequential(
            nn.Linear(441 + 256 + 748, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 8)
        )
        self.terrain_encoder = nn.Sequential(nn.Linear(27, 128), nn.SiLU(), nn.Linear(128, 256))
        with torch.no_grad():
            self.terrain_selector[-1].weight.mul_(0.01)
            self.terrain_selector[-1].bias.copy_(
                torch.atanh(torch.tensor([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]))
            )
        self.register_buffer("terrain_gate", torch.tensor(0.0))
        y, x = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing="ij")
        self.register_buffer("patch_offsets", torch.stack((x, y), -1).float() * 0.1)

    def forward(self, obs):
        history = unpack_terms(obs[:, :930], self.prop_widths, 10)
        reference = unpack_terms(obs[:, 930:1678], self.reference_widths, 11)
        heights = obs[:, 1678:]
        tokens = self.state_embedding(history)
        state = self.history_attention(tokens[:, -1:], tokens)
        centers = 0.8 * self.terrain_selector(
            torch.cat((heights, state[:, 0], reference.flatten(1)), -1)
        ).tanh().reshape(-1, 4, 2)
        grid = centers[:, :, None, None] + self.patch_offsets[None, None]
        patches = F.grid_sample(
            heights.reshape(-1, 1, 21, 21), grid.reshape(-1, 20, 5, 2), padding_mode="border", align_corners=True
        ).reshape(-1, 4, 25)
        terrain_tokens = self.terrain_encoder(torch.cat((patches, centers), -1))
        source = torch.cat((state, self.command_embedding(reference), terrain_tokens), 1)
        bias = obs.new_zeros(1, 16)
        bias[:, -4:] = self.terrain_gate.log()
        positions = torch.cat((self.key_positions, self.key_positions.new_zeros(4)))
        intent = self.intent_attention(state, source, self.query_positions, positions, attention_bias=bias)
        # PPO calls the actor immediately before evaluate; the graph is shared, never detached.
        self.intent = intent[:, 0]
        return self.action_head(torch.cat((history[:, -1], self.intent), -1))


class TerrainActorCritic(CrossAttentionActorCritic):
    def __init__(self, *args, num_rewards=4, **kwargs):
        if num_rewards != 4:
            raise ValueError("PGMT terrain requires upper/lower/auxiliary/terrain reward groups")
        # One shared trunk with four outputs, rather than four duplicate critic trunks.
        super().__init__(*args, num_rewards=1, **kwargs)

    def _build_actor(self, num_actions):
        if self.mlp_input_dim_a != 2119:
            raise ValueError(f"Expected 2119 policy inputs, got {self.mlp_input_dim_a}")
        return TerrainActor(num_actions)

    def _build_critic(self, num_values=1):
        dims = (self.mlp_input_dim_c + 256, *self.critic_hidden_dims, 4)
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(torch.cat((critic_observations, self.actor.intent), -1))
