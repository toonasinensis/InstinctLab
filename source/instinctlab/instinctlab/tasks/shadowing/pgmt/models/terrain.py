"""Terrain glimpses and a shared-intent grouped-value PGMT model."""

import torch
from torch import nn
from torch.nn import functional as F

from .cross_attention import CrossAttentionActor, CrossAttentionActorCritic, unpack_terms


class TerrainActor(CrossAttentionActor):
    def __init__(self, num_joints=29):
        super().__init__(num_joints)
        self.amp = False
        self.amp_heads = True
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
        action, self.intent = self.action_and_intent(obs)
        return action

    def action(self, obs):
        # BPTT has an independent critic: no shared-intent side effect inside
        # a compiled or activation-checkpointed policy forward.
        return self.action_and_intent(obs)[0]

    def action_and_intent(self, obs):
        return self.action_and_intent_from_inputs(*self.inputs(obs))

    def inputs(self, obs):
        history = unpack_terms(obs[:, :930], self.prop_widths, 10)
        reference = unpack_terms(obs[:, 930:1678], self.reference_widths, 11)
        return history, reference, obs[:, 1678:]

    def action_from_inputs(self, history, reference, heights):
        return self.action_and_intent_from_inputs(history, reference, heights)[0]

    def output_head(self, layers, features):
        if not self.amp or self.amp_heads:
            return layers(features)
        for index in range(len(layers) - 1):
            features = layers[index](features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return layers[-1](features.float())

    def action_and_intent_from_inputs(self, history, reference, heights):
        with torch.autocast(
            device_type=heights.device.type, dtype=torch.bfloat16, enabled=self.amp and heights.is_cuda
        ):
            tokens = self.state_embedding(history)
            state = self.history_attention(tokens[:, -1:], tokens)
            centers = self.output_head(
                self.terrain_selector, torch.cat((heights, state[:, 0], reference.flatten(1)), -1)
            ).float()
        # Sampling arithmetic remains FP32; amp_heads controls the final linear.
        centers = 0.8 * centers.tanh().reshape(-1, 4, 2)
        grid = centers[:, :, None, None] + self.patch_offsets[None, None]
        patches = F.grid_sample(
            heights.reshape(-1, 1, 21, 21), grid.reshape(-1, 20, 5, 2), padding_mode="border", align_corners=True
        ).reshape(-1, 4, 25)
        bias = heights.new_zeros(1, 16)
        bias[:, -4:] = self.terrain_gate.log()
        positions = torch.cat((self.key_positions, self.key_positions.new_zeros(4)))
        with torch.autocast(
            device_type=heights.device.type, dtype=torch.bfloat16, enabled=self.amp and heights.is_cuda
        ):
            terrain_tokens = self.terrain_encoder(torch.cat((patches, centers), -1))
            source = torch.cat((state, self.command_embedding(reference), terrain_tokens), 1)
            intent = self.intent_attention(state, source, self.query_positions, positions, attention_bias=bias)
            action = self.output_head(self.action_head, torch.cat((history[:, -1], intent[:, 0]), -1)).float()
        # PPO shares this graph with evaluate; BPTT uses the pure action entry.
        return action, intent[:, 0].float()


class TerrainActorCritic(CrossAttentionActorCritic):
    def __init__(self, *args, num_rewards=4, **kwargs):
        if num_rewards not in (3, 4):
            raise ValueError("PGMT terrain requires upper/lower/auxiliary and an optional terrain reward group")
        self.num_value_groups = num_rewards
        # One shared trunk with one output per active group.
        super().__init__(*args, num_rewards=1, **kwargs)

    def _build_actor(self, num_actions):
        if self.mlp_input_dim_a != 2119:
            raise ValueError(f"Expected 2119 policy inputs, got {self.mlp_input_dim_a}")
        return TerrainActor(num_actions)

    def _build_critic(self, num_values=1):
        dims = (self.mlp_input_dim_c + 256, *self.critic_hidden_dims, self.num_value_groups)
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(torch.cat((critic_observations, self.actor.intent), -1))
