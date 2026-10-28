"""Stage-one CrossAttention math ported from whole_body_tracking/utils/pgmt_cross_attention.py."""

import torch
from torch import nn
from torch.nn import functional as F

from instinct_rl.modules.actor_critic import ActorCritic


def unpack_terms(value: torch.Tensor, widths: tuple[int, ...], frames: int) -> torch.Tensor:
    expected = sum(widths) * frames
    if value.ndim != 2 or value.shape[-1] != expected:
        raise ValueError(f"Expected [B,{expected}], got {tuple(value.shape)}")
    chunks = value.split([width * frames for width in widths], dim=-1)
    return torch.cat([chunk.reshape(-1, frames, width) for chunk, width in zip(chunks, widths)], dim=-1)


def apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs; value is [B, heads, tokens, head_dim]."""
    width = value.shape[-1]
    if width % 2:
        raise ValueError("RoPE requires an even head dimension")
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width)
    angle = positions.to(device=value.device, dtype=torch.float32)[:, None] * frequencies[None, :]
    cosine, sine = angle.cos().to(value.dtype), angle.sin().to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, feedforward: int):
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model must be divisible by heads with even head_dim")
        self.heads = heads
        self.query_norm = nn.LayerNorm(d_model)
        self.source_norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, feedforward), nn.SiLU(), nn.Linear(feedforward, d_model))

    def forward(
        self, query, source, query_positions=None, key_positions=None, attention_bias=None, return_weights=False
    ):
        q_input, kv_input = self.query_norm(query), self.source_norm(source)

        def split_heads(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)

        q, k, v = split_heads(self.q(q_input)), split_heads(self.k(kv_input)), split_heads(self.v(kv_input))
        if query_positions is not None:
            q, k = apply_rope(q, query_positions), apply_rope(k, key_positions)
        weights = None
        if return_weights:
            logits = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
            if attention_bias is not None:
                logits = logits + attention_bias
            weights = logits.softmax(dim=-1)
            attended = weights @ v
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(query.shape)
        result = query + self.out(attended)
        result = result + self.ff(self.ff_norm(result))
        return (result, weights) if return_weights else result


class CrossAttentionActor(nn.Module):
    def __init__(self, num_joints=29, d_model=256, num_heads=4, dim_feedforward=1024):
        super().__init__()
        self.prop_widths = (3, 3, num_joints, num_joints, num_joints)
        self.reference_widths = (num_joints, num_joints, 6, 1)
        self.state_embedding = nn.Linear(sum(self.prop_widths), d_model)
        self.command_embedding = nn.Linear(sum(self.reference_widths), d_model)
        self.history_attention = CrossAttentionBlock(d_model, num_heads, dim_feedforward)
        self.intent_attention = CrossAttentionBlock(d_model, num_heads, dim_feedforward)
        self.register_buffer("query_positions", torch.tensor([0.0]))
        self.register_buffer("key_positions", torch.tensor([0.0, *range(11)], dtype=torch.float32))
        dims = (sum(self.prop_widths) + d_model, 1024, 512, 256, num_joints)
        layers = []
        for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.action_head = nn.Sequential(*layers)

    def forward(self, obs):
        prop_dim = sum(self.prop_widths) * 10
        history = unpack_terms(obs[:, :prop_dim], self.prop_widths, 10)
        references = unpack_terms(obs[:, prop_dim:], self.reference_widths, 11)
        tokens = self.state_embedding(history)
        state = self.history_attention(tokens[:, -1:], tokens)
        source = torch.cat((state, self.command_embedding(references)), dim=1)
        intent = self.intent_attention(state, source, self.query_positions, self.key_positions)
        return self.action_head(torch.cat((history[:, -1], intent[:, 0]), dim=-1))


class CrossAttentionActorCritic(ActorCritic):
    """Instinct-RL PPO interface; the critic remains an independent scalar MLP."""

    def _build_actor(self, num_actions):
        expected = (6 + 3 * num_actions) * 10 + (2 * num_actions + 7) * 11
        if self.mlp_input_dim_a != expected:
            raise ValueError(f"Stage-one policy requires {expected} inputs, got {self.mlp_input_dim_a}")
        return CrossAttentionActor(num_actions)

    def _build_critic(self, num_values=1):
        dims = (self.mlp_input_dim_c, *self.critic_hidden_dims, num_values)
        layers = []
        for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)
