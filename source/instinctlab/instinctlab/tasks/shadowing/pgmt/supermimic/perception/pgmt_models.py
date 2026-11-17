"""PGMT policy/WM adapters. Actor never consumes privileged height labels."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from instinctlab.tasks.shadowing.pgmt.supermimic.pgmt_models import PGMTDynamics, PGMTPolicy, PolicyNormalizer, mlp

from .config import DepthCfg
from .pgmt_perception import GeometricPerception, PGMTDepthEncoder


class PGMTDepthActor(nn.Module):
    def __init__(self, num_actions, perception=None):
        super().__init__()
        self.cfg = DepthCfg(**(perception or {}))
        self.encoder = PGMTDepthEncoder()
        width = self.cfg.history_width + self.cfg.reference_width + self.cfg.latent_width
        self.action_head = mlp((width, *self.cfg.actor_hidden, num_actions))
        self.register_buffer("terrain_gate", torch.tensor(1.0))
        self.amp, self.amp_heads = False, True

    def inputs(self, observation):
        if observation.shape[-1] != self.cfg.physical_width + self.cfg.image_width:
            raise ValueError("Depth policy expects physical observation followed by one raw image")
        physical = observation[..., : self.cfg.physical_width]
        image = observation[..., self.cfg.physical_width :].reshape(-1, *self.cfg.image_shape)
        return physical, image

    def forward(self, observation):
        physical, image = self.inputs(observation)
        self.intent = self.encoder(image)
        width = self.cfg.history_width + self.cfg.reference_width
        return self.action_from_inputs(physical[..., :width], self.intent)

    def action(self, observation):
        return self.forward(observation)

    def action_from_inputs(self, history_reference, latent):
        if self.cfg.actor_latent_detach:
            latent = latent.detach()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and latent.is_cuda):
            return self.action_head(torch.cat((history_reference, latent), -1)).float()


class DepthPolicyNormalizer(PolicyNormalizer):
    def __init__(self, shape, scan_width, perception, **kwargs):
        self.perception = perception
        if shape != perception.physical_width + perception.image_width:
            raise ValueError(f"Unexpected native depth observation shape: {shape}")
        super().__init__(perception.physical_width, scan_width, **kwargs)

    def update(self, observation):
        super().update(observation[..., : self.perception.physical_width])

    def forward(self, observation):
        return super().forward(observation[..., : self.perception.physical_width])


class PGMTDepthPolicy(PGMTPolicy):
    def forward(self, observation):
        if isinstance(observation, torch.Tensor):
            return self.collect(observation)
        physical = self.normalizer(observation["physical"])
        history_reference = physical[..., : self.network.cfg.history_width + self.network.cfg.reference_width]
        args = history_reference, observation["latent"]
        if self.execution.actor_recompute and torch.is_grad_enabled():
            return checkpoint(self.network.action_from_inputs, *args, use_reentrant=False, preserve_rng_state=False)
        return self.network.action_from_inputs(*args)

    def collect(self, observation):
        physical, image = self.network.inputs(observation)
        latent = self.network.encoder(image)
        self.network.intent = latent
        width = self.network.cfg.history_width + self.network.cfg.reference_width
        return self.network.action_from_inputs(self.normalizer(physical)[..., :width], latent)


class PGMTDepthDynamics(PGMTDynamics):
    def __init__(self, *args, encoder, perception, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual = GeometricPerception(encoder, perception)
        first = self.trunk[0]
        self.trunk[0] = nn.Linear(first.in_features + perception.latent_width, first.out_features)
        self.latent_head = nn.Linear(self.cfg.hidden_width, perception.latent_width)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)

    def predict_joint_delta(self, normalized, heights, latent):
        visual = self.visual
        z = (latent - visual.latent_mean) / visual.latent_std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.execution.bf16 and heights.is_cuda):
            hidden = self.trunk(torch.cat((normalized, self.terrain_encoder(heights), z), -1))
            if self.execution.bf16 and not self.execution.bf16_heads:
                hidden = hidden.float()
            delta, logits = self.predict_heads(hidden)
            return delta, logits, self.latent_head(hidden).float()

    def joint_transition(self, state, action, previous_action, heights, context, latent, *, recompute=False):
        normalized = self.input_norm(self.inputs(state, action, previous_action, heights, context))
        args = normalized, heights, latent
        if recompute and torch.is_grad_enabled():
            delta, logits, dz = checkpoint(
                self.predict_joint_delta, *args, use_reentrant=False, preserve_rng_state=False
            )
        else:
            delta, logits, dz = self.predict_joint_delta(*args)
        prediction = delta * self.delta_norm.std() + self.delta_norm.mean
        return self.representation.integrate(state, prediction), logits, latent + dz * self.visual.latent_delta_std

    def predict_transition(self, *args, **kwargs):
        raise RuntimeError("Depth WM requires a latent state; use joint_transition")
