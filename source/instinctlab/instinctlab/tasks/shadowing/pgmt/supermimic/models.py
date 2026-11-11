"""Neural models contain no training loop, simulator, or reward policy."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from instinctlab.tasks.shadowing.pgmt.wm_dynamics import RunningMoments

from .config import DynamicsCfg, ExecutionCfg
from .interfaces import DynamicsModel, Policy


def mlp(widths):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PGMTPolicy(Policy):
    """Reuse PGMT CNN/TF weights and observation semantics through one policy interface."""

    def __init__(self, network, normalizer, execution: ExecutionCfg):
        super().__init__()
        self.network, self.execution = network, execution
        object.__setattr__(self, "normalizer", normalizer)
        self.network.amp, self.network.amp_heads = execution.bf16, execution.bf16_heads

    def forward(self, observation):
        inputs = self.network.inputs(self.normalizer(observation))
        if self.execution.actor_recompute and torch.is_grad_enabled():
            return checkpoint(
                self.network.action_from_inputs,
                *inputs,
                use_reentrant=False,
                preserve_rng_state=False,
                early_stop=False
            )
        return self.network.action_from_inputs(*inputs)

    def collect(self, observation):
        # Native forward records encoder/selector diagnostics used by the viewer.
        return self.network(self.normalizer(observation))


class Dynamics(DynamicsModel):
    def __init__(
        self,
        representation,
        terrain,
        cfg: DynamicsCfg,
        context_width: int,
        contact_ids,
        scan_points: int,
        execution: ExecutionCfg,
    ):
        super().__init__()
        self.representation, self.cfg, self.execution = representation, cfg, execution
        object.__setattr__(self, "terrain", terrain)
        layout = representation.layout
        self.predict_contact, self.predict_force = cfg.predict_contact, False
        self.contact_input = False
        self.contact_indices = tuple(contact_ids)
        self.input_norm = RunningMoments(
            layout.feature_width + 2 * layout.joints + context_width, representation.cfg.state_norm_epsilon
        )
        self.delta_norm = RunningMoments(representation.prediction_dim, representation.cfg.delta_norm_epsilon)
        self.error_norm = (
            None
            if representation.cfg.kind == "delta"
            else RunningMoments(layout.delta.width, representation.cfg.delta_norm_epsilon)
        )
        self.terrain_encoder = mlp((scan_points, *cfg.terrain_widths))
        # The legacy terrain encoder ends with SiLU; retain that exact architecture.
        self.terrain_encoder.append(nn.SiLU())
        layers, width = [], self.input_norm.mean.numel() + cfg.terrain_widths[-1]
        for _ in range(cfg.hidden_layers):
            layers.extend((nn.Linear(width, cfg.hidden_width), nn.LayerNorm(cfg.hidden_width), nn.SiLU()))
            width = cfg.hidden_width
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(width, representation.prediction_dim)
        self.contact_head = nn.Linear(width, len(contact_ids)) if cfg.predict_contact else None
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def inputs(self, state, action, previous_action, heights, context):
        position = state[..., self.representation.layout.state["position"]]
        ground, _ = self.terrain.query(position[..., :-1])
        return torch.cat((self.representation.represent(state, ground), action, previous_action, context), -1)

    def predict_delta(self, normalized, heights):
        with torch.autocast(
            device_type=heights.device.type, dtype=torch.bfloat16, enabled=self.execution.bf16 and heights.is_cuda
        ):
            hidden = self.trunk(torch.cat((normalized, self.terrain_encoder(heights)), -1))
            if self.execution.bf16 and not self.execution.bf16_heads:
                with torch.autocast(device_type=heights.device.type, enabled=False):
                    return self.predict_heads(hidden.float())
            return self.predict_heads(hidden)

    def predict_heads(self, hidden):
        return self.delta_head(hidden).float(), (
            self.contact_head(hidden).float() if self.contact_head is not None else None
        )

    def forward(self, state, action, previous_action, heights, context, *, recompute=False):
        normalized = self.input_norm(self.inputs(state, action, previous_action, heights, context))
        if recompute and torch.is_grad_enabled():
            delta, logits = checkpoint(
                self.predict_delta, normalized, heights, use_reentrant=False, preserve_rng_state=False, early_stop=False
            )
        else:
            delta, logits = self.predict_delta(normalized, heights)
        prediction = delta * self.delta_norm.std() + self.delta_norm.mean
        return self.representation.apply_prediction(state, prediction), logits


class Value(nn.Module):
    def __init__(self, width: int, hidden: tuple[int, ...]):
        super().__init__()
        self.network = mlp((width, *hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features).squeeze(-1)
