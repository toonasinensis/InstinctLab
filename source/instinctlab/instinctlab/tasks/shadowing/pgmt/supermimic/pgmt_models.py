"""Neural models contain no training loop, simulator, or reward policy."""

import torch
import torch.distributed as dist
from dataclasses import replace
from torch import nn
from torch.utils.checkpoint import checkpoint

from instinct_rl.modules.normalizer import EmpiricalNormalization

from .config import DynamicsCfg, ExecutionCfg
from .models import Dynamics as SuperMimicDynamics
from .pgmt_representation import DeltaRepresentation


class RunningMoments(nn.Module):
    """SuperMimic's online global moments, collected independently of forwards."""

    def __init__(self, width, epsilon=1e-2, max_count=2e8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.epsilon = epsilon
        self.max_count = max_count

    @torch.no_grad()
    def update(self, values):
        if self.count >= self.max_count:
            return
        values = values.reshape(-1, self.mean.numel()).float()
        shifted = values - self.mean
        stats = torch.cat((shifted.sum(0), shifted.square().sum(0), values.new_tensor([len(values)])))
        if dist.is_initialized():
            dist.all_reduce(stats)
        total_batch = stats[-1]
        if total_batch == 0:
            return
        if not torch.isfinite(stats).all():
            raise FloatingPointError("Nonfinite world-model normalization data")
        summed, squared = stats[:-1].chunk(2)
        delta = summed / total_batch
        variance = (squared / total_batch - delta.square()).clamp_min(0)
        total = self.count + total_batch
        self.variance.copy_(
            (self.variance * self.count + variance * total_batch + delta.square() * self.count * total_batch / total)
            / total
        )
        self.mean.add_(delta * (total_batch / total))
        self.count.copy_(total)

    def std(self):
        return (self.variance + self.epsilon**2).sqrt()

    def forward(self, values):
        return (values - self.mean) / self.std()


def mlp(widths):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PGMTPolicy(nn.Module):
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

    def finish_collection(self):
        self.network.intent = self.network.intent.detach()

    def collect(self, observation):
        # Native forward records encoder/selector diagnostics used by the viewer.
        return self.network(self.normalizer(observation))


class PGMTDynamics(SuperMimicDynamics):
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
        nn.Module.__init__(self)
        self.representation, self.cfg, self.execution = representation, cfg, execution
        self.context_width = context_width
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

    def predict_transition(self, state, action, previous_action, heights, context, *, recompute=False):
        normalized = self.input_norm(self.inputs(state, action, previous_action, heights, context))
        if recompute and torch.is_grad_enabled():
            delta, logits = checkpoint(
                self.predict_delta, normalized, heights, use_reentrant=False, preserve_rng_state=False, early_stop=False
            )
        else:
            delta, logits = self.predict_delta(normalized, heights)
        prediction = delta * self.delta_norm.std() + self.delta_norm.mean
        return self.representation.integrate(state, prediction), logits

    def forward(self, rep_state, action, aux_state, sim_ctx, *, previous_action, recompute=False):
        state = self.representation.reconstruct_full_state(aux_state, rep_state)
        from instinctlab.tasks.shadowing.pgmt.tensor_math import yaw_angle

        heights, _ = self.terrain.local(
            state[..., self.representation.layout.state["position"]],
            yaw_angle(state[..., self.representation.layout.state["rotation"]]),
        )
        successor, logits = self.predict_transition(
            state, action, previous_action, heights, sim_ctx, recompute=recompute
        )
        return self.representation.extract_aux_state(successor), self.representation.represent(successor)


class Value(nn.Module):
    def __init__(self, width: int, hidden: tuple[int, ...]):
        super().__init__()
        self.network = mlp((width, *hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features).squeeze(-1)


class PolicyNormalizer(EmpiricalNormalization):
    def __init__(self, shape, scan_width, **kwargs):
        super().__init__(shape, **kwargs)
        self.scan_width = scan_width

    def update(self, x):
        with torch.inference_mode(False), torch.no_grad():
            super().update(x)

    def forward(self, x):
        normalized = super().forward(x)
        return torch.cat((normalized[..., : -self.scan_width], x[..., -self.scan_width :]), -1)


def initialize_shared_dynamics(dynamics, seed: int, substeps: int):
    """Keep shared hidden/contact tensors invariant to prediction-head width.

    This optional comparison rule preserves the established PGMT initialization;
    it does not belong to the supervised learner or run at any training update.
    """
    rep = dynamics.representation
    if not dynamics.cfg.match_delta_initialization or rep.cfg.kind != "velocity":
        return
    device = next(dynamics.parameters()).device
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        reference_rep = DeltaRepresentation(
            rep.layout, replace(rep.cfg, kind="delta"), rep.dt, substeps, rep.root_com
        ).to(device)
        reference = PGMTDynamics(
            reference_rep,
            dynamics.terrain,
            dynamics.cfg,
            dynamics.context_width,
            dynamics.contact_indices,
            dynamics.terrain_encoder[0].in_features,
            dynamics.execution,
        ).to(device)
    source = dict(reference.named_parameters())
    with torch.no_grad():
        for name, value in dynamics.named_parameters():
            if source[name].shape == value.shape:
                value.copy_(source[name])
