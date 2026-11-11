"""Typed algorithm configuration. Task geometry/rewards remain in the environment config."""

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class OptimizerCfg:
    kind: Literal["adam", "adamw", "muon"] = "adam"
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_grad_value: float = 0.0


@dataclass(frozen=True)
class ExecutionCfg:
    compile: Literal["none", "selective", "chain"] = "selective"
    cuda_graph: bool = True
    capture_warmup: int = 3
    actor_recompute: bool = True
    wm_recompute: bool = True
    bf16: bool = False
    bf16_heads: bool = False


@dataclass(frozen=True)
class RepresentationCfg:
    kind: Literal["delta", "velocity", "structured"] = "velocity"
    loss_scale: Literal["running", "bryson"] = "bryson"
    nominal_error: tuple[float, ...] = (0.01, 0.02, 0.1, 0.3, 0.1, 3.14)
    error_floor: tuple[float, ...] = (0.003, 0.005, 0.02, 0.05, 0.003, 0.05)
    state_norm_epsilon: float = 1e-2
    delta_norm_epsilon: float = 1e-4
    height_clip_m: float = 5.0


@dataclass(frozen=True)
class DynamicsCfg:
    hidden_width: int = 2048
    hidden_layers: int = 3
    terrain_widths: tuple[int, ...] = (128, 64)
    predict_contact: bool = True
    contact_bce_weight: float = 0.1
    probability_epsilon: float = 1e-6
    contact_threshold: float = 0.5


@dataclass(frozen=True)
class PolicyLearnerCfg:
    horizon: int = 32
    batch_size: int = 4096
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    value_upper_bound: float | None = None
    value_gradient_weight: float = 0.0
    gradient_epsilon: float = 0.05
    gradient_batch_size: int = 4096
    gradient_scale_floor: float = 1e-3
    denominator_epsilon: float = 1e-12
    ema_decay: float = 0.995


@dataclass(frozen=True)
class WorldModelLearnerCfg:
    horizon: int = 8
    batch_size: int = 2048
    num_batches: int = 10
    holdout_stride: int = 8
    validation_horizons: tuple[int, ...] = (8, 16, 32)
    validation_batch: int = 256
    validation_interval: int = 100


@dataclass(frozen=True)
class CollectionCfg:
    replay_steps: int = 1000
    history_length: int = 10
    steps_per_update: float = 2.5
    initial_steps: int = 32
    warmup_iterations: int = 32
    action_noise: float = 0.1
    update_normalizer_during_warmup: bool = True
    sampling_attempts: int = 8
    sampling_oversample: int = 2


@dataclass(frozen=True)
class ScheduleCfg:
    duration: int = 250000
    min_factor: float = 0.03


@dataclass(frozen=True)
class SeedCfg:
    world_model: int = 1741
    replay: int = 2909
    actor: int = 8009
    collection: int = 9001
    critic: int = 10009
    value: int = 11003
    observation: int = 12007
    gradient: int = 15013


@dataclass(frozen=True)
class SuperMimicCfg:
    representation: RepresentationCfg = field(default_factory=RepresentationCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)
    policy: PolicyLearnerCfg = field(default_factory=PolicyLearnerCfg)
    world_model: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
    collection: CollectionCfg = field(default_factory=CollectionCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    seeds: SeedCfg = field(default_factory=SeedCfg)
    actor_optimizer: OptimizerCfg = field(
        default_factory=lambda: OptimizerCfg(kind="muon", betas=(0.9, 0.95), clip_grad_value=0.1)
    )
    wm_optimizer: OptimizerCfg = field(default_factory=lambda: OptimizerCfg(kind="muon"))
    critic_optimizer: OptimizerCfg = field(default_factory=lambda: OptimizerCfg(kind="muon"))

    def validate(self):
        if self.execution.compile not in ("none", "selective", "chain"):
            raise ValueError("Unknown compilation mode")
        if self.representation.kind not in ("delta", "velocity", "structured"):
            raise ValueError("Unknown state representation")
        if self.representation.loss_scale not in ("running", "bryson"):
            raise ValueError("Unknown WM loss scale")
        if not 0 <= self.policy.ema_decay < 1 or not 0 < self.policy.gamma <= 1:
            raise ValueError("Invalid EMA decay or discount")
        if self.collection.steps_per_update < 1:
            raise ValueError("At least one real step is required per update")
        required = max(self.policy.horizon, self.world_model.horizon, *self.world_model.validation_horizons)
        if not required <= self.collection.initial_steps <= self.collection.replay_steps:
            raise ValueError("Initial collection and replay must cover every horizon")
        if (
            min(
                self.policy.horizon,
                self.policy.batch_size,
                self.policy.critic_epochs,
                self.policy.critic_minibatches,
                self.world_model.horizon,
                self.world_model.batch_size,
                self.world_model.num_batches,
                self.world_model.validation_interval,
                self.schedule.duration,
            )
            < 1
        ):
            raise ValueError("Horizons, batch sizes, update counts and schedule duration must be positive")
        if self.collection.history_length < 2:
            raise ValueError("Causal replay requires at least two history samples")
        if not 0 <= self.policy.lam <= 1 or not 0 <= self.schedule.min_factor <= 1:
            raise ValueError("Lambda and minimum LR factor must be in [0, 1]")
        if self.world_model.holdout_stride < 2:
            raise ValueError("WM holdout must leave training environments")
        if self.execution.cuda_graph and self.policy.value_gradient_weight:
            raise ValueError("Value-gradient updates currently require eager optimizer updates")


def from_dict(values: dict) -> SuperMimicCfg:
    defaults = asdict(SuperMimicCfg())
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown SuperMimic configuration sections: {sorted(unknown)}")
    defaults.update(values)
    result = SuperMimicCfg(
        representation=RepresentationCfg(**defaults["representation"]),
        dynamics=DynamicsCfg(**defaults["dynamics"]),
        policy=PolicyLearnerCfg(**defaults["policy"]),
        world_model=WorldModelLearnerCfg(**defaults["world_model"]),
        collection=CollectionCfg(**defaults["collection"]),
        execution=ExecutionCfg(**defaults["execution"]),
        schedule=ScheduleCfg(**defaults["schedule"]),
        seeds=SeedCfg(**defaults["seeds"]),
        actor_optimizer=OptimizerCfg(**defaults["actor_optimizer"]),
        wm_optimizer=OptimizerCfg(**defaults["wm_optimizer"]),
        critic_optimizer=OptimizerCfg(**defaults["critic_optimizer"]),
    )
    result.validate()
    return result
