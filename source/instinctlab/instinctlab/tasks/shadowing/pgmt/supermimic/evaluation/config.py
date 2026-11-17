"""Restore training configuration and apply only explicitly requested evaluation overrides."""

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentOverrides:
    """None inherits the saved value; runtime phase/level changes are explicit too."""

    num_envs: int | None = None
    episode_length_s: float | None = None
    seed: int | None = None
    device: str | None = None
    terrain_level: int | None = None
    terrain_proportions: dict[str, float] | None = None
    terrain_columns: int | None = None
    terrain_sampling_weights: dict[str, float] | None = None
    motion_buffer: str = "LAFAN"
    motion_names: list[str] | None = None
    finish_flat_warmup: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    task: str
    checkpoint: Path
    env_config: Path
    output: Path
    environment: EnvironmentOverrides = field(default_factory=EnvironmentOverrides)
    weights: str = "ema"
    contacts: bool = False
    steps: int = 0
    smoke: bool = False
    observe_check: bool = False


def canonical(value):
    if hasattr(value, "to_dict"):
        return canonical(value.to_dict())
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if callable(value):
        return value.__module__ + ":" + value.__qualname__
    return value


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, prefix + "." + key if prefix else key))
        return result
    return {prefix: value}


def apply_overrides(cfg, overrides):
    """Environment fields are opt-in. Never infer a common cohort from the task name."""
    for value, target, name in (
        (overrides.num_envs, cfg.scene, "num_envs"),
        (overrides.episode_length_s, cfg, "episode_length_s"),
        (overrides.seed, cfg, "seed"),
        (overrides.device, cfg.sim, "device"),
    ):
        if value is not None:
            setattr(target, name, value)
    generator = cfg.scene.terrain.terrain_generator
    if overrides.terrain_columns is not None:
        generator.num_cols = overrides.terrain_columns
    if overrides.terrain_proportions is not None:
        unknown = overrides.terrain_proportions.keys() - generator.sub_terrains.keys()
        if unknown:
            raise ValueError(f"Unknown terrain families: {sorted(unknown)}")
        for name, terrain in generator.sub_terrains.items():
            terrain.proportion = overrides.terrain_proportions.get(name, 0.0)
    if overrides.terrain_sampling_weights is not None:
        cfg.commands.motion.terrain_sampling_weights = copy.deepcopy(overrides.terrain_sampling_weights)
    if overrides.motion_names is not None:
        cfg.scene.motion_reference.motion_buffers[overrides.motion_buffer].motion_names = list(overrides.motion_names)


def prepare_config(settings):
    from instinctlab.utils import instinct_rl_cli as cli

    saved_env = cli.load_saved_yaml(str(settings.env_config))
    # Dynamic mapping keys must be restored after the strict saved-config loader.
    weights = saved_env["commands"]["motion"].pop("terrain_sampling_weights", {})
    cfg, _ = cli.resolve_saved_cfg(settings.task, [], env_yaml=copy.deepcopy(saved_env), play_mode=False)
    cfg.commands.motion.terrain_sampling_weights = weights
    before = canonical(cfg)
    for section in ("observations", "terminations", "events", "actions"):
        if canonical(saved_env[section]) != before[section]:
            raise ValueError(f"Saved training {section} differs after loading; refuse silent defaults")
    apply_overrides(cfg, settings.environment)
    cfg.log_dir = str(settings.output)
    after = canonical(cfg)
    a, b = flatten(before), flatten(after)
    differences = {
        key: {"training": a.get(key), "evaluation": b.get(key)}
        for key in sorted(a.keys() | b.keys())
        if a.get(key) != b.get(key)
    }
    (settings.output / "config_diff.json").write_text(json.dumps(differences, indent=2) + "\n")
    (settings.output / "overrides.json").write_text(json.dumps(asdict(settings.environment), indent=2) + "\n")
    (settings.output / "training_env.yaml").write_bytes(settings.env_config.read_bytes())
    return cfg
