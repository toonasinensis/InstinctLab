"""Command-line configuration for native-training-task evaluation."""

import argparse
import torch
from pathlib import Path

from .config import EnvironmentOverrides, EvaluationConfig, prepare_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settings", type=Path, help="YAML environment overrides; omitted fields inherit training")
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episode-length-s", type=float)
    parser.add_argument("--level", type=int, choices=range(10))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--contacts", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--observe-check", action="store_true", help="Smoke baseline with observer disabled")
    args = parser.parse_args(argv)
    if (args.steps or args.observe_check) and not args.smoke:
        parser.error("Partial or equivalence checks require --smoke")
    overrides = {}
    if args.settings:
        from instinctlab.utils.instinct_rl_cli import load_saved_yaml

        overrides = load_saved_yaml(str(args.settings))["environment"]
    for flag, field in (
        ("num_envs", "num_envs"),
        ("episode_length_s", "episode_length_s"),
        ("level", "terrain_level"),
        ("seed", "seed"),
        ("device", "device"),
    ):
        value = getattr(args, flag)
        if value is not None:
            overrides[field] = value
    return EvaluationConfig(
        task=args.task,
        checkpoint=args.checkpoint,
        env_config=args.env_config or args.checkpoint.parent / "params/env.yaml",
        output=args.output,
        environment=EnvironmentOverrides(**overrides),
        weights=args.weights,
        contacts=args.contacts,
        steps=args.steps,
        smoke=args.smoke,
        observe_check=args.observe_check,
    )


def main():
    settings = parse_args()
    settings.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml

    import instinctlab.tasks  # noqa: F401

    from .runner import run

    cfg = prepare_config(settings)
    dump_yaml(str(settings.output / "effective_env.yaml"), cfg)
    # Preserve the existing artifact name for downstream report tooling.
    dump_yaml(str(settings.output / "env.yaml"), cfg)
    torch.set_num_threads(2)
    with launch_simulation(cfg, dict(device=cfg.sim.device, visualizer="none")):
        run(settings, cfg, saved)
