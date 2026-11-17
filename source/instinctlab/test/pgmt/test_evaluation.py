"""Configuration inheritance and pre-reset first-episode accounting contracts."""

import copy
import torch
from types import SimpleNamespace as NS

import pytest

from instinctlab.tasks.shadowing.pgmt.supermimic.evaluation.cli import parse_args
from instinctlab.tasks.shadowing.pgmt.supermimic.evaluation.config import EnvironmentOverrides, apply_overrides
from instinctlab.tasks.shadowing.pgmt.supermimic.evaluation.metrics import EpisodeMetrics
from instinctlab.tasks.shadowing.pgmt.supermimic.evaluation.report import summarize, write_report


def environment_config():
    return NS(
        seed=7,
        episode_length_s=10,
        sim=NS(device="cpu"),
        scene=NS(
            num_envs=4096,
            terrain=NS(
                terrain_generator=NS(
                    num_cols=20,
                    sub_terrains={
                        "flat": NS(proportion=0.8, platform_width=1.3),
                        "boxes": NS(proportion=0.2, platform_width=1.0),
                    },
                )
            ),
            motion_reference=NS(motion_buffers={"LAFAN": NS(motion_names=None)}),
        ),
        commands=NS(motion=NS(terrain_sampling_weights={"boxes": 2.0})),
    )


def test_empty_overrides_preserve_all_training_fields():
    cfg = environment_config()
    before = copy.deepcopy(cfg)
    apply_overrides(cfg, EnvironmentOverrides())
    assert cfg == before


def test_overrides_are_explicit_and_preserve_geometry():
    cfg = environment_config()
    apply_overrides(cfg, EnvironmentOverrides(num_envs=32, seed=0, terrain_proportions={"flat": 1.0}))
    assert cfg.scene.num_envs == 32
    assert cfg.seed == 0
    assert cfg.scene.terrain.terrain_generator.sub_terrains["flat"].proportion == 1.0
    assert cfg.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion == 0.0
    assert cfg.scene.terrain.terrain_generator.sub_terrains["flat"].platform_width == 1.3
    assert cfg.commands.motion.terrain_sampling_weights == {"boxes": 2.0}
    assert cfg.scene.motion_reference.motion_buffers["LAFAN"].motion_names is None
    with pytest.raises(ValueError, match="Unknown terrain"):
        apply_overrides(cfg, EnvironmentOverrides(terrain_proportions={"typo": 1.0}))


def test_cli_defaults_inherit_and_flags_override_preset(tmp_path):
    argv = ["--task", "training-task", "--checkpoint", "model.pt", "--output", "out"]
    assert parse_args(argv).environment == EnvironmentOverrides()
    preset = tmp_path / "settings.yaml"
    preset.write_text("environment:\n  num_envs: 4000\n  seed: 42\n  terrain_level: 9\n")
    result = parse_args(argv + ["--settings", str(preset), "--num-envs", "32", "--seed", "0"])
    assert result.environment.num_envs == 32
    assert result.environment.seed == 0
    assert result.environment.terrain_level == 9


def test_terminal_step_counted_once_and_timeouts_are_not_failures():
    reference = torch.zeros(2, 2, 3)
    robot = reference.clone()
    robot[:, :, 0] = 0.1
    command = NS(
        body_pos_relative_w=reference,
        body_pos_w=reference,
        robot_body_pos_w=robot,
        anchor_pos_w=reference[:, 0],
        robot_anchor_pos_w=robot[:, 0],
        upper_ids=[0],
        lower_ids=[1],
        motion_anchor_body_index=0,
    )
    failed = torch.tensor([True, False])
    timeout = torch.tensor([False, False])
    env = NS(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        command_manager=NS(get_term=lambda _: command),
        termination_manager=NS(
            active_terms=["fall", "time_out"], terminated=failed, get_term=lambda n: failed if n == "fall" else timeout
        ),
        reward_manager=NS(_reward_buf={"native": torch.tensor([1.0, 2.0])}),
        reset_buf=torch.tensor([True, False]),
    )
    metric = EpisodeMetrics(env)
    metric(env.step_dt)
    failed.zero_()
    timeout[1] = True
    env.reset_buf[:] = True
    robot[0] = 100  # Reset state must not contaminate the completed first episode.
    robot[1, :, 0] = 0.3
    metric(env.step_dt)
    rows = metric.rows([{"terrain": "flat"}, {"terrain": "boxes"}])
    assert rows[0]["error_all_cm"] == pytest.approx(10)
    assert rows[1]["error_all_cm"] == pytest.approx(20)
    assert [r["steps"] for r in rows] == [1, 2]
    assert [r["fail"] for r in rows] == [True, False]
    assert [r["duration_s"] for r in rows] == [0.02, 0.04]
    assert [r["native_env_return"] for r in rows] == [1, 4]
    assert summarize(rows)["macro"]["fail_percent"] == 50


def test_report_serializes_native_environment_metadata(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    env_yaml = tmp_path / "training.yaml"
    env_yaml.write_text("seed: 42\n")
    args = NS(
        checkpoint=checkpoint,
        env_config=env_yaml,
        weights="ema",
        output=tmp_path,
        environment=EnvironmentOverrides(terrain_level=9),
        task="training-task",
    )
    native = NS(
        num_envs=1,
        step_dt=0.02,
        physics_dt=0.005,
        cfg=NS(decimation=4),
        command_manager=NS(get_term=lambda _: NS(flat_warmup_active=False)),
    )
    row = dict(
        terrain="flat",
        error_upper_cm=1.0,
        error_lower_cm=2.0,
        error_all_cm=1.5,
        error_pelvis_cm=3.0,
        root_xy_cm=2.0,
        native_env_return=4.0,
        fail=False,
        duration_s=1.0,
        termination_flags=["motion_end"],
    )
    metric = NS(rows=lambda _: [row], finished=torch.tensor([True]), trace=[], active_trace=[])
    report = write_report(args, NS(seed=42, episode_length_s=9999), {"iter": 30000}, native, metric, [], 1.0)
    assert report["envs"] == 1
    assert report["physics"] == dict(dt=0.005, control_dt=0.02, decimation=4, physics_handles_decimation=None)
    assert (tmp_path / "episodes.csv").exists()
    assert (tmp_path / "error_trace.pt").exists()
