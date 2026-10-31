"""Exercise terrain curriculum boundaries with real PGMT reset placement."""

import gymnasium as gym
import hashlib
import inspect
import json
import torch
from pathlib import Path

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.shadowing.pgmt.mdp.terrain_commands import TerrainMotion
from instinctlab.utils import instinct_rl_cli as cli


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    args.distributed = False
    cfg, agent = cli.resolve_train_cfg(args.task, overrides)
    cfg.scene.num_envs = args.num_envs or 128
    cfg.seed = agent.seed
    if cfg.scene.num_envs < 32:
        parser.error("At least32 environments are required for mixed reset cases")
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            env.reset()
            base = env.unwrapped
            c = base.command_manager.get_term("motion")
            t = base.scene.terrain
            assert c.cfg.enable_terrain_curriculum and t.max_terrain_level == 10
            assert c.sampling_contract["terrain_curriculum_version"] == "max_level_resample_v1"
            ids = torch.arange(base.num_envs, device=base.device)
            levels = torch.full_like(ids, 9)
            levels[0], levels[2], levels[8:11] = 8, 0, 4
            types = ids % len(c.terrain.column_names)
            types[:11] = 2  # Nonflat slope: exercise the platform-departure requirement.
            types[11] = 0  # Flat terrain does not require platform departure.
            t.terrain_levels.copy_(levels)
            t.terrain_types.copy_(types)
            t.env_origins[:] = t.terrain_origins[levels, types]
            c.terrain.level.copy_(levels)
            families = c.terrain.column_families[types]
            c.terrain.family.copy_(families)
            c.terrain.outside.zero_()
            c.terrain.outside[5] = True
            c.terrain.off_platform_steps.fill_(c.cfg.curriculum_off_platform_steps + 1)
            c.terrain.off_platform_steps[4] = c.cfg.curriculum_off_platform_steps
            c.terrain.off_platform_steps[11] = 0
            c.started.fill_(True)
            c.started[7] = False
            failures = torch.zeros_like(ids, dtype=torch.bool)
            failures[[1, 2, 6, 9]] = True
            timeouts = torch.ones_like(failures)
            timeouts[[1, 2, 3, 8, 9]] = False
            base.termination_manager.terminated.copy_(failures)
            base.termination_manager.get_term("time_out").copy_(timeouts)
            visits_before, failures_before = c.stratum_visits.clone(), c.stratum_failure.clone()
            expected_visits, expected_failures = torch.zeros_like(visits_before), torch.zeros_like(failures_before)
            for i in range(base.num_envs):
                if i != 7:  # First placement must not settle an episode.
                    expected_visits[families[i], levels[i], c.motion_ids[i]] += 1
                    expected_failures[families[i], levels[i], c.motion_ids[i]] += failures[i]
            torch.manual_seed(1234)
            c.reset_placement(ids)
            # Advancing into L9 must not randomize; success at L9 must randomize.
            assert t.terrain_levels[:11].tolist() == [9, 8, 0, 9, 9, 9, 8, 9, 4, 3, 5]
            randomized = t.terrain_levels[11:]
            assert ((randomized >= 0) & (randomized < 10)).all()
            assert randomized.unique().numel() > 1 and (randomized < 9).any()
            torch.testing.assert_close(t.terrain_types, types)
            torch.testing.assert_close(t.env_origins, t.terrain_origins[t.terrain_levels, types])
            torch.testing.assert_close(base.scene.env_origins, t.env_origins)
            torch.testing.assert_close(c.terrain.level, t.terrain_levels)
            torch.testing.assert_close(c.stratum_visits, visits_before + expected_visits)
            decay = c.cfg.motion_failure_decay
            torch.testing.assert_close(c.stratum_failure, decay * failures_before + (1 - decay) * expected_failures)
            after = t.terrain_levels.clone()
            c.cfg.enable_terrain_curriculum = False
            base.termination_manager.terminated.fill_(True)
            base.termination_manager.get_term("time_out").fill_(True)
            c.reset_placement(ids)
            torch.testing.assert_close(t.terrain_levels, after)
            source = Path(inspect.getsourcefile(TerrainMotion))
            report = dict(
                passed=True,
                envs=base.num_envs,
                device=str(base.device),
                curriculum_version=c.sampling_contract["terrain_curriculum_version"],
                boundary_cases=after[:11].tolist(),
                randomized_level_histogram=torch.bincount(randomized, minlength=10).tolist(),
                column_and_origin_consistent=True,
                old_level_failure_and_visit_accounting=True,
                frozen_curriculum_preserved=True,
                curriculum_source=str(source),
                curriculum_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                importer_source=inspect.getsourcefile(type(t).update_env_origins),
                quality_evaluation=False,
            )
            root = Path(args.report_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "curriculum-regression.json").write_text(json.dumps(report, indent=2) + "\n")
            print("CURRICULUM_REGRESSION " + json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
