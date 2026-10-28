"""Bounded real-simulator validation of scheme-one updates, distributed state and resume."""

import copy
import gymnasium as gym
import json
import os
import torch
import torch.distributed as dist
from pathlib import Path

import warp as wp

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.shadowing.pgmt.world_model_runner import WorldModelRunner
from instinctlab.utils import instinct_rl_cli as cli


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    args.distributed = "LOCAL_RANK" in os.environ
    cfg, agent = cli.resolve_train_cfg(args.task, overrides)
    cli.update_instinct_rl_cfg(agent, args)
    cfg.scene.num_envs = args.num_envs or 16
    cfg.seed = agent.seed
    with launch_simulation(cfg, args):
        if args.distributed:
            wp.set_device(cfg.sim.device)
            dist.init_process_group("nccl")
            cfg.seed += dist.get_rank()
            agent.device = cfg.sim.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

        root = Path(args.report_dir)
        root.mkdir(parents=True, exist_ok=True)
        env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        try:
            runner = WorldModelRunner(env, agent.to_dict(), str(root), agent.device)
            if agent.resume:
                runner.load(cli.checkpoint_path(str(root), agent.load_run, agent.load_checkpoint))
            with torch.no_grad():
                from instinctlab.tasks.shadowing.pgmt.mdp.terrain_observations import critic_state

                native = env.unwrapped
                rebuilt = runner.mdp.critic_observations(
                    runner.mdp.state(),
                    native.action_manager.action,
                    runner.mdp.command.terrain.support_z,
                    runner.mdp.context(),
                )
                expected = critic_state(native)
                critic_parity = float((rebuilt - expected).abs().max())
                torch.testing.assert_close(rebuilt, expected, atol=5e-5, rtol=1e-5)
            runner.learn(4)
            assert runner.imagination_updates > 0, "Smoke must exercise actual imagined updates"
            modules = (runner.alg.actor_critic, runner.wm, runner.teacher, *runner.normalizers.values())
            saved = copy.deepcopy([m.state_dict() for m in modules])
            optimizers = (runner.alg.optimizer, runner.wm_optimizer, runner.actor_optimizer)
            saved_optimizers = copy.deepcopy([o.state_dict() for o in optimizers])
            counters = (runner.wm_updates, runner.imagination_updates)
            path = str(root / "acceptance.pt")
            runner.save(path)
            runner.load(path)
            torch.testing.assert_close([m.state_dict() for m in modules], saved, rtol=0, atol=0)
            torch.testing.assert_close([o.state_dict() for o in optimizers], saved_optimizers, rtol=0, atol=0)
            assert (runner.wm_updates, runner.imagination_updates) == counters
            assert not runner.replay and not runner.pending
            runner.learn(2)
            assert runner.wm_updates > counters[0] and runner.imagination_updates > counters[1]
            if runner.wm_cfg["teacher_ema"] > 0:
                torch.testing.assert_close(
                    runner.teacher.terrain_gate, runner.alg.actor_critic.actor.terrain_gate, rtol=0, atol=0
                )
            for module in modules:
                for value in module.state_dict().values():
                    assert torch.isfinite(value).all()
                    if dist.is_initialized():
                        expected = value.clone()
                        dist.broadcast(expected, src=0)
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
            report = dict(
                rank=rank,
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
                envs_per_rank=env.num_envs,
                source_checkpoint=runner.source_checkpoint,
                wm_updates=runner.wm_updates,
                imagination_updates=runner.imagination_updates,
                exact_module_optimizer_reload=True,
                cross_rank_equality=True,
                observation_parity=runner.observation_parity,
                validation_threshold=runner.wm_cfg["validation_threshold"],
                teacher_ema=runner.wm_cfg["teacher_ema"],
                maximum_imagination_kl=runner.wm_cfg["max_imagination_kl"],
                bootstrap_value=runner.wm_cfg["bootstrap_value"],
                critic_observation_max_error=critic_parity,
                scope="Short correctness smoke, not a performance result",
            )
            (root / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            env.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
