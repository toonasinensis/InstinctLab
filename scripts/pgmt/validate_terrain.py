"""PGMT terrain acceptance: warm-start parity, shared gradients, terminal semantics and DDP resume."""

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
from instinctlab.tasks.shadowing.pgmt.models.cross_attention import CrossAttentionActor
from instinctlab.tasks.shadowing.pgmt.terrain_ppo import TerrainStorage
from instinctlab.tasks.shadowing.pgmt.terrain_runner import TerrainRunner
from instinctlab.utils import instinct_rl_cli as cli


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    args.distributed = "LOCAL_RANK" in os.environ
    cfg, agent = cli.resolve_train_cfg(args.task, overrides)
    cli.update_instinct_rl_cfg(agent, args)
    cfg.scene.num_envs = args.num_envs or 32
    cfg.seed = agent.seed
    with launch_simulation(cfg, args):
        if args.distributed:
            wp.set_device(cfg.sim.device)
            dist.init_process_group("nccl")
            cfg.seed += dist.get_rank()
            agent.device = cfg.sim.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        root = Path(args.report_dir)
        root.mkdir(parents=True, exist_ok=True)
        runner = TerrainRunner(env, agent.to_dict(), str(root), agent.device)
        base = env.unwrapped
        c = base.command_manager.get_term("motion")
        obs, extras = env.get_observations()
        assert obs.shape == (env.num_envs, 2119)
        assert extras["observations"]["critic"].shape == (env.num_envs, 296)
        assert env.num_rewards == 4
        assert len(c.motion._all_motion_files) == 40
        report = dict(
            rank=rank,
            world=world,
            envs=env.num_envs,
            motions=40,
            policy_dim=2119,
            critic_dim=296,
            reward_groups=list(base.reward_manager.active_terms),
        )
        # Check deterministic stage-one action preservation after channel/normalizer expansion.
        saved = torch.load(agent.warm_start, map_location=env.device, weights_only=True)
        source = CrossAttentionActor().to(env.device)
        source.load_state_dict(
            {k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")}
        )
        raw = torch.randn(16, 2119, device=env.device)
        runner.eval_mode()
        with torch.no_grad():
            normalized = runner.normalizers["policy"](raw)
            expected = source(normalized[:, :1645])
            actual = runner.alg.actor_critic.act_inference(normalized)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        report["warm_start_max_action_error"] = (actual - expected).abs().max().item()
        assert torch.equal(normalized[:, 1678:], raw[:, 1678:])
        report["terrain_not_empirically_normalized"] = True
        # Critic-only gradients reach shared intent parameters, exactly one optimizer registration.
        model = runner.alg.actor_critic
        model.act_inference(torch.randn(16, 2119, device=env.device))
        model.evaluate(torch.randn(16, 296, device=env.device)).square().mean().backward()
        assert model.actor.intent_attention.q.weight.grad.abs().sum() > 0
        params = [p for g in runner.alg.optimizer.param_groups for p in g["params"]]
        assert len(params) == len({id(p) for p in params})
        runner.alg.optimizer.zero_grad()
        report["shared_critic_gradient"] = True
        report["unique_optimizer_parameters"] = True
        # Independent Monte Carlo return example: two steps, gamma=lambda=1, terminal at t=1.
        storage = TerrainStorage(2, 2, [1], [1], [1], num_rewards=4, device=env.device)
        storage.rewards[:] = torch.tensor(
            [[[1.0, 2, 3, 4], [2.0, 4, 6, 8]], [[4.0, 3, 2, 1], [8.0, 6, 4, 2]]], device=env.device
        )
        storage.values.zero_()
        storage.dones[0] = 0
        storage.dones[1] = 1
        storage.compute_returns(torch.zeros(2, 4, device=env.device), 1.0, 1.0)
        torch.testing.assert_close(storage.returns[0], storage.rewards.sum(0))
        torch.testing.assert_close(storage.returns[1], storage.rewards[1])
        assert torch.equal(storage.advantages[..., 0], storage.advantages[..., 3])
        report["summed_group_gae"] = True
        # Exercise timeout observations before reset and bootstrap of four terminal values.
        if not cfg.flat_precheck:
            terrain = base.scene.terrain
            indices = torch.arange(env.num_envs, device=env.device)
            terrain.terrain_levels[:] = indices % 10
            if env.num_envs >= 100:
                terrain.terrain_types[:] = (indices // 10) % 10
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            c.started.zero_()
            env.reset()
            obs, extras = env.get_observations()
            report["all_ten_levels_assigned"] = len(torch.unique(terrain.terrain_levels)) == 10
        base.episode_length_buf.fill_(base.max_episode_length - 1)
        runner.train_mode()
        pack = extras["observations"]
        normalized = runner._prepare_extra_obs(pack)
        with torch.inference_mode():
            _, _, rewards, dones, info = runner.rollout_step(normalized, pack)
        assert info["bootstrap_time_outs"].any()
        assert "terminal_observations" in info
        assert torch.isfinite(runner.alg.storage.rewards[0]).all()
        report["terminal_bootstrap"] = True
        report["terminal_timeout_count"] = int(info["bootstrap_time_outs"].sum())
        runner.alg.storage.clear()
        env.reset()
        runner.learn(3, init_at_random_ep_len=False)
        c.stratum_failure[0, 0, 0] = 0.5 + rank
        c.stratum_visits[0, 0, 0] = 2 + rank
        checkpoint = root / "acceptance.pt"
        runner.save(str(checkpoint))
        before = copy.deepcopy(model.state_dict())
        norm_before = copy.deepcopy({k: v.state_dict() for k, v in runner.normalizers.items()})
        failure = c.stratum_failure.clone()
        visits = c.stratum_visits.clone()
        optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
        runner.load(str(checkpoint))
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        torch.testing.assert_close({k: v.state_dict() for k, v in runner.normalizers.items()}, norm_before, rtol=0, atol=0)
        torch.testing.assert_close(c.stratum_failure, failure, rtol=0, atol=0)
        torch.testing.assert_close(c.stratum_visits, visits, rtol=0, atol=0)
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, runner.alg.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
        report["exact_model_optimizer_normalizers_sampler_reload"] = True
        runner.learn(2, init_at_random_ep_len=False)
        report["finite_model"] = all(torch.isfinite(p).all().item() for p in model.parameters())
        if dist.is_initialized():
            states = [model.state_dict(), *(v.state_dict() for v in runner.normalizers.values())]
            for state in states:
                for value in state.values():
                    expected = value.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            report["all_ranks_synchronized"] = True
        report["terrain_family"] = torch.unique(c.terrain.family).tolist()
        report["terrain_levels"] = torch.unique(c.terrain.level).tolist()
        (root / f"rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        env.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
