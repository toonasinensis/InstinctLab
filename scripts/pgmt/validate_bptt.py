"""Bounded real/imagined BPTT acceptance and independent gradient checks."""

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
from instinctlab.tasks.shadowing.pgmt.bptt_mdp import lambda_returns, raw_history
from instinctlab.tasks.shadowing.pgmt.bptt_runner import BPTTRunner
from instinctlab.tasks.shadowing.pgmt.wm_dynamics import yaw_angle
from instinctlab.utils import instinct_rl_cli as cli


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    args.distributed = "LOCAL_RANK" in os.environ
    cfg, agent = cli.resolve_train_cfg(args.task, overrides)
    cli.update_instinct_rl_cfg(agent, args)
    cfg.scene.num_envs = args.num_envs or 128
    cfg.seed = agent.seed
    with launch_simulation(cfg, args):
        if args.distributed:
            wp.set_device(cfg.sim.device)
            dist.init_process_group("nccl")
            cfg.seed += dist.get_rank()
            agent.device = cfg.sim.device
        from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        root = Path(args.report_dir)
        root.mkdir(parents=True, exist_ok=True)
        runner = BPTTRunner(env, agent.to_dict(), str(root), agent.device)
        ids = torch.arange(env.num_envs, device=runner.device)
        start = runner.mdp.placement() | {"env_id": ids}
        context = runner.mdp.context(start)
        state = runner.rewards.state()
        obs, valid = runner.mdp.observations(
            state, raw_history(runner.observation), runner.rewards.command.terrain.support_z, context
        )
        # The raster approximation is measured separately; all nonterrain channels must match.
        torch.testing.assert_close(obs[:, :1678], runner.observation[:, :1678], atol=3e-4, rtol=1e-5)
        assert valid.all() and torch.isfinite(obs).all()
        generator = cfg.scene.terrain.terrain_generator
        origins = env.unwrapped.scene.terrain.terrain_origins[..., :2].reshape(-1, 2)
        half_tile = origins.new_tensor(generator.size) / 2
        low, high = origins.amin(0) - half_tile, origins.amax(0) + half_tile
        middle = (low + high) / 2
        margin = generator.border_width / 2
        probes = torch.stack(
            (
                torch.stack((low[0] - margin, middle[1])),
                torch.stack((high[0] + margin, middle[1])),
                torch.stack((middle[0], low[1] - margin)),
                torch.stack((middle[0], high[1] + margin)),
            )
        )
        baked, baked_valid = runner.heightfield.query(probes)
        native, _, native_valid = runner.rewards.command.terrain.query(probes)
        assert baked_valid.all() and native_valid.all()
        torch.testing.assert_close(baked, native, atol=2e-4, rtol=0)
        outside = runner.heightfield.origin[None] - 0.1
        assert not runner.heightfield.query(outside)[1].any()
        report = dict(
            observation_parity_max_error=float((obs[:, :1678] - runner.observation[:, :1678]).abs().max()),
            reward_count=len(runner.rewards.terms),
            critic_source=runner.bptt["critic_source"],
            horizon=runner.bptt["horizon"],
            rank=runner.rank,
            world=runner.world,
            physical_border_probes_valid=True,
            physical_border_height_max_error=float((baked - native).abs().max()),
            beyond_raster_rejected=True,
        )
        assert runner.wm.contact_head is None and runner.wm.force_head is None
        assert not {id(p) for p in runner.value.parameters()} & {id(p) for p in runner.actor.parameters()}
        # A timeout bootstraps its terminal value, never the next episode's return.
        reward = torch.tensor([[1.0, 10.0]], device=runner.device)
        values = torch.tensor([[0.0, 5.0, 7.0]], device=runner.device)
        result = lambda_returns(
            reward, torch.ones_like(reward), values, torch.tensor([[True, False]], device=runner.device), 1.0, 1.0
        )
        torch.testing.assert_close(result, torch.tensor([[6.0, 17.0]], device=runner.device))
        actor_before = copy.deepcopy(runner.actor.state_dict())
        norms_before = copy.deepcopy({k: v.state_dict() for k, v in runner.normalizers.items()})
        runner.collect()
        runner.fit_world_model()
        runner.collect()
        runner.fit_world_model()
        data = runner.sample_actor()
        runner.wm_optimizer.zero_grad(set_to_none=True)
        runner.wm.requires_grad_(False)
        runner.value.requires_grad_(False)
        loss, _, _, _, valid, states, actions = runner.imagine(data)
        future_map = runner.heightfield.local(states[1][:, :3], yaw_angle(states[1][:, 3:7]))[0]
        map_gradient = torch.autograd.grad(future_map.square().mean(), actions[0], retain_graph=True)[0]
        assert torch.isfinite(map_gradient).all() and map_gradient.abs().sum() > 0
        action_gradient = torch.autograd.grad(loss, actions[0], retain_graph=True)[0]
        assert valid.all() and torch.isfinite(action_gradient).all() and action_gradient.abs().sum() > 0
        loss.backward()
        assert all(p.grad is None for p in runner.wm.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in runner.actor.parameters())
        report.update(
            first_action_gradient_norm=float(action_gradient.norm()),
            future_heightmap_gradient_norm=float(map_gradient.norm()),
            wm_parameters_frozen_in_actor_backward=True,
        )
        runner.actor_optimizer.zero_grad(set_to_none=True)
        runner.wm.requires_grad_(True)
        runner.value.requires_grad_(True)
        # Finite-difference a policy parameter through a fixed, closed-loop trajectory.
        parameter = runner.actor.action_head[-1].bias
        direction = torch.linspace(-1, 1, parameter.numel(), device=runner.device)
        direction = direction / direction.norm()
        runner.wm.requires_grad_(False)
        runner.value.requires_grad_(False)
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        value = runner.imagine(data)[0]
        gradient = torch.autograd.grad(value, parameter)[0]
        expected = (gradient * direction).sum().item()
        epsilon = 0.003
        with torch.no_grad():
            parameter.add_(epsilon * direction)
            plus = float(runner.imagine(data)[0])
            parameter.add_(-2 * epsilon * direction)
            minus = float(runner.imagine(data)[0])
            parameter.add_(epsilon * direction)
        actual = (plus - minus) / (2 * epsilon)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        runner.wm.requires_grad_(True)
        runner.value.requires_grad_(True)
        assert abs(actual - expected) < 0.05 * max(abs(expected), 0.01), (actual, expected)
        report.update(
            policy_direction_autograd=expected,
            policy_direction_finite_difference=actual,
            gradient_scope="Computational graph consistency; not a real-simulator Jacobian accuracy claim",
        )
        runner.warmup_complete = True
        runner.learn(args.max_iterations or 3)
        assert runner.actor_updates == (args.max_iterations or 3) and runner.ppo_updates == 0
        assert any(not torch.equal(value, runner.actor.state_dict()[key]) for key, value in actor_before.items())
        torch.testing.assert_close(
            {k: v.state_dict() for k, v in runner.normalizers.items()}, norms_before, rtol=0, atol=0
        )
        saved_actor = copy.deepcopy(runner.actor.state_dict())
        saved_value = copy.deepcopy(runner.value.state_dict())
        saved_optimizer = copy.deepcopy(runner.actor_optimizer.state_dict())
        checkpoint = root / "acceptance.pt"
        runner.save(str(checkpoint))
        runner.load(str(checkpoint))
        torch.testing.assert_close(runner.actor.state_dict(), saved_actor, rtol=0, atol=0)
        torch.testing.assert_close(runner.value.state_dict(), saved_value, rtol=0, atol=0)
        torch.testing.assert_close(runner.actor_optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
        if dist.is_initialized():
            for module in (runner.actor, runner.wm, runner.value):
                for value in module.state_dict().values():
                    expected = value.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
        # A collective invalid window must preserve weights/counters and retain evidence.
        local_map = runner.heightfield.local

        def invalid_local(*args, **kwargs):
            heights, valid = local_map(*args, **kwargs)
            return heights, torch.zeros_like(valid)

        runner.heightfield.local = invalid_local
        counters = (runner.actor_updates, runner.critic_updates, runner.wm_updates)
        try:
            runner.update_bptt()
        except RuntimeError as error:
            assert "no actor update was applied" in str(error)
        else:
            raise AssertionError("Invalid imagined window was accepted")
        finally:
            runner.heightfield.local = local_map
        assert counters == (runner.actor_updates, runner.critic_updates, runner.wm_updates)
        torch.testing.assert_close(runner.actor.state_dict(), saved_actor, rtol=0, atol=0)
        torch.testing.assert_close(runner.value.state_dict(), saved_value, rtol=0, atol=0)
        window_path = root / f"invalid-rollout-{runner.actor_updates}-rank-{runner.rank}.pt"
        window = torch.load(window_path, map_location="cpu", weights_only=True)
        assert not window["valid"].any() and len(window["states"]) == runner.bptt["horizon"] + 1
        assert (root / f"invalid-rollout-{runner.actor_updates}.pt").is_file()
        report.update(
            actor_updates=runner.actor_updates,
            critic_updates=runner.critic_updates,
            ppo_updates=0,
            exact_reload=True,
            normalizers_frozen=True,
            independent_mlp=True,
            invalid_window_capture=True,
            parameters_finite=all(
                torch.isfinite(p).all().item()
                for module in (runner.actor, runner.wm, runner.value)
                for p in module.parameters()
            ),
        )
        (root / f"rank_{runner.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        env.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
