"""Real-rollout WM learning, reward parity, policy isolation and checkpoint acceptance."""

import copy
import gymnasium as gym
import json
import math
import os
import torch
import torch.distributed as dist
from pathlib import Path

import warp as wp

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.shadowing.pgmt.wm_dynamics import yaw_angle
from instinctlab.tasks.shadowing.pgmt.wm_runner import SupervisedWMRunner
from instinctlab.utils import instinct_rl_cli as cli


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--offline-iterations", type=int, default=100)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    args.distributed = "LOCAL_RANK" in os.environ
    cfg, agent = cli.resolve_train_cfg(args.task, overrides)
    cli.update_instinct_rl_cfg(agent, args)
    cfg.scene.num_envs = args.num_envs or 256
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
            runner = SupervisedWMRunner(env, agent.to_dict(), str(root), agent.device)
            base = env.unwrapped
            c = runner.rewards.command
            terrain = base.scene.terrain
            ids = torch.arange(env.num_envs, device=env.device)
            # Diagnostic coverage includes full geometric difficulty; training defaults unchanged.
            terrain.terrain_levels[:] = ids % 10
            terrain.terrain_types[:] = (ids // 10) % 10
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            c.started.zero_()
            env.reset()
            # Deliberately include real timeout -> reset boundaries in collected transitions.
            base.episode_length_buf[::8] = base.max_episode_length - 2
            runner.learn(args.max_iterations or 4, init_at_random_ep_len=False)
            baseline = runner.validate()
            policy = copy.deepcopy(runner.alg.actor_critic.state_dict())
            normalizers = copy.deepcopy({k: v.state_dict() for k, v in runner.normalizers.items()})
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(runner.device)
            policy_gradients = [
                p.grad.clone() if p.grad is not None else None for p in runner.alg.actor_critic.parameters()
            ]
            moment_counts = [
                m.count.clone()
                for m in (runner.wm.input_norm, runner.wm.delta_norm, runner.wm.force_norm)
                if m is not None
            ]
            for iteration in range(args.offline_iterations):
                runner.fit_world_model(update_moments=False)
                if (iteration + 1) % 25 == 0:
                    print(f"WM_OFFLINE rank={rank} iteration={iteration+1} loss={runner.training_loss}", flush=True)
            final = runner.validate()
            assert all(math.isfinite(value) for value in (*baseline.values(), *final.values()))
            torch.testing.assert_close(
                [p.grad for p in runner.alg.actor_critic.parameters()], policy_gradients, rtol=0, atol=0
            )
            torch.testing.assert_close(
                [m.count for m in (runner.wm.input_norm, runner.wm.delta_norm, runner.wm.force_norm) if m is not None],
                moment_counts,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(runner.alg.actor_critic.state_dict(), policy, rtol=0, atol=0)
            torch.testing.assert_close(
                {k: v.state_dict() for k, v in runner.normalizers.items()}, normalizers, rtol=0, atol=0
            )
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(runner.device))
            # All fitted tensors are detached; no policy parameter is owned by the WM optimizer.
            assert not {id(p) for p in runner.wm.parameters()} & {id(p) for p in runner.alg.actor_critic.parameters()}
            horizon = max(runner.options["validation_horizons"])
            data = runner.sample(horizon, validation=True)
            assert (data["env_id"] % 8 == 0).all()
            train = runner.sample(runner.options["train_horizon"])
            assert (train["env_id"] % 8 != 0).all()
            assert not data["state"].requires_grad and not data["state"].is_inference()
            after_end = data["done"].cumsum(1)[:, :-1] > 0
            assert (data["mask"][:, 1:][after_end] == 0).all()
            adjacent = data["mask"][:, 1:].bool()
            assert adjacent.any()
            torch.testing.assert_close(
                data["next_state"][:, :-1][adjacent], data["state"][:, 1:][adjacent], atol=1e-5, rtol=0
            )
            assert horizon > runner.num_steps_per_env and len(runner.replay) >= horizon
            assert (runner.wm.contact_head is not None) == runner.options["predict_contact"]
            assert (runner.wm.force_head is not None) == runner.options["predict_force"]
            if not runner.wm.predict_force:
                modified = dict(data, next_force=data["next_force"] + 123.0)
                if not runner.wm.predict_contact:
                    modified.update(contact=1 - data["contact"], next_contact=1 - data["next_contact"])
                with torch.no_grad():
                    original_output = runner.predict(data, runner.options["train_horizon"])
                    changed_output = runner.predict(modified, runner.options["train_horizon"])
                torch.testing.assert_close(original_output, changed_output, atol=0, rtol=0)
            common = runner.rewards.supported_keys(False, False)
            supported = runner.rewards.supported_keys(runner.wm.predict_contact, runner.wm.predict_force)
            assert len(common) == 19
            assert len(supported) == (27 if runner.wm.predict_force else 24 if runner.wm.predict_contact else 19)
            for h in runner.options["validation_horizons"]:
                assert f"fixed/h{h}/reward_common_mae" in final
                if not runner.wm.predict_force:
                    assert f"fixed/h{h}/reward_predicted_mae" not in final
            # Native mesh scan versus the baked map, at arbitrary current held-out states.
            state = data["state"][:, 0].clone().requires_grad_(True)
            heights, valid = runner.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            heights.sum().backward()
            assert torch.isfinite(state.grad).all() and state.grad[:, 2].abs().sum() > 0
            with torch.no_grad():
                yaw = yaw_angle(state[:, 3:7])
                x, y = runner.heightfield.local_xy.unbind(-1)
                xy = (
                    torch.stack(
                        (
                            yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                            yaw.sin()[:, None] * x + yaw.cos()[:, None] * y,
                        ),
                        -1,
                    )
                    + state[:, None, :2]
                )
                mesh, _, mesh_valid = c.terrain.query(xy)
                mesh = (mesh - state[:, 2:3]).clamp(-5, 5)
                heightmap_mesh_mae = float((mesh - heights.detach()).abs()[mesh_valid].mean())
            # Exact WM/optimizer continuation; simulator/replay intentionally refill on resume.
            wm_before = copy.deepcopy(runner.wm.state_dict())
            optimizer_before = copy.deepcopy(runner.wm_optimizer.state_dict())
            checkpoint = root / "acceptance.pt"
            runner.save(str(checkpoint))
            actual_digest = runner.terrain_digest
            if rank == (dist.get_world_size() - 1 if dist.is_initialized() else 0):
                runner.terrain_digest = "deliberately-invalid-terrain-for-resume-test"
            rejected = False
            try:
                runner.load(str(checkpoint))
            except ValueError as error:
                rejected = "terrain raster" in str(error)
            finally:
                runner.terrain_digest = actual_digest
            assert rejected, "Every rank must reject a changed rank-local terrain without hanging"
            runner.load(str(checkpoint))
            torch.testing.assert_close(runner.wm.state_dict(), wm_before, rtol=0, atol=0)
            torch.testing.assert_close(runner.wm_optimizer.state_dict(), optimizer_before, rtol=0, atol=0)
            assert not runner.replay
            for value in runner.wm.state_dict().values():
                assert torch.isfinite(value).all()
                if dist.is_initialized():
                    expected = value.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            report = dict(
                rank=rank,
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
                envs=env.num_envs,
                source_checkpoint=runner.source_checkpoint,
                wm_updates=runner.wm_updates,
                ppo_only_policy_updates=True,
                offline_policy_and_normalizers_unchanged=True,
                offline_policy_gradients_unchanged=True,
                offline_moment_counts_unchanged=True,
                global_rng_unchanged=True,
                reset_prefix_mask_checked=True,
                cross_ppo_rollout_continuity_checked=True,
                predict_contact=runner.wm.predict_contact,
                predict_force=runner.wm.predict_force,
                disabled_auxiliary_targets_do_not_leak=True,
                validation_horizons=runner.options["validation_horizons"],
                supported_reward_terms=supported,
                common_reward_terms=common,
                heldout_environments_excluded_from_training=True,
                heightmap_coordinate_gradient_finite=True,
                heightmap_mesh_mae_m=heightmap_mesh_mae,
                exact_wm_optimizer_reload=True,
                rank_local_terrain_contract_and_collective_rejection=True,
                all_ranks_wm_equal=True,
                baseline=baseline,
                final=final,
                scope="Bounded diagnostic; no WM/BPTT actor updates; not a tracking quality benchmark",
            )
            (root / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                "WM_ACCEPTANCE " + json.dumps({k: v for k, v in report.items() if k not in ("baseline", "final")}),
                flush=True,
            )
        finally:
            env.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
