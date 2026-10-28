"""Native PGMT integration and checkpoint acceptance on the compute host."""

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
from instinctlab.tasks.shadowing.pgmt.runner import PGMTRunner
from instinctlab.utils import instinct_rl_cli as cli


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    args.distributed = "LOCAL_RANK" in os.environ
    cfg, agent = cli.resolve_train_cfg(args.task, overrides)
    agent = cli.update_instinct_rl_cfg(agent, args)
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
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        base = env.unwrapped
        motion = base.scene["motion_reference"]
        buffer = motion.motion_buffers["LAFAN"]
        obs, extras = env.get_observations()
        assert obs.shape == (env.num_envs, 1645)
        assert extras["observations"]["critic"].shape == (env.num_envs, 296)
        assert len(buffer._all_motion_files) == 40
        assert "height_scanner" not in base.scene.sensors and "camera" not in base.scene.sensors
        max_error = 0.0
        # Check native reference poses against Newton on every LAFAN clip.
        for first in range(0, 40, env.num_envs):
            ids = torch.arange(min(env.num_envs, 40 - first), device=env.device)
            buffer._assigned_env_motion_selection[ids] = torch.arange(first, first + len(ids), device=env.device)
            buffer._motion_buffer_start_time_s[ids] = 0.1
            state = motion.get_init_reference_state(ids)
            base.scene["robot"].write_root_state_to_sim(
                torch.cat((state.base_pos_w, state.base_quat_w, state.base_lin_vel_w, state.base_ang_vel_w), -1),
                env_ids=ids,
            )
            base.scene["robot"].write_joint_state_to_sim(state.joint_pos, state.joint_vel, env_ids=ids)
            base.sim.forward()
            base.scene.update(base.physics_dt)
            links = base.scene["robot"].find_bodies(motion.cfg.link_of_interests, preserve_order=True)[0]
            data = buffer._all_motion_sequences.link_pos_w
            index = buffer._assigned_env_motion_selection[ids]
            expected = data[index, torch.full_like(index, 5)] + base.scene.env_origins[ids, None]
            error = (expected - base.scene["robot"].data.body_pos_w.torch[ids][:, links]).norm(dim=-1).max().item()
            max_error = max(max_error, error)
        assert max_error < 1e-4, max_error
        env.reset()
        for _ in range(12):
            env.step(torch.zeros(env.num_envs, 29, device=env.device))
        ids = torch.tensor([0, 2], device=env.device)
        base._reset_idx(ids)
        base.sim.forward()
        base.scene.update(base.physics_dt)
        history = base.observation_manager.compute(update_history=True)["policy"]
        for name, width in [
            ("projected_gravity", 3),
            ("base_ang_vel", 3),
            ("joint_pos", 29),
            ("joint_vel", 29),
            ("last_action", 29),
        ]:
            value = history[name][ids].reshape(2, 10, width)
            torch.testing.assert_close(value, value[:, :1].expand_as(value))
        env.reset()
        folder = Path(args.report_dir)
        folder.mkdir(parents=True, exist_ok=True)
        runner = PGMTRunner(env, copy.deepcopy(agent.to_dict()), log_dir=str(folder / "train"), device=agent.device)
        runner.learn(3, init_at_random_ep_len=False)
        saved = folder / "validation.pt"
        runner.save(str(saved))
        model = copy.deepcopy(runner.alg.actor_critic.state_dict())
        optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
        if world > 1:
            dist.barrier()
        saved_task = torch.load(saved, map_location="cpu", weights_only=True)["infos"]["rank_states"][rank]

        normalizers = copy.deepcopy({k: n.state_dict() for k, n in runner.normalizers.items()})
        resumed = PGMTRunner(env, copy.deepcopy(agent.to_dict()), log_dir=str(folder / "resume"), device=agent.device)
        resumed.load(str(saved))
        torch.testing.assert_close(resumed.alg.actor_critic.state_dict(), model, rtol=0, atol=0)
        torch.testing.assert_close(resumed.alg.optimizer.state_dict(), optimizer, rtol=0, atol=0)
        torch.testing.assert_close({k: n.state_dict() for k, n in resumed.normalizers.items()}, normalizers, rtol=0, atol=0)
        assert resumed.alg.learning_rate == runner.alg.learning_rate
        restored_task = resumed._task_state()
        torch.testing.assert_close(restored_task["failures"], saved_task["failures"], atol=0, rtol=0)
        resumed.learn(2, init_at_random_ep_len=False)
        assert resumed.current_learning_iteration == 5
        assert any(not torch.equal(v, model[k]) for k, v in resumed.alg.actor_critic.state_dict().items())
        assert all(torch.isfinite(x).all() for x in resumed.alg.actor_critic.state_dict().values())
        if world > 1:
            for value in resumed.alg.actor_critic.state_dict().values():
                expected = value.clone()
                dist.broadcast(expected, src=0)
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        report = {
            "rank": rank,
            "world_size": world,
            "synchronized_parameters": True,
            "sampler_failures_reload": "exact",
            "task": args.task,
            "envs": env.num_envs,
            "lafan_motions": 40,
            "policy_dim": 1645,
            "critic_dim": 296,
            "max_fk_error_m": max_error,
            "partial_reset_history": "pass",
            "train_updates": 3,
            "resume_updates": 2,
            "model_optimizer_normalizer_lr_reload": "exact",
            "updated_weights_finite": True,
        }
        (folder / f"rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        env.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
