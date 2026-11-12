"""Newton integration regression against the previous PGMT backend.

Run with an explicit validation output directory. This test constructs a legacy
calculation oracle without calling its runner constructor or starting another job.
"""

import copy
import gymnasium as gym
import json
import torch
from pathlib import Path
from tensordict import TensorDict
from types import MethodType

from _legacy_source import legacy_module

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.learner_utils import _capture_stream_and_pool
from instinctlab.tasks.shadowing.pgmt.supermimic.runners.pgmt_runner import PGMTSuperMimicRunner as SuperMimicRunner
from instinctlab.utils import instinct_rl_cli as cli


def legacy_oracle(runner):
    """Same inputs and parameters, previous implementation of all numerical paths."""
    old = object.__new__(legacy_module("desktop_runner").DesktopRunner)
    options = runner.options
    old.device = runner.device
    old.wm = (
        legacy_module("wm_dynamics")
        .TerrainDynamics(
            runner.context.shape[-1],
            hidden=options.dynamics.hidden_width,
            amp=False,
            predict_contact=options.dynamics.predict_contact,
            predict_force=False,
            contact_indices=runner.command.foot_ids,
            scan_points=len(runner.rewards.heightfield.local_xy),
            ground_query=runner.rewards.heightfield.query,
            root_com=runner.rewards.root_com[0],
            structured=options.representation.kind == "structured",
            prediction_mode="velocity" if options.representation.kind == "velocity" else "delta",
        )
        .to(runner.device)
    )
    source = runner._dynamics.state_dict()
    old.wm.load_state_dict(
        {
            key: source["representation." + key] if "representation." + key in source else source[key]
            for key in old.wm.state_dict()
        }
    )
    old.actor = copy.deepcopy(runner.policy.network)
    old.actor.action_from_inputs = MethodType(type(old.actor).action_from_inputs, old.actor)
    old.normalizers = runner.normalizers
    old.rewards, old.heightfield, old.xi = (
        runner.rewards,
        runner.rewards.heightfield,
        runner.context,
    )
    old.mdp = legacy_module("bptt_mdp").BPTTMDP(old.rewards, old.normalizers["policy"], old.wm)
    old.foot_rewards, old.analytic_foot = runner.foot_rewards, runner.analytic_foot
    old.cfg = {**runner.train_cfg, "experiment": {"value_gradient_weight": 0.0}}
    old.options = dict(loss_scale=options.representation.loss_scale, train_horizon=options.world_model.horizon)
    old.bptt = dict(
        horizon=options.policy.horizon,
        gamma=options.policy.gamma,
        **{"lambda": options.policy.lam},
        checkpoint_activations=True,
    )
    old.execution = dict(
        actor_checkpoint=options.execution.actor_recompute, wm_checkpoint=options.execution.wm_recompute
    )
    old.value = copy.deepcopy(runner.value)
    old._features_forward, old._imagine_step = old.features, old.imagine_step
    old._bodies_forward = old.rewards.bodies
    old._terminal_value_forward = old._trajectory_value_forward = old.value
    old._rollout_objective = old.rollout_objective
    return old


def flat_grad(module):
    return torch.cat(
        [p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in module.parameters()]
    )


def compare(left, right, report, name, tolerance=2e-4):
    error = float(((left - right).norm() / right.norm().clamp_min(1e-12)).detach())
    report[name] = error
    print(f"{name}: relative L2 = {error:.8g}", flush=True)
    report[name + "/max_abs"] = float((left - right).abs().amax().detach())
    assert torch.isfinite(left).all() and error <= tolerance, (name, error, tolerance)


def run(runner, output, tolerance):
    report = {"relative_tolerance": tolerance}
    runner._before_iteration(False)
    runner.collect(runner.options.collection.initial_steps)
    runner.world_model_learner.update_normalization(runner.replay_buffer, runner.ids)
    batch = runner.world_model_learner.sample(runner.replay_buffer, runner.ids, runner.generators["replay"])
    runner.world_model_learner.update(torch.stack([TensorDict(batch, batch_size=[])]))
    old = legacy_oracle(runner)
    old_loss = old.predict(batch, runner.options.world_model.horizon)[0]
    new_loss = runner.world_model_learner.loss(batch)[0]
    compare(new_loss, old_loss, report, "wm_loss", tolerance)
    old_loss.backward()
    runner._dynamics.zero_grad(set_to_none=True)
    new_loss.backward()
    compare(flat_grad(runner._dynamics), flat_grad(old.wm), report, "wm_gradient", tolerance)
    old.wm.zero_grad(set_to_none=True)
    runner._dynamics.zero_grad(set_to_none=True)
    old.wm.requires_grad_(False)
    runner._dynamics.requires_grad_(False)
    runner.value.requires_grad_(False)
    old.value.requires_grad_(False)
    data = runner.wm_env.sample_initial_states()
    inputs = runner.wm_env.prepare(data, runner.generators["observation"])
    for scale in (0.0, 0.5, 1.0):
        runner.foot_rewards.scale.fill_(scale)
        before = old.prepared_imagination(inputs)
        after = runner.policy_learner.imagine(inputs)
        for key, value in zip(
            ("loss", "features", "targets", "survival", "valid", "reward_means"), before, strict=True
        ):
            if key == "valid":
                assert torch.equal(after[key], value)
            else:
                compare(after[key], value, report, f"contact{scale}/{key}", tolerance)
        old.actor.zero_grad(set_to_none=True)
        runner.policy.zero_grad(set_to_none=True)
        before[0].backward()
        after["loss"].backward()
        compare(
            flat_grad(runner.policy.network),
            flat_grad(old.actor),
            report,
            f"contact{scale}/actor_gradient",
            tolerance=tolerance,
        )
        assert all(p.grad is None for p in runner._dynamics.parameters())
    runner.value.requires_grad_(True)
    runner._dynamics.requires_grad_(True)
    # A real learner update and atomic round-trip include optimizers, EMA, clocks and curriculum.
    runner.warmup_complete = True
    runner.policy_learner.update(inputs)
    observation = runner.observation.clone()
    expected = runner.policy(observation).detach()
    runner.save(output / "roundtrip.pt")
    saved = torch.load(output / "roundtrip.pt", weights_only=True, map_location=runner.device)
    legacy_actor = copy.deepcopy(runner.policy.network)
    legacy_actor.load_state_dict(
        {k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")}
    )
    compare(
        legacy_actor.action(runner.normalizers["policy"](observation)),
        expected,
        report,
        "legacy_play_export",
    )
    from instinctlab.tasks.shadowing.pgmt.supermimic.runners.pgmt_checkpoint import initialize_legacy_weights

    initialize_legacy_weights(runner, output / "roundtrip.pt")
    compare(runner.policy(observation), expected, report, "legacy_weight_import")
    with torch.no_grad():
        next(runner.policy.parameters()).add_(0.1)
    runner.load(output / "roundtrip.pt")
    compare(runner.policy(observation), expected, report, "checkpoint_actor")
    assert len(runner.replay_buffer) == 0
    assert runner.policy_learner.actor_updates == 1
    assert runner.world_model_learner.updates == 1
    assert all(p.grad is None for p in runner._dynamics.parameters())
    runner.collect(runner.options.collection.initial_steps)
    updates = runner.options.execution.capture_warmup + 2 if runner.options.execution.cuda_graph else 1
    for _ in range(updates):
        metrics = runner.iteration(True)
        assert all(torch.isfinite(torch.as_tensor(x)).all() for x in metrics.values())
    if runner.options.execution.cuda_graph:
        assert all(value for key, value in metrics.items() if key.startswith("Performance/cuda_graph_"))
        runner.save(output / "captured.pt")
        runner.load(output / "captured.pt")
        assert not any(runner.world_model_learner.capture_status.values())
        assert not any(runner.policy_learner.capture_status.values())
    report["resume_actor_updates"] = runner.policy_learner.actor_updates
    report["metrics"] = {k: float(v) for k, v in metrics.items()}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2), flush=True)


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    env_cfg, agent_cfg = cli.resolve_train_cfg(args.task, overrides)
    agent_cfg = cli.update_instinct_rl_cfg(agent_cfg, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed
    output = Path(args.logroot)
    output.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with launch_simulation(env_cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        runner = SuperMimicRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        try:
            # Oracle backward and captured backward must use the same CUDA stream.
            # The oracle deliberately differentiates through the same model outside
            # the learner; unlike normal training it can seed default-stream
            # dependencies into compiled autograd before capture begins.
            stream, _ = _capture_stream_and_pool(runner.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run(runner, output, args.relative_tolerance)
            torch.cuda.current_stream().wait_stream(stream)
        finally:
            runner._cleanup()
            env.close()


if __name__ == "__main__":
    main()
