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

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.shadowing.pgmt.bptt_mdp import BPTTMDP
from instinctlab.tasks.shadowing.pgmt.desktop_runner import DesktopRunner
from instinctlab.tasks.shadowing.pgmt.supermimic.runner import SuperMimicRunner
from instinctlab.tasks.shadowing.pgmt.wm_dynamics import TerrainDynamics
from instinctlab.utils import instinct_rl_cli as cli


def legacy_oracle(runner):
    """Same inputs and parameters, previous implementation of all numerical paths."""
    old = object.__new__(DesktopRunner)
    options = runner.options
    old.device = runner.device
    old.wm = TerrainDynamics(
        runner.components.context.shape[-1],
        hidden=options.dynamics.hidden_width,
        amp=False,
        predict_contact=options.dynamics.predict_contact,
        predict_force=False,
        contact_indices=runner.components.collector.command.foot_ids,
        scan_points=len(runner.components.rewards.heightfield.local_xy),
        ground_query=runner.components.rewards.heightfield.query,
        root_com=runner.components.rewards.root_com[0],
        structured=options.representation.kind == "structured",
        prediction_mode="velocity" if options.representation.kind == "velocity" else "delta",
    ).to(runner.device)
    source = runner.components.dynamics.state_dict()
    old.wm.load_state_dict(
        {
            key: source["representation." + key] if "representation." + key in source else source[key]
            for key in old.wm.state_dict()
        }
    )
    old.actor = copy.deepcopy(runner.components.policy.network)
    old.actor.action_from_inputs = MethodType(type(old.actor).action_from_inputs, old.actor)
    old.normalizers = runner.components.normalizers
    old.rewards, old.heightfield, old.xi = (
        runner.components.rewards,
        runner.components.rewards.heightfield,
        runner.components.context,
    )
    old.mdp = BPTTMDP(old.rewards, old.normalizers["policy"], old.wm)
    old.foot_rewards, old.analytic_foot = runner.components.foot_rewards, runner.components.analytic_foot
    old.cfg = {**runner.cfg, "experiment": {"value_gradient_weight": 0.0}}
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
    old.value = copy.deepcopy(runner.components.value)
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
    error = float((left - right).norm() / right.norm().clamp_min(1e-12))
    report[name] = error
    print(f"{name}: relative L2 = {error:.8g}", flush=True)
    report[name + "/max_abs"] = float((left - right).abs().amax())
    assert torch.isfinite(left).all() and error <= tolerance, (name, error, tolerance)


def run(runner, output, tolerance):
    report = {"relative_tolerance": tolerance}
    runner.components.collector.collect(runner.options.collection.initial_steps, 0, False)
    runner.components.wm_learner.update_normalization(runner.components.replay, runner.components.collector.ids)
    batch = runner.components.wm_learner.sample(
        runner.components.replay, runner.components.collector.ids, runner.components.generators["replay"]
    )
    runner.components.wm_learner.update(torch.stack([TensorDict(batch, batch_size=[])]))
    old = legacy_oracle(runner)
    old_loss = old.predict(batch, runner.options.world_model.horizon)[0]
    new_loss = runner.components.wm_learner.loss(batch)[0]
    compare(new_loss, old_loss, report, "wm_loss", tolerance)
    old_loss.backward()
    runner.components.dynamics.zero_grad(set_to_none=True)
    new_loss.backward()
    compare(flat_grad(runner.components.dynamics), flat_grad(old.wm), report, "wm_gradient", tolerance)
    old.wm.zero_grad(set_to_none=True)
    runner.components.dynamics.zero_grad(set_to_none=True)
    old.wm.requires_grad_(False)
    runner.components.dynamics.requires_grad_(False)
    runner.components.value.requires_grad_(False)
    old.value.requires_grad_(False)
    data = runner.components.collector.sample_actor(
        runner.options.policy, runner.options.world_model, runner.components.generators["actor"]
    )
    inputs = runner.components.world_env.prepare(data, runner.components.generators["observation"])
    for scale in (0.0, 0.5, 1.0):
        runner.components.foot_rewards.scale.fill_(scale)
        before = old.prepared_imagination(inputs)
        after = runner.components.world_env.rollout(inputs)
        for key, value in zip(
            ("loss", "features", "targets", "survival", "valid", "reward_means"), before, strict=True
        ):
            if key == "valid":
                assert torch.equal(after[key], value)
            else:
                compare(after[key], value, report, f"contact{scale}/{key}", tolerance)
        old.actor.zero_grad(set_to_none=True)
        runner.components.policy.zero_grad(set_to_none=True)
        before[0].backward()
        after["loss"].backward()
        compare(
            flat_grad(runner.components.policy.network),
            flat_grad(old.actor),
            report,
            f"contact{scale}/actor_gradient",
            tolerance=tolerance,
        )
        assert all(p.grad is None for p in runner.components.dynamics.parameters())
    runner.components.value.requires_grad_(True)
    runner.components.dynamics.requires_grad_(True)
    # A real learner update and atomic round-trip include optimizers, EMA, clocks and curriculum.
    runner.warmup_complete = True
    runner.components.policy_learner.update(inputs)
    observation = runner.components.collector.observation.clone()
    expected = runner.components.policy(observation).detach()
    runner.save(output / "roundtrip.pt")
    saved = torch.load(output / "roundtrip.pt", weights_only=True, map_location=runner.device)
    legacy_actor = copy.deepcopy(runner.components.policy.network)
    legacy_actor.load_state_dict(
        {k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")}
    )
    compare(
        legacy_actor.action(runner.components.normalizers["policy"](observation)),
        expected,
        report,
        "legacy_play_export",
    )
    with torch.no_grad():
        next(runner.components.policy.parameters()).add_(0.1)
    runner.load(output / "roundtrip.pt")
    compare(runner.components.policy(observation), expected, report, "checkpoint_actor")
    assert len(runner.components.replay) == 0
    assert runner.components.policy_learner.actor_updates == 1
    assert runner.components.wm_learner.updates == 1
    assert all(p.grad is None for p in runner.components.dynamics.parameters())
    runner.components.collector.collect(
        runner.options.collection.initial_steps, runner.components.policy_learner.actor_updates, True
    )
    metrics = runner.iteration(True)
    assert all(torch.isfinite(torch.as_tensor(x)).all() for x in metrics.values())
    report["resume_actor_updates"] = runner.components.policy_learner.actor_updates
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
            run(runner, output, args.relative_tolerance)
        finally:
            runner.components.collector.close()
            env.close()


if __name__ == "__main__":
    main()
