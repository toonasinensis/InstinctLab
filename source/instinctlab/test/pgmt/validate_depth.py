"""Newton depth integration: ownership, terminal images, BPTT and checkpoint reload.

Use --logroot pointing to an external validation directory, never a training root.
"""

import gymnasium as gym
import json
import torch
from pathlib import Path
from tensordict import TensorDict

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.shadowing.pgmt.supermimic.runners.pgmt_depth_runner import PGMTDepthRunner
from instinctlab.utils import instinct_rl_cli as cli


def run(runner, output):
    runner._before_iteration(False)
    runner.collect(runner.options.collection.initial_steps)
    learner, dynamics = runner.world_model_learner, runner._dynamics
    learner.update_normalization(runner.replay_buffer, runner.ids)
    assert runner.actor.encoder is dynamics.visual.encoder
    actor_params = {id(p) for g in runner.policy_learner.actor_optimizer.param_groups for p in g["params"]}
    wm_params = {id(p) for g in learner.optimizer.param_groups for p in g["params"]}
    assert not actor_params & wm_params
    assert {id(p) for p in runner.actor.encoder.parameters()} <= wm_params
    batch = learner.sample(runner.replay_buffer, runner.ids, runner.generators["replay"])
    data = learner.prepare_latents(batch)
    prediction = learner.predict(data, runner.options.world_model.horizon)
    dynamics.zero_grad(set_to_none=True)
    prediction["loss"].backward()
    assert all(p.grad is None for p in runner.actor.encoder.parameters())
    del prediction, data
    dynamics.zero_grad(set_to_none=True)
    scales = {name: value.clone() for name, value in dynamics.visual.named_buffers()}
    loss, _ = learner.loss(batch)
    loss.backward()
    compiled_error = None
    if runner.options.execution.compile != "none":
        eager = torch.cat(
            [p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in dynamics.parameters()]
        )
        for name, value in dynamics.visual.named_buffers():
            value.copy_(scales[name])
        dynamics.zero_grad(set_to_none=True)
        compiled_loss, _ = learner.loss_forward(batch)
        compiled_loss.backward()
        compiled = torch.cat(
            [p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in dynamics.parameters()]
        )
        compiled_error = float((compiled - eager).norm() / eager.norm().clamp_min(1e-12))
        assert compiled_error <= 0.002, compiled_error
        del compiled_loss, eager, compiled
    encoder_norm = torch.stack([p.grad.square().sum() for p in runner.actor.encoder.parameters()]).sum().sqrt()
    assert torch.isfinite(encoder_norm) and encoder_norm > 0
    del loss
    dynamics.zero_grad(set_to_none=True)
    wm = learner.update(torch.stack([TensorDict(batch, batch_size=[])]))
    inputs = runner.wm_env.prepare(runner.wm_env.sample_initial_states(), runner.generators["observation"])
    actor = runner.policy_learner.update(inputs)
    assert all(p.grad is None for p in dynamics.parameters())
    if runner.options.execution.cuda_graph:
        for _ in range(runner.options.execution.capture_warmup + 1):
            batch = learner.sample(runner.replay_buffer, runner.ids, runner.generators["replay"])
            wm = learner.update(torch.stack([TensorDict(batch, batch_size=[])]))
            actor = runner.policy_learner.update(inputs)
        assert all(learner.capture_status.values())
        assert all(runner.policy_learner.capture_status.values())
    contact_gradients = {}
    for scale in (0.0, 0.5, 1.0):
        runner.foot_rewards.scale.fill_(scale)
        actor = runner.policy_learner.update(inputs)
        assert all(torch.isfinite(value).all() for value in actor.values())
        assert all(p.grad is None for p in dynamics.parameters())
        contact_gradients[str(scale)] = float(actor["actor_grad_norm"])
    captured = dict(learner.capture_status, **runner.policy_learner.capture_status)
    with torch.no_grad():
        observation = runner.observation.clone()
        expected = runner.policy(observation).clone()
    runner.save(output / "roundtrip.pt")
    with torch.no_grad():
        next(runner.actor.action_head.parameters()).add_(0.1)
    runner.load(output / "roundtrip.pt")
    with torch.no_grad():
        torch.testing.assert_close(runner.policy(observation), expected)
    assert len(runner.replay_buffer) == 0
    report = dict(
        actor_shape=list(expected.shape),
        encoder_grad_norm=float(encoder_norm),
        actor_wm_optimizer_disjoint=True,
        wm_loss_does_not_train_encoder=True,
        actor_update_freezes_wm=True,
        checkpoint_reload_matches=True,
        cuda_graph=captured,
        tf32=torch.backends.cuda.matmul.allow_tf32,
        contact_scale_actor_gradients=contact_gradients,
        compiled_gradient_relative_l2=compiled_error,
        penetration=runner.train_cfg["foot_contact"]["penetration"],
        reconstruction_mse=float(wm["Perception/height_reconstruction_mse"]),
        actor_grad_norm=float(actor["actor_grad_norm"]),
        critic_grad_norm=float(actor["critic_grad_norm"]),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--disable-tf32", action="store_true")
    args, overrides = cli.parse_cli(parser, require_run_name=True)
    torch.backends.cuda.matmul.allow_tf32 = not args.disable_tf32
    torch.backends.cudnn.allow_tf32 = not args.disable_tf32
    env_cfg, agent_cfg = cli.resolve_train_cfg(args.task, overrides)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed
    output = Path(args.logroot)
    output.mkdir(parents=True, exist_ok=True)
    with launch_simulation(env_cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        runner = PGMTDepthRunner(env, agent_cfg.to_dict(), log_dir=str(output), device=agent_cfg.device)
        try:
            run(runner, output)
        finally:
            runner._cleanup()
            env.close()


if __name__ == "__main__":
    main()
