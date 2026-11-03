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
from instinctlab.tasks.shadowing.pgmt.bptt_mdp import OBSERVATION_WIDTHS, lambda_returns, raw_history
from instinctlab.tasks.shadowing.pgmt.bptt_runner import BPTTRunner
from instinctlab.tasks.shadowing.pgmt.wm_dynamics import average_gradients, yaw_angle
from instinctlab.utils import instinct_rl_cli as cli


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    return copy.deepcopy(value)


def check_ddp_parity(runner, data, diagnostic=False):
    """Compare the old native/manual reducer and DDP on identical local data."""
    report = {}

    def compare(name, network, optimizer, native, distributed):
        weights, optimizer_state = cpu_state(network.state_dict()), cpu_state(optimizer.state_dict())
        original_optimizer_state = copy.deepcopy(optimizer_state)
        noise = runner.noise_generator.get_state().clone()
        gradients, updates, losses, norms = [], [], [], []
        calls = []
        hook = distributed.register_forward_hook(lambda *args: calls.append(1))
        try:
            forwards = (native, native, distributed, distributed) if diagnostic else (native, distributed)
            for forward in forwards:
                network.load_state_dict(weights)
                # load_state_dict retains CPU Adam step tensors. Each fork must
                # own its state so the first update cannot advance the next fork.
                optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                optimizer.zero_grad(set_to_none=True)
                runner.noise_generator.set_state(noise)
                output = forward()
                loss = output[0] if isinstance(output, tuple) else output
                if isinstance(output, tuple):
                    assert output[4].all(), "Parity batch left the valid terrain/reference domain"
                loss.backward()
                if forward is native:
                    # Reference only: the production BPTT path has no manual reducer.
                    average_gradients(network)
                else:
                    for parameter in network.parameters():
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                gradients.append([p.grad.detach().cpu().clone() for p in network.parameters()])
                losses.append(float(loss.detach()))
                norms.append(float(torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)))
                if name == "actor":
                    torch.nn.utils.clip_grad_value_(network.parameters(), 0.1)
                optimizer.step()
                updates.append(cpu_state(network.state_dict()))
                torch.testing.assert_close(optimizer_state, original_optimizer_state, rtol=0, atol=0)
                del output, loss
            if diagnostic:
                comparisons = {}
                for label, left, right in (("native_repeat", 0, 1), ("native_ddp", 0, 2), ("ddp_repeat", 2, 3)):
                    a = torch.cat([value.flatten().double() for value in gradients[left]])
                    b = torch.cat([value.flatten().double() for value in gradients[right]])
                    mismatches = {}
                    for index, (key, _) in enumerate(network.named_parameters()):
                        difference = (updates[left][key] - updates[right][key]).abs()
                        bad = difference > 2e-7 + 1e-5 * updates[right][key].abs()
                        if bad.any():
                            maximum = int(difference.flatten().argmax())
                            mismatches[key] = dict(
                                count=int(bad.sum()),
                                elements=bad.numel(),
                                max_update_error=float(difference.flatten()[maximum]),
                                reference_gradient=float(gradients[left][index].flatten()[maximum]),
                                actual_gradient=float(gradients[right][index].flatten()[maximum]),
                                reference_update=float((updates[left][key] - weights[key]).flatten()[maximum]),
                                actual_update=float((updates[right][key] - weights[key]).flatten()[maximum]),
                            )
                    comparisons[label] = dict(
                        gradient_relative_l2_error=float((a - b).norm() / a.norm().clamp_min(1e-20)),
                        gradient_cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                        parameter_update_gate_passed=not mismatches,
                        mismatches=mismatches,
                    )
                report[name] = dict(
                    purpose="Diagnostic only; original acceptance thresholds unchanged",
                    acceptance_passed=False,
                    passes=["native", "native", "ddp", "ddp"],
                    losses=losses,
                    gradient_norms=norms,
                    comparisons=comparisons,
                    optimizer_reference_immutable=True,
                )
                torch.save(
                    dict(
                        report=report[name],
                        weights=weights,
                        optimizer=optimizer_state,
                        gradients=gradients,
                        updates=updates,
                        noise=noise,
                        data=cpu_state(data),
                        wm=cpu_state(runner.wm.state_dict()),
                        value=cpu_state(runner.value.state_dict()),
                        xi=runner.xi.cpu(),
                        execution=runner.execution,
                        learning_rates=runner.learning_rates,
                    ),
                    runner.log_root / f"{name}-diagnostic-rank-{runner.rank}.pt",
                )
                return
            reference = torch.cat([value.flatten().double() for value in gradients[0]])
            actual = torch.cat([value.flatten().double() for value in gradients[1]])
            relative = float((actual - reference).norm() / reference.norm().clamp_min(1e-20))
            cosine = float(torch.nn.functional.cosine_similarity(actual, reference, dim=0))
            assert relative < 1e-4 and cosine > 0.999999, (name, relative, cosine)
            torch.testing.assert_close(updates[0], updates[1], rtol=1e-5, atol=2e-7)
            assert losses[0] == losses[1] and len(calls) == 1, (name, losses, calls)
            for parameter in network.parameters():
                expected = parameter.detach().clone()
                dist.broadcast(expected, src=0)
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            report[name] = dict(
                gradient_relative_l2_error=relative,
                gradient_cosine=cosine,
                identical_local_loss=True,
                optimizer_update_close=True,
                rank_parameters_identical=True,
                ddp_forward_calls=len(calls),
                optimizer_reference_immutable=True,
            )
        finally:
            hook.remove()
            network.load_state_dict(weights)
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            optimizer.zero_grad(set_to_none=True)
            runner.noise_generator.set_state(noise)

    # Register hooks on the actual DDP entries, while presenting zero-argument
    # calls to the comparison so every path receives the same tensors.
    class Call:
        def __init__(self, entry, *inputs):
            self.entry, self.inputs = entry, inputs

        def __call__(self):
            return self.entry(*self.inputs)

        def register_forward_hook(self, hook):
            return self.entry.register_forward_hook(hook)

    assert dist.is_initialized() and runner.world in (2, 4)
    runner.wm_optimizer.zero_grad(set_to_none=True)
    runner.value_optimizer.zero_grad(set_to_none=True)
    runner.wm.requires_grad_(False)
    runner.value.requires_grad_(False)
    try:
        compare(
            "actor",
            runner.actor,
            runner.actor_optimizer,
            lambda: runner.imagine(data),
            Call(runner.actor_objective, data),
        )
        assert all(p.grad is None for module in (runner.wm, runner.value) for p in module.parameters())
    finally:
        runner.wm.requires_grad_(True)
        runner.value.requires_grad_(True)
    if diagnostic:
        report["learning_rates"] = runner.learning_rates
        return report
    wm_data = runner.sample(runner.options["train_horizon"])
    compare(
        "world_model",
        runner.wm,
        runner.wm_optimizer,
        lambda: runner._world_model_loss(wm_data),
        Call(runner.wm_objective, wm_data),
    )
    features, targets, mask = runner.real_critic_data(data)
    valid = mask.flatten().bool()
    features, targets = features.flatten(0, 1)[valid], targets.flatten()[valid]
    batch = max(1, len(targets) // runner.bptt["critic_minibatches"])
    features, targets = features[:batch], targets[:batch].detach()
    compare(
        "critic",
        runner.value,
        runner.value_optimizer,
        lambda: runner.value_loss(features, targets),
        Call(runner.value_objective, features, targets),
    )
    report["learning_rates"] = runner.learning_rates
    return report


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--prefill-steps", type=int, default=32)
    parser.add_argument("--check-ddp-parity", action="store_true")
    parser.add_argument("--ddp-diagnostic", action="store_true")
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
        # The real history already contains noise. Compare remaining channels against
        # a read-only, noise-disabled native computation without advancing that history.
        terms = runner.native.observation_manager._group_obs_term_cfgs["policy"]
        noises = [term.noise for term in terms]
        try:
            for term in terms:
                term.noise = None
            clean = runner.native.observation_manager.compute_group("policy", update_history=False)
            clean = torch.cat([value.flatten(1) for value in clean.values()], -1)
        finally:
            for term, noise in zip(terms, noises):
                term.noise = noise
        # The raster approximation is measured separately; all nonterrain channels must match.
        torch.testing.assert_close(obs[:, :1678], clean[:, :1678], atol=3e-4, rtol=1e-5)
        noise_state = runner.noise_generator.get_state()
        noisy = runner.mdp.observations(
            state,
            raw_history(runner.observation),
            runner.rewards.command.terrain.support_z,
            context,
            runner.noise_generator,
        )[0]
        after_noise = runner.noise_generator.get_state()
        runner.noise_generator.set_state(noise_state)
        prepared = runner.mdp.draw_noise(env.num_envs, OBSERVATION_WIDTHS, runner.noise_generator)
        prepared_obs = runner.mdp.observations(
            state, raw_history(runner.observation), runner.rewards.command.terrain.support_z, context, prepared
        )[0]
        torch.testing.assert_close(prepared_obs, noisy, rtol=0, atol=0)
        torch.testing.assert_close(runner.noise_generator.get_state(), after_noise, rtol=0, atol=0)
        assert not torch.equal(noisy[:, 930:1678], obs[:, 930:1678])
        torch.testing.assert_close(noisy[:, :930], obs[:, :930], rtol=0, atol=0)
        torch.testing.assert_close(noisy[:, 1678:], obs[:, 1678:], rtol=0, atol=0)
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
            observation_parity_max_error=float((obs[:, :1678] - clean[:, :1678]).abs().max()),
            observation_noise=True,
            imagined_noise_changes_only_configured_channels=True,
            reward_count=len(runner.rewards.terms),
            critic_source=runner.bptt["critic_source"],
            horizon=runner.bptt["horizon"],
            rank=runner.rank,
            world=runner.world,
            physical_border_probes_valid=True,
            physical_border_height_max_error=float((baked - native).abs().max()),
            beyond_raster_rejected=True,
            prepared_observation_noise_exact=True,
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
        replay_add = runner.replay.add

        def check_history(record):
            replay_add(record)
            ids = torch.arange(env.num_envs, device=runner.device)
            latest = torch.full_like(ids, len(runner.replay) - 1)
            torch.testing.assert_close(runner.replay.histories(latest, ids), record["history"], rtol=0, atol=0)

        runner.replay.add = check_history
        runner.collect(args.prefill_steps)
        runner.fit_world_model()
        runner.collect()
        runner.fit_world_model()
        runner.replay.add = replay_add
        report["native_noisy_history_reconstruction_exact"] = True
        data = runner.sample_actor()
        if args.ddp_diagnostic:
            report["ddp_diagnostic"] = check_ddp_parity(runner, data, diagnostic=True)
            (root / f"diagnostic-rank-{runner.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            env.close()
            if dist.is_initialized():
                dist.destroy_process_group()
            return
        if args.check_ddp_parity:
            report["ddp_parity"] = check_ddp_parity(runner, data)
        # Recorded real exploration must have no effect on imagined actions/objective.
        noise_state = runner.noise_generator.get_state()
        with torch.no_grad():
            original = runner.imagine(data)
            runner.noise_generator.set_state(noise_state)
            altered = dict(data, action_noise=torch.full_like(data["action_noise"], 100))
            without = runner.imagine(altered)
            torch.testing.assert_close(original[0], without[0], rtol=0, atol=0)
            torch.testing.assert_close(torch.stack(original[-1]), torch.stack(without[-1]), rtol=0, atol=0)
        report["no_imagined_action_exploration"] = True
        del original, without
        runner.wm_optimizer.zero_grad(set_to_none=True)
        runner.wm.requires_grad_(False)
        runner.value.requires_grad_(False)
        if runner.bptt["actor_batch"] <= 128:
            # A small batch fits both forms. Full-size validation must not run
            # the uncheckpointed form which already demonstrably exhausts VRAM.
            parameter = runner.actor.action_head[-1].bias
            noise_state = runner.noise_generator.get_state()
            runner.bptt["checkpoint_activations"] = False
            direct_loss = runner.imagine(data)[0]
            direct_gradient = torch.autograd.grad(direct_loss, parameter)[0]
            runner.noise_generator.set_state(noise_state)
            runner.bptt["checkpoint_activations"] = True
            recomputed_loss = runner.imagine(data)[0]
            recomputed_gradient = torch.autograd.grad(recomputed_loss, parameter)[0]
            torch.testing.assert_close(direct_loss, recomputed_loss, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(direct_gradient, recomputed_gradient, rtol=1e-4, atol=1e-6)
            report["checkpoint_gradient_parity"] = True
            del direct_loss, direct_gradient, recomputed_loss, recomputed_gradient
        noise_state = runner.noise_generator.get_state()
        loss, _, _, _, valid, states, actions = runner.imagine(data)
        future_map = runner.heightfield.local(states[1][:, :3], yaw_angle(states[1][:, 3:7]))[0]
        # The closed-loop transition is native even when the analytic objective
        # is compiled. Test the first action directly across the complete horizon.
        gradient_input = actions[0]
        map_gradient = torch.autograd.grad(future_map.square().mean(), gradient_input)[0]
        assert torch.isfinite(map_gradient).all() and map_gradient.abs().sum() > 0
        # Compiled backward may donate saved buffers. Use one backward per fresh
        # graph, with identical noise, rather than disabling production donation.
        runner.noise_generator.set_state(noise_state)
        loss, _, _, _, valid, states, actions = runner.imagine(data)
        action_gradient = torch.autograd.grad(loss, actions[0])[0]
        assert valid.all() and torch.isfinite(action_gradient).all() and action_gradient.abs().sum() > 0
        runner.noise_generator.set_state(noise_state)
        loss, _, _, _, valid, states, actions = runner.imagine(data)
        terminal_gradient = torch.autograd.grad(states[-1].square().mean(), actions[0])[0]
        assert torch.isfinite(terminal_gradient).all() and terminal_gradient.abs().sum() > 0
        runner.noise_generator.set_state(noise_state)
        loss, _, _, _, valid, states, actions = runner.imagine(data)
        loss.backward()
        assert all(p.grad is None for p in runner.wm.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in runner.actor.parameters())
        report.update(
            objective_gradient_norm=float(action_gradient.norm()),
            future_heightmap_gradient_norm=float(map_gradient.norm()),
            wm_parameters_frozen_in_actor_backward=True,
            gradient_input="first action",
            first_action_gradient_norm=float(action_gradient.norm()),
            terminal_state_first_action_gradient_norm=float(terminal_gradient.norm()),
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
        execution = runner.execution.copy()
        # BF16 quantization is not a smooth finite-difference oracle. Verify the
        # FP32 eager computational graph separately, then restore actual execution.
        runner.configure_execution(dict(execution, bf16=False, compile=False))
        noise_state = runner.noise_generator.get_state()
        value = runner.imagine(data)[0]
        gradient = torch.autograd.grad(value, parameter)[0]
        expected = (gradient * direction).sum().item()
        epsilon = 0.003
        with torch.no_grad():
            parameter.add_(epsilon * direction)
            runner.noise_generator.set_state(noise_state)
            plus = float(runner.imagine(data)[0])
            parameter.add_(-2 * epsilon * direction)
            runner.noise_generator.set_state(noise_state)
            minus = float(runner.imagine(data)[0])
            parameter.add_(epsilon * direction)
        actual = (plus - minus) / (2 * epsilon)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        runner.configure_execution(execution)
        runner.wm.requires_grad_(True)
        runner.value.requires_grad_(True)
        assert abs(actual - expected) < 0.05 * max(abs(expected), 0.01), (actual, expected)
        report.update(
            policy_direction_autograd=expected,
            policy_direction_finite_difference=actual,
            gradient_scope="Computational graph consistency; not a real-simulator Jacobian accuracy claim",
            finite_difference_execution="FP32 eager, TF32 disabled",
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
        saved_noise_rng = runner.noise_generator.get_state().clone()
        checkpoint = root / "acceptance.pt"
        runner.save(str(checkpoint))
        # A collective invalid window must preserve weights/counters and retain evidence.
        # Check before load(), which deliberately clears the simulator replay.
        sample_actor = runner.sample_actor
        invalid_data = sample_actor()
        invalid_data = dict(invalid_data, state=invalid_data["state"].clone())
        invalid_data["state"][:, 0, :2] = runner.heightfield.origin - 2.0
        runner.sample_actor = lambda: invalid_data
        counters = (runner.actor_updates, runner.critic_updates, runner.wm_updates)
        try:
            runner.update_bptt()
        except RuntimeError as error:
            assert "no actor update was applied" in str(error)
        else:
            raise AssertionError("Invalid imagined window was accepted")
        finally:
            runner.sample_actor = sample_actor
        assert counters == (runner.actor_updates, runner.critic_updates, runner.wm_updates)
        torch.testing.assert_close(runner.actor.state_dict(), saved_actor, rtol=0, atol=0)
        torch.testing.assert_close(runner.value.state_dict(), saved_value, rtol=0, atol=0)
        window_path = root / f"invalid-rollout-{runner.actor_updates}-rank-{runner.rank}.pt"
        window = torch.load(window_path, map_location="cpu", weights_only=True)
        report["invalid_rejection_probe"] = "actual initial state outside raster; no geometry monkeypatch"
        assert not window["valid"].any() and len(window["states"]) == runner.bptt["horizon"] + 1
        assert (root / f"invalid-rollout-{runner.actor_updates}.pt").is_file()
        runner.load(str(checkpoint))
        torch.testing.assert_close(runner.actor.state_dict(), saved_actor, rtol=0, atol=0)
        torch.testing.assert_close(runner.value.state_dict(), saved_value, rtol=0, atol=0)
        torch.testing.assert_close(runner.actor_optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
        torch.testing.assert_close(runner.noise_generator.get_state(), saved_noise_rng, rtol=0, atol=0)
        if dist.is_initialized():
            for module in (runner.actor, runner.wm, runner.value):
                for value in module.state_dict().values():
                    expected = value.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
        report.update(
            actor_updates=runner.actor_updates,
            critic_updates=runner.critic_updates,
            ppo_updates=0,
            exact_reload=True,
            normalizers_frozen=True,
            critic_architecture_unchanged=True,
            replay_capacity=runner.replay.capacity,
            wm_hidden=runner.options["hidden"],
            wm_amp=runner.options["amp"],
            checkpoint_activations=runner.bptt["checkpoint_activations"],
            execution=runner.execution,
            observation_noise_rng_exact_reload=True,
            independent_mlp=True,
            distributed=runner.distributed,
            learning_rates=runner.learning_rates,
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
