"""Atomic checkpoints with explicit ownership and a legacy inference export."""

import torch
from dataclasses import asdict
from pathlib import Path

SCHEMA = "pgmt-supermimic-composed-v1"


def environment_state(runner):
    c, terrain = runner.components.collector.command, runner.env.unwrapped.scene.terrain
    return dict(
        failures=c.stratum_failure,
        visits=c.stratum_visits,
        levels=terrain.terrain_levels,
        types=terrain.terrain_types,
        original_types=c.original_types,
        original_levels=c.original_levels,
        original_curriculum=c.original_curriculum,
        original_randomize=c.original_randomize,
        flat_warmup_active=c.flat_warmup_active,
    )


def restore_environment(runner, state):
    c, terrain = runner.components.collector.command, runner.env.unwrapped.scene.terrain
    c.stratum_failure.copy_(state["failures"])
    c.stratum_visits.copy_(state["visits"])
    terrain.terrain_levels.copy_(state["levels"])
    terrain.terrain_types.copy_(state["types"])
    terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
    c.original_types.copy_(state["original_types"])
    c.original_levels.copy_(state["original_levels"])
    for name in ("original_curriculum", "original_randomize", "flat_warmup_active"):
        setattr(c, name, state[name])
    c.cfg.enable_terrain_curriculum = c.original_curriculum and not c.flat_warmup_active
    c.cfg.randomize_terrain_on_reset = c.original_randomize and not c.flat_warmup_active
    c.metrics["flat_warmup"].fill_(float(c.flat_warmup_active))
    c.started.zero_()
    runner.env.unwrapped.termination_manager.terminated.zero_()
    runner.components.collector.observation, _ = runner.env.reset()
    runner.components.collector.episode_rewards.running.zero_()
    runner.components.replay.clear()


def contract(runner):
    return dict(
        options=asdict(runner.options),
        policy=runner.cfg["policy"],
        num_envs=runner.env.num_envs,
        sampling=runner.components.collector.command.sampling_contract,
        rewards=runner.components.rewards.contract,
        contact=runner.cfg["foot_contact"],
        analytic=runner.cfg["analytic_foot"],
        flat_warmup_updates=runner.cfg["flat_warmup_updates"],
        termination_softness=runner.cfg["termination_softness"],
    )


def save(runner, path):
    path = Path(path)
    payload = dict(
        schema=SCHEMA,
        contract=contract(runner),
        policy=runner.components.policy_learner.state_dict(),
        world_model=runner.components.wm_learner.state_dict(),
        environment=environment_state(runner),
        normalizers={key: module.state_dict() for key, module in runner.components.normalizers.items()},
        generators={key: generator.get_state() for key, generator in runner.components.generators.items()},
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(runner.device),
        iteration=runner.current_learning_iteration,
        learner_round=runner.learner_round,
        warmup_complete=runner.warmup_complete,
        collection_steps=runner.components.collector.steps,
        step_accumulator=runner.components.collector.step_accumulator,
        normalize_during_warmup=runner.components.collector.normalize_during_warmup,
    )
    # Existing evaluation/play readers can extract the raw or EMA actor as before.
    output = dict(
        supermimic=payload,
        iter=runner.current_learning_iteration,
        model_state_dict=runner.components.networks.state_dict(),
        **{
            key + "_normalizer_state_dict": normalizer.state_dict()
            for key, normalizer in runner.components.normalizers.items()
        },
        infos=dict(
            schema=runner.env.unwrapped.cfg.mdp_schema,
            sampling_contract=runner.components.collector.command.sampling_contract,
            user=dict(foot_contact=runner.cfg["foot_contact"], analytic_foot=runner.cfg["analytic_foot"]),
        ),
        scan187=dict(
            actor_ema=runner.components.policy_learner.ema.state_dict(),
            ema_updates=runner.components.policy_learner.actor_updates,
            scan=runner.env.unwrapped.cfg.height_scan,
            heightfield_resolution=runner.env.unwrapped.cfg.heightfield_resolution,
        ),
        bptt=dict(
            actor_updates=runner.components.policy_learner.actor_updates,
            critic_updates=runner.components.policy_learner.critic_updates,
            value=runner.components.value.state_dict(),
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(path)


def load(runner, path):
    payload = torch.load(path, map_location=runner.device, weights_only=True)["supermimic"]
    if payload["schema"] != SCHEMA or payload["contract"] != contract(runner):
        raise ValueError("Resume configuration differs; use explicit weight initialization for a changed experiment")
    runner.components.policy_learner.load_state_dict(payload["policy"])
    runner.components.wm_learner.load_state_dict(payload["world_model"])
    for name, state in payload["normalizers"].items():
        runner.components.normalizers[name].load_state_dict(state)
    for name, state in payload["generators"].items():
        runner.components.generators[name].set_state(state.cpu())
    torch.set_rng_state(payload["torch_rng"].cpu())
    torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), runner.device)
    runner.current_learning_iteration = payload["iteration"]
    runner.learner_round = payload["learner_round"]
    runner.warmup_complete = payload["warmup_complete"]
    runner.components.collector.steps = payload["collection_steps"]
    runner.components.collector.step_accumulator = payload["step_accumulator"]
    runner.components.collector.normalize_during_warmup = payload["normalize_during_warmup"]
    restore_environment(runner, payload["environment"])
    # Simulator episodes/replay restart; model, optimizer, clocks and curriculum continue.


def initialize_legacy_weights(runner, path):
    """Explicit weight initialization; does not claim optimizer/replay continuation."""
    saved = torch.load(path, map_location=runner.device, weights_only=True)
    source = {
        key.removeprefix("actor."): value
        for key, value in saved["model_state_dict"].items()
        if key.startswith("actor.")
    }
    stage_one = saved.get("infos", {}).get("schema") == "pgmt-lafan-native-v1"
    if stage_one:
        target = runner.components.policy.network.state_dict()
        for key, value in target.items():
            if key in source:
                if key == "command_embedding.weight":
                    value.zero_()
                    value[:, : source[key].shape[-1]].copy_(source[key])
                else:
                    value.copy_(source[key])
            elif not key.startswith(("terrain_", "patch_offsets")):
                raise ValueError(f"Stage-one actor tensor missing: {key}")
        source = target
    runner.components.policy.network.load_state_dict(source)
    for name, normalizer in runner.components.normalizers.items():
        source = saved[name + "_normalizer_state_dict"]
        if stage_one and name == "policy":
            target = normalizer.state_dict()
            for key, value in target.items():
                if value.ndim and value.shape != source[key].shape:
                    value[..., : source[key].shape[-1]].copy_(source[key])
                else:
                    value.copy_(source[key])
            source = target
        normalizer.load_state_dict(source)
    runner.components.collector.normalize_during_warmup = False
    if "supervised_wm" in saved:
        source = saved["supervised_wm"]["model"]
        mapped = {}
        for key, value in runner.components.dynamics.state_dict().items():
            old_key = key.removeprefix("representation.")
            if old_key == "nominal_error":
                mapped[key] = value
            elif old_key in source:
                mapped[key] = source[old_key]
            elif old_key == "root_com":
                mapped[key] = value
            else:
                raise ValueError(f"Legacy WM is missing {old_key}")
        runner.components.dynamics.load_state_dict(mapped)
    if "bptt" in saved:
        runner.components.value.load_state_dict(saved["bptt"]["value"])
    runner.components.policy_learner.ema.load_state_dict(
        saved.get("scan187", {}).get("actor_ema", runner.components.networks.actor.state_dict())
    )
