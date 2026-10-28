"""Real PGMT PPO plus terrain-conditioned differentiable imagination updates."""

import json
import math
import time
import torch
import torch.distributed as dist
from collections import deque
from copy import deepcopy
from pathlib import Path
from torch.nn import functional as F

from .heightfield import HeightField
from .imagination import ImaginationMDP, raw_history
from .terrain_runner import TerrainRunner
from .world_model import TerrainWorldModel, average_gradients, yaw_angle


def ordinary(tensor):
    # Rollouts normally use inference_mode. Replay must contain ordinary tensors
    # that autograd can save during subsequent supervised and actor updates.
    with torch.inference_mode(False):
        return tensor.detach().clone()


def frozen_terminal_value(actor_critic, policy_obs, critic_obs):
    """Keep state derivatives through V, without optimizing its shared feature map.

    The PGMT critic consumes actor intent. Freezing both networks for this forward
    prevents a direct actor-feature shortcut that changes V without changing an
    imagined action. Earlier action graphs still train the actor through state.
    """
    parameters = list(actor_critic.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        actor_critic.actor(policy_obs)
        return actor_critic.evaluate(critic_obs).sum(-1)
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)


@torch.no_grad()
def bounded_actor_step(actor, optimizer, observations, std, maximum_kl):
    """Backtrack an auxiliary Adam step against fixed replay observations.

    Imagination does not change action noise, so this is the exact diagonal-Gaussian
    KL on this batch. It bounds a policy change, not the true closed-loop return.
    """
    if maximum_kl == 0:
        optimizer.step()
        return 0.0, 1.0
    parameters = list(actor.parameters())
    original = [p.clone() for p in parameters]
    old_action = actor(observations).clone()
    optimizer_state = deepcopy(optimizer.state_dict())
    optimizer.step()
    delta = [p - old for p, old in zip(parameters, original, strict=True)]
    scale = 1.0
    for _ in range(8):
        kl = 0.5 * ((actor(observations) - old_action) / std).square().sum(-1).mean()
        if dist.is_initialized():
            dist.all_reduce(kl)
            kl.div_(dist.get_world_size())
        value = float(kl)
        if math.isfinite(value) and value <= maximum_kl:
            return value, scale
        if not math.isfinite(value):
            break
        scale *= min(0.5, 0.9 * math.sqrt(maximum_kl / value))
        for parameter, old, change in zip(parameters, original, delta, strict=True):
            parameter.copy_(old + scale * change)
    for parameter, old in zip(parameters, original, strict=True):
        parameter.copy_(old)
    optimizer.load_state_dict(optimizer_state)
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite imagined policy proposal; actor and optimizer were restored")
    return 0.0, 0.0


class WorldModelRunner(TerrainRunner):
    surrogate_version = 3

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if train_cfg.get("resume"):
            # A complete scheme-one checkpoint must not depend on its old initialization file.
            train_cfg = dict(train_cfg, warm_start=None, stage2_initialization=None)
        elif train_cfg.get("warm_start") and train_cfg.get("stage2_initialization"):
            raise ValueError("Choose stage-one warm-start or stage-two initialization, not both")
        super().__init__(env, train_cfg, log_dir, device)
        self.wm_cfg = dict(
            hidden=512,
            resolution=0.05,
            warmup=50,
            updates=16,
            batch_size=256,
            horizon=4,
            actor_lr=1e-5,
            anchor=1.0,
            teacher_ema=0.0,
            max_imagination_kl=0.0,
            bootstrap_value=0.0,
            replay_chunks=8,
            actor_updates=1,
            validation_threshold=1.0,
        )
        self.wm_cfg.update(train_cfg.get("world_model", {}))
        if not 0 <= self.wm_cfg["teacher_ema"] <= 1 or self.wm_cfg["max_imagination_kl"] < 0:
            raise ValueError("teacher_ema must be in [0,1] and max_imagination_kl must be nonnegative")
        if not 0 <= self.wm_cfg["bootstrap_value"] <= 1:
            raise ValueError("bootstrap_value must be in [0,1]")
        self.heightfield = HeightField.from_context(
            env.unwrapped.command_manager.get_term("motion").terrain, self.wm_cfg["resolution"]
        )
        self.mdp = ImaginationMDP(env.unwrapped, self.heightfield)
        self.wm = TerrainWorldModel(self.wm_cfg["hidden"]).to(device)
        self.wm_optimizer = torch.optim.AdamW(self.wm.parameters(), lr=3e-4, weight_decay=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.alg.actor_critic.actor.parameters(), lr=self.wm_cfg["actor_lr"])
        self.replay = deque(maxlen=self.wm_cfg["replay_chunks"])
        self.pending = []
        self.wm_updates = 0
        self.imagination_updates = 0
        self.stage2_initialization = train_cfg.get("stage2_initialization")
        self.start_gate = 0.0
        if self.stage2_initialization:
            self.initialize_stage2(self.stage2_initialization)
        self.teacher = deepcopy(self.alg.actor_critic.actor).eval().requires_grad_(False)
        self._capture_terminal_state()
        ppo_update = self.alg.update

        def update(iteration):
            start = time.monotonic()
            metrics = self.fit_world_model(iteration)
            losses, stats = ppo_update(iteration)
            if self.wm_cfg["teacher_ema"]:
                self.update_teacher()
            if (
                iteration >= self.wm_cfg["warmup"]
                and metrics["wm_validation"] < self.wm_cfg["validation_threshold"]
                and metrics["validation_valid_fraction"] >= 0.5
            ):
                for _ in range(self.wm_cfg["actor_updates"]):
                    metrics.update(self.imagine())
            metrics["model_update_seconds"] = time.monotonic() - start
            metrics["imagination_updates"] = self.imagination_updates
            stats.update({"Scheme1/" + k: torch.tensor(v, device=self.device) for k, v in metrics.items()})
            if self.log_dir and not self.is_mp_rank_other_process():
                with (Path(self.log_dir) / "world-model.jsonl").open("a") as file:
                    file.write(json.dumps(dict(iteration=iteration, **metrics)) + "\n")
            return losses, stats

        self.alg.update = update

    @torch.no_grad()
    def update_teacher(self):
        actor = self.alg.actor_critic.actor
        for target, source in zip(self.teacher.parameters(), actor.parameters(), strict=True):
            target.lerp_(source, self.wm_cfg["teacher_ema"])
        # In particular, terrain attention must use the current gate, not the flat gate.
        source_buffers = dict(actor.named_buffers())
        for name, target in self.teacher.named_buffers():
            target.copy_(source_buffers[name])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if dist.is_initialized():
            # Each rank has different terrain/data, but one shared dynamics model and teacher.
            for module in (self.wm, self.teacher):
                for value in module.state_dict().values():
                    dist.broadcast(value, src=0)
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def _prepare_extra_obs(self, extra_obs):
        # Keep the actual observation: after a distributed normalizer sync, inverting
        # an observation with the new moments does not recover its original values.
        self.raw_policy_obs = ordinary(extra_obs["policy"])
        return super()._prepare_extra_obs(extra_obs)

    def initialize_stage2(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved["infos"]["schema"] != "pgmt-lafan-terrain-v1":
            raise ValueError("Stage-two initialization requires the PGMT terrain schema")
        self.alg.actor_critic.load_state_dict(saved["model_state_dict"], strict=True)
        for key, normalizer in self.normalizers.items():
            normalizer.load_state_dict(saved[key + "_normalizer_state_dict"])
        self.start_gate = float(self.alg.actor_critic.actor.terrain_gate)
        self.alg.learning_rate = saved["infos"]["rank_states"][0]["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        self.source_checkpoint = str(path)
        print(f"[Scheme1] Initialized existing terrain policy from {path}; WM and optimizers start fresh", flush=True)

    def _capture_terminal_state(self):
        env = self.env.unwrapped
        original = env.reward_manager.compute

        def compute(dt):
            reward = original(dt)
            self.terminal_state = ordinary(self.mdp.state())
            self.terminal_contact = ordinary(self.mdp.command.contacts.contact.float())
            return reward

        env.reward_manager.compute = compute

    def rollout_step(self, obs, extra_obs):
        gate = max(
            self.start_gate, min(1.0, self.current_learning_iteration / max(1, self.cfg.get("terrain_gate_ramp", 200)))
        )
        self.alg.actor_critic.actor.terrain_gate.fill_(gate)
        self.mdp.command.metrics["terrain_gate"][:] = gate
        raw = self.raw_policy_obs
        record = {key: ordinary(value) for key, value in self.mdp.context().items()}
        record.update(
            state=ordinary(self.mdp.state()),
            history=ordinary(raw_history(raw)),
            previous_action=ordinary(self.env.unwrapped.action_manager.action),
            support=ordinary(self.mdp.command.terrain.support_z),
            contact=ordinary(self.mdp.command.contacts.contact.float()),
        )
        if not hasattr(self, "observation_parity"):
            reconstructed, valid = self.mdp.observations(record["state"], record["history"], record)
            self.observation_parity = dict(
                non_terrain_max=float((reconstructed[:, :1678] - raw[:, :1678]).abs().max()),
                terrain_mae=float((reconstructed[:, 1678:] - raw[:, 1678:]).abs().mean()),
                terrain_max=float((reconstructed[:, 1678:] - raw[:, 1678:]).abs().max()),
                valid_fraction=float(valid.float().mean()),
            )
            print("[Scheme1] Observation reconstruction:", self.observation_parity, flush=True)
            if (
                not self.env.unwrapped.cfg.observations.policy.enable_corruption
                and self.observation_parity["non_terrain_max"] > 2e-4
            ):
                raise ValueError("Imagination observations disagree with native PGMT")
        actions = self.alg.act(obs, extra_obs)
        record["action"] = ordinary(actions)
        obs, rewards, dones, infos = self.env.step(actions)
        record.update(next_state=self.terminal_state, next_contact=self.terminal_contact, done=ordinary(dones.bool()))
        self.pending.append(record)
        if len(self.pending) == self.num_steps_per_env:
            with torch.inference_mode(False):
                self.replay.append({key: torch.stack([r[key] for r in self.pending]) for key in record})
            self.pending.clear()
        extra_obs = infos["observations"]
        extra_obs.setdefault("policy", obs)
        obs = self._prepare_extra_obs(extra_obs)
        self.alg.process_env_step(rewards, dones, infos, obs, extra_obs)
        return obs, extra_obs, rewards, dones, infos

    def sample(self, length, validation=False):
        chunk = self.replay[-1] if validation else self.replay[int(torch.randint(len(self.replay), ()).item())]
        steps, envs = chunk["state"].shape[:2]
        if length >= steps:
            raise ValueError("Imagination horizon must be shorter than the collected rollout")
        # Entire held-out environments, rather than overlapping windows from the same trajectories.
        ids = torch.arange(envs, device=self.device)
        eligible = ids[ids % 8 == 0] if validation else ids[ids % 8 != 0]
        if not len(eligible):
            raise ValueError("Use at least two environments for separate WM training and validation")
        e = eligible[torch.randint(len(eligible), (self.wm_cfg["batch_size"],), device=self.device)]
        t = torch.randint(steps - length, e.shape, device=self.device)
        return [{key: value[t + i, e] for key, value in chunk.items()} for i in range(length + 1)]

    def wm_loss(self, sequence):
        state = sequence[0]["state"]
        contact = sequence[0]["contact"]
        alive = state.new_ones(state.shape[0])
        losses = []
        for data in sequence[:-1]:
            heights, valid = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            predicted, contact_logits = self.wm(state, data["action"], data["previous_action"], contact, heights)
            target = self.wm.delta_norm(self.wm.target(state, data["next_state"]))
            loss = (predicted - target).square().mean(-1)
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                contact_logits, data["next_contact"], reduction="none"
            ).mean(-1)
            weight = alive * valid
            losses.append((loss * weight).sum() / weight.sum().clamp_min(1))
            state, contact = self.wm.step(state, data["action"], data["previous_action"], contact, heights)
            alive = alive * (~data["done"]).float()
        return torch.stack(losses).mean()

    def fit_world_model(self, iteration):
        latest = self.replay[-1]
        if iteration <= self.wm_cfg["warmup"]:
            with torch.no_grad():
                train_envs = torch.arange(latest["state"].shape[1], device=self.device) % 8 != 0
                data = {key: value[:, train_envs].flatten(0, 1) for key, value in latest.items()}
                heights, _ = self.heightfield.local(data["state"][:, :3], yaw_angle(data["state"][:, 3:7]))
                self.wm.input_norm.update(
                    self.wm.inputs(data["state"], data["action"], data["previous_action"], data["contact"], heights)
                )
                self.wm.delta_norm.update(self.wm.target(data["state"], data["next_state"]))
        self.wm.train().requires_grad_(True)
        for _ in range(self.wm_cfg["updates"]):
            loss = self.wm_loss(self.sample(min(3, self.wm_cfg["horizon"])))
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            average_gradients(self.wm)
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0, error_if_nonfinite=True)
            self.wm_optimizer.step()
            self.wm_updates += 1
        with torch.no_grad():
            sequence = self.sample(self.wm_cfg["horizon"], validation=True)
            validation = self.wm_loss(sequence)
            diagnostics = self.validation_errors(sequence)
        metrics = dict(
            wm_loss=float(loss.detach()), wm_validation=float(validation), wm_updates=self.wm_updates, **diagnostics
        )
        if dist.is_initialized():
            # All ranks must make the same imagination-gate decision before collectives.
            values = torch.tensor(list(metrics.values()), device=self.device)
            dist.all_reduce(values)
            metrics = dict(zip(metrics, (values / dist.get_world_size()).tolist()))
        return metrics

    @torch.no_grad()
    def validation_errors(self, sequence):
        """Physical end-of-horizon errors, using the same held-out open-loop actions."""
        state, contact = sequence[0]["state"], sequence[0]["contact"]
        alive = state.new_ones(state.shape[0])
        for data in sequence[:-1]:
            heights, valid = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            alive = alive * valid
            state, contact = self.wm.step(state, data["action"], data["previous_action"], contact, heights)
            error = self.wm.target(state, data["next_state"])
            weight = alive
            alive = alive * (~data["done"])
        denominator = weight.sum().clamp_min(1)
        return {
            "validation_root_rmse_m": float(((error[:, :3].square().mean(-1) * weight).sum() / denominator).sqrt()),
            "validation_rotation_rmse_rad": float(
                ((error[:, 3:6].square().sum(-1) * weight).sum() / denominator).sqrt()
            ),
            "validation_joint_rmse_rad": float(
                ((error[:, 12:41].square().mean(-1) * weight).sum() / denominator).sqrt()
            ),
            "validation_valid_fraction": float(weight.mean()),
        }

    def imagine(self):
        self.wm.eval().requires_grad_(False)
        actor = self.alg.actor_critic.actor
        actor.intent = actor.intent.detach()
        sequence = self.sample(self.wm_cfg["horizon"])
        state, history, previous_action, contact, support = (
            sequence[0][key] for key in ("state", "history", "previous_action", "contact", "support")
        )
        normalizer = self.normalizers["policy"]
        was_training = normalizer.training
        normalizer.eval()
        discount = state.new_ones(state.shape[0])
        objective = state.new_zeros(state.shape[0])
        anchor_loss = state.new_zeros(())
        try:
            for index, data in enumerate(sequence[:-1]):
                obs, valid = self.mdp.observations(state, history, data)
                obs = normalizer(obs)
                if index == 0:
                    initial_obs = obs.detach()
                action = actor(obs)
                with torch.no_grad():
                    target_action = self.teacher(obs.detach())
                # No teacher loss on predicted states paired with a post-reset reference.
                # Detach the weight so predicting death cannot reduce the anchor penalty.
                weight = (discount * valid).detach()
                anchor_loss = anchor_loss + (
                    (action - target_action).square().mean(-1) * weight
                ).sum() / weight.sum().clamp_min(1)
                heights, _ = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
                previous_state = state
                state, contact = self.wm.step(state, action, previous_action, contact, heights)
                support, support_valid = self.mdp.update_support(state, support, sequence[index + 1])
                reward, survival = self.mdp.reward(
                    state, action, previous_action, support, sequence[index + 1], previous_state, contact
                )
                discount = discount * valid * support_valid * (~data["done"])
                objective = objective + discount * reward
                discount = discount * 0.99 * survival
                history = torch.cat(
                    (history[:, 1:], self.mdp.proprioception(state, action, sequence[index + 1])[:, None]), 1
                )
                previous_action = action
            reward_return = objective.mean().detach()
            bootstrap = state.new_zeros(state.shape[0])
            if self.wm_cfg["bootstrap_value"]:
                terminal_obs, valid = self.mdp.observations(state, history, sequence[-1])
                critic_obs = self.mdp.critic_observations(state, previous_action, support, sequence[-1])
                critic_normalizer = self.normalizers["critic"]
                critic_was_training = critic_normalizer.training
                try:
                    critic_normalizer.eval()
                    value = frozen_terminal_value(
                        self.alg.actor_critic, normalizer(terminal_obs), critic_normalizer(critic_obs)
                    )
                finally:
                    critic_normalizer.train(critic_was_training)
                bootstrap = self.wm_cfg["bootstrap_value"] * discount * valid * value
                objective = objective + bootstrap
            loss = -objective.mean() + self.wm_cfg["anchor"] * anchor_loss / self.wm_cfg["horizon"]
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            average_gradients(actor)
            grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5, error_if_nonfinite=True)
            kl, step_scale = bounded_actor_step(
                actor,
                self.actor_optimizer,
                initial_obs,
                self.alg.actor_critic.std.detach(),
                self.wm_cfg["max_imagination_kl"],
            )
            self.imagination_updates += int(step_scale > 0)
            # Release the mutable shared-intent graph before the next PPO call.
            actor.intent = actor.intent.detach()
        finally:
            normalizer.train(was_training)
            self.wm.requires_grad_(True)
        return dict(
            imagined_return=float(objective.mean().detach()),
            imagined_reward_return=float(reward_return),
            imagined_bootstrap=float(bootstrap.mean().detach()),
            imagination_grad=float(grad),
            anchor_loss=float(anchor_loss.detach()),
            imagination_kl=kl,
            imagination_step_scale=step_scale,
        )

    def save(self, path, infos=None):
        # Publish only the complete checkpoint; evaluators/resumers can read `path`
        # concurrently while the base runner gathers per-rank simulator state.
        base_path = str(path) + ".scheme1-base"
        super().save(base_path, infos)
        if self.is_mp_rank_other_process():
            dist.barrier()
            return
        saved = torch.load(base_path, map_location="cpu", weights_only=True)
        saved["scheme1"] = dict(
            world_model=self.wm.state_dict(),
            wm_optimizer=self.wm_optimizer.state_dict(),
            teacher=self.teacher.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            config=self.wm_cfg,
            wm_updates=self.wm_updates,
            imagination_updates=self.imagination_updates,
            start_gate=self.start_gate,
            surrogate_version=self.surrogate_version,
        )
        temporary = str(path) + ".scheme1.tmp"
        torch.save(saved, temporary)
        Path(temporary).replace(path)
        Path(base_path).unlink()
        if dist.is_initialized():
            dist.barrier()

    def load(self, path):
        # Optimizer.load_state_dict places parameter moments on the parameter device;
        # keep non-capturable Adam's scalar step counters on CPU, as on a fresh run.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = saved["scheme1"]
        saved_config = {"teacher_ema": 0.0, "max_imagination_kl": 0.0, "bootstrap_value": 0.0, **model["config"]}
        if model.get("surrogate_version", 1) != self.surrogate_version and not self.cfg.get(
            "allow_surrogate_change", False
        ):
            raise ValueError("Surrogate reward changed; explicitly set allow_surrogate_change for a new experiment")
        changed = {
            key for key in saved_config.keys() | self.wm_cfg.keys() if saved_config.get(key) != self.wm_cfg.get(key)
        }
        if changed and not (
            changed <= {"anchor", "teacher_ema", "max_imagination_kl", "bootstrap_value"}
            and self.cfg.get("allow_surrogate_change", False)
        ):
            raise ValueError(f"World model configuration changed on resume: {sorted(changed)}")
        result = super().load(path)
        self.wm.load_state_dict(model["world_model"])
        self.teacher.load_state_dict(model["teacher"])
        if saved_config["teacher_ema"] == 0 and self.wm_cfg["teacher_ema"] > 0:
            # Explicit activation after a PPO/WM warm-up starts from that learned policy,
            # not from an untrained terrain branch in the original flat teacher.
            self.teacher.load_state_dict(self.alg.actor_critic.actor.state_dict())
            print("[Scheme1] Activated EMA teacher from the resumed actor", flush=True)
        self.wm_optimizer.load_state_dict(model["wm_optimizer"])
        self.actor_optimizer.load_state_dict(model["actor_optimizer"])
        self.wm_updates = model["wm_updates"]
        self.imagination_updates = model["imagination_updates"]
        self.start_gate = model["start_gate"]
        # Simulator and replay restart together; no sequence crosses this boundary.
        self.replay.clear()
        self.pending.clear()
        return result
