"""Real PGMT PPO plus terrain-conditioned differentiable imagination updates."""

import json
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
from .world_model import TerrainWorldModel, yaw_angle


def ordinary(tensor):
    # Rollouts normally use inference_mode. Replay must contain ordinary tensors
    # that autograd can save during subsequent supervised and actor updates.
    with torch.inference_mode(False):
        return tensor.detach().clone()


class WorldModelRunner(TerrainRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if dist.is_initialized():
            raise NotImplementedError("Scheme-one distributed gradient synchronization is not yet implemented")
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
            replay_chunks=8,
            actor_updates=1,
            validation_threshold=1.0,
        )
        self.wm_cfg.update(train_cfg.get("world_model", {}))
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
            if iteration >= self.wm_cfg["warmup"] and metrics["wm_validation"] < self.wm_cfg["validation_threshold"]:
                for _ in range(self.wm_cfg["actor_updates"]):
                    metrics.update(self.imagine())
            metrics["model_update_seconds"] = time.monotonic() - start
            metrics["imagination_updates"] = self.imagination_updates
            stats.update({"Scheme1/" + k: torch.tensor(v, device=self.device) for k, v in metrics.items()})
            if self.log_dir:
                with (Path(self.log_dir) / "world-model.jsonl").open("a") as file:
                    file.write(json.dumps(dict(iteration=iteration, **metrics)) + "\n")
            return losses, stats

        self.alg.update = update

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
        norm = self.normalizers["policy"]
        # EmpiricalNormalization's exact inverse; heights bypass it.
        raw = torch.cat((obs[:, :1678] * (norm._std[:, :1678] + norm.eps) + norm._mean[:, :1678], obs[:, 1678:]), -1)
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
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0, error_if_nonfinite=True)
            self.wm_optimizer.step()
            self.wm_updates += 1
        with torch.no_grad():
            validation = self.wm_loss(self.sample(self.wm_cfg["horizon"], validation=True))
        return dict(wm_loss=float(loss.detach()), wm_validation=float(validation), wm_updates=self.wm_updates)

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
                action = actor(obs)
                with torch.no_grad():
                    target_action = self.teacher(obs.detach())
                anchor_loss = anchor_loss + (action - target_action).square().mean()
                heights, _ = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
                state, contact = self.wm.step(state, action, previous_action, contact, heights)
                support, support_valid = self.mdp.update_support(state, support, sequence[index + 1])
                reward, survival = self.mdp.reward(state, action, previous_action, support, sequence[index + 1])
                discount = discount * valid * support_valid * (~data["done"])
                objective = objective + discount * reward
                discount = discount * 0.99 * survival
                history = torch.cat((history[:, 1:], self.mdp.proprioception(state, action)[:, None]), 1)
                previous_action = action
            loss = -objective.mean() + self.wm_cfg["anchor"] * anchor_loss / self.wm_cfg["horizon"]
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5, error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.imagination_updates += 1
            # Release the mutable shared-intent graph before the next PPO call.
            actor.intent = actor.intent.detach()
        finally:
            normalizer.train(was_training)
            self.wm.requires_grad_(True)
        return dict(
            imagined_return=float(objective.mean().detach()),
            imagination_grad=float(grad),
            anchor_loss=float(anchor_loss.detach()),
        )

    def save(self, path, infos=None):
        super().save(path, infos)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["scheme1"] = dict(
            world_model=self.wm.state_dict(),
            wm_optimizer=self.wm_optimizer.state_dict(),
            teacher=self.teacher.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            config=self.wm_cfg,
            wm_updates=self.wm_updates,
            imagination_updates=self.imagination_updates,
            start_gate=self.start_gate,
        )
        temporary = str(path) + ".scheme1.tmp"
        torch.save(saved, temporary)
        Path(temporary).replace(path)

    def load(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        model = saved["scheme1"]
        if model["config"] != self.wm_cfg:
            raise ValueError("World model configuration changed on resume")
        result = super().load(path)
        self.wm.load_state_dict(model["world_model"])
        self.teacher.load_state_dict(model["teacher"])
        self.wm_optimizer.load_state_dict(model["wm_optimizer"])
        self.actor_optimizer.load_state_dict(model["actor_optimizer"])
        self.wm_updates = model["wm_updates"]
        self.imagination_updates = model["imagination_updates"]
        self.start_gate = model["start_gate"]
        # Simulator and replay restart together; no sequence crosses this boundary.
        self.replay.clear()
        self.pending.clear()
        return result
