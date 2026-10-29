"""Standalone SuperMimic learner: real dynamics data, imagined actor gradients.

There is deliberately no PPO algorithm, teacher policy, or shared-intent critic
in this runner. Native simulation rewards are logged only.
"""

import hashlib
import json
import math
import os
import subprocess
import time
import torch
import torch.distributed as dist
from pathlib import Path
from torch.nn import functional as F

import warp as wp

from isaaclab.utils.math import quat_apply_inverse

from .heightfield import HeightField
from .imagination import raw_history
from .models.cross_attention import CrossAttentionActor
from .models.terrain import TerrainActor
from .pure_wm_mdp import PureImaginationMDP
from .pure_wm_model import ImaginedValue, TerrainDynamics, lambda_returns
from .pure_wm_replay import TransitionReplay
from .terrain_runner import TerrainNormalization
from .world_model import average_gradients, yaw_angle

STAGE1_SHA256 = "a4620507acd757baf31c239f8503eb4ccd72b950e21dfd8ece94d2f0ebf54305"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PureWorldModelRunner:
    schema = "pgmt-purewm-v1"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.device, self.cfg = env, torch.device(device), train_cfg
        self.native = env.unwrapped
        self.log_dir = Path(log_dir) if log_dir else None
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.options = dict(train_cfg["pure_wm"])
        if train_cfg.get("stage2_initialization"):
            raise ValueError("Pure WM only accepts original stage1 initialization")
        if self.native.cfg.observations.policy.enable_corruption:
            raise ValueError("The initial pure-WM contract uses identical noiseless real/imagined observations")
        for key in ("wm_batch", "actor_batch"):
            if self.options[key] % self.world_size:
                raise ValueError(f"Global {key} must be divisible by rank count")
        self.actor = TerrainActor().to(device)
        self.policy_normalizer = TerrainNormalization(2119).to(device).eval()
        self.lineage = None
        if not train_cfg.get("resume"):
            self.load_stage1(train_cfg["warm_start"])
        heightfield = HeightField.from_context(
            self.native.command_manager.get_term("motion").terrain, self.options["resolution"]
        )
        self.mdp = PureImaginationMDP(self.native, heightfield)
        material = self.native.event_manager.get_term_cfg("physics_material").func._impl
        robot = self.native.scene["robot"].data
        com_event = self.native.event_manager.get_term_cfg("base_com").func
        mu = wp.to_torch(material._friction_binding).clone()
        bias = robot.default_joint_pos.torch - robot.default_joint_pos_nominal
        com = (robot.body_com_pos_b.torch - com_event.default_com[..., :3])[:, com_event.asset_cfg.body_ids].flatten(1)
        self.xi = torch.cat((mu, bias, com), -1).clone()
        self.xi_schema = dict(
            friction=mu.shape[-1],
            joint_bias=bias.shape[-1],
            torso_com=com.shape[-1],
            restitution="not applied by pinned MuJoCo Warp solver",
        )
        self.wm = TerrainDynamics(self.xi.shape[-1], self.options["hidden"], self.options["amp"]).to(device)
        self.critic = ImaginedValue(296 + 14 + self.xi.shape[-1] + 441).to(device)
        lr = self.options["learning_rate"] / math.sqrt(self.world_size)
        self.wm_optimizer = torch.optim.Adam(self.wm.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.replay = TransitionReplay(self.options["replay_steps"], env.num_envs, device)
        self.iteration = self.real_frames = self.wm_updates = self.actor_updates = self.critic_updates = 0
        self.frozen_validation = None
        self.episode_step = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self.raw_observation, _ = env.get_observations()
        self.capture_terminal()
        for module in (self.actor, self.policy_normalizer, self.wm, self.critic):
            if dist.is_initialized():
                for value in module.state_dict().values():
                    dist.broadcast(value, src=0)
        self.reward_hash = hashlib.sha256(json.dumps(self.mdp.contract, sort_keys=True).encode()).hexdigest()
        if self.log_dir and self.rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "pure-wm-contract.json").write_text(
                json.dumps(
                    dict(
                        schema=self.schema,
                        options=self.options,
                        xi=self.xi_schema,
                        rewards=self.mdp.contract,
                        reward_hash=self.reward_hash,
                        lineage=self.lineage,
                        world_size=self.world_size,
                        local_envs=env.num_envs,
                        effective_lr=lr,
                        control_dt=self.native.step_dt,
                        actor_update="WM BPTT only",
                        critic="independent scalar imagined lambda return",
                        actor_normalizer="stage1 moments frozen; new velocity channels identity; terrain raw",
                        action="(mean + N(0, action_noise)).clamp(-20,20); deterministic deployment",
                    ),
                    indent=2,
                )
                + "\n"
            )

    @torch.no_grad()
    def load_stage1(self, path):
        if not path or digest(path) != STAGE1_SHA256:
            raise ValueError("Fresh pure-WM training requires the original stage1 model_2700 SHA256")
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved["infos"]["schema"] != "pgmt-lafan-native-v1":
            raise ValueError("Stage1 checkpoint schema mismatch")
        source = saved["model_state_dict"]
        for name, value in self.actor.state_dict().items():
            if name.startswith(("terrain_", "patch_offsets")):
                continue
            original = source["actor." + name]
            if name == "command_embedding.weight":
                value.zero_()
                value[:, :65].copy_(original)
            else:
                value.copy_(original)
        for name, value in self.policy_normalizer.state_dict().items():
            original = saved["policy_normalizer_state_dict"][name]
            if value.ndim and value.shape[-1] == 2119:
                value[..., :1645].copy_(original)
                torch.testing.assert_close(value[..., :1645], original, atol=0, rtol=0)
            else:
                value.copy_(original)
        baseline = CrossAttentionActor().to(self.device)
        baseline.load_state_dict({key[6:]: value for key, value in source.items() if key.startswith("actor.")})
        sample = torch.randn(64, 1645, device=self.device)
        old_norm = saved["policy_normalizer_state_dict"]
        raw_sample = sample * (old_norm["_std"] + 0.01) + old_norm["_mean"]
        expanded = torch.cat((raw_sample, sample.new_zeros(64, 474)), -1)
        # The embedding grows from 65 to 68 columns. Compare in full fp32:
        # TF32 selects different GEMM kernels for those equivalent shapes.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            difference = (self.actor(self.policy_normalizer(expanded)) - baseline(sample)).abs().max()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        if difference > 1e-5:
            raise ValueError(f"Expanded actor changed stage1 actions: {float(difference)}")
        self.actor.intent = self.actor.intent.detach()
        self.lineage = dict(
            path=str(path),
            sha256=STAGE1_SHA256,
            schema=saved["infos"]["schema"],
            iteration=saved["iter"],
            expanded_action_max_error=float(difference),
        )

    def capture_terminal(self):
        original = self.native.reward_manager.compute

        def compute(dt):
            reward = original(dt)
            c = self.mdp.command
            self.terminal = dict(
                next_state=self.mdp.state(),
                next_contact=c.contacts.contact.float().clone(),
                next_force=c.contacts.force.flatten(1).clone(),
            )
            self.real_metrics = {key: float(value) for key, value in self.native.extras["step"].items()}
            for key in self.mdp.omitted:
                group, name = key.split("/")
                self.real_metrics["omitted/" + key] = float(
                    self.native.reward_manager._termwise_reward_buf[group][name].mean() * dt
                )
            return reward

        self.native.reward_manager.compute = compute

    def action(self, observation, noise=True):
        action = self.actor(self.policy_normalizer(observation))
        if noise:
            action = action + self.options["action_noise"] * torch.randn_like(action)
        return action.clamp(-20, 20)

    @torch.no_grad()
    def collect(self, steps):
        for _ in range(steps):
            c = self.mdp.command
            state = self.mdp.state()
            record = self.mdp.windows.placement()
            history = raw_history(self.raw_observation)
            record.update(
                state=state,
                previous_action=self.native.action_manager.action.clone(),
                contact=c.contacts.contact.float().clone(),
                support=c.terrain.support_z.clone(),
                proprioception=history[:, -1].clone(),
                episode_step=self.episode_step.clone(),
            )
            if self.real_frames == 0:
                # Real collection may approach a clipped native reference window.
                # Full-window eligibility belongs to imagined start sampling.
                context = self.mdp.context()
                reconstructed, _ = self.mdp.observations(state, history, context)
                torch.testing.assert_close(reconstructed[:, :1678], self.raw_observation[:, :1678], atol=2e-4, rtol=0)
                torch.testing.assert_close(
                    self.mdp.proprioception(state, record["previous_action"], context),
                    history[:, -1],
                    atol=2e-4,
                    rtol=0,
                )
            action = self.action(self.raw_observation)
            self.raw_observation, reward, done, _ = self.env.step(action)
            # The action manager stores the normalized target before PD scaling.
            torch.testing.assert_close(
                self.native.action_manager.prev_action[~done.bool()],
                record["previous_action"][~done.bool()],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                self.native.action_manager.action[~done.bool()], action[~done.bool()], atol=0, rtol=0
            )
            record.update(self.terminal, action=action.clone(), done=done.bool().clone())
            self.replay.add(record)
            heights, _ = self.mdp.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            # Online statistics count each real transition once, following the base.
            self.wm.input_norm.update(
                self.wm.inputs(state, action, record["previous_action"], record["contact"], heights, self.xi)
            )
            self.wm.delta_norm.update(self.wm.target(state, record["next_state"]))
            rotation = record["next_state"][:, None, 3:7].expand(-1, 14, -1)
            force = quat_apply_inverse(rotation, record["next_force"].reshape(-1, 14, 3)).flatten(1)
            self.wm.force_norm.update(force)
            self.episode_step = torch.where(done.bool(), 0, self.episode_step + 1)
            self.real_frames += self.env.num_envs * self.world_size
            self.real_metrics["real_reward"] = float(reward.sum(-1).mean())
            self.real_metrics["clipped_action_fraction"] = float((action.abs() >= 20).float().mean())
            self.actor.intent = self.actor.intent.detach()

    def critic_features(self, state, action, support, contact, context, xi):
        native = self.mdp.critic_observations(state, action, support, context)
        heights, _ = self.mdp.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
        return torch.cat((native, contact, xi, heights), -1)

    def wm_sequence(self, data, length):
        state, contact = data["state"][:, 0], data["contact"][:, 0]
        xi = self.xi[data["env_id"]]
        errors, masks, contact_loss, force_loss = [], [], [], []
        valid_prefix = torch.ones(len(state), device=self.device)
        for step in range(length):
            heights, valid = self.mdp.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            state, logits, forces = self.wm(
                state, data["action"][:, step], data["previous_action"][:, step], contact, heights, xi
            )
            contact = logits.sigmoid()
            error = self.wm.state_error(state, data["next_state"][:, step])
            valid_prefix = valid_prefix * valid
            errors.append(error)
            masks.append(data["mask"][:, step] * valid_prefix)
            contact_loss.append(
                F.binary_cross_entropy_with_logits(logits, data["next_contact"][:, step], reduction="none").mean(-1)
            )
            target_rotation = data["next_state"][:, step, None, 3:7].expand(-1, 14, -1)
            target_force = quat_apply_inverse(target_rotation, data["next_force"][:, step].reshape(-1, 14, 3)).flatten(
                1
            )
            force_loss.append((forces - self.wm.force_norm(target_force)).square().mean(-1))
        errors, masks = torch.stack(errors, 1), torch.stack(masks, 1)
        scale = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
        loss = (
            (errors / scale).square().mean(-1) + 0.1 * torch.stack(contact_loss, 1) + 0.1 * torch.stack(force_loss, 1)
        )
        return (loss * masks).sum() / len(state), errors, masks

    def optimizer_step(self, module, optimizer, value_clip=False):
        average_gradients(module)
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
        if value_clip:
            torch.nn.utils.clip_grad_value_(module.parameters(), 0.1)
        optimizer.step()
        return float(norm)

    def fit_world_model(self):
        batch = self.options["wm_batch"] // self.world_size
        length = self.options["wm_window"]
        total = 0.0
        for _ in range(self.options["wm_updates"]):
            self.wm_optimizer.zero_grad(set_to_none=True)
            for begin in range(0, batch, self.options["wm_microbatch"]):
                count = min(batch - begin, self.options["wm_microbatch"])
                data = self.replay.sequences(count, length)
                loss, _, _ = self.wm_sequence(data, length)
                (loss * count / batch).backward()
                total += float(loss.detach()) * count / batch / length
            self.optimizer_step(self.wm, self.wm_optimizer)
            self.wm_updates += 1
        return total / self.options["wm_updates"]

    @torch.no_grad()
    def validation(self):
        metrics = {}
        checks = [
            (f"h{horizon}", self.frozen_validation, horizon)
            for horizon in sorted({1, self.options["horizon"], self.options["validation_horizon"]})
        ]
        checks.append(
            (
                "current",
                self.replay.sequences(self.options["validation_batch"], self.options["horizon"], validation=True),
                self.options["horizon"],
            )
        )
        for prefix, data, horizon in checks:
            _, error, mask = self.wm_sequence(data, horizon)
            weight = mask[:, -1]
            error = error[:, -1]
            stats = torch.stack(
                (
                    (error[:, :3].square().mean(-1) * weight).sum(),
                    (error[:, 3:6].square().sum(-1) * weight).sum(),
                    (error[:, 12:41].square().mean(-1) * weight).sum(),
                    weight.sum(),
                    weight.new_tensor(len(weight)),
                )
            )
            if dist.is_initialized():
                dist.all_reduce(stats)
            physical = (stats[:3] / stats[3].clamp_min(1)).sqrt().tolist()
            for name, value in zip(("root_m", "rotation_rad", "joint_rad"), physical):
                metrics[f"{prefix}/{name}"] = value
            metrics[f"{prefix}/valid_fraction"] = float(stats[3] / stats[4])
            null_error = self.wm.state_error(data["state"][:, 0], data["next_state"][:, horizon - 1])
            null_stats = torch.stack(
                (
                    (null_error[:, :3].square().mean(-1) * weight).sum(),
                    (null_error[:, 12:41].square().mean(-1) * weight).sum(),
                    weight.sum(),
                )
            )
            if dist.is_initialized():
                dist.all_reduce(null_stats)
            metrics[f"{prefix}/stationary_root_m"] = float((null_stats[0] / null_stats[2].clamp_min(1)).sqrt())
            metrics[f"{prefix}/stationary_joint_rad"] = float((null_stats[1] / null_stats[2].clamp_min(1)).sqrt())
            if prefix == "current":
                for family in range(5):
                    selected = (data["family"][:, 0] == family).float() * weight
                    family_stats = torch.stack(
                        (
                            (error[:, :3].square().mean(-1) * selected).sum(),
                            (error[:, 12:41].square().mean(-1) * selected).sum(),
                            selected.sum(),
                        )
                    )
                    if dist.is_initialized():
                        dist.all_reduce(family_stats)
                    metrics[f"current/family{family}/count"] = float(family_stats[2])
                    if family_stats[2] > 0:
                        metrics[f"current/family{family}/root_m"] = float((family_stats[0] / family_stats[2]).sqrt())
                        metrics[f"current/family{family}/joint_rad"] = float((family_stats[1] / family_stats[2]).sqrt())
        h = self.options["horizon"]
        thresholds = self.options["physical_gate"]
        passed = all(
            math.isfinite(metrics[f"{prefix}/{name}"]) and metrics[f"{prefix}/{name}"] < value
            for prefix in (f"h{h}", "current")
            for name, value in thresholds.items()
        )
        passed = passed and min(metrics[f"h{h}/valid_fraction"], metrics["current/valid_fraction"]) >= 0.5
        return passed, metrics

    def imagine_batch(self, count):
        horizon, gamma = self.options["horizon"], self.options["gamma"]
        start, history = self.replay.starts(count, horizon, self.mdp.windows.lengths)
        state, previous_action, contact, support = (
            start[key] for key in ("state", "previous_action", "contact", "support")
        )
        xi = self.xi[start["env_id"]]
        context = self.mdp.reference_context(start, 0)
        discount, objective = state.new_ones(count), state.new_zeros(count)
        features = [self.critic_features(state, previous_action, support, contact, context, xi).detach()]
        rewards, survival_steps = [], []
        for step in range(horizon):
            observation, valid = self.mdp.observations(state, history, context)
            action = self.action(observation)
            heights, _ = self.mdp.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            previous_state, previous_contact, previous_context = state, contact, context
            state, logits, _ = self.wm(state, action, previous_action, contact, heights, xi)
            contact = logits.sigmoid()
            context = self.mdp.reference_context(start, step + 1)
            support, support_valid = self.mdp.update_support(state, support, context)
            reward, survival, _ = self.mdp.objective(
                state,
                action,
                previous_action,
                support,
                context,
                previous_state,
                contact,
                previous_contact,
                previous_context,
                (start["episode_step"] + step > 0).float(),
            )
            # Pay the terminating step's reward, then absorb. Real replay done
            # describes another action trajectory and is never used here.
            objective = objective + discount * reward
            survival = survival * valid * support_valid
            discount = discount * gamma * survival
            rewards.append(reward.detach())
            survival_steps.append(survival.detach())
            history = torch.cat((history[:, 1:], self.mdp.proprioception(state, action, context)[:, None]), 1)
            previous_action = action
            feature = self.critic_features(state, action, support, contact, context, xi)
            features.append(feature.detach())
        objective = objective + discount * self.critic(feature)
        features = torch.stack(features)
        with torch.no_grad():
            values = self.critic(features.flatten(0, 1)).reshape(horizon + 1, count)
            targets = lambda_returns(
                torch.stack(rewards), torch.stack(survival_steps), values, gamma, self.options["lambda"]
            )
        return (
            -objective.mean(),
            features[:-1].flatten(0, 1),
            targets.flatten(),
            float((discount / gamma**horizon).detach().mean()),
        )

    def update_actor(self):
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.wm_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        batch = self.options["actor_batch"] // self.world_size
        all_features, all_targets, total, alive = [], [], 0.0, 0.0
        try:
            for begin in range(0, batch, self.options["actor_microbatch"]):
                count = min(batch - begin, self.options["actor_microbatch"])
                loss, features, targets, survival = self.imagine_batch(count)
                (loss * count / batch).backward()
                self.actor.intent = self.actor.intent.detach()
                all_features.append(features)
                all_targets.append(targets)
                total += float(loss.detach()) * count / batch
                alive += survival * count / batch
            norm = self.optimizer_step(self.actor, self.actor_optimizer, value_clip=True)
            self.actor_updates += 1
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        features, targets = torch.cat(all_features), torch.cat(all_targets)
        # Online critic moments use detached data; never change them mid-BPTT.
        self.critic.normalizer.update(features)
        value_loss = 0.0
        for _ in range(self.options["critic_epochs"]):
            permutation = torch.randperm(len(features), device=self.device)
            for ids in permutation.chunk(self.options["critic_minibatches"]):
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss = (self.critic(features[ids]) - targets[ids]).square().mean()
                loss.backward()
                self.optimizer_step(self.critic, self.critic_optimizer)
                self.critic_updates += 1
                value_loss += float(loss.detach())
        return dict(
            actor_loss=total,
            actor_gradient_norm=norm,
            imagination_survival=alive,
            critic_loss=value_loss / (self.options["critic_epochs"] * self.options["critic_minibatches"]),
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("Pure-WM history requires actual episode age")
        started = time.monotonic()
        self.collect(self.options["fill_steps"])
        self.frozen_validation = self.replay.sequences(
            self.options["validation_batch"], self.options["validation_horizon"], validation=True
        )
        gate, metrics = False, {}
        end = self.iteration + num_learning_iterations
        for iteration in range(self.iteration, end):
            self.iteration = iteration
            # UTD .4: one learner iteration for 2.5 new steps per environment.
            steps = math.floor((iteration + 1) / self.options["utd"]) - math.floor(iteration / self.options["utd"])
            self.collect(steps)
            loss = self.fit_world_model()
            if iteration % self.options["validation_interval"] == 0:
                gate, metrics = self.validation()
            actor_metrics = {}
            if iteration >= self.options["warmup_iterations"] and gate:
                self.actor.terrain_gate.fill_(min(1.0, (self.actor_updates + 1) / self.options["terrain_gate_ramp"]))
                actor_metrics = self.update_actor()
            self.mdp.command.metrics["terrain_gate"][:] = self.actor.terrain_gate
            row = dict(
                iteration=iteration,
                wall_seconds=time.monotonic() - started,
                real_frames=self.real_frames,
                wm_loss=loss,
                gate=gate,
                wm_updates=self.wm_updates,
                actor_updates=self.actor_updates,
                critic_updates=self.critic_updates,
                **metrics,
                **self.real_metrics,
                **actor_metrics,
            )
            if self.rank == 0 and iteration % self.cfg["log_interval"] == 0:
                print("[PureWM] " + json.dumps(row), flush=True)
                if self.log_dir:
                    with (self.log_dir / "pure-wm.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
            self.iteration = iteration + 1
            if self.log_dir and (self.iteration % self.cfg["save_interval"] == 0 or self.iteration == end):
                self.save(self.log_dir / f"model_{self.iteration}.pt")

    def save(self, path):
        c, terrain = self.mdp.command, self.native.scene.terrain
        local = dict(
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device),
            failures=c.stratum_failure.cpu(),
            visits=c.stratum_visits.cpu(),
            levels=terrain.terrain_levels.cpu(),
            types=terrain.terrain_types.cpu(),
        )
        ranks = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        if self.rank == 0:
            checkpoint = dict(
                schema=self.schema,
                actor=self.actor.state_dict(),
                policy_normalizer=self.policy_normalizer.state_dict(),
                wm=self.wm.state_dict(),
                critic=self.critic.state_dict(),
                actor_optimizer=self.actor_optimizer.state_dict(),
                wm_optimizer=self.wm_optimizer.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict(),
                options=self.options,
                xi_schema=self.xi_schema,
                lineage=self.lineage,
                reward_hash=self.reward_hash,
                ranks=ranks,
                local_envs=self.env.num_envs,
                iteration=self.iteration,
                real_frames=self.real_frames,
                wm_updates=self.wm_updates,
                actor_updates=self.actor_updates,
                critic_updates=self.critic_updates,
                resume_semantics="fresh episodes and replay refill; optimizer continuation, not simulator-exact replay",
            )
            torch.save(checkpoint, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
        if dist.is_initialized():
            dist.barrier()

    def load(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved.get("schema") != self.schema or saved["lineage"]["sha256"] != STAGE1_SHA256:
            raise ValueError("Resume requires a pure-WM checkpoint with original stage1 lineage")
        if (
            saved["options"] != self.options
            or saved["reward_hash"] != self.reward_hash
            or saved["xi_schema"] != self.xi_schema
        ):
            raise ValueError("Pure-WM resume changed algorithm/reward/context contract")
        if len(saved["ranks"]) != self.world_size or saved["local_envs"] != self.env.num_envs:
            raise ValueError("Pure-WM resume changed rank or environment count")
        for name in (
            "actor",
            "policy_normalizer",
            "wm",
            "critic",
            "actor_optimizer",
            "wm_optimizer",
            "critic_optimizer",
        ):
            getattr(self, name).load_state_dict(saved[name])
        for name in ("iteration", "real_frames", "wm_updates", "actor_updates", "critic_updates", "lineage"):
            setattr(self, name, saved[name])
        local = saved["ranks"][self.rank]
        c, terrain = self.mdp.command, self.native.scene.terrain
        c.stratum_failure.copy_(local["failures"])
        c.stratum_visits.copy_(local["visits"])
        terrain.terrain_levels.copy_(local["levels"])
        terrain.terrain_types.copy_(local["types"])
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        torch.set_rng_state(local["torch_rng"].cpu())
        torch.cuda.set_rng_state(local["cuda_rng"].cpu(), self.device)
        c.started.zero_()
        self.native.termination_manager.terminated.zero_()
        self.raw_observation, _ = self.env.reset()
        self.episode_step.zero_()
        self.replay.clear()
        self.frozen_validation = None
        if self.rank == 0 and self.log_dir:
            contract_path = self.log_dir / "pure-wm-contract.json"
            contract = json.loads(contract_path.read_text())
            contract.update(lineage=self.lineage, resume_from=str(path), resume_semantics=saved["resume_semantics"])
            contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        print(f"[PureWM] resumed {path}; fresh episodes, refill {self.options['fill_steps']} steps", flush=True)

    def add_git_repo_to_log(self, path):
        if self.rank == 0 and self.log_dir:
            commit = subprocess.run(
                ["git", "-C", str(Path(path).parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            )
            with (self.log_dir / "source-revisions.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(path=str(path), commit=commit.stdout.strip())) + "\n")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.actor.to(device)
            self.policy_normalizer.to(device)
        self.actor.eval()
        return lambda observations: self.action(observations, noise=False)
