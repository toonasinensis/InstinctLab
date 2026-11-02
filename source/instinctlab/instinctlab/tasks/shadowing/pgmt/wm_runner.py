"""PPO policy updates plus an independent, supervised Scheme A world model."""

import hashlib
import json
import time
import torch
import torch.distributed as dist
from pathlib import Path
from torch.nn import functional as F

import warp as wp
from instinct_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from .heightfield import HeightField
from .terrain_runner import TerrainRunner
from .wm_dynamics import TerrainDynamics, average_gradients, rotation_increment, yaw_angle
from .wm_replay import WMReplay
from .wm_rewards import AnalyticRewards


def ordinary(value):
    # Rollouts are inference tensors. Supervised autograd needs ordinary copies.
    with torch.inference_mode(False):
        return value.detach().clone()


class SupervisedWMRunner(TerrainRunner):
    schema = "pgmt-supervised-wm-v3"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.options = train_cfg["world_model"]
        horizons = [self.options["train_horizon"], *self.options["validation_horizons"]]
        capacity = self.options.get("replay_steps", self.options["replay_chunks"] * self.num_steps_per_env)
        if env.num_envs < 2 or min(horizons) < 1 or max(horizons) > capacity:
            raise ValueError("WM requires >=2 environments and replay capacity covering all horizons")
        if self.options["train_horizon"] > self.num_steps_per_env:
            raise ValueError("WM fitting must be possible after the first PPO rollout")
        if self.options["predict_force"] and not self.options["predict_contact"]:
            raise ValueError("The configured force ablation requires contact prediction")
        if (
            min(
                self.options[key]
                for key in ("updates", "batch_size", "validation_batch", "validation_interval", "replay_chunks")
            )
            < 1
        ):
            raise ValueError("WM update, batch, interval and replay sizes must be positive")
        if train_cfg.get("ppo_checkpoint") and train_cfg.get("warm_start"):
            raise ValueError("Choose terrain PPO initialization or native stage1 warm-start, not both")
        self.native = env.unwrapped
        self.device = torch.device(device)
        self.initial_gate = 0.0
        if train_cfg.get("ppo_checkpoint"):
            self.initialize_ppo(train_cfg["ppo_checkpoint"])
        # WM initialization/sampling must not consume the policy/reset RNG stream.
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(train_cfg["seed"] + 1741)
            self.heightfield = HeightField.from_context(
                self.native.command_manager.get_term("motion").terrain, self.options["resolution"]
            ).to(self.device)
            self.rewards = AnalyticRewards(self.native, self.heightfield)
            c = self.rewards.command
            self.context_modes = {
                name: self.native.event_manager.get_term_cfg(name).mode
                for name in ("physics_material", "add_joint_default_pos", "base_com")
            }
            if any(mode != "startup" for mode in self.context_modes.values()):
                raise ValueError(
                    "WM environment context requires startup-only randomization; snapshot per transition if changed"
                )
            material = self.native.event_manager.get_term_cfg("physics_material").func._impl
            com_event = self.native.event_manager.get_term_cfg("base_com").func
            robot = c.robot.data
            # Actual solver friction binding, randomized actuator zero and torso COM.
            self.xi = ordinary(
                torch.cat(
                    (
                        wp.to_torch(material._friction_binding),
                        robot.default_joint_pos.torch - robot.default_joint_pos_nominal,
                        (robot.body_com_pos_b.torch - com_event.default_com[..., :3])[
                            :, com_event.asset_cfg.body_ids
                        ].flatten(1),
                    ),
                    -1,
                )
            )
            self.wm = TerrainDynamics(
                self.xi.shape[-1],
                self.options["hidden"],
                self.options["amp"],
                self.options["predict_contact"],
                self.options["predict_force"],
            ).to(self.device)
        self.terrain_digest = hashlib.sha256(self.heightfield.heights.cpu().numpy().tobytes()).hexdigest()
        self.wm_optimizer = torch.optim.AdamW(self.wm.parameters(), lr=self.options["learning_rate"], weight_decay=1e-5)
        self.generator = torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + 2909)
        self.replay = WMReplay(capacity)
        self.frozen_validation = None
        self.wm_updates = 0
        self._world_model_loss = self.world_model_loss
        self.latest_metrics = {}
        self.capture_terminal()
        for value in self.wm.state_dict().values():
            if dist.is_initialized():
                dist.broadcast(value, src=0)
        update_ppo = self.alg.update

        def update(iteration):
            losses, stats = update_ppo(iteration)
            started = time.perf_counter()
            self.fit_world_model()
            fitted = time.perf_counter()
            ready = len(self.replay) >= max(self.options["validation_horizons"])
            validating = ready and (
                self.frozen_validation is None or iteration % self.options["validation_interval"] == 0
            )
            if validating:
                self.latest_metrics = self.validate()
                self.write_metrics(iteration)
            stats["wm/train_loss"] = torch.tensor(self.training_loss, device=self.device)
            stats["wm/fit_seconds"] = torch.tensor(fitted - started, device=self.device)
            stats["wm/validation_seconds"] = torch.tensor(time.perf_counter() - fitted, device=self.device)
            stats["wm/validation_ran"] = torch.tensor(float(validating), device=self.device)
            stats["wm/updates"] = torch.tensor(self.wm_updates, device=self.device)
            stats.update(
                {
                    "wm/" + key: torch.tensor(value, device=self.device)
                    for key, value in (self.latest_metrics.items() if validating else ())
                    if key not in {"updates", "train_loss"}
                    and not any("/" + name + "/" in key for name in ("oracle", "raster", "predicted"))
                }
            )
            stats["wm/seconds"] = torch.tensor(time.perf_counter() - started, device=self.device)
            return losses, stats

        self.alg.update = update
        if log_dir and not self.is_mp_rank_other_process():
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / "wm-contract.json").write_text(
                json.dumps(
                    dict(
                        schema=self.schema,
                        options=self.options,
                        rewards=self.rewards.contract,
                        predicted_reward_terms=self.rewards.supported_keys(
                            self.wm.predict_contact, self.wm.predict_force
                        ),
                        common_reward_terms=self.rewards.supported_keys(False, False),
                        contact_input=self.wm.predict_contact,
                        replay="chronological ring across PPO rollouts; terminating step included, successors masked",
                        validation_ready_after_steps=max(self.options["validation_horizons"]),
                        state="p3, quaternion_xyzw4, world_COM_v3, world_w3, q29, qd29",
                        context_width=self.xi.shape[-1],
                        context_modes=self.context_modes,
                        terrain_sha256=self.terrain_digest,
                        actor_update="PPO only; detached WM data",
                        holdout="local env_id % 8 == 0; excluded from fitting AND WM moments",
                        resume="optimizer continuation, fresh simulator episodes and replay",
                    ),
                    indent=2,
                )
                + "\n"
            )

    def initialize_ppo(self, path):
        """New WM experiment from learned terrain weights; fresh PPO optimizer/simulator."""
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved["infos"]["schema"] != self.native.cfg.mdp_schema:
            raise ValueError("PPO initialization requires a terrain PPO checkpoint")
        if saved["infos"].get("sampling_contract") != self.native.command_manager.get_term("motion").sampling_contract:
            raise ValueError("PPO initialization changed motion/terrain sampling contract")
        self.alg.actor_critic.load_state_dict(saved["model_state_dict"])
        for group, normalizer in self.normalizers.items():
            normalizer.load_state_dict(saved[group + "_normalizer_state_dict"])
        self.initial_gate = float(self.alg.actor_critic.actor.terrain_gate)
        self.alg.learning_rate = saved["infos"]["rank_states"][0]["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        self.source_checkpoint = str(path)

    def capture_terminal(self):
        original = self.native.reward_manager.compute

        def compute(dt):
            result = original(dt)
            c = self.rewards.command
            self.terminal = {"ref_" + key: ordinary(value) for key, value in self.rewards.context().items()}
            self.terminal.update(
                next_state=ordinary(self.rewards.state()),
                next_contact=ordinary(c.contacts.contact.float()),
                next_force=ordinary(c.contacts.force),
                next_support=ordinary(c.terrain.support_z),
                derivative_valid=ordinary(c.contacts.valid_derivative.float()),
                action=ordinary(self.native.action_manager.action),
                native_reward=ordinary(
                    torch.stack(
                        [
                            self.native.reward_manager._termwise_reward_buf[group][name] * dt
                            for key in self.rewards.terms
                            for group, name in [key.split("/")]
                        ],
                        -1,
                    )
                ),
            )
            return result

        self.native.reward_manager.compute = compute

    def rollout_step(self, obs, extra_obs):
        c = self.rewards.command
        record = dict(
            state=ordinary(self.rewards.state()),
            contact=ordinary(c.contacts.contact.float()),
            previous_action=ordinary(self.native.action_manager.action),
            support=ordinary(c.terrain.support_z),
            previous_ref_velocity=ordinary(c.contacts.previous_ref_velocity),
        )
        gate = max(
            self.initial_gate,
            min(1.0, self.current_learning_iteration / max(1, self.cfg.get("terrain_gate_ramp", 200))),
        )
        self.alg.actor_critic.actor.terrain_gate.fill_(gate)
        c.metrics["terrain_gate"][:] = gate
        result = OnPolicyRunner.rollout_step(self, obs, extra_obs)
        record.update(self.terminal, done=ordinary(result[3].bool()))
        self.replay.add(record)
        return result

    def sample(self, horizon, validation=False):
        eligible = torch.arange(self.env.num_envs, device=self.device)
        eligible = eligible[eligible % 8 == 0] if validation else eligible[eligible % 8 != 0]
        count = self.options["validation_batch"] if validation else self.options["batch_size"]
        ids = eligible[torch.randint(len(eligible), (count,), device=self.device, generator=self.generator)]
        return self.replay.sequences(ids, horizon, self.generator, history=False)

    @torch.no_grad()
    def update_moments(self):
        ids = torch.arange(self.env.num_envs, device=self.device)
        ids = ids[ids % 8 != 0]
        # Bound the heightmap temporary allocation at 4096 environments/GPU.
        inputs, deltas, forces = [], [], []
        rows = self.replay.rows(
            torch.arange(len(self.replay) - self.replay.added, len(self.replay), device=self.device)
        )
        for row in rows:
            data = {
                key: self.replay.data[key][row, ids]
                for key in ("state", "action", "previous_action", "contact", "next_state", "next_force")
            }
            heights, valid = self.heightfield.local(data["state"][:, :3], yaw_angle(data["state"][:, 3:7]))
            inputs.append(
                self.wm.inputs(
                    data["state"], data["action"], data["previous_action"], data["contact"], heights, self.xi[ids]
                )[valid]
            )
            deltas.append(self.wm.target(data["state"], data["next_state"])[valid])
            if self.wm.predict_force:
                rotation = data["next_state"][:, None, 3:7].expand(-1, 14, -1)
                forces.append(quat_apply_inverse(rotation, data["next_force"]).flatten(1)[valid])
        self.wm.input_norm.update(torch.cat(inputs))
        self.wm.delta_norm.update(torch.cat(deltas))
        if self.wm.predict_force:
            self.wm.force_norm.update(torch.cat(forces))
        self.replay.added = 0

    def predict(self, data, horizon):
        state = data["state"][:, 0]
        contact = data["contact"][:, 0] if self.wm.predict_contact else None
        states, contacts, forces, errors, losses, masks = [], [], [], [], [], []
        valid_prefix = torch.ones(len(state), device=self.device)
        scale = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
        for step in range(horizon):
            heights, valid = self.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
            state, logits, force = self.wm(
                state,
                data["action"][:, step],
                data["previous_action"][:, step],
                contact,
                heights,
                self.xi[data["env_id"]],
            )
            contact = logits.sigmoid() if logits is not None else None
            error = self.wm.state_error(state, data["next_state"][:, step])
            loss = (error / scale).square().mean(-1)
            if self.wm.predict_contact:
                loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                    logits, data["next_contact"][:, step], reduction="none"
                ).mean(-1)
                contacts.append(contact)
            if self.wm.predict_force:
                target_rotation = data["next_state"][:, step, None, 3:7].expand(-1, 14, -1)
                target_force = quat_apply_inverse(target_rotation, data["next_force"][:, step]).flatten(1)
                loss = loss + 0.1 * (force - self.wm.force_norm(target_force)).square().mean(-1)
                forces.append(force * self.wm.force_norm.std() + self.wm.force_norm.mean)
            valid_prefix = valid_prefix * valid
            masks.append(data["mask"][:, step] * valid_prefix)
            states.append(state)
            errors.append(error)
            losses.append(loss)
        masks = torch.stack(masks, 1)
        loss = (torch.stack(losses, 1) * masks).sum() / masks.sum().clamp_min(1)
        return (
            loss,
            torch.stack(errors, 1),
            masks,
            torch.stack(states, 1),
            torch.stack(contacts, 1) if contacts else None,
            torch.stack(forces, 1) if forces else None,
            torch.stack(losses, 1),
        )

    def world_model_loss(self, data):
        return self.predict(data, self.options["train_horizon"])[0]

    def fit_world_model(self, update_moments=True):
        if update_moments:
            self.update_moments()
        self.wm.train()
        for _ in range(self.options["updates"]):
            data = self.sample(self.options["train_horizon"])
            loss = self._world_model_loss(data)
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            average_gradients(self.wm)
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0, error_if_nonfinite=True)
            self.wm_optimizer.step()
            self.wm_updates += 1
        self.training_loss = float(loss.detach())

    def reduce_mean(self, value, weight):
        sums = torch.stack(((value * weight).sum(), weight.sum()))
        if dist.is_initialized():
            dist.all_reduce(sums)
        return float(sums[0] / sums[1].clamp_min(1))

    @torch.no_grad()
    def validate(self):
        if self.frozen_validation is None:
            self.frozen_validation = self.sample(max(self.options["validation_horizons"]), validation=True)
        metrics = {"updates": self.wm_updates, "train_loss": getattr(self, "training_loss", 0.0)}
        for dataset, data in (
            ("fixed", self.frozen_validation),
            ("current", self.sample(max(self.options["validation_horizons"]), validation=True)),
        ):
            # One H32 open-loop rollout supplies every requested prefix.
            _, all_errors, all_masks, all_states, all_contacts, all_forces, all_losses = self.predict(
                data, max(self.options["validation_horizons"])
            )
            all_supports = self.prediction_supports(data, all_states)
            for horizon in self.options["validation_horizons"]:
                errors, masks, states = all_errors[:, :horizon], all_masks[:, :horizon], all_states[:, :horizon]
                contacts = all_contacts[:, :horizon] if all_contacts is not None else None
                forces = all_forces[:, :horizon] if all_forces is not None else None
                loss = (all_losses[:, :horizon] * masks).sum() / masks.sum().clamp_min(1)
                weight, error = masks[:, -1], errors[:, -1]
                prefix = f"{dataset}/h{horizon}"
                metrics[prefix + "/loss"] = float(loss)
                metrics[prefix + "/valid_fraction"] = self.reduce_mean(weight, torch.ones_like(weight))
                initial = data["state"][:, 0]
                duration = horizon * self.native.step_dt
                spin = initial[:, 10:13]
                link_velocity = initial[:, 7:10] - torch.cross(
                    spin, quat_apply(initial[:, 3:7], self.rewards.root_com[data["env_id"]]), dim=-1
                )
                constant_velocity = torch.cat(
                    (
                        initial[:, :3] + link_velocity * duration,
                        quat_mul(
                            initial[:, 3:7], rotation_increment(quat_apply_inverse(initial[:, 3:7], spin) * duration)
                        ),
                        initial[:, 7:13],
                        initial[:, 13:42] + initial[:, 42:71] * duration,
                        initial[:, 42:71],
                    ),
                    -1,
                )
                cv_error = self.wm.state_error(constant_velocity, data["next_state"][:, horizon - 1])
                for name, section in (
                    ("root_m", slice(0, 3)),
                    ("rotation_rad", slice(3, 6)),
                    ("velocity_mps", slice(6, 9)),
                    ("angular_radps", slice(9, 12)),
                    ("joint_rad", slice(12, 41)),
                    ("joint_velocity_radps", slice(41, 70)),
                ):
                    metrics[prefix + "/" + name] = self.reduce_mean(error[:, section].square().mean(-1), weight) ** 0.5
                    stationary = self.wm.state_error(data["state"][:, 0], data["next_state"][:, horizon - 1])
                    metrics[prefix + "/stationary_" + name] = (
                        self.reduce_mean(stationary[:, section].square().mean(-1), weight) ** 0.5
                    )
                    metrics[prefix + "/constant_velocity_" + name] = (
                        self.reduce_mean(cv_error[:, section].square().mean(-1), weight) ** 0.5
                    )
                metrics.update(
                    self.reward_metrics(data, states, contacts, forces, masks, prefix, all_supports[:, horizon - 1])
                )
        return metrics

    def prediction_supports(self, data, states):
        support = data["support"][:, 0]
        supports = []
        for step in range(states.shape[1]):
            context = {key[4:]: value[:, step] for key, value in data.items() if key.startswith("ref_")}
            bodies = self.rewards.bodies(states[:, step], data["env_id"])
            support, _ = self.rewards.support(states[:, step], support, context, bodies)
            supports.append(support)
        return torch.stack(supports, 1)

    @torch.no_grad()
    def reward_metrics(self, data, states, contacts, forces, masks, prefix, support):
        h = states.shape[1]
        state = states[:, -1]
        previous = states[:, -2] if h > 1 else data["state"][:, 0]
        context = {key[4:]: value[:, h - 1] for key, value in data.items() if key.startswith("ref_")}
        contact, previous_contact = None, None
        if contacts is not None:
            contact = (contacts[:, -1] > 0.5).float()
            previous_contact = (contacts[:, -2] > 0.5).float() if h > 1 else data["contact"][:, 0]
        force = (
            None
            if forces is None
            else quat_apply(state[:, None, 3:7].expand(-1, 14, -1), forces[:, -1].reshape(-1, 14, 3))
        )
        predicted, _ = self.rewards.evaluate(
            state,
            previous,
            data["action"][:, h - 1],
            data["previous_action"][:, h - 1],
            contact,
            previous_contact,
            force,
            support,
            context,
            data["previous_ref_velocity"][:, h - 1],
            data["derivative_valid"][:, h - 1],
            data["env_id"],
        )
        predicted_keys = self.rewards.supported_keys(self.wm.predict_contact, self.wm.predict_force)
        metrics = {}
        # At H, first quantify formula/FK parity with true state + true mesh queries.
        real = data["next_state"][:, h - 1]
        kwargs = dict(
            state=real,
            previous_state=data["state"][:, h - 1],
            action=data["action"][:, h - 1],
            previous_action=data["previous_action"][:, h - 1],
            contact=data["next_contact"][:, h - 1],
            previous_contact=data["contact"][:, h - 1],
            force=data["next_force"][:, h - 1],
            support=data["next_support"][:, h - 1],
            context=context,
            previous_ref_velocity=data["previous_ref_velocity"][:, h - 1],
            derivative_valid=data["derivative_valid"][:, h - 1],
            ids=data["env_id"],
        )
        oracle, _ = self.rewards.evaluate(**kwargs, exact_surface=True)
        raster, _ = self.rewards.evaluate(**kwargs)
        weight = masks[:, -1]
        target = data["native_reward"][:, h - 1]
        metrics[prefix + "/reward_native_abs_mean"] = self.reduce_mean(target.sum(-1).abs(), weight)
        if contacts is not None:
            metrics[prefix + "/contact_brier"] = self.reduce_mean(
                (contacts[:, -1] - data["next_contact"][:, h - 1]).square().mean(-1), weight
            )
            feet = self.rewards.command.foot_ids
            pc = (contacts[:, -1, feet] > 0.5).float()
            rc = data["next_contact"][:, h - 1, feet]
            tp = self.reduce_mean((pc * rc).sum(-1), weight)
            pp = self.reduce_mean(pc.sum(-1), weight)
            rp = self.reduce_mean(rc.sum(-1), weight)
            metrics[prefix + "/foot_contact_f1"] = 2 * tp / max(pp + rp, 1e-8)
            metrics[prefix + "/contact_accuracy"] = self.reduce_mean(
                ((contacts[:, -1] > 0.5) == data["next_contact"][:, h - 1].bool()).float().mean(-1), weight
            )
        if forces is not None:
            metrics[prefix + "/force_rmse_N"] = (
                self.reduce_mean((force - data["next_force"][:, h - 1]).square().mean((-1, -2)), weight) ** 0.5
            )
        keys = list(self.rewards.terms)
        for name, output in (("oracle", oracle), ("raster", raster)):
            metrics[prefix + "/reward_" + name + "_mae"] = self.reduce_mean(
                (output.sum(-1) - target.sum(-1)).abs(), weight
            )
            for index, key in enumerate(keys):
                metrics[prefix + "/" + name + "/" + key] = self.reduce_mean(
                    (output[:, index] - target[:, index]).abs(), weight
                )
        target_indices = [keys.index(key) for key in predicted_keys]
        metrics[prefix + "/reward_supported_mae"] = self.reduce_mean(
            (predicted.sum(-1) - target[:, target_indices].sum(-1)).abs(), weight
        )
        metrics[prefix + "/reward_supported_native_abs_mean"] = self.reduce_mean(
            target[:, target_indices].sum(-1).abs(), weight
        )
        common = self.rewards.supported_keys(False, False)
        pred_common = [predicted_keys.index(key) for key in common]
        native_common = [keys.index(key) for key in common]
        metrics[prefix + "/reward_common_mae"] = self.reduce_mean(
            (predicted[:, pred_common].sum(-1) - target[:, native_common].sum(-1)).abs(), weight
        )
        metrics[prefix + "/reward_common_native_abs_mean"] = self.reduce_mean(
            target[:, native_common].sum(-1).abs(), weight
        )
        if len(predicted_keys) == len(keys):
            metrics[prefix + "/reward_predicted_mae"] = metrics[prefix + "/reward_supported_mae"]
        for index, key in enumerate(predicted_keys):
            metrics[prefix + "/predicted/" + key] = self.reduce_mean(
                (predicted[:, index] - target[:, target_indices[index]]).abs(), weight
            )
        true_map, valid = self.heightfield.local(real[:, :3], yaw_angle(real[:, 3:7]))
        predicted_map, pv = self.heightfield.local(states[:, -1, :3], yaw_angle(states[:, -1, 3:7]))
        metrics[prefix + "/heightmap_mae_m"] = self.reduce_mean(
            (true_map - predicted_map).abs().mean(-1), weight * valid * pv
        )
        return metrics

    def write_metrics(self, iteration):
        if self.log_dir and not self.is_mp_rank_other_process():
            with (Path(self.log_dir) / "wm-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(iteration=iteration, **self.latest_metrics)) + "\n")

    def save(self, path, infos=None):
        # TerrainRunner performs collective rank-state capture and atomic base save.
        base_path = str(path) + ".ppo"
        local = dict(sampling_rng=self.generator.get_state().cpu(), terrain_sha256=self.terrain_digest)
        ranks = [None] * dist.get_world_size() if dist.is_initialized() else [local]
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        super().save(base_path, infos)
        if not self.is_mp_rank_other_process():
            saved = torch.load(base_path, map_location="cpu", weights_only=True)
            saved["supervised_wm"] = dict(
                schema=self.schema,
                options=self.options,
                model=self.wm.state_dict(),
                optimizer=self.wm_optimizer.state_dict(),
                updates=self.wm_updates,
                ranks=ranks,
                reward_contract=self.rewards.contract,
                initial_gate=self.initial_gate,
            )
            temporary = str(path) + ".tmp"
            torch.save(saved, temporary)
            Path(temporary).replace(path)
            Path(base_path).unlink()
        if dist.is_initialized():
            dist.barrier()

    def load(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)["supervised_wm"]
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        changed = [
            key
            for key, expected in (
                ("schema", self.schema),
                ("options", self.options),
                ("reward_contract", self.rewards.contract),
            )
            if saved[key] != expected
        ]
        if len(saved["ranks"]) != world:
            changed.append("rank count")
        elif saved["ranks"][rank]["terrain_sha256"] != self.terrain_digest:
            changed.append(f"rank{rank} terrain raster")
        # Reject collectively: another rank must not enter TerrainRunner's
        # reset/normalizer collectives while this rank throws a local exception.
        if dist.is_initialized():
            differences = [None] * world
            dist.all_gather_object(differences, changed)
            changed = [f"rank{i}: {items}" for i, items in enumerate(differences) if items]
        if changed:
            raise ValueError(f"Supervised WM resume contract changed: {changed}")
        result = super().load(path)
        self.wm.load_state_dict(saved["model"])
        self.wm_optimizer.load_state_dict(saved["optimizer"])
        self.wm_updates = saved["updates"]
        self.initial_gate = saved["initial_gate"]
        self.generator.set_state(saved["ranks"][rank]["sampling_rng"])
        self.replay.clear()
        self.frozen_validation = None
        return result
