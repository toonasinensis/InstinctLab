"""Pure model-gradient actor updates with real/imagined critic supervision."""

import json
import time
import torch
import torch.distributed as dist
from pathlib import Path
from torch import nn
from torch.utils.checkpoint import checkpoint

from tensorboardX import SummaryWriter

from .bptt_distributed import backward, distributed_objective, learning_rates, scheduled_learning_rates
from .bptt_mdp import BPTTMDP, OBSERVATION_WIDTHS, PROPRIO_WIDTHS, lambda_returns, raw_history
from .wm_runner import SupervisedWMRunner, ordinary


def compile_forward(forward, *, dynamic=None):
    from torch._dynamo import config as dynamo_config
    from torch._functorch import config as functorch_config

    compiled = torch.compile(forward, fullgraph=True, dynamic=dynamic, options={"triton.cudagraphs": False})

    def call(*args):
        # AOTAutograd captures the backward assumption during the lazy first
        # invocation. All backward calls run outside the network autocast scopes.
        # DDP wraps the native horizon outside these accepted compile regions.
        # Keep its active-forward context from repartitioning their graphs.
        with (
            dynamo_config.patch(optimize_ddp=False),
            functorch_config.patch(backward_pass_autocast="off"),
        ):
            return compiled(*args)

    return call


class IndependentValue(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.amp = False
        self.amp_heads = True
        self.network = nn.Sequential(
            nn.Linear(width, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        with torch.autocast(
            device_type=features.device.type, dtype=torch.bfloat16, enabled=self.amp and features.is_cuda
        ):
            if not self.amp or self.amp_heads:
                return self.network(features).float().squeeze(-1)
            for index in range(len(self.network) - 1):
                features = self.network[index](features)
            with torch.autocast(device_type=features.device.type, enabled=False):
                return self.network[-1](features.float()).squeeze(-1)


class BPTTRunner(SupervisedWMRunner):
    """Reuse verified WM fitting, but never call the PPO optimizer or PPO rollout."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.bptt = train_cfg["bptt"]
        self.bptt.setdefault("lr_scaling", "none")
        self.bptt.setdefault("lr_schedule", None)
        train_collect_steps = self.bptt.setdefault("train_collect_steps", None)
        if train_collect_steps is not None and (type(train_collect_steps) is not int or train_collect_steps <= 0):
            raise ValueError("BPTT train_collect_steps must be a positive integer or None")
        if self.bptt["critic_source"] not in ("real", "imagined"):
            raise ValueError("BPTT critic source must be real or imagined")
        horizon = self.bptt["horizon"]
        if not isinstance(horizon, int) or not 1 <= horizon <= self.replay.capacity:
            raise ValueError("BPTT horizon must be a positive integer within replay capacity")
        self.replay_horizon = max(horizon, self.options["train_horizon"], *self.options["validation_horizons"])
        if (
            self.wm.predict_force
            or len(self.rewards.terms) != 19
            or (self.wm.predict_contact and not self.options.get("feet_only"))
        ):
            raise ValueError("BPTT requires exactly 19 noncontact rewards and no contact or force heads")
        if not self.native.cfg.observations.policy.enable_corruption:
            raise ValueError("BPTT requires real and imagined policy observation noise")
        if not (
            train_cfg.get("ppo_checkpoint") or train_cfg.get("warm_start") or train_cfg.get("random_initialization")
        ):
            raise ValueError("Choose an explicit terrain PPO or stage-one actor initialization source")
        self.pretrained_wm_updates = 0
        if train_cfg.get("wm_checkpoint"):
            saved = torch.load(train_cfg["wm_checkpoint"], map_location=self.device, weights_only=True)["supervised_wm"]
            if saved["options"]["predict_contact"] or saved["options"]["predict_force"]:
                raise ValueError("WM initialization source must have neither contact nor force heads")
            if saved["options"]["hidden"] != self.options["hidden"]:
                raise ValueError("WM source width differs; initialize the new width from scratch, not a partial load")
            self.wm.load_state_dict(saved["model"], strict=True)
            self.pretrained_wm_updates = saved["updates"]
        self.mdp = BPTTMDP(self.rewards, self.normalizers["policy"], self.wm)
        self.actor = self.alg.actor_critic.actor
        self.alg.actor_critic.critic.requires_grad_(False)
        self.alg.actor_critic.std.requires_grad_(False)
        # No PPO transitions or optimizer steps are used by this runner.
        self.alg.storage = None
        for normalizer in self.normalizers.values():
            normalizer.eval()
        for value in self.wm.state_dict().values():
            if dist.is_initialized():
                dist.broadcast(value, src=0)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.actor_generator = torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + 8009 + self.rank)
        self.real_generator = torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + 9001 + self.rank)
        self.critic_generator = torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + 10009 + self.rank)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + 12007 + self.rank)
        self.critic_noise_generator = torch.Generator(device=self.device).manual_seed(
            train_cfg["seed"] + 13001 + self.rank
        )
        self.step_accumulator = 0.0
        if self.bptt["utd"] <= 0 or self.bptt["initial_collect_steps"] < self.replay_horizon:
            raise ValueError(f"BPTT requires positive UTD and at least {self.replay_horizon} initial replay steps")
        width = int(self.alg.actor_critic.mlp_input_dim_a) + 68 + self.xi.shape[-1] + 8
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(train_cfg["seed"] + 11003)
            self.value = IndependentValue(width).to(self.device)
        for value in self.value.state_dict().values():
            if dist.is_initialized():
                dist.broadcast(value, src=0)
        self.learning_rates = learning_rates(self.bptt, self.options, self.world)
        rates = self.learning_rates["effective"]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rates["actor"], betas=(0.9, 0.95))
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=rates["critic"])
        for group in self.wm_optimizer.param_groups:
            group["lr"] = rates["world_model"]
        self.configure_execution(train_cfg.get("bptt_execution", {}))
        self.actor_objective = distributed_objective(self.actor, self.actor_loss, self.device)
        self.wm_objective = distributed_objective(self.wm, self.world_model_training_loss, self.device)
        self.value_objective = distributed_objective(self.value, self.value_loss, self.device)
        self.distributed = dict(
            method="DistributedDataParallel" if dist.is_initialized() else "single_process",
            objectives=["actor_horizon", "world_model_loss", "value_loss"],
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            loss_reduction="local mean, equal rank average",
            compile_ddp_optimizer=False,
        )
        self.actor_updates = self.critic_updates = self.collection_steps = 0
        self.learner_round = 0
        self.warmup_complete = False
        self.observation, _ = self.env.get_observations()
        self.log_root = Path(log_dir) if log_dir else None
        self.ppo_updates = 0
        if self.log_root:
            self.log_root.mkdir(parents=True, exist_ok=True)
            if self.rank == 0:
                contract_path = self.log_root / "wm-contract.json"
                contract = json.loads(contract_path.read_text())
                contract["actor_update"] = "BPTT only through frozen WM parameters; no PPO update"
                contract["critic_source"] = self.bptt["critic_source"]
                contract["policy_horizon"] = self.bptt["horizon"]
                contract["options"] = self.options
                contract["bptt_execution"] = self.execution
                contract["compiled_regions"] = self.compiled_regions
                contract["distributed"] = self.distributed
                contract["learning_rates"] = self.learning_rates
                contract_path.write_text(json.dumps(contract, indent=2) + "\n")
            (self.log_root / f"bptt-rank-{self.rank}-contract.json").write_text(
                json.dumps(
                    dict(
                        bptt=self.bptt,
                        world_model=self.options,
                        reward_contract=self.rewards.contract,
                        ppo_checkpoint=train_cfg["ppo_checkpoint"],
                        stage1_checkpoint=train_cfg.get("warm_start"),
                        wm_checkpoint=train_cfg["wm_checkpoint"],
                        actor_updates="BPTT only",
                        ppo_updates=0,
                        value="independent scalar MLP 512/256/128",
                        value_width=width,
                        num_envs=env.num_envs,
                        world_size=self.world,
                        sampling_contract=self.rewards.command.sampling_contract,
                        observation_noise=True,
                        action_noise="Gaussian exploration in real collection only; no added imagined action noise",
                        replay_capacity=self.replay.capacity,
                        wm_initialization="checkpoint" if train_cfg.get("wm_checkpoint") else "fresh",
                        execution=self.execution,
                        compiled_regions=self.compiled_regions,
                        distributed=self.distributed,
                        learning_rates=self.learning_rates,
                        precision=(
                            "BF16 networks"
                            + (" including heads" if self.execution["bf16_heads"] else ", FP32 output heads")
                            + "; FP32 geometry/returns/parameters"
                            if self.wm.amp
                            else "FP32"
                        ),
                    ),
                    indent=2,
                )
                + "\n"
            )

    def configure_execution(self, options):
        """Select execution before a run (or an explicitly recorded benchmark fork).

        Raw modules retain parameter/optimizer/checkpoint ownership. Compiled
        callables never replace modules, so state_dict keys remain portable.
        """
        self.execution = dict(
            bf16=False,
            bf16_heads=False,
            compile=False,
            compile_mode="none",
            actor_checkpoint=self.bptt["checkpoint_activations"],
            wm_checkpoint=self.bptt["checkpoint_activations"],
        )
        unknown = options.keys() - self.execution.keys()
        if unknown or any(type(value) is not bool for key, value in options.items() if key != "compile_mode"):
            raise ValueError(f"Invalid BPTT execution options: {options}")
        self.execution.update(options)
        mode = self.execution["compile_mode"]
        if mode not in ("none", "wm", "chain") or (self.execution["compile"] and mode != "none"):
            raise ValueError("Choose legacy compile or compile_mode=none|wm|chain")
        self.actor.amp = self.value.amp = self.wm.amp = self.execution["bf16"]
        self.actor.amp_heads = self.value.amp_heads = self.wm.amp_heads = self.execution["bf16_heads"]
        self.options["amp"] = self.wm.amp
        self.cfg["bptt_execution"] = self.execution.copy()
        self.compiled_regions = ["rollout_objective", "world_model_loss", "value"] if self.execution["compile"] else []
        self._rollout_objective = self.rollout_objective
        self._world_model_loss = self.world_model_loss
        self._value_forward = self.value.forward
        self._terminal_value_forward = self.value.forward
        self._trajectory_value_forward = self.value.forward
        self._features_forward = self.features
        self._imagine_step = self.imagine_step
        self._bodies_forward = self.rewards.bodies
        if self.execution["compile"]:
            # Keep the closed-loop transition native: compiling it changed H32
            # gradients beyond the numerical gate, even with FP32 networks.
            # Rewards/returns and the independent learners compile separately;
            # autograd still connects their backward graphs across all H32 steps.
            self._rollout_objective = compile_forward(self.rollout_objective)
            self._world_model_loss = compile_forward(self.world_model_loss)
            self._value_forward = compile_forward(self.value.forward)
        elif mode in ("wm", "chain"):
            self._world_model_loss = compile_forward(self.world_model_loss)
            self.compiled_regions = ["world_model_loss"]
            if mode == "chain":
                # These batches/horizons are fixed. Automatic symbolic-shape
                # promotion can select cudnn_grid_sampler without a fake kernel.
                self._features_forward = compile_forward(self.features, dynamic=False)
                self._imagine_step = compile_forward(self.imagine_step, dynamic=False)
                self._bodies_forward = compile_forward(self.rewards.bodies, dynamic=False)
                self._rollout_objective = compile_forward(self.rollout_objective, dynamic=False)
                self._value_forward = compile_forward(self.value.forward, dynamic=False)
                self._terminal_value_forward = self._value_forward
                self._trajectory_value_forward = self._value_forward
                self.compiled_regions += [
                    "initial_features",
                    "imagine_step",
                    "trajectory_fk",
                    "rollout_objective",
                    "value",
                ]

    def actor_loss(self, data):
        loss, features, targets, alive, valid, states, actions = self.imagine(data)
        # Only the scalar objective participates in DDP's output graph. Keep
        # detached evidence for invalid-window reporting without a second root.
        return loss, features, targets, alive, valid, [x.detach() for x in states], [x.detach() for x in actions]

    def world_model_training_loss(self, data):
        return self._world_model_loss(data)

    def world_model_update_loss(self, data):
        return self.wm_objective(data)

    def world_model_backward(self, loss):
        backward(loss, self.wm)

    def value_loss(self, features, targets):
        return (self._value_forward(features) - targets).square().mean()

    @torch.inference_mode()
    def collect(self, steps=None):
        c = self.rewards.command
        rows = []
        self.actor.eval()
        self.actor.terrain_gate.fill_(self.initial_gate)
        c.metrics["terrain_gate"][:] = self.initial_gate
        if steps is None:
            self.step_accumulator += 1 / self.bptt["utd"]
            steps = int(self.step_accumulator)
            self.step_accumulator -= steps
        self.last_collection_steps = steps
        for _ in range(steps):
            if self.cfg.get("random_initialization") and not self.warmup_complete:
                self.normalizers["policy"].update(self.observation)
            record = dict(
                state=ordinary(self.rewards.state()),
                contact=ordinary(c.contacts.contact.float()),
                previous_action=ordinary(self.native.action_manager.action),
                support=ordinary(c.terrain.support_z),
                previous_ref_velocity=ordinary(c.contacts.previous_ref_velocity),
                history=ordinary(raw_history(self.observation)),
                episode_step=ordinary(self.native.episode_length_buf),
                **{k: ordinary(v) for k, v in self.mdp.placement().items()},
            )
            noise = torch.randn(self.env.num_envs, 29, device=self.device, generator=self.real_generator)
            noise = noise * self.bptt["action_noise"]
            action = self.actor(self.normalizers["policy"](self.observation)) + noise
            self.observation, reward, done, info = self.env.step(action)
            record.update(
                self.terminal,
                done=ordinary(done.bool()),
                action_noise=ordinary(noise),
                failure=ordinary(self.native.reset_terminated.bool()),
            )
            self.replay.add(record)
            rows.append({k: ordinary(v) for k, v in info["step"].items()})
            self.collection_steps += 1
        self.actor.intent = self.actor.intent.detach()
        return {k: torch.stack([x[k] for x in rows]).mean() for k in rows[0]}

    def sample_actor(self):
        # Keep complete windows covering policy learning and WM validation.
        count, horizon = self.bptt["actor_batch"], self.replay_horizon
        eligible = torch.arange(self.env.num_envs, device=self.device)
        eligible = eligible[eligible % 8 != 0]
        pieces, total = [], 0
        for _ in range(8):
            ids = eligible[
                torch.randint(len(eligible), (2 * count,), device=self.device, generator=self.actor_generator)
            ]
            ages = torch.randint(
                len(self.replay) - horizon + 1, ids.shape, device=self.device, generator=self.actor_generator
            )
            rows = self.replay.rows(ages)
            frame, motion = (self.replay.data[key][rows, ids] for key in ("frame", "motion"))
            valid = frame + horizon + 11 < self.rewards.command.label_lengths[motion]
            selected = valid.nonzero().flatten()[: count - total]
            pieces.append((ids[selected], ages[selected]))
            total += len(selected)
            if total == count:
                return self.replay.sequences(
                    torch.cat([part[0] for part in pieces]),
                    horizon,
                    self.actor_generator,
                    ages=torch.cat([part[1] for part in pieces]),
                )
        raise RuntimeError("Insufficient eligible motion windows for BPTT; refusing a smaller batch")

    @staticmethod
    def start(data, step=0):
        return {
            key: data[key][:, step]
            for key in ("motion", "frame", "placement_position", "placement_rotation", "level", "family")
        } | {"env_id": data["env_id"]}

    def features(self, state, history, support, context, ids, generator=None):
        obs, valid = self.mdp.observations(state, history, support, context, generator)
        return self.mdp.value_features(state, obs, support, context, self.xi[ids]), obs, valid

    def network_forward(self, module, *inputs):
        # These two forwards have no RNG. Observation noise is drawn outside the
        # recomputed region, so backward uses exactly the original noisy inputs.
        enabled = self.execution["actor_checkpoint" if module is self.actor else "wm_checkpoint"]
        enabled = enabled and self.bptt["checkpoint_activations"]
        if module is self.wm:
            return self.wm(*inputs, checkpoint_network=enabled)
        # Keep tuple-producing view operations outside the recomputed network.
        # PyTorch2.11's min-cut partitioner cannot save/recompute a split tuple
        # reused across the boundary. Geometry and all gradients stay connected.
        inputs = self.actor.inputs(*inputs)
        forward = self.actor.action_from_inputs
        if enabled and torch.is_grad_enabled():
            return checkpoint(forward, *inputs, use_reentrant=False, preserve_rng_state=False, early_stop=False)
        return forward(*inputs)

    def imagine(self, data):
        start = self.start(data)
        contexts = [self.mdp.context(start, step) for step in range(self.bptt["horizon"] + 1)]
        batch = len(data["env_id"])
        observation_noise = [self.mdp.draw_noise(batch, self.mdp.observation_widths, self.noise_generator)]
        proprio_noise = []
        for _ in range(self.bptt["horizon"]):
            proprio_noise.append(self.mdp.draw_noise(batch, PROPRIO_WIDTHS, self.noise_generator))
            observation_noise.append(self.mdp.draw_noise(batch, self.mdp.observation_widths, self.noise_generator))
        return self.imagine_rollout(data, contexts, proprio_noise, observation_noise)

    def imagine_step(
        self, state, history, support, previous_action, observation, context, ids, proprio_noise, observation_noise
    ):
        action = self.network_forward(self.actor, self.normalizers["policy"](observation))
        successor, contact_logits, _ = self.network_forward(
            self.wm, state, action, previous_action, None, observation[:, 1678:], self.xi[ids]
        )
        next_support, support_valid = self.rewards.support_from_state(successor, support, context, ids)
        history = torch.cat(
            (history[:, 1:], self.mdp.proprioception(successor, action, ids, proprio_noise)[:, None]), 1
        )
        features, observation, valid = self.features(successor, history, next_support, context, ids, observation_noise)
        return (
            successor,
            history,
            next_support,
            action,
            features,
            observation,
            support_valid & valid & context["reference_valid"],
            contact_logits,
        )

    def imagine_rollout(self, data, contexts, proprio_noise, observation_noise):
        horizon = self.bptt["horizon"]
        ids = data["env_id"]
        state, history, support, previous_action = (
            data[k][:, 0].contiguous() for k in ("state", "history", "support", "previous_action")
        )
        context = contexts[0]
        features, observation, valid = self._features_forward(
            state, history, support, context, ids, observation_noise[0]
        )
        feature_rows, states, actions = [features], [state], []
        reward_contexts, supports, previous_actions, logits = [], [], [], []
        reference_velocities = [context["ref_body_velocity"]]
        for step in range(horizon):
            next_context = contexts[step + 1]
            successor, history, next_support, action, features, observation, next_valid, contact_logits = (
                self._imagine_step(
                    state,
                    history,
                    support,
                    previous_action,
                    observation,
                    next_context,
                    ids,
                    proprio_noise[step],
                    observation_noise[step + 1],
                )
            )
            valid = valid & next_valid
            feature_rows.append(features)
            states.append(successor)
            actions.append(action)
            if contact_logits is not None:
                logits.append(contact_logits)
            previous_actions.append(previous_action)
            supports.append(next_support)
            reward_contexts.append(next_context)
            reference_velocities.append(next_context["ref_body_velocity"])
            state, previous_action, support, context = successor, action, next_support, next_context

        # Rewards and survival do not feed the next policy observation. Evaluate
        # all steps together, sharing one FK pass for current/previous bodies.
        batch = len(ids)
        all_states = torch.stack(states, 1)
        reward_context = {
            key: torch.stack([row[key] for row in reward_contexts], 1).flatten(0, 1)
            for key in (
                "ref_position",
                "ref_rotation",
                "ref_body_position",
                "ref_body_rotation",
                "ref_body_velocity",
                "ref_body_angular",
                "ref_velocity",
                "ref_q",
                "ref_qd",
                "ref_contact",
                "ref_valid",
                "level",
                "family",
            )
        }
        if self.cfg.get("analytic_foot", {}).get("enabled"):
            reward_context["foot_acceleration_memory"] = data["foot_acceleration_memory"][:, 0]
        supports = torch.stack(supports, 1).flatten()
        derivative_valid = state.new_ones(batch, horizon)
        derivative_valid[:, 0] = data["derivative_valid"][:, 0]
        all_bodies = self._bodies_forward(all_states.flatten(0, 1), ids[:, None].expand(-1, horizon + 1).flatten())
        features = torch.stack(feature_rows, 1)
        terminal_value = self._terminal_value_forward(features[:, -1])
        with torch.no_grad():
            values = self._trajectory_value_forward(features.detach())
        loss, targets, alive, reward_means = self._rollout_objective(
            all_states,
            torch.stack(actions, 1).flatten(0, 1),
            torch.stack(previous_actions, 1).flatten(0, 1),
            supports,
            reward_context,
            torch.stack(reference_velocities[:-1], 1).flatten(0, 1),
            derivative_valid.flatten(),
            ids,
            all_bodies,
            terminal_value,
            values,
            torch.stack(logits, 1).flatten(0, 1) if logits else None,
        )
        self.record_imagined_rewards(reward_means)
        return loss, features[:, :-1].detach(), targets, alive, valid, states, actions

    def rollout_objective(
        self,
        all_states,
        actions,
        previous_actions,
        supports,
        reward_context,
        reference_velocities,
        derivative_valid,
        ids,
        all_bodies,
        terminal_value,
        values,
        contact_logits=None,
    ):
        """Batched rewards and returns; keep gradients to every imagined step."""
        batch, steps = all_states.shape[:2]
        horizon = steps - 1
        body_rows = [value.reshape(batch, steps, *value.shape[1:]) for value in all_bodies]
        bodies = tuple(value[:, 1:].flatten(0, 1) for value in body_rows)
        previous_bodies = tuple(value[:, :-1].flatten(0, 1) for value in body_rows)
        successor = all_states[:, 1:].flatten(0, 1)
        rewards, _ = self.rewards.evaluate(
            successor,
            all_states[:, :-1].flatten(0, 1),
            actions,
            previous_actions,
            None,
            None,
            None,
            supports,
            reward_context,
            reference_velocities,
            derivative_valid,
            ids[:, None].expand(-1, horizon).flatten(),
            bodies=bodies,
            previous_bodies=previous_bodies,
        )
        rewards, reward_means = self.augment_imagined_rewards(
            rewards, successor, reward_context, ids, horizon, bodies, contact_logits, previous_bodies, derivative_valid
        )
        rewards = rewards.sum(-1).reshape(batch, horizon)
        survival = self.mdp.soft_survival(successor, supports, reward_context, bodies).reshape(batch, horizon)
        discount = (self.bptt["gamma"] * survival).cumprod(1)
        before = torch.cat((discount.new_ones(batch, 1), discount[:, :-1]), 1)
        objective = (before * rewards).sum(1) + discount[:, -1] * terminal_value
        with torch.no_grad():
            targets = lambda_returns(
                rewards.detach(),
                survival.detach(),
                values,
                torch.zeros_like(survival, dtype=torch.bool),
                self.bptt["gamma"],
                self.bptt["lambda"],
            )
        return (
            -objective.mean(),
            targets,
            survival.prod(1).detach().mean(),
            reward_means,
        )

    def augment_imagined_rewards(
        self, rewards, state, context, ids, horizon, bodies, logits, previous_bodies=None, derivative_valid=None
    ):
        return rewards, rewards.detach().mean(0)

    def record_imagined_rewards(self, means):
        pass

    def scalar_tag(self, key):
        return "BPTT/" + key

    @torch.no_grad()
    def real_critic_data(self, data):
        features, successors = [], []
        for t in range(self.bptt["horizon"]):
            start = self.start(data, t)
            ids = data["env_id"]
            context, next_context = self.mdp.context(start), self.mdp.context(start, 1)
            features.append(
                self.features(
                    data["state"][:, t],
                    data["history"][:, t],
                    data["support"][:, t],
                    context,
                    ids,
                    self.critic_noise_generator,
                )[0]
            )
            history = torch.cat(
                (
                    data["history"][:, t, 1:],
                    self.mdp.proprioception(
                        data["next_state"][:, t], data["action"][:, t], ids, self.critic_noise_generator
                    )[:, None],
                ),
                1,
            )
            successors.append(
                self.features(
                    data["next_state"][:, t],
                    history,
                    data["next_support"][:, t],
                    next_context,
                    ids,
                    self.critic_noise_generator,
                )[0]
            )
        features, successors = torch.stack(features, 1), torch.stack(successors, 1)
        next_values = self._value_forward(successors)
        # Values after a reset are excluded; each transition has its own pre-reset successor.
        future, targets = next_values[:, -1], []
        for t in range(self.bptt["horizon"] - 1, -1, -1):
            future = data["native_reward"][:, t].sum(-1) + self.bptt["gamma"] * (~data["failure"][:, t]).float() * (
                (1 - self.bptt["lambda"]) * next_values[:, t]
                + self.bptt["lambda"] * torch.where(data["done"][:, t], next_values[:, t], future)
            )
            targets.append(future)
        return features, torch.stack(targets[::-1], 1), data["mask"][:, : self.bptt["horizon"]]

    def fit_value(self, features, targets, mask):
        features, targets = features.flatten(0, 1), targets.flatten()
        selected = mask.flatten().bool()
        features, targets = features[selected], targets[selected]
        loss_sum = features.new_zeros(())
        for _ in range(self.bptt["critic_epochs"]):
            order = torch.randperm(len(targets), device=self.device, generator=self.critic_generator)
            for ids in order.tensor_split(self.bptt["critic_minibatches"]):
                loss = self.value_objective(features[ids], targets[ids])
                self.value_optimizer.zero_grad(set_to_none=True)
                backward(loss, self.value)
                nn.utils.clip_grad_norm_(self.value.parameters(), 1.0, error_if_nonfinite=True)
                self.value_optimizer.step()
                self.critic_updates += 1
                loss_sum += loss.detach()
        return float(loss_sum / (self.bptt["critic_epochs"] * self.bptt["critic_minibatches"]))

    def update_bptt(self, actor=True):
        data = self.sample_actor()
        self.wm_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.wm.requires_grad_(False)
        self.value.requires_grad_(False)
        try:
            with torch.set_grad_enabled(actor):
                objective = self.actor_objective if actor else self.imagine
                loss, features, targets, alive, valid, states, actions = objective(data)
            bad = (~valid).any().float()
            if dist.is_initialized():
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad:
                if self.log_root:
                    prefix = self.log_root / f"invalid-rollout-{self.actor_updates}"
                    torch.save(
                        dict(
                            data={key: value.detach().cpu() for key, value in data.items()},
                            states=[value.detach().cpu() for value in states],
                            actions=[value.detach().cpu() for value in actions],
                            valid=valid.detach().cpu(),
                            raster_origin=self.heightfield.origin.cpu(),
                            raster_extent=self.heightfield.extent.cpu(),
                            rank=self.rank,
                            actor_updates=self.actor_updates,
                        ),
                        str(prefix) + f"-rank-{self.rank}.pt",
                    )
                    self.save(str(prefix) + ".pt")
                raise RuntimeError(
                    "Imagined rollout left the terrain raster/reference domain; "
                    "no actor update was applied; model and per-rank windows saved in log directory"
                )
            norm = 0.0
            if actor:
                backward(loss, self.actor)
                norm = float(nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0, error_if_nonfinite=True))
                nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
                self.actor_optimizer.step()
                self.actor_updates += 1
            self.actor.intent = self.actor.intent.detach()
            if any(p.grad is not None for p in self.wm.parameters()):
                raise AssertionError("BPTT wrote WM parameter gradients")
        finally:
            self.wm.requires_grad_(True)
            self.value.requires_grad_(True)
        if self.bptt["critic_source"] == "real":
            features, targets, mask = self.real_critic_data(data)
        else:
            mask = torch.ones_like(targets)
        value_loss = self.fit_value(features, targets.detach(), mask)
        return dict(
            actor_loss=float(loss.detach()),
            critic_loss=value_loss,
            actor_grad_norm=norm,
            imagination_survival=float(alive),
            critic_valid_fraction=float(mask.float().mean()),
        )

    def update_learning_rates(self, iteration):
        rates = scheduled_learning_rates(self.learning_rates["effective"], self.bptt["lr_schedule"], iteration)
        for name, optimizer in (
            ("actor", self.actor_optimizer),
            ("critic", self.value_optimizer),
            ("world_model", self.wm_optimizer),
        ):
            for group in optimizer.param_groups:
                group["lr"] = rates[name]

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("BPTT experiments retain real reset/episode history semantics")
        stream = (self.log_root / f"bptt-rank-{self.rank}-metrics.jsonl").open("a", buffering=1)
        writer = SummaryWriter(str(self.log_root / "tensorboard")) if self.rank == 0 else None
        phases = [] if self.warmup_complete else [("warmup", self.bptt["warmup_iterations"])]
        phases.append(("train", num_learning_iterations))
        previous_record_time = None
        try:
            if len(self.replay) < self.replay_horizon:
                self.collect(self.bptt["initial_collect_steps"])
            for phase, iterations in phases:
                for iteration in range(iterations):
                    started = time.perf_counter()
                    self.update_learning_rates(self.learner_round)
                    metrics = self.collect(self.bptt["train_collect_steps"] if phase == "train" else None)
                    collected = time.perf_counter()
                    self.fit_world_model()
                    fitted = time.perf_counter()
                    stats = self.update_bptt(actor=phase == "train") if len(self.replay) >= self.replay_horizon else {}
                    updated = time.perf_counter()
                    validating = (
                        len(self.replay) >= self.replay_horizon and iteration % self.options["validation_interval"] == 0
                    )
                    if validating:
                        self.latest_metrics = self.validate()
                        self.write_metrics(self.current_learning_iteration)
                    row = dict(
                        phase=phase,
                        iteration=self.current_learning_iteration,
                        phase_iteration=iteration,
                        learner_round=self.learner_round,
                        actor_updates=self.actor_updates,
                        critic_updates=self.critic_updates,
                        wm_updates=self.wm_updates,
                        pretrained_wm_updates=self.pretrained_wm_updates,
                        ppo_updates=0,
                        num_envs=self.env.num_envs,
                        world_size=self.world,
                        critic_source=self.bptt["critic_source"],
                        policy_horizon=self.bptt["horizon"],
                        collection_steps=self.collection_steps,
                        collected_this_round=self.last_collection_steps,
                        replay_size=len(self.replay),
                        replay_capacity=self.replay.capacity,
                        collection_seconds=collected - started,
                        wm_seconds=fitted - collected,
                        policy_critic_seconds=updated - fitted,
                        validation_ran=validating,
                        wm_train_loss=self.training_loss,
                        actor_learning_rate=float(self.actor_optimizer.param_groups[0]["lr"]),
                        critic_learning_rate=float(self.value_optimizer.param_groups[0]["lr"]),
                        wm_learning_rate=float(self.wm_optimizer.param_groups[0]["lr"]),
                        **stats,
                        **{key: float(value) for key, value in metrics.items()},
                    )
                    scalar_keys = [key for key, value in row.items() if isinstance(value, (int, float))]
                    scalars = torch.tensor([row[key] for key in scalar_keys], device=self.device)
                    if not torch.isfinite(scalars).all():
                        raise FloatingPointError("Nonfinite BPTT training metrics")
                    if dist.is_initialized():
                        dist.all_reduce(scalars)
                        scalars /= self.world
                    row.update(zip(scalar_keys, scalars.cpu().tolist()))
                    recorded_at = time.perf_counter()
                    row.update(
                        seconds=recorded_at - started,
                        # End-to-end cycle includes the preceding log/checkpoint
                        # write, which happens after the previous metric record.
                        iteration_seconds=(
                            recorded_at - previous_record_time
                            if previous_record_time is not None
                            else recorded_at - started
                        ),
                        time=time.time(),
                        finite=True,
                        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                    )
                    previous_record_time = recorded_at
                    stream.write(json.dumps(row) + "\n")
                    if writer:
                        for key in scalar_keys:
                            writer.add_scalar(self.scalar_tag(key), row[key], self.current_learning_iteration)
                        if phase == "train":
                            writer.add_scalar(
                                "Performance/iteration_seconds",
                                row["iteration_seconds"],
                                self.current_learning_iteration,
                            )
                        print(
                            f"[BPTT] {phase} {iteration+1}/{iterations} H{self.bptt['horizon']} "
                            f"critic={self.bptt['critic_source']} body={row['PGMT/body_position_error_m']:.4f} "
                            f"loss={row.get('actor_loss', 0):.4f} seconds={row['seconds']:.3f}",
                            flush=True,
                        )
                    self.write_extra_scalars(writer)
                    self.learner_round += 1
                    if phase == "train":
                        self.current_learning_iteration += 1
                        if self.current_learning_iteration % self.save_interval == 0:
                            self.save(str(self.log_root / f"model_{self.current_learning_iteration}.pt"))
                if phase == "warmup":
                    self.warmup_complete = True
                    self.save(str(self.log_root / "warmup.pt"))
            self.save(str(self.log_root / f"model_{self.current_learning_iteration}.pt"))
        finally:
            stream.close()
            if writer:
                writer.close()

    def write_extra_scalars(self, writer):
        pass

    def save(self, path, infos=None):
        intermediate = str(path) + ".bptt-base"
        super().save(intermediate, infos)
        local = {
            name: getattr(self, name).get_state().cpu()
            for name in (
                "actor_generator",
                "real_generator",
                "critic_generator",
                "noise_generator",
                "critic_noise_generator",
            )
        }
        ranks = [None] * self.world
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        if self.rank == 0:
            saved = torch.load(intermediate, map_location="cpu", weights_only=True)
            saved["bptt"] = dict(
                schema="pgmt-bptt-v2",
                options=self.bptt,
                execution=self.execution,
                compiled_regions=self.compiled_regions,
                distributed=self.distributed,
                learning_rates=self.learning_rates,
                value=self.value.state_dict(),
                actor_optimizer=self.actor_optimizer.state_dict(),
                value_optimizer=self.value_optimizer.state_dict(),
                actor_updates=self.actor_updates,
                learner_round=self.learner_round,
                critic_updates=self.critic_updates,
                collection_steps=self.collection_steps,
                pretrained_wm_updates=self.pretrained_wm_updates,
                wm_checkpoint=self.cfg["wm_checkpoint"],
                warmup_complete=self.warmup_complete,
                step_accumulator=self.step_accumulator,
                ppo_updates=0,
                ranks=ranks,
            )
            temporary = str(path) + ".tmp"
            torch.save(saved, temporary)
            Path(temporary).replace(path)
            Path(intermediate).unlink()
        if dist.is_initialized():
            dist.barrier()

    def load(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)["bptt"]
        saved_options = dict(saved["options"])
        saved_options.setdefault("lr_scaling", "none")
        saved_options.setdefault("lr_schedule", None)
        saved_options.setdefault("train_collect_steps", None)
        if saved["schema"] != "pgmt-bptt-v2" or saved_options != self.bptt:
            raise ValueError("BPTT resume changed critic source or training options")
        saved_rates = saved.get("learning_rates")
        if saved_rates != self.learning_rates and not (saved_rates is None and self.bptt["lr_scaling"] == "none"):
            raise ValueError("BPTT resume changed base/effective learning rates or world size")
        legacy_execution = dict(
            bf16=False,
            compile=False,
            actor_checkpoint=saved["options"]["checkpoint_activations"],
            wm_checkpoint=saved["options"]["checkpoint_activations"],
        )
        saved_execution = saved.get("execution", legacy_execution).copy()
        # Early BF16 forks autocast every output head. Never silently change
        # their precision on resume; old FP32 checkpoints remain compatible.
        saved_execution.setdefault("bf16_heads", saved_execution["bf16"])
        saved_execution.setdefault("compile_mode", "none")
        if saved_execution != self.execution:
            raise ValueError("BPTT resume changed execution options; use an explicit benchmark fork instead")
        if saved.get("compiled_regions", []) != self.compiled_regions:
            raise ValueError("BPTT resume changed compilation boundaries; use an explicit benchmark fork instead")
        result = super().load(path)
        self.value.load_state_dict(saved["value"])
        self.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        self.value_optimizer.load_state_dict(saved["value_optimizer"])
        self.learner_round = saved.get(
            "learner_round", saved["actor_updates"] + self.bptt["warmup_iterations"] * saved["warmup_complete"]
        )
        expected_rates = scheduled_learning_rates(
            self.learning_rates["effective"], self.bptt["lr_schedule"], max(0, self.learner_round - 1)
        )
        for name, optimizer in (
            ("actor", self.actor_optimizer),
            ("critic", self.value_optimizer),
            ("world_model", self.wm_optimizer),
        ):
            if any(group["lr"] != expected_rates[name] for group in optimizer.param_groups):
                raise ValueError(f"Checkpoint optimizer learning rate differs for {name}")
        for name in ("actor_updates", "critic_updates", "collection_steps", "pretrained_wm_updates", "warmup_complete"):
            setattr(self, name, saved[name])
        self.step_accumulator = saved["step_accumulator"]
        for name, state in saved["ranks"][self.rank].items():
            getattr(self, name).set_state(state)
        for normalizer in self.normalizers.values():
            normalizer.eval()
        self.observation, _ = self.env.get_observations()
        return result
