"""IsaacLab/Newton task specialization of the copied SuperMimic runner."""

import copy
import torch
from pathlib import Path
from tensordict import TensorDict

from instinct_rl.modules.normalizer import EmpiricalNormalization
from instinct_rl.utils.utils import get_subobs_size

from isaaclab.utils import string_to_callable

from instinctlab.tasks.shadowing.pgmt.analytic_foot_rewards import AnalyticFootRewards
from instinctlab.tasks.shadowing.pgmt.foot_rewards import FootContactRewards
from instinctlab.tasks.shadowing.pgmt.heightfield import HeightField

from ..config import from_dict
from ..envs.pgmt_world_model_env import PGMTWorldModelEnv
from ..layouts import RobotLayout
from ..learners.pgmt_learner_utils import compile_function, set_learning_rate
from ..learners.pgmt_policy_learner import PGMTActorCriticPolicyLearner
from ..learners.pgmt_wm_learner import PGMTWorldModelLearner
from ..pgmt_models import PGMTDynamics, PGMTPolicy, PolicyNormalizer, Value, initialize_shared_dynamics
from ..pgmt_replay_buffer import PGMTReplayBuffer
from ..pgmt_representation import REPRESENTATIONS
from ..pgmt_sim_context import PGMTSimContext, ordinary
from ..utils.pgmt_logger import PGMTLogger
from ..utils.pgmt_mdp import PGMTMDP
from ..utils.pgmt_rewards import AnalyticRewards
from ..utils.pgmt_terrain import BilinearTerrain
from . import pgmt_checkpoint
from .supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg, WorldModel


class PGMTSuperMimicRunner(SuperMimicRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.train_cfg = train_cfg
        self.options = from_dict(train_cfg["supermimic"])
        self.log_root = None if log_dir is None else Path(log_dir)
        self.device = torch.device(device)
        self.generators = {
            name: torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + getattr(self.options.seeds, name))
            for name in ("replay", "actor", "collection", "critic", "observation", "gradient")
        }
        self.current_learning_iteration = self.learner_round = self.collection_steps = 0
        self.step_accumulator = 0.0
        self.warmup_complete = False
        self.normalize_during_warmup = self.options.collection.update_normalizer_during_warmup
        self.latest_metrics, self.terminal = {}, {}
        cfg = SuperMimicRunnerCfg(
            max_iterations=train_cfg["max_iterations"],
            save_interval=train_cfg["save_interval"],
            run_name=train_cfg["run_name"],
            history_length=self.options.collection.history_length,
        )
        super().__init__(env, cfg, task_id="PGMT", device=self.device)
        self.ids = torch.arange(env.num_envs, device=self.device)
        self.observation, _ = env.get_observations()
        self.env.unwrapped.add_transition_observer(self._capture_transition)
        if train_cfg.get("initialize_checkpoint"):
            pgmt_checkpoint.initialize_legacy_weights(self, train_cfg["initialize_checkpoint"])

    def _build_world_model(self):
        env, cfg, options, device = self.env, self.train_cfg, self.options, self.device
        native = env.unwrapped
        scan = native.cfg.height_scan
        scan_width = 1
        for size in scan["size"]:
            scan_width *= round(size / scan["spacing"]) + 1
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(cfg["seed"])
            policy_cfg = copy.deepcopy(cfg["policy"])
            actor_type = string_to_callable(policy_cfg.pop("class_name"))
            actor = actor_type(env.num_actions, **policy_cfg).to(device)
            normalizers = {}
            for group, values in cfg["normalizers"].items():
                values = {key: value for key, value in values.items() if key != "class_name"}
                shape = get_subobs_size(env.get_obs_format()[group])
                normalizer = (
                    PolicyNormalizer(shape, scan_width, **values)
                    if group == "policy"
                    else EmpiricalNormalization(shape, **values)
                )
                normalizers[group] = normalizer.to(device).eval()
            env.reset()
        layout = RobotLayout(env.num_actions)
        command = native.command_manager.get_term("motion")
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(cfg["seed"] + options.seeds.world_model)
            field = (
                native.pgmt_heightfield
                if hasattr(native, "pgmt_heightfield")
                else HeightField.from_context(command.terrain, native.cfg.heightfield_resolution)
            )
            terrain = BilinearTerrain(field.to(device))
            rewards = AnalyticRewards(native, field)
            sim_context = PGMTSimContext(native)
            context = sim_context.read(native)
            representation = REPRESENTATIONS[options.representation.kind](
                layout, options.representation, native.step_dt, native.cfg.decimation, rewards.root_com[0]
            ).to(device)
            dynamics = PGMTDynamics(
                representation,
                terrain,
                options.dynamics,
                context.shape[-1],
                command.foot_ids,
                scan_width,
                options.execution,
            ).to(device)
        self.actor, self.normalizers, self.terrain, self.rewards = actor, normalizers, terrain, rewards
        self.context, self.command = context, command
        object.__setattr__(representation, "terrain", terrain)
        initialize_shared_dynamics(dynamics, cfg["seed"] + options.seeds.world_model, native.cfg.decimation)
        return WorldModel(representation, dynamics, sim_context, command.reference.cfg.num_frames, env.num_actions)

    def _build_replay_buffer(self):
        return PGMTReplayBuffer(self.options.collection.replay_steps, self.options.collection.history_length)

    def _build_wm_learner(self):
        return PGMTWorldModelLearner(
            self._dynamics, self.context, self.options.world_model, self.options.wm_optimizer, self.options.execution
        )

    def _init_policy(self):
        env, cfg, options, device = self.env, self.train_cfg, self.options, self.device
        native, actor, normalizers = env.unwrapped, self.actor, self.normalizers
        rewards, context, command, layout = self.rewards, self.context, self.command, self._representation.layout
        dynamics = self._dynamics
        policy = PGMTPolicy(actor, normalizers["policy"], options.execution)
        reference_frames = command.reference.cfg.num_frames
        mdp = PGMTMDP(
            rewards,
            normalizers["policy"],
            dynamics,
            options.collection.history_length,
            reference_frames,
            cfg["termination_softness"],
        )
        width = mdp.observation_layout.width + layout.feature_width + context.shape[-1] + mdp.critic_context_width
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(cfg["seed"] + options.seeds.value)
            value = Value(width, options.policy.critic_hidden).to(device)
        foot_rewards = FootContactRewards(rewards, cfg["foot_contact"])
        analytic_foot = AnalyticFootRewards(rewards, cfg["analytic_foot"]) if cfg["analytic_foot"]["enabled"] else None
        if (cfg["foot_contact"]["enabled"] or analytic_foot is not None) and not options.dynamics.predict_contact:
            raise ValueError("Enabled contact rewards require the foot probability head")
        self.policy, self.mdp, self.value = policy, mdp, value
        self.foot_rewards, self.analytic_foot = foot_rewards, analytic_foot
        self.wm_env = PGMTWorldModelEnv(
            dynamics, mdp, context, self.replay_buffer, foot_rewards, analytic_foot, options
        )
        self.wm_env.sample_generator = self.generators["actor"]
        self.policy_learner = PGMTActorCriticPolicyLearner(
            policy,
            value,
            self.wm_env,
            options.policy,
            options.actor_optimizer,
            options.critic_optimizer,
            options.execution,
            self.generators["critic"],
            self.generators["gradient"],
        )
        if options.execution.compile != "none":
            policy.network.action_from_inputs = compile_function(policy.network.action_from_inputs, True)
            dynamics.predict_delta = compile_function(dynamics.predict_delta, True)

    def _build_logger(self, **kwargs):
        return PGMTLogger(self.env, self.mdp, self.foot_rewards, self.analytic_foot, self.log_root, **kwargs)

    def _motion_times(self):
        return self.command.frame_ids.clone()

    def _capture_transition(self, dt):
        self.terminal, terms = self.mdp.capture_transition(self.foot_rewards, self.analytic_foot, self.ids, dt)
        self.logger.episode_rewards.add(terms, self.env.unwrapped.reset_buf.bool())

    def _post_step_fields(self, dones, extras):
        return dict(
            self.terminal, done=ordinary(dones.bool()), failure=ordinary(self.env.unwrapped.reset_terminated.bool())
        )

    @torch.inference_mode()
    def step(self):
        native = self.env.unwrapped
        if not self.warmup_complete and self.normalize_during_warmup:
            self.policy.normalizer.update(self.observation)
        record = dict(
            contact=ordinary(self.command.contacts.contact.float()),
            previous_action=ordinary(native.action_manager.action),
            support=ordinary(self.command.terrain.support_z),
            previous_ref_velocity=ordinary(self.command.contacts.previous_ref_velocity),
            history=ordinary(self.mdp.raw_history(self.observation)),
            episode_step=ordinary(native.episode_length_buf),
            **{key: ordinary(value) for key, value in self.mdp.placement().items()},
        )
        noise = (
            torch.randn(
                self.env.num_envs, self.env.num_actions, device=self.device, generator=self.generators["collection"]
            )
            * self.options.collection.action_noise
        )
        record["action_noise"] = ordinary(noise)
        action = self.policy.collect(self.observation) + noise
        self.observation, _, done, info = self._step_and_store(
            action, ordinary(self.rewards.state()), extra_fields=record
        )
        self._step_metrics = {key: ordinary(value) for key, value in info["step"].items()}
        self.logger.process_env_step(self.terminal["native_reward"].sum(-1), done.bool(), info)
        return self.env.num_envs

    def _finish_collection(self):
        self.policy.finish_collection()

    def _updates_this_iteration(self):
        return 1

    def _steps_this_iteration(self):
        self.step_accumulator += self.options.collection.steps_per_update
        steps = int(self.step_accumulator)
        self.step_accumulator -= steps
        return steps

    def _before_iteration(self, train_actor):
        self._actor_updates_before_collection = self.policy_learner.actor_updates
        self._finish_flat_warmup()
        self.foot_rewards.set_updates(self.policy_learner.actor_updates)
        gate = (
            min(1.0, (self.policy_learner.actor_updates + 1) / self.train_cfg["terrain_gate_ramp"])
            if self.warmup_complete
            else 0.0
        )
        self.policy.network.eval()
        self.policy.network.terrain_gate.fill_(gate)
        self.command.metrics["terrain_gate"][:] = gate
        result = self.update_learning_rates()
        return {"Train/" + name + "_learning_rate": value for name, value in result.items()}

    def _finish_flat_warmup(self):
        if (
            not self.command.flat_warmup_active
            or self.policy_learner.actor_updates < self.train_cfg["flat_warmup_updates"]
        ):
            return
        if len(self.replay_buffer):
            self.replay_buffer.data["done"][(self.replay_buffer.pointer - 1) % self.replay_buffer.capacity].fill_(True)
        self.command.finish_flat_warmup()
        self.observation, _ = self.env.reset()
        self.logger.episode_rewards.running.zero_()
        if self.env.unwrapped.episode_length_buf.any() or self.command.terrain.outside.any():
            raise AssertionError("Terrain transition did not reset native episode state")

    def update_learning_rates(self):
        pairs = dict(
            actor=(self.policy_learner.actor_optimizer, self.options.actor_optimizer),
            critic=(self.policy_learner.value_optimizer, self.options.critic_optimizer),
            world_model=(self.world_model_learner.optimizer, self.options.wm_optimizer),
        )
        return {
            name: set_learning_rate(optimizer, cfg.lr, self.options.schedule, self.learner_round)
            for name, (optimizer, cfg) in pairs.items()
        }

    def _sample_wm_batches(self):
        self.world_model_learner.update_normalization(self.replay_buffer, self.ids)
        return torch.stack(
            [
                TensorDict(
                    self.world_model_learner.sample(self.replay_buffer, self.ids, self.generators["replay"]),
                    batch_size=[],
                )
                for _ in range(self.options.world_model.num_batches)
            ]
        )

    def _update_policy(self, train_actor):
        data = self.wm_env.sample_initial_states()
        inputs = self.wm_env.prepare(data, self.generators["observation"])
        return self.policy_learner.update(inputs, train_actor=train_actor)

    def _combine_metrics(self, metrics, wm, policy):
        metrics.update(self.logger.metrics(self.replay_buffer, self.last_steps, policy.pop("reward_means")))
        metrics.update({key if "/" in key else "Train/" + key: value for key, value in policy.items()})
        metrics.update({"WorldModel/" + key: value for key, value in wm.items()})
        metrics.update(
            {
                "Performance/cuda_graph_" + key: float(value)
                for key, value in (self.world_model_learner.capture_status | self.policy_learner.capture_status).items()
            }
        )
        metrics.update(
            {
                "EMA/updates": self.policy_learner.actor_updates,
                "Curriculum/flat_warmup": float(self.command.flat_warmup_active),
                "Curriculum/actor_updates_before_collection": float(self._actor_updates_before_collection),
            }
        )
        return metrics

    def _prepare_training(self):
        if self.log_root is None:
            raise ValueError("Training requires an explicit output directory")
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.logger.open()
        if not (self.log_root / "initial.pt").exists():
            self.save(self.log_root / "initial.pt")
        if len(self.replay_buffer) < self.options.collection.initial_steps:
            self._before_iteration(False)
            self.collect(self.options.collection.initial_steps)

    def _phases(self):
        phases = [] if self.warmup_complete else [("warmup", self.options.collection.warmup_iterations)]
        return phases + [("train", self._requested_iterations)]

    def _diagnostics(self, index, metrics):
        if index % self.options.world_model.validation_interval == 0:
            self.latest_metrics = self.world_model_learner.validate(
                self.replay_buffer, self.ids, self.generators["replay"]
            )
            self.logger.diagnostics(self.latest_metrics, self.current_learning_iteration)
            for key, value in self.latest_metrics.items():
                if key.startswith("current/") and (
                    key.startswith(f"current/h{self.options.world_model.horizon}/") or key.endswith("/root_m")
                ):
                    metrics["WorldModel/" + key] = value

    def _log_iteration(self, metrics, phase, index):
        metrics["Performance/peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        self.logger.record(
            metrics, phase, self.current_learning_iteration, self.learner_round, self.policy_learner.actor_updates
        )

    def _after_phase(self, phase):
        if phase == "warmup":
            self.warmup_complete = True
            self.save(self.log_root / "warmup.pt")

    def _save_training(self, path):
        self.save(path)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("SM collection requires native reset/history semantics")
        self._requested_iterations = num_learning_iterations
        self.run()

    def _cleanup(self):
        self.env.unwrapped.remove_transition_observer(self._capture_transition)
        self.logger.close()

    def save(self, path):
        pgmt_checkpoint.save(self, path)

    def load(self, path):
        pgmt_checkpoint.load(self, path)

    def get_inference_policy(self, device=None):
        self.policy.eval()
        return self.policy

    def add_git_repo_to_log(self, path):
        # Source snapshots are saved by the external experiment launcher.
        pass
