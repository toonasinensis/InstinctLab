"""PGMT terrain imagination. Actor, critic, returns and optimizer live in learners.

Rewards are finalized over the trajectory to retain batched FK. The step extras
explicitly declare this deferred reward contract; no zero reward is presented as
an actual environment reward.
"""

import torch
from tensordict import TensorDict

from ..learners.pgmt_learner_utils import compile_function
from .motion_tracking_env import MotionTrackingEnv


class PGMTWorldModelEnv(MotionTrackingEnv):
    def __init__(self, dynamics, mdp, context, replay, foot_rewards, analytic_foot, options):
        self.dynamics, self.mdp, self.xi, self.replay_buffer = dynamics, mdp, context, replay
        self.representation = dynamics.representation
        self.rewards, self.foot_rewards, self.analytic_foot = mdp.rewards, foot_rewards, analytic_foot
        self.options, self.execution = options, options.execution
        self.horizon, self.num_envs = options.policy.horizon, options.policy.batch_size
        self.history_length = options.collection.history_length
        self.device = context.device
        self._num_actions = self.representation.layout.joints
        chain = options.execution.compile == "chain"
        self.features_forward = compile_function(self.features, chain)
        self.step_forward = compile_function(self.transition, chain)
        self.bodies_forward = compile_function(self.rewards.bodies, chain)
        self.rewards_forward = compile_function(self.evaluate_rewards, options.execution.compile != "none")

    @staticmethod
    def start(data):
        names = ("motion", "frame", "placement_position", "placement_rotation", "level", "family")
        return {key: data[key][:, 0] for key in names} | {"env_id": data["env_id"]}

    def prepare(self, data, generator):
        batch = len(data["env_id"])
        contexts = [
            TensorDict(self.mdp.context(self.start(data), step), batch_size=[batch]) for step in range(self.horizon + 1)
        ]
        observation = [self.mdp.draw_noise(batch, self.mdp.observation_widths, generator)]
        proprio = []
        for _ in range(self.horizon):
            proprio.append(self.mdp.draw_noise(batch, self.mdp.proprio_widths, generator))
            observation.append(self.mdp.draw_noise(batch, self.mdp.observation_widths, generator))
        return TensorDict(
            dict(
                data=TensorDict(data, batch_size=[]),
                prev_action=data["previous_action"][:, 0],
                sim_ctx=self.xi[data["env_id"]],
                contexts=torch.stack(contexts),
                proprio_noise=torch.stack([TensorDict(n, batch_size=[batch]) for n in proprio]),
                observation_noise=torch.stack([TensorDict(n, batch_size=[batch]) for n in observation]),
            ),
            batch_size=[],
        )

    def features(self, state, history, support, context, ids, noise):
        obs, valid = self.mdp.observations(state, history, support, context, noise)
        return self.mdp.value_features(state, obs, support, context, self.xi[ids]), obs, valid

    def transition(
        self,
        state,
        history,
        support,
        previous_action,
        action,
        observation,
        context,
        ids,
        proprio_noise,
        observation_noise,
    ):
        successor, logits = self.dynamics.predict_transition(
            state,
            action,
            previous_action,
            observation[:, self.mdp.observation_layout["terrain"]],
            self.xi[ids],
            recompute=self.execution.wm_recompute,
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
            logits,
        )

    def _on_reset(self, inputs):
        self.data = data = inputs["data"].to_dict()
        self.contexts = [row.to_dict() for row in inputs["contexts"].unbind(0)]
        self.proprio_noise = [row.to_dict() for row in inputs["proprio_noise"].unbind(0)]
        self.observation_noise = [row.to_dict() for row in inputs["observation_noise"].unbind(0)]
        self.ids = data["env_id"]
        self.state, self.history, self.support, self.previous_action = (
            data[k][:, 0].contiguous() for k in ("state", "history", "support", "previous_action")
        )
        features, self.observation, self.valid = self.features_forward(
            self.state, self.history, self.support, self.contexts[0], self.ids, self.observation_noise[0]
        )
        self.feature_rows, self.states = [features], [self.state]
        self.actions, self.previous_actions, self.supports, self.logits = [], [], [], []
        self._rollout_step = 0

    def _unpack_initial_states(self, inputs):
        state = inputs["data", "state"][:, 0].contiguous()
        return self.representation.extract_aux_state(state), self.representation.represent(state)

    def get_observations(self):
        return TensorDict(dict(actor=self.observation, critic=self.feature_rows[-1]), batch_size=[len(self.ids)])

    def _advance_history(self):
        # Native noised history is advanced once inside transition().
        pass

    def _compute_commands(self):
        self._rollout_step += 1

    def _step_extras(self):
        return {"deferred_rewards": True}

    def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
        step = self._rollout_step
        successor, history, support, action, features, observation, valid, logits = self.step_forward(
            self.state,
            self.history,
            self.support,
            self.previous_action,
            action,
            self.observation,
            self.contexts[step + 1],
            self.ids,
            self.proprio_noise[step],
            self.observation_noise[step + 1],
        )
        self.valid = self.valid & valid
        self.feature_rows.append(features)
        self.states.append(successor)
        self.actions.append(action)
        self.previous_actions.append(self.previous_action)
        self.supports.append(support)
        if logits is not None:
            self.logits.append(logits)
        self.state, self.history, self.support = successor, history, support
        self.previous_action, self.observation = action, observation
        return (
            self.representation.extract_aux_state(successor),
            self.representation.represent(successor),
            self.representation.joint_state(self.representation.represent(successor)),
            None,
            None,
        )

    def evaluate_trajectory(self):
        batch, horizon, ids = len(self.ids), self.horizon, self.ids
        all_states = torch.stack(self.states, 1)
        reward_keys = (
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
        context = {key: torch.stack([row[key] for row in self.contexts[1:]], 1).flatten(0, 1) for key in reward_keys}
        if self.analytic_foot is not None:
            context["foot_acceleration_memory"] = self.data["foot_acceleration_memory"][:, 0]
        derivative_valid = self.state.new_ones(batch, horizon)
        derivative_valid[:, 0] = self.data["derivative_valid"][:, 0]
        bodies = self.bodies_forward(all_states.flatten(0, 1), ids[:, None].expand(-1, horizon + 1).flatten())
        features = torch.stack(self.feature_rows, 1)
        reward, survival, means = self.rewards_forward(
            all_states,
            torch.stack(self.actions, 1).flatten(0, 1),
            torch.stack(self.previous_actions, 1).flatten(0, 1),
            torch.stack(self.supports, 1).flatten(),
            context,
            torch.stack([row["ref_body_velocity"] for row in self.contexts[:-1]], 1).flatten(0, 1),
            derivative_valid.flatten(),
            ids,
            bodies,
            torch.stack(self.logits, 1).flatten(0, 1) if self.logits else None,
        )
        return dict(features=features, rewards=reward, survival=survival, valid=self.valid, reward_means=means)

    def evaluate_rewards(
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
        return rewards, survival, reward_means

    def augment_imagined_rewards(
        self, rewards, state, context, ids, horizon, bodies, logits, previous_bodies=None, derivative_valid=None
    ):
        rows = ids[:, None].expand(-1, horizon).flatten()
        contact = (
            self.foot_rewards(state, logits.sigmoid(), context["ref_contact"], context["ref_valid"], rows, bodies)
            if self.dynamics.predict_contact
            else state.new_zeros(len(state), len(self.foot_rewards.names))
        )
        terms = [rewards, contact]
        if self.analytic_foot is not None:
            assert previous_bodies is not None
            acceleration = (bodies[2] - previous_bodies[2]) / self.rewards.dt
            analytic, _ = self.analytic_foot.evaluate(
                bodies,
                acceleration,
                derivative_valid,
                rows,
                context["foot_acceleration_memory"],
                horizon,
                contact=logits.sigmoid(),
            )
            terms.append(analytic)
        combined = torch.cat(terms, -1)
        return combined, combined.detach().mean(0)

    def sample_initial_states(self):
        cfg, wm_cfg = self.options.policy, self.options.world_model
        generator = self.sample_generator
        command = self.mdp.command
        ids_all = torch.arange(len(self.xi), device=self.device)
        horizon = max(cfg.horizon, wm_cfg.horizon, *wm_cfg.validation_horizons)
        eligible = ids_all[ids_all % wm_cfg.holdout_stride != 0]
        pieces, total = [], 0
        for _ in range(self.options.collection.sampling_attempts):
            ids = eligible[
                torch.randint(
                    len(eligible),
                    (self.options.collection.sampling_oversample * cfg.batch_size,),
                    device=self.device,
                    generator=generator,
                )
            ]
            ages = torch.randint(
                len(self.replay_buffer) - horizon + 1, ids.shape, device=self.device, generator=generator
            )
            rows = self.replay_buffer.rows(ages)
            frame, motion = (self.replay_buffer.data[key][rows, ids] for key in ("frame", "motion"))
            valid = frame + horizon + self.mdp.reference_frames < command.label_lengths[motion]
            selected = valid.nonzero().flatten()[: cfg.batch_size - total]
            pieces.append((ids[selected], ages[selected]))
            total += len(selected)
            if total == cfg.batch_size:
                return self.replay_buffer.sequences(
                    torch.cat([part[0] for part in pieces]),
                    horizon,
                    generator,
                    ages=torch.cat([part[1] for part in pieces]),
                )
        raise RuntimeError("Insufficient motion windows for the configured actor batch")
