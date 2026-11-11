"""PGMT differentiable environment using SuperMimic's reset/step boundary.

FK for trajectory rewards is batched after the rollout. Support height still
feeds the next observation at every step. Only parameters of the policy receive
actor gradients; state, terrain-query coordinates and support remain connected.
"""

import torch
from tensordict import TensorDict

from .execution import compile_function
from .interfaces import WorldModelEnv
from .mdp import lambda_returns


class PGMTWorldModelEnv(WorldModelEnv):
    def __init__(self, policy, dynamics, value, mdp, context, foot_rewards, analytic_foot, cfg, execution):
        self.policy, self.dynamics, self.value, self.mdp = policy, dynamics, value, mdp
        self.xi, self.rewards = context, mdp.rewards
        self.foot_rewards, self.analytic_foot = foot_rewards, analytic_foot
        self.cfg, self.execution = cfg, execution
        chain = execution.compile == "chain"
        self.features_forward = compile_function(self.features, chain)
        self.step_forward = compile_function(self.transition, chain)
        self.bodies_forward = compile_function(self.rewards.bodies, chain)
        self.objective_forward = compile_function(self.rollout_objective, execution.compile != "none")
        self.value_forward = compile_function(value.forward, chain)

    @staticmethod
    def start(data):
        names = ("motion", "frame", "placement_position", "placement_rotation", "level", "family")
        return {key: data[key][:, 0] for key in names} | {"env_id": data["env_id"]}

    def prepare(self, data, generator):
        batch = len(data["env_id"])
        contexts = [
            TensorDict(self.mdp.context(self.start(data), step), batch_size=[batch])
            for step in range(self.cfg.horizon + 1)
        ]
        observation = [self.mdp.draw_noise(batch, self.mdp.observation_widths, generator)]
        proprio = []
        for _ in range(self.cfg.horizon):
            proprio.append(self.mdp.draw_noise(batch, self.mdp.proprio_widths, generator))
            observation.append(self.mdp.draw_noise(batch, self.mdp.observation_widths, generator))
        return TensorDict(
            dict(
                data=TensorDict(data, batch_size=[]),
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
        self, state, history, support, previous_action, observation, context, ids, proprio_noise, observation_noise
    ):
        action = self.policy(observation)
        successor, logits = self.dynamics(
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

    def reset(self, inputs):
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

    def step(self, step):
        successor, history, support, action, features, observation, valid, logits = self.step_forward(
            self.state,
            self.history,
            self.support,
            self.previous_action,
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

    def rollout(self, inputs):
        self.reset(inputs)
        for step in range(self.cfg.horizon):
            self.step(step)
        batch, horizon, ids = len(self.ids), self.cfg.horizon, self.ids
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
        terminal_value = self.value_forward(features[:, -1])
        if self.cfg.value_upper_bound is not None:
            terminal_value = terminal_value.clamp(max=self.cfg.value_upper_bound)
        with torch.no_grad():
            values = self.value_forward(features.detach())
        loss, targets, alive, means = self.objective_forward(
            all_states,
            torch.stack(self.actions, 1).flatten(0, 1),
            torch.stack(self.previous_actions, 1).flatten(0, 1),
            torch.stack(self.supports, 1).flatten(),
            context,
            torch.stack([row["ref_body_velocity"] for row in self.contexts[:-1]], 1).flatten(0, 1),
            derivative_valid.flatten(),
            ids,
            bodies,
            terminal_value,
            values,
            torch.stack(self.logits, 1).flatten(0, 1) if self.logits else None,
        )
        return dict(
            loss=loss,
            features=features[:, :-1].detach(),
            targets=targets,
            survival=alive,
            valid=self.valid,
            reward_means=means,
        )

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
        discount = (self.cfg.gamma * survival).cumprod(1)
        before = torch.cat((discount.new_ones(batch, 1), discount[:, :-1]), 1)
        objective = (before * rewards).sum(1) + discount[:, -1] * terminal_value
        with torch.no_grad():
            targets = lambda_returns(
                rewards.detach(),
                survival.detach(),
                values,
                torch.zeros_like(survival, dtype=torch.bool),
                self.cfg.gamma,
                self.cfg.lam,
            )
        if self.cfg.value_upper_bound is not None:
            targets = targets.clamp(max=self.cfg.value_upper_bound)
        return (
            -objective.mean(),
            targets,
            survival.prod(1).detach().mean(),
            reward_means,
        )

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
