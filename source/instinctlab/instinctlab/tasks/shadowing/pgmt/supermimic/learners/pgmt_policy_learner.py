"""Actor/critic learner adapted from SuperMimic's ActorCriticPolicyLearner.

BPTT traverses the environment, then critic regression uses detached imagined
features and lambda-return targets. Sampling, random permutations and EMA updates
remain outside captured optimizer updates.
"""

import copy
import torch
from tensordict import TensorDict

from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_learner_utils import (
    CapturedUpdate,
    compile_function,
    graph_state,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.policy_learner import (
    ActorCriticPolicyLearner as SuperMimicPolicyLearner,
)

from .pgmt_learner_utils import build_optimizers, hidden_matrices, optimize
from .policy_learner import gae_returns


class PGMTActorCriticPolicyLearner(SuperMimicPolicyLearner):
    def __init__(self, policy, value, wm_env, cfg, actor_cfg, critic_cfg, execution, generator, gradient_generator):
        self.policy, self.value, self.wm_env = policy, value, wm_env
        self.cfg, self.actor_cfg, self.critic_cfg = cfg, actor_cfg, critic_cfg
        self.generator, self.gradient_generator = generator, gradient_generator
        self.execution = execution
        super().__init__(policy.network, value, wm_env, cfg, device=str(next(value.parameters()).device))
        self.actor_optimizer, self.value_optimizer = self.actor_optimizers[0], self.critic_optimizers[0]
        self._actor_update = self._captured_update
        self.ema = copy.deepcopy(policy.network).eval().requires_grad_(False)
        self.value_forward = compile_function(value.forward, execution.compile != "none")
        self.objective_forward = compile_function(self.objective, execution.compile != "none")
        self.actor_updates = self.critic_updates = 0
        self.gradient_data = None
        self._critic_update = self._capture(self._critic, lambda: graph_state([value], [self.value_optimizer]))

    def _build_optimizers(self, module):
        if module is self.value:
            return [build_optimizers(module, hidden_matrices(module.network), self.critic_cfg)]
        return [build_optimizers(module, hidden_matrices(module.action_head), self.actor_cfg)]

    def _build_critic_optimizers(self, module):
        return self._build_optimizers(module)

    def _compile_actor(self, module):
        # PGMT compiles its structured action_from_inputs callable in setup.
        return module

    def _capture(self, fn, state):
        return CapturedUpdate(fn, state, self.execution.capture_warmup) if self.execution.cuda_graph else fn

    def _value_bound(self):
        return self.cfg.value_upper_bound

    def _graph_state(self):
        return self.graph_state()

    def _summarize(self, totals, count):
        return {key: value / count for key, value in totals.items()}

    @property
    def capture_status(self) -> dict[str, bool]:
        return {
            name: isinstance(update, CapturedUpdate) and update.captured
            for name, update in (("actor", self._actor_update), ("critic", self._critic_update))
        }

    def graph_state(self):
        env = self.wm_env
        return graph_state(
            [self.policy.network, self.policy.normalizer, self.value, env.dynamics, env.dynamics.terrain],
            [self.actor_optimizer],
        ) + [env.xi, env.foot_rewards.scale]

    def update(self, inputs, *, train_actor=True):
        self.policy.network.train()
        self.policy.normalizer.eval()
        self.wm_env.dynamics.eval().requires_grad_(False)
        self.value.requires_grad_(False)
        # Clear old supervised gradients so frozen-model checks have meaning.
        for parameter in self.wm_env.dynamics.parameters():
            parameter.grad = None
        try:
            if train_actor:
                result = super().update(torch.stack([inputs]))
                self.actor_updates += 1
            else:
                with torch.no_grad():
                    result = self.imagine(inputs)
                result["grad_norm"] = result["loss"].new_zeros(())
            if any(parameter.grad is not None for parameter in self.wm_env.dynamics.parameters()):
                raise AssertionError("Actor update wrote world-model parameter gradients")
        finally:
            self.wm_env.dynamics.requires_grad_(True)
            self.value.requires_grad_(True)
        features, targets = result["features"].flatten(0, 1), result["targets"].flatten()
        orders = torch.stack(
            [
                torch.randperm(len(targets), device=targets.device, generator=self.generator)
                for _ in range(self.cfg.critic_epochs)
            ]
        )
        critic_input = TensorDict(dict(features=features, targets=targets, orders=orders), batch_size=[])
        critic = (self._critic_update if train_actor else self._critic)(critic_input)
        self.critic_updates += self.cfg.critic_epochs * self.cfg.critic_minibatches
        if train_actor:
            self.update_ema()
        return dict(
            actor_loss=result["loss"].detach(),
            critic_loss=critic["loss"],
            actor_grad_norm=result["grad_norm"],
            critic_grad_norm=critic["grad_norm"],
            imagination_survival=result["survival"],
            reward_means=result["reward_means"],
            **{key: value for key, value in critic.items() if key.startswith("ValueGradient/")},
            **{key: value for key, value in result.items() if key.startswith(("Encoder/",))}
        )

    def imagine(self, inputs):
        """The learner owns actor execution, value bootstrap and return targets."""
        observation = self.wm_env.reset(inputs)
        for _ in range(self.cfg.horizon):
            action = self.policy(observation["actor"])
            observation, _, _, _ = self.wm_env.step(action)
        trajectory = self.wm_env.evaluate_trajectory()
        features = trajectory.pop("features")
        terminal_value = self.value_forward(features[:, -1])
        if self.cfg.value_upper_bound is not None:
            terminal_value = terminal_value.clamp(max=self.cfg.value_upper_bound)
        with torch.no_grad():
            values = self.value_forward(features.detach())
        loss, targets = self.objective_forward(trajectory["rewards"], trajectory["survival"], terminal_value, values)
        return dict(
            loss=loss,
            features=features[:, :-1].detach(),
            targets=targets,
            survival=trajectory["survival"].prod(1).detach().mean(),
            valid=trajectory["valid"],
            reward_means=trajectory["reward_means"],
        )

    def objective(self, rewards, survival, terminal_value, values):
        discount = (self.cfg.gamma * survival).cumprod(1)
        before = torch.cat((discount.new_ones(len(discount), 1), discount[:, :-1]), 1)
        objective = (before * rewards).sum(1) + discount[:, -1] * terminal_value
        with torch.no_grad():
            targets = gae_returns(
                values[:, :-1].T,
                values[:, -1],
                rewards.detach().T,
                1 - survival.detach().T,
                self.cfg.gamma,
                self.cfg.lam,
            ).T
        if self.cfg.value_upper_bound is not None:
            targets = targets.clamp(max=self.cfg.value_upper_bound)
        return -objective.mean(), targets

    def _train_batch(self, inputs):
        if self.cfg.value_gradient_weight:
            inputs = inputs.clone()
            initial = inputs["data", "state"][:, 0].detach().requires_grad_()
            inputs["data", "state"] = torch.cat((initial[:, None], inputs["data", "state"][:, 1:]), 1)
        result = self.imagine(inputs)
        torch._assert_async(result["valid"].all(), "Imagined rollout left valid reference/terrain domain")
        if self.cfg.value_gradient_weight:
            target = torch.autograd.grad(-len(initial) * result["loss"], initial, retain_graph=True)[0]
            self.gradient_data = dict(
                state=initial.detach(),
                target=target.detach(),
                history=inputs["data", "history"][:, 0].detach(),
                support=inputs["data", "support"][:, 0].detach(),
                context=inputs["contexts"][0].to_dict(),
                noise=inputs["observation_noise"][0].to_dict(),
                ids=inputs["data", "env_id"],
                scale=(inputs["data", "next_state"][:, 0] - initial.detach())
                .std(0)
                .clamp_min(self.cfg.gradient_scale_floor),
            )
        result["grad_norm"] = optimize(
            self.actor_optimizer, result["loss"], list(self.policy.network.parameters()), self.actor_cfg
        )
        encoder = getattr(self.policy.network, "encoder", None)
        if encoder is not None:
            gradients = [p.grad.square().sum() for p in encoder.parameters() if p.grad is not None]
            result["Encoder/encoder_gradient_norm"] = (
                torch.stack(gradients).sum().sqrt().detach() if gradients else result["loss"].new_zeros(())
            )
        result["loss"] = result["loss"].detach()
        return result

    def _critic(self, inputs):
        orders = [order.tensor_split(self.cfg.critic_minibatches) for order in inputs["orders"].unbind(0)]
        result = self._regress_critic(inputs["features"], inputs["targets"], orders)
        self.gradient_data = None
        return result

    def _critic_loss(self, features, targets, ids, epoch):
        loss = (self.value_forward(features[ids]) - targets[ids]).square().mean()
        stats = dict(value_loss=loss.detach())
        if self.gradient_data is not None and epoch == 0:
            count = min(self.cfg.gradient_batch_size, len(self.gradient_data["state"]))
            indices = torch.randperm(
                len(self.gradient_data["state"]), device=ids.device, generator=self.gradient_generator
            )[:count]
            surrogate, error, cosine = self.gradient_loss(indices)
            loss = loss + self.cfg.value_gradient_weight * surrogate
            stats.update({"ValueGradient/relative_error": error, "ValueGradient/cosine": cosine})
        return loss, stats

    def _critic_optimize(self, loss):
        return optimize(self.value_optimizer, loss, list(self.value.parameters()), self.critic_cfg)

    def _critic_summary(self, statistics):
        result = dict(
            loss=torch.stack([s["value_loss"] for s in statistics]).mean(),
            grad_norm=torch.stack([s["grad_norm"] for s in statistics]).amax(),
        )
        for key in ("ValueGradient/relative_error", "ValueGradient/cosine"):
            values = [s[key] for s in statistics if key in s]
            if values:
                result[key] = torch.stack(values).mean()
        return result

    def gradient_loss(self, indices):
        data = self.gradient_data
        assert data is not None
        state = data["state"][indices].detach().requires_grad_()
        context = {key: value[indices] for key, value in data["context"].items()}
        noise = {key: value[indices] for key, value in data["noise"].items()}

        def features(value):
            return self.wm_env.features(
                value, data["history"][indices], data["support"][indices], context, data["ids"][indices], noise
            )[0]

        x = features(state)
        leaf = x.detach().requires_grad_()
        u = torch.autograd.grad(self.value(leaf).sum(), leaf, create_graph=True)[0]
        prediction = torch.autograd.grad(x, state, grad_outputs=u.detach())[0].detach().requires_grad_()
        scale = data["scale"]
        rotation_slice = self.wm_env.dynamics.representation.layout.state["rotation"]

        def tangent(value):
            rotation = state[:, rotation_slice].detach()
            angular = value[:, rotation_slice] - (value[:, rotation_slice] * rotation).sum(-1, keepdim=True) * rotation
            return torch.cat((value[:, : rotation_slice.start], angular, value[:, rotation_slice.stop :]), -1)

        predicted, target = tangent(prediction) * scale, tangent(data["target"][indices]) * scale
        error = (predicted - target).square().sum() / target.square().sum().clamp_min(self.cfg.denominator_epsilon)
        direction = tangent(torch.autograd.grad(error, prediction)[0])
        with torch.no_grad():
            epsilon = self.cfg.gradient_epsilon / (direction / scale).norm(dim=-1).clamp_min(
                self.cfg.denominator_epsilon
            )

            def shifted(sign):
                value = state + sign * epsilon[:, None] * direction
                rotation = torch.nn.functional.normalize(value[:, rotation_slice], dim=-1)
                return torch.cat((value[:, : rotation_slice.start], rotation, value[:, rotation_slice.stop :]), -1)

            jacobian = (features(shifted(1)) - features(shifted(-1))) / (2 * epsilon[:, None])
            cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        return (u * jacobian).sum(), error.detach(), cosine

    @torch.no_grad()
    def update_ema(self):
        torch._foreach_lerp_(
            list(self.ema.parameters()), list(self.policy.network.parameters()), 1 - self.cfg.ema_decay
        )
        for averaged, current in zip(self.ema.buffers(), self.policy.network.buffers(), strict=True):
            averaged.copy_(current)

    def state_dict(self):
        return dict(
            actor=self.policy.network.state_dict(),
            value=self.value.state_dict(),
            ema=self.ema.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            value_optimizer=self.value_optimizer.state_dict(),
            actor_updates=self.actor_updates,
            critic_updates=self.critic_updates,
        )

    def load_state_dict(self, state):
        if isinstance(self._actor_update, CapturedUpdate):
            self._actor_update.reset()
        if isinstance(self._critic_update, CapturedUpdate):
            self._critic_update.reset()

        self.policy.network.load_state_dict(state["actor"])
        self.value.load_state_dict(state["value"])
        self.ema.load_state_dict(state["ema"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.actor_updates, self.critic_updates = state["actor_updates"], state["critic_updates"]
