"""Actor/critic learner adapted from SuperMimic's ActorCriticPolicyLearner.

BPTT traverses the environment, then critic regression uses detached imagined
features and lambda-return targets. Sampling, random permutations and EMA updates
remain outside captured optimizer updates.
"""

import copy
import torch
from tensordict import TensorDict

from instinctlab.tasks.shadowing.pgmt.supermimic.execution import CapturedUpdate, compile_function, graph_state
from instinctlab.tasks.shadowing.pgmt.supermimic.interfaces import Learner

from .optimizers import build_optimizers, hidden_matrices, optimize


class PolicyLearner(Learner):
    def __init__(self, policy, value, wm_env, cfg, actor_cfg, critic_cfg, execution, generator, gradient_generator):
        self.policy, self.value, self.wm_env = policy, value, wm_env
        self.cfg, self.actor_cfg, self.critic_cfg = cfg, actor_cfg, critic_cfg
        self.generator, self.gradient_generator = generator, gradient_generator
        self.actor_optimizer = build_optimizers(policy.network, hidden_matrices(policy.network.action_head), actor_cfg)
        self.value_optimizer = build_optimizers(value, hidden_matrices(value.network), critic_cfg)
        self.ema = copy.deepcopy(policy.network).eval().requires_grad_(False)
        self.value_forward = compile_function(value.forward, execution.compile != "none")
        self.actor_updates = self.critic_updates = 0
        self.gradient_data = None
        self._actor_update = (
            CapturedUpdate(self._actor, self.graph_state, execution.capture_warmup)
            if execution.cuda_graph
            else self._actor
        )
        self._critic_update = (
            CapturedUpdate(self._critic, lambda: graph_state([value], [self.value_optimizer]), execution.capture_warmup)
            if execution.cuda_graph
            else self._critic
        )

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
                result = self._actor_update(inputs)
                self.actor_updates += 1
            else:
                with torch.no_grad():
                    result = self.wm_env.rollout(inputs)
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

    def _actor(self, inputs):
        if self.cfg.value_gradient_weight:
            inputs = inputs.clone()
            initial = inputs["data", "state"][:, 0].detach().requires_grad_()
            inputs["data", "state"] = torch.cat((initial[:, None], inputs["data", "state"][:, 1:]), 1)
        result = self.wm_env.rollout(inputs)
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
        losses, norms, errors, cosines = [], [], [], []
        for epoch, order in enumerate(inputs["orders"].unbind(0)):
            for ids in order.tensor_split(self.cfg.critic_minibatches):
                loss = (self.value_forward(inputs["features"][ids]) - inputs["targets"][ids]).square().mean()
                reported_loss = loss.detach()
                if self.gradient_data is not None and epoch == 0:
                    count = min(self.cfg.gradient_batch_size, len(self.gradient_data["state"]))
                    indices = torch.randperm(
                        len(self.gradient_data["state"]), device=ids.device, generator=self.gradient_generator
                    )[:count]
                    surrogate, error, cosine = self.gradient_loss(indices)
                    loss = loss + self.cfg.value_gradient_weight * surrogate
                    errors.append(error)
                    cosines.append(cosine)
                norms.append(optimize(self.value_optimizer, loss, list(self.value.parameters()), self.critic_cfg))
                losses.append(reported_loss)
        self.gradient_data = None
        metrics = dict(loss=torch.stack(losses).mean(), grad_norm=torch.stack(norms).amax())
        if errors:
            metrics["ValueGradient/relative_error"] = torch.stack(errors).mean()
            metrics["ValueGradient/cosine"] = torch.stack(cosines).mean()
        return metrics

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
