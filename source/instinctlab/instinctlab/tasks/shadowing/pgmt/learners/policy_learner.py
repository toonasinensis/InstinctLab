"""BPTT actor and independently regressed imagined critic, following desktop SM."""

import torch

from .learner_utils import CapturedUpdate, graph_state, optimize


class PolicyLearner:
    def __init__(
        self,
        actor,
        dynamics,
        value,
        actor_optimizer,
        value_optimizer,
        imagine,
        value_loss,
        capture=False,
        state_modules=(),
        state_tensors=(),
    ):
        self.actor, self.dynamics, self.value = actor, dynamics, value
        self.actor_optimizer, self.value_optimizer = actor_optimizer, value_optimizer
        self.imagine, self.value_loss = imagine, value_loss
        self.actor_parameters = list(actor.parameters())
        state = lambda: graph_state([actor, dynamics, value, *state_modules], [actor_optimizer]) + list(state_tensors)
        self.update_actor = CapturedUpdate(self._actor, state) if capture else self._actor
        state = lambda: graph_state([value], [value_optimizer])
        self.update_value = CapturedUpdate(self._value, state) if capture else self._value

    def _actor(self, inputs):
        loss, features, targets, alive, valid, reward_means = self.imagine(inputs)
        torch._assert_async(valid.all(), "Imagined rollout left valid terrain/reference domain")
        norm = optimize(self.actor_optimizer, loss, self.actor_parameters, 0.1)
        encoder_norm = torch.stack([p.grad.square().sum() for p in self.actor.encoder.parameters()]).sum().sqrt()
        return dict(
            loss=loss.detach(),
            features=features,
            targets=targets.detach(),
            survival=alive,
            valid=valid,
            grad_norm=norm.detach(),
            encoder_grad_norm=encoder_norm.detach(),
            reward_means=reward_means,
        )

    def _value(self, inputs):
        features, targets = inputs["features"], inputs["targets"]
        losses, norms = [], []
        for order in inputs["orders"].unbind(0):
            for ids in order.chunk(4):
                loss = self.value_loss(features[ids], targets[ids])
                norm = optimize(self.value_optimizer, loss, list(self.value.parameters()))
                losses.append(loss.detach())
                norms.append(norm.detach())
        return dict(loss=torch.stack(losses).mean(), grad_norm=torch.stack(norms).amax())
