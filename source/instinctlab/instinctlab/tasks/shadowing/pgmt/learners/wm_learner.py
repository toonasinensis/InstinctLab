"""Supervised H8 dynamics updates; sampling and moments belong to the runner."""

import torch

from .learner_utils import CapturedUpdate, graph_state, optimize


class WorldModelLearner:
    def __init__(self, dynamics, optimizer, loss, capture=False):
        self.dynamics, self.optimizer, self.loss = dynamics, optimizer, loss
        self.parameters = list(dynamics.parameters())
        self.update = (
            CapturedUpdate(self._update, lambda: graph_state([dynamics], [optimizer])) if capture else self._update
        )

    def _update(self, batches):
        losses, contacts, norms = [], [], []
        for batch in batches.unbind(0):
            loss, contact = self.loss(batch.to_dict())
            norm = optimize(self.optimizer, loss, self.parameters)
            losses.append(loss.detach())
            contacts.append(contact.detach())
            norms.append(norm.detach())
        return dict(
            loss=losses[-1],
            contact_bce=contacts[-1],
            grad_norm=torch.stack(norms).amax(),
        )
