"""Actor owns the action head; the shared geometric encoder belongs to the WM optimizer."""

from .pgmt_learner_utils import build_optimizers, hidden_matrices
from .pgmt_policy_learner import PGMTActorCriticPolicyLearner


class DepthPolicyLearner(PGMTActorCriticPolicyLearner):
    def _build_optimizers(self, module):
        if module is self.value:
            return super()._build_optimizers(module)
        return [build_optimizers(module.action_head, hidden_matrices(module.action_head), self.actor_cfg)]
