"""Regression checks for the upstream foundation and PGMT inheritance boundary."""

import torch
import unittest
from tensordict import TensorDict
from types import SimpleNamespace

from instinctlab.tasks.shadowing.pgmt.supermimic.learners.learner_utils import CapturedUpdate as CoreCapture
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_learner_utils import CapturedUpdate, lambda_returns
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_policy_learner import (
    PGMTActorCriticPolicyLearner as PolicyLearner,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_wm_learner import (
    PGMTWorldModelLearner as WorldModelLearner,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.policy_learner import ActorCriticPolicyLearner, gae_returns
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.wm_learner import WorldModelLearner as CoreWM


class ToyWM(CoreWM):
    """Change task loss and execution; keep the upstream update loop."""

    def _build_optimizers(self):
        return [torch.optim.SGD(self.dynamics.parameters(), lr=0.1)]

    def _loss_weights(self):
        return {}

    def _compile(self, fn):
        return fn

    def _capture(self, fn, state):
        return fn

    def _batch_loss(self, data):
        loss = (self.dynamics(data["x"]) - data["y"]).square().mean()
        return loss, loss.new_empty(0)


class InheritanceTests(unittest.TestCase):
    def test_upstream_math_keeps_wxyz_at_the_isaaclab_boundary(self):
        from instinctlab.tasks.shadowing.pgmt.supermimic.utils import math

        half = 2**-0.5
        rotation = torch.tensor([[half, 0.0, 0.0, half]])
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[0.0, 1.0, 0.0]])
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(math.quat_apply(rotation, x), y)
        torch.testing.assert_close(math.quat_apply_inverse(rotation, y), x)
        torch.testing.assert_close(math.quat_mul(rotation, math.quat_inv(rotation)), identity)
        torch.testing.assert_close(math.yaw_quat(rotation), rotation)
        torch.testing.assert_close(
            math.matrix_from_quat(rotation), torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        )
        position, orientation = math.subtract_frame_transforms(
            torch.tensor([[1.0, 2.0, 0.0]]), rotation, torch.tensor([[1.0, 3.0, 0.0]]), rotation
        )
        torch.testing.assert_close(position, x)
        torch.testing.assert_close(orientation, identity)

    def test_runner_and_environment_execute_the_upstream_templates(self):
        import inspect

        from instinctlab.tasks.shadowing.pgmt.supermimic.envs.pgmt_world_model_env import PGMTWorldModelEnv
        from instinctlab.tasks.shadowing.pgmt.supermimic.envs.world_model_env import WorldModelEnv
        from instinctlab.tasks.shadowing.pgmt.supermimic.pgmt_replay_buffer import PGMTReplayBuffer
        from instinctlab.tasks.shadowing.pgmt.supermimic.replay_buffer import ReplayBuffer
        from instinctlab.tasks.shadowing.pgmt.supermimic.runners.pgmt_runner import PGMTSuperMimicRunner
        from instinctlab.tasks.shadowing.pgmt.supermimic.runners.supermimic_runner import BaseRunner, SuperMimicRunner

        self.assertIs(PGMTSuperMimicRunner.run, SuperMimicRunner.run)
        self.assertIs(PGMTSuperMimicRunner.iteration, SuperMimicRunner.iteration)
        self.assertIs(PGMTSuperMimicRunner.collect, SuperMimicRunner.collect)
        self.assertIs(PGMTSuperMimicRunner._step_and_store, BaseRunner._step_and_store)
        self.assertIs(PGMTWorldModelEnv.reset, WorldModelEnv.reset)
        self.assertIs(PGMTWorldModelEnv.step, WorldModelEnv.step)
        self.assertTrue(issubclass(PGMTReplayBuffer, ReplayBuffer))
        arguments = inspect.signature(PGMTWorldModelEnv).parameters
        self.assertNotIn("policy", arguments)
        self.assertNotIn("value", arguments)

    def test_task_extensions_reuse_the_update_and_capture_loops(self):
        self.assertIs(WorldModelLearner._update_from, CoreWM._update_from)
        self.assertIs(PolicyLearner._update_from, ActorCriticPolicyLearner._update_from)
        self.assertIs(PolicyLearner._regress_critic, ActorCriticPolicyLearner._regress_critic)
        self.assertIs(CapturedUpdate.__call__, CoreCapture.__call__)

    def test_base_world_model_updates_every_minibatch_and_reduces_losses(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        learner = ToyWM(model, SimpleNamespace(represent=lambda x: x), SimpleNamespace())
        data = TensorDict(
            dict(
                state=torch.zeros(2, 1, 1),
                action=torch.zeros(2, 1, 1),
                sim_ctx=TensorDict(dict(x=torch.ones(2, 1, 1), y=torch.zeros(2, 1, 1)), [2]),
            ),
            [2],
        )
        # The original default batch selection stays unchanged; the task extends it.
        learner._prepare_batches = lambda batches: batches["sim_ctx"]
        metrics = learner.update(data)
        torch.testing.assert_close(model.weight, torch.tensor([[0.8]]))
        torch.testing.assert_close(metrics["Loss/state"], torch.tensor((1 + 0.81) / 2))

    def test_upstream_gae_and_pgmt_returns_agree_for_soft_termination(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.4, 0.3, 0.2]])
        rewards = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.3, 0.2]])
        survival = torch.tensor([[1.0, 0.6, 0.0], [1.0, 1.0, 1.0]])
        actual = gae_returns(values[:, :-1].T, values[:, -1], rewards.T, 1 - survival.T, 0.99, 0.95)
        expected = lambda_returns(rewards, survival, values, torch.zeros_like(rewards, dtype=torch.bool), 0.99, 0.95)
        torch.testing.assert_close(actual.T, expected)


if __name__ == "__main__":
    unittest.main()
