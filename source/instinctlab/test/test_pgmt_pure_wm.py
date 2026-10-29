"""Essential CPU contracts for standalone world-model training components."""

import torch
import unittest

from instinctlab.tasks.shadowing.pgmt.pure_wm_model import (
    ImaginedValue,
    RunningMoments,
    TerrainDynamics,
    lambda_returns,
)
from instinctlab.tasks.shadowing.pgmt.pure_wm_replay import TransitionReplay


class ModelContractTest(unittest.TestCase):
    def test_online_normalization_and_stop_count(self):
        norm = RunningMoments(2, epsilon=0.01, max_count=5)
        data = torch.tensor([[1.0, 3.0], [2.0, 8.0], [4.0, -1.0], [9.0, 5.0], [10.0, 6.0]])
        norm.update(data[:2])
        norm.update(data[2:])
        torch.testing.assert_close(norm.mean, data.mean(0))
        torch.testing.assert_close(norm.std(), (data.var(0, unbiased=False) + 0.01**2).sqrt())
        norm.update(data + 100)
        torch.testing.assert_close(norm.mean, data.mean(0))
        self.assertEqual(norm.count, 5)

    def test_integrated_error_uses_ground_truth_frame(self):
        reference = torch.zeros(2, 71)
        reference[:, 6] = 1
        reference[1, 5:7] = torch.tensor([2**-0.5, 2**-0.5])
        prediction = reference.clone()
        prediction[:, 0] += 0.2
        error = TerrainDynamics.state_error(prediction, reference)
        torch.testing.assert_close(error[0, :3], torch.tensor([0.2, 0.0, 0.0]))
        torch.testing.assert_close(error[1, :3], torch.tensor([0.0, -0.2, 0.0]), atol=1e-6, rtol=0)

    def test_frozen_models_keep_actor_input_derivatives(self):
        torch.manual_seed(8)
        wm = TerrainDynamics(64, hidden=16, amp=False)
        torch.nn.init.normal_(wm.delta_head.weight, std=0.01)
        wm.requires_grad_(False)
        value = ImaginedValue(71).requires_grad_(False)
        torch.nn.init.normal_(value.net[-1].weight, std=0.01)
        actor = torch.nn.Linear(71, 29)
        state = torch.zeros(2, 71)
        state[:, 6] = 1
        previous = torch.zeros(2, 29)
        contact, heights, context = torch.zeros(2, 14), torch.zeros(2, 441), torch.zeros(2, 64)
        for _ in range(3):
            action = actor(state)
            state, logits, _ = wm(state, action, previous, contact, heights, context)
            contact, previous = logits.sigmoid(), action
        value(state).sum().backward()
        self.assertGreater(float(actor.weight.grad.norm()), 0)
        self.assertTrue(torch.isfinite(actor.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in wm.parameters()))
        self.assertTrue(all(p.grad is None for p in value.parameters()))

    def test_lambda_targets_terminal_reward_and_detachment(self):
        rewards = torch.tensor([[1.0, 3.0], [2.0, 100.0]], requires_grad=True)
        survival = torch.tensor([[1.0, 0.0], [0.7, 1.0]], requires_grad=True)
        values = torch.tensor([[99.0, 88.0], [4.0, 200.0], [3.0, 400.0]], requires_grad=True)
        actual = lambda_returns(rewards, survival, values)
        last = 2 + 0.99 * 0.7 * 3
        self.assertAlmostEqual(float(actual[0, 0]), 1 + 0.99 * (0.05 * 4 + 0.95 * last), places=5)
        self.assertEqual(float(actual[0, 1]), 3)
        self.assertFalse(actual.requires_grad)
        with self.assertRaises(ValueError):
            lambda_returns(rewards, survival, values[:-1])


class ReplayContractTest(unittest.TestCase):
    @staticmethod
    def record(step):
        return dict(
            state=torch.full((8, 1), float(step)),
            done=torch.full((8,), step % 4 == 3),
            episode_step=torch.full((8,), step % 4),
            frame=torch.full((8,), step + 10),
            motion=torch.zeros(8, dtype=torch.long),
            proprioception=torch.full((8, 3), float(step)),
        )

    def test_wrap_windows_include_terminal_and_exclude_reset(self):
        replay = TransitionReplay(6, 8, "cpu")
        for step in range(10):
            replay.add(self.record(step))
        data = replay.sequences(128, 4)
        states = data["state"][..., 0]
        self.assertGreaterEqual(float(states.min()), 4)
        torch.testing.assert_close(states[:, 1:] - states[:, :-1], torch.ones(128, 3))
        for index in range(4):
            expected = (~data["done"][:, :index]).all(-1).float()
            torch.testing.assert_close(data["mask"][:, index], expected)
        self.assertTrue((data["env_id"] % 8 != 0).all())
        self.assertTrue((replay.sequences(8, 2, validation=True)["env_id"] == 0).all())

    def test_history_backfills_first_frame_without_crossing_episode(self):
        replay = TransitionReplay(12, 8, "cpu")
        for step in range(12):
            replay.add(self.record(step))
        samples, history = replay.starts(64, 32, torch.tensor([1000]), history_length=3)
        current = samples["state"][:, 0]
        since_reset = samples["episode_step"]
        for slot, back in enumerate((2, 1, 0)):
            expected = current - torch.minimum(since_reset, torch.full_like(since_reset, back))
            torch.testing.assert_close(history[:, slot, 0], expected)
        self.assertTrue((samples["env_id"] % 8 != 0).all())

    def test_bad_reference_coverage_fails_boundedly(self):
        replay = TransitionReplay(12, 8, "cpu")
        for step in range(12):
            replay.add(self.record(step))
        with self.assertRaisesRegex(RuntimeError, "reference coverage"):
            replay.starts(2, 32, torch.tensor([20]))
        replay.clear()
        with self.assertRaises(ValueError):
            replay.sequences(1, 8)


if __name__ == "__main__":
    unittest.main()
