"""Numerical regression tests for the composed SM components; no simulator required."""

import copy
import os
import torch
import unittest
from tensordict import TensorDict

from _legacy_source import legacy_module

from instinctlab.tasks.shadowing.pgmt.supermimic.config import OptimizerCfg, RepresentationCfg, ScheduleCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.layouts import RobotLayout
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_learner_utils import (
    CapturedUpdate,
    build_optimizers,
    graph_state,
    hidden_matrices,
    optimize,
    set_learning_rate,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.pgmt_replay_buffer import PGMTReplayBuffer as ReplayBuffer
from instinctlab.tasks.shadowing.pgmt.supermimic.pgmt_representation import REPRESENTATIONS


class RepresentationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PGMT_LEGACY_SOURCE"), "Requires an external legacy comparison snapshot")
    def test_predictions_targets_and_gradients_match_previous_models(self):
        TerrainDynamics = legacy_module("wm_dynamics").TerrainDynamics
        torch.manual_seed(13)
        for kind in REPRESENTATIONS:
            with self.subTest(kind=kind):
                layout = RobotLayout(29)
                offset = torch.tensor([0.01, -0.02, 0.08])
                rep = REPRESENTATIONS[kind](layout, RepresentationCfg(kind=kind), 0.02, 4, offset)
                old = TerrainDynamics(
                    1,
                    hidden=16,
                    amp=False,
                    predict_force=False,
                    contact_indices=(0, 1),
                    structured=kind == "structured",
                    root_com=offset,
                    prediction_mode="velocity" if kind == "velocity" else "delta",
                )
                x = torch.randn(8, layout.state.width)
                x[:, layout.state["rotation"]] = torch.nn.functional.normalize(x[:, layout.state["rotation"]], dim=-1)
                y = x.clone()
                y[:, layout.state["position"]] += 0.02
                y[:, layout.state["linear_velocity"]] += 0.1
                torch.testing.assert_close(rep.target(x, y), old.target(x, y))
                delta = (torch.randn(8, rep.prediction_dim) * 0.01).requires_grad_()
                # Feed the same unnormalized delta through the legacy integration path.
                old.delta_norm.mean.zero_()
                old.delta_norm.variance.fill_(1 - old.delta_norm.epsilon**2)
                old.predict_delta = lambda *args: (delta, None, None)
                a = torch.zeros(8, 29)
                before = old(x, a, a, None, torch.zeros(8, 441), torch.zeros(8, 1))[0]
                after = rep.integrate(x, delta)
                torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-6)
                weight = torch.randn_like(after)
                g0 = torch.autograd.grad((before * weight).sum(), delta, retain_graph=True)[0]
                g1 = torch.autograd.grad((after * weight).sum(), delta)[0]
                torch.testing.assert_close(g0, g1, rtol=2e-5, atol=2e-5)

    def test_identity_rotation_has_finite_gradients(self):
        layout = RobotLayout(29)
        for kind, cls in REPRESENTATIONS.items():
            rep = cls(layout, RepresentationCfg(kind=kind), 0.02, 4, torch.zeros(3))
            state = torch.zeros(2, layout.state.width)
            state[:, layout.state["rotation"].stop - 1] = 1
            delta = torch.zeros(2, rep.prediction_dim, requires_grad=True)
            rep.integrate(state, delta).sum().backward()
            self.assertTrue(torch.isfinite(delta.grad).all())


class ReplayTests(unittest.TestCase):
    def test_terminal_step_reset_history_and_ring_wrap(self):
        replay = ReplayBuffer(capacity=5, history_length=3)
        histories = []
        history = torch.zeros(1, 3, 1)
        episode_step = 0
        for step in range(9):
            if step == 4:
                episode_step = 0
                history.fill_(step)
            else:
                history = torch.cat((history[:, 1:], torch.full((1, 1, 1), float(step))), 1)
            histories.append(history.clone())
            replay.insert(
                dict(
                    history=history,
                    episode_step=torch.tensor([episode_step]),
                    done=torch.tensor([step == 3]),
                    state=torch.tensor([[float(step)]]),
                )
            )
            episode_step += 1
            for age in range(len(replay)):
                reconstructed = replay.histories(torch.tensor([age]), torch.tensor([0]))
                torch.testing.assert_close(reconstructed, histories[step - len(replay) + 1 + age])
            if step == 4:
                data = replay.sequences(torch.tensor([0]), 5, torch.Generator(), ages=torch.tensor([0]))
                torch.testing.assert_close(data["mask"], torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]]))
        replay.clear()
        self.assertEqual(len(replay), 0)


class ReturnTests(unittest.TestCase):
    def test_timeout_bootstraps_terminal_value_without_reset_leak(self):
        from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_learner_utils import lambda_returns

        rewards = torch.tensor([[1.0, 2.0, 100.0]])
        values = torch.tensor([[0.0, 10.0, 20.0, 999.0]])
        continuation = torch.ones_like(rewards)
        done = torch.tensor([[False, True, False]])
        result = lambda_returns(rewards, continuation, values, done, gamma=0.5, lam=1.0)
        # Step 1 bootstraps its terminal value 20; the following reset's reward 100 is excluded.
        torch.testing.assert_close(result[0, :2], torch.tensor([7.0, 12.0]))
        continuation[0, 1] = 0
        result = lambda_returns(rewards, continuation, values, done, gamma=0.5, lam=1.0)
        torch.testing.assert_close(result[0, :2], torch.tensor([2.0, 2.0]))

    def test_episode_reward_uses_configured_duration(self):
        from instinctlab.tasks.shadowing.pgmt.episode_rewards import EpisodeRewards

        terms = ("base", "match", "support", "penetration", "slip")
        accumulator = EpisodeRewards(2, terms, 1, 10.0, "cpu")
        reward = torch.tensor([[0.02, 0.0, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0, 0.0]])
        for step in range(500):
            accumulator.add(reward, torch.tensor([step == 249 or step == 499, step == 499]))
            if step == 249:
                result = accumulator.take()
                self.assertAlmostEqual(result["Episode_Reward/base"], 0.5, places=5)
        result = accumulator.take()
        self.assertAlmostEqual(result["Episode_Reward/base"], 0.75, places=5)
        self.assertTrue(torch.equal(accumulator.running, torch.zeros_like(accumulator.running)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA capture requires a GPU")
class OptimizerTests(unittest.TestCase):
    def test_captured_muon_tracks_changing_learning_rate(self):
        torch.manual_seed(42)
        original = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 16), torch.nn.SiLU(), torch.nn.Linear(16, 2)
        ).cuda()
        captured = copy.deepcopy(original)
        cfg = OptimizerCfg(kind="muon", lr=0.002)
        optimizers = [build_optimizers(net, hidden_matrices(net), cfg) for net in (original, captured)]

        def update(net, optimizer, data):
            loss = (net(data["x"]) - data["y"]).square().mean()
            norm = optimize(optimizer, loss, list(net.parameters()), cfg)
            return dict(loss=loss.detach(), norm=norm)

        graph = CapturedUpdate(
            lambda data: update(captured, optimizers[1], data),
            lambda: graph_state([captured], [optimizers[1]]),
            warmup=3,
        )
        schedule = ScheduleCfg(duration=8, min_factor=0.03)
        for step in range(9):
            data = TensorDict(
                dict(x=torch.randn(32, 8, device="cuda"), y=torch.randn(32, 2, device="cuda")), batch_size=[]
            )
            for opt in optimizers:
                set_learning_rate(opt, cfg.lr, schedule, step)
            eager = update(original, optimizers[0], data)
            actual = graph(data)
            torch.testing.assert_close(actual["loss"], eager["loss"], rtol=2e-5, atol=2e-5)
            for before, after in zip(original.parameters(), captured.parameters(), strict=True):
                torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-5)
        self.assertIsNotNone(graph._graph)
        # Loading an optimizer may rebind its state tensors; reset and capture again.
        optimizers[1].load_state_dict(copy.deepcopy(optimizers[1].state_dict()))
        graph.reset()
        self.assertFalse(graph.captured)
        for _ in range(4):
            data = TensorDict(
                dict(x=torch.randn(32, 8, device="cuda"), y=torch.randn(32, 2, device="cuda")), batch_size=[]
            )
            eager = update(original, optimizers[0], data)
            actual = graph(data)
            torch.testing.assert_close(actual["loss"], eager["loss"], rtol=2e-5, atol=2e-5)
            for before, after in zip(original.parameters(), captured.parameters(), strict=True):
                torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-5)
        self.assertTrue(graph.captured)


if __name__ == "__main__":
    unittest.main()
