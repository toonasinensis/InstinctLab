"""CPU contracts for scheme-one geometry, dynamics, and checkpoint actor layout."""

import torch
import unittest

from isaaclab.sensors.ray_caster.patterns import GridPatternCfg, grid_pattern

from instinctlab.tasks.shadowing.pgmt.heightfield import HeightField
from instinctlab.tasks.shadowing.pgmt.imagination import pack_history, raw_history, stance_slip
from instinctlab.tasks.shadowing.pgmt.world_model import TerrainWorldModel, rotation_increment
from instinctlab.tasks.shadowing.pgmt.world_model_runner import WorldModelRunner


class GeometryTest(unittest.TestCase):
    def setUp(self):
        xy = torch.linspace(-3, 3, 121, dtype=torch.float64)
        yy, xx = torch.meshgrid(xy, xy, indexing="ij")
        self.field = HeightField(0.3 * xx - 0.2 * yy, [-3, -3], 0.05)

    def test_native_scan_order(self):
        rays, _ = grid_pattern(GridPatternCfg(resolution=0.1, size=(2, 2), ordering="yx"), "cpu")
        torch.testing.assert_close(self.field.local_xy.float(), rays[:, :2], atol=2e-7, rtol=0)

    def test_values_and_coordinate_gradients(self):
        pose = torch.tensor([[0.137, -0.243, 0.8, 0.273]], dtype=torch.float64, requires_grad=True)
        weights = torch.linspace(0.1, 1, 441, dtype=torch.float64)[None]

        def objective(value):
            h, valid = self.field.local(value[:, :3], value[:, 3])
            self.assertTrue(valid.all())
            return (h * weights).mean()

        loss = objective(pose)
        actual = torch.autograd.grad(loss, pose)[0]
        differences = []
        for i in range(4):
            delta = torch.zeros_like(pose)
            delta[0, i] = 1e-5
            differences.append((objective(pose + delta) - objective(pose - delta)) / 2e-5)
        torch.testing.assert_close(actual[0], torch.stack(differences), atol=2e-8, rtol=1e-7)
        self.assertGreater(actual[0, 3].abs().item(), 1e-3)

    def test_bounds_are_explicit(self):
        _, valid = self.field.query(torch.tensor([[4.0, 0.0]], dtype=torch.float64))
        self.assertFalse(valid.item())

    def test_future_perception_backpropagates_to_action(self):
        action = torch.tensor([[0.2, -0.1, 0.04]], dtype=torch.float64, requires_grad=True)
        # A known, nonzero dynamics Jacobian isolates the coordinate path from actor parameters.
        future_position = torch.tensor([[0.0, 0.0, 0.8]], dtype=torch.float64) + 0.02 * action
        heights, _ = self.field.local(future_position, action[:, 0] * 0.1)
        next_action = (heights * torch.linspace(0, 1, 441)).mean()
        gradient = torch.autograd.grad(next_action, action)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs() > 1e-5).all())

    def test_frozen_world_model_preserves_multistep_input_gradients(self):
        torch.manual_seed(9)
        model = TerrainWorldModel(hidden=16).double()
        # A fitted model has a nonzero dynamics Jacobian; the constructor deliberately
        # initializes its output head to zero before any supervised updates.
        torch.nn.init.normal_(model.dynamics.weight, std=0.002)
        model.requires_grad_(False)
        initial = torch.zeros(1, 71, dtype=torch.float64)
        initial[:, 2], initial[:, 6] = 0.8, 1.0

        def future_perception(action):
            state = initial
            previous = torch.zeros_like(action)
            contact = torch.zeros(1, 14, dtype=torch.float64)
            for _ in range(3):
                heights, _ = self.field.local(state[:, :3], state[:, 3] * 0)
                state, contact = model.step(state, action, previous, contact, heights)
                previous = action
            heights, _ = self.field.local(state[:, :3], state[:, 3] * 0)
            return heights.square().mean()

        action = torch.randn(1, 29, dtype=torch.float64, requires_grad=True) * 0.1
        derivative = torch.autograd.grad(future_perception(action), action)[0]
        component = derivative.abs().argmax().item()
        perturbation = torch.zeros_like(action)
        perturbation[0, component] = 1e-5
        numerical = (future_perception(action + perturbation) - future_perception(action - perturbation)) / 2e-5
        self.assertGreater(derivative.abs().max().item(), 1e-6)
        torch.testing.assert_close(derivative[0, component], numerical, atol=1e-9, rtol=1e-5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class DynamicsTest(unittest.TestCase):
    def test_stance_slip_gradient_cannot_reduce_contact_probability(self):
        previous = torch.zeros(1, 2, 3, requires_grad=True)
        feet = torch.tensor([[[0.03, 0.04, 0.1], [1.0, 1.0, 0.0]]], requires_grad=True)
        contact = torch.tensor([[0.9, 0.0]], requires_grad=True)
        cost = stance_slip(previous, feet, contact, 0.02).sum()
        torch.testing.assert_close(cost, torch.tensor(5.625))
        cost.backward()
        torch.testing.assert_close(feet.grad, torch.tensor([[[135.0, 180.0, 0.0], [0.0, 0.0, 0.0]]]))
        torch.testing.assert_close(previous.grad, -feet.grad)
        self.assertIsNone(contact.grad)

    def test_ambiguous_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            WorldModelRunner(None, dict(warm_start="stage1.pt", stage2_initialization="stage2.pt"))

    def test_history_term_order(self):
        history = torch.randn(4, 10, 93)
        torch.testing.assert_close(raw_history(pack_history(history)), history)

    def test_delta_target_and_integration(self):
        torch.manual_seed(3)
        wm = TerrainWorldModel(hidden=16)
        state = torch.randn(1, 71) * 0.1
        state[:, 3:7] = rotation_increment(torch.tensor([[0.2, -0.1, 1.1]]))
        delta = torch.randn(1, 70) * 0.02
        with torch.no_grad():
            wm.dynamics.bias.copy_(delta[0])
        action = torch.randn(1, 29, requires_grad=True)
        next_state, contact = wm.step(state, action, torch.zeros_like(action), torch.zeros(1, 14), torch.zeros(1, 441))
        torch.testing.assert_close(wm.target(state, next_state), delta, atol=2e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(contact).all())
        next_state.sum().backward()
        self.assertTrue(torch.isfinite(action.grad).all())

    def test_zero_rotation_derivative(self):
        vector = torch.zeros(2, 3, requires_grad=True)
        rotation_increment(vector)[:, :3].sum().backward()
        torch.testing.assert_close(vector.grad, torch.full_like(vector, 0.5))


if __name__ == "__main__":
    unittest.main()
