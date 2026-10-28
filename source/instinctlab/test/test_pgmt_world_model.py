"""CPU contracts for scheme-one geometry, dynamics, and checkpoint actor layout."""

import torch
import unittest

from isaaclab.sensors.ray_caster.patterns import GridPatternCfg, grid_pattern
from isaaclab.utils.math import quat_apply, quat_mul, yaw_quat

from instinctlab.tasks.shadowing.pgmt.heightfield import HeightField
from instinctlab.tasks.shadowing.pgmt.imagination import pack_history, raw_history
from instinctlab.tasks.shadowing.pgmt.world_model import TerrainWorldModel, rotation_increment


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


class DynamicsTest(unittest.TestCase):
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
