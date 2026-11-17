"""Regression tests for masking, optimizer ownership and the v010 gradient boundary."""

import torch
import unittest
from dataclasses import replace

from instinctlab.tasks.shadowing.pgmt.supermimic.perception.config import DepthCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_models import PGMTDepthActor
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_perception import (
    GeometricPerception,
    PGMTDepthEncoder,
    height_labels,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_replay import DepthReplayBuffer


class DepthTests(unittest.TestCase):
    def test_actor_latent_detach_only_changes_actor_gradient_path(self):
        actor = PGMTDepthActor(29)
        history = torch.randn(2, actor.cfg.history_width + actor.cfg.reference_width)
        for detach in (True, False):
            actor.cfg = replace(actor.cfg, actor_latent_detach=detach)
            latent = torch.randn(2, actor.cfg.latent_width, requires_grad=True)
            action = actor.action_from_inputs(history, latent)
            self.assertEqual(action.shape, (2, 29))
            action.square().sum().backward()
            self.assertEqual(latent.grad is None, detach)
            if not detach:
                self.assertGreater(float(latent.grad.norm()), 0)

    def test_mask_ignores_nan_labels_and_excludes_empty_camera(self):
        cfg = DepthCfg()
        visual = GeometricPerception(PGMTDepthEncoder(), cfg)
        height = torch.full((2, cfg.height_points), float("nan"))
        valid = torch.zeros_like(height, dtype=torch.bool)
        image = torch.randn(2, *cfg.image_shape)
        loss = visual.reconstruction_loss(image, height, valid)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in visual.parameters()))
        hits = torch.zeros(2, cfg.height_points, 3)
        hits[0, 0, 2] = float("nan")
        target, mask = height_labels(torch.ones(2), hits, torch.tensor([True, False]), cfg)
        self.assertFalse(bool(mask[0, 0]))
        self.assertFalse(bool(mask[1].any()))
        self.assertTrue(bool(torch.isfinite(target).all()))

    def test_all_invalid_latent_stats_remain_unchanged(self):
        cfg = DepthCfg()
        visual = GeometricPerception(PGMTDepthEncoder(), cfg)
        before = {k: v.clone() for k, v in visual.named_buffers()}
        visual.update_latent_stats(torch.randn(2, 64), torch.randn(2, 3, 64), torch.zeros(2, 3))
        for key, value in visual.named_buffers():
            torch.testing.assert_close(value, before[key])

    def test_host_images_preserve_terminal_successors_after_ring_wrap(self):
        replay = DepthReplayBuffer(3, 2)
        for step in range(5):
            replay.insert(
                dict(
                    state=torch.full((2, 1), float(step)),
                    action=torch.zeros(2, 1),
                    done=torch.tensor([step == 3, False]),
                    depth_image=torch.full((2, 1, 2, 2), float(step)),
                    next_depth_image=torch.full((2, 1, 2, 2), step + 0.5),
                )
            )
        ids, ages = torch.tensor([0, 1]), torch.tensor([0, 0])
        data = replay.sequences(ids, 3, None, ages=ages)
        window = replay.image_windows(data, 3)
        torch.testing.assert_close(window[:, :, 0, 0, 0], torch.tensor([[2.0, 2.5, 3.5, 4.5], [2.0, 2.5, 3.5, 4.5]]))
        torch.testing.assert_close(data["mask"], torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]))
        replay.clear()
        self.assertEqual(len(replay), 0)


if __name__ == "__main__":
    unittest.main()
