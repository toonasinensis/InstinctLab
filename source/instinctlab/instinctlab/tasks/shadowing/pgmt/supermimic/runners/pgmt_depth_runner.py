"""Depth specialization of the existing runner; collection/training loop is inherited."""

import torch

from instinctlab.tasks.shadowing.pgmt.supermimic.envs.pgmt_depth_env import PGMTDepthWorldModelEnv
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_depth_policy_learner import DepthPolicyLearner
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_depth_wm_learner import DepthWorldModelLearner
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_learner_utils import compile_function
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.config import DepthCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_models import (
    DepthPolicyNormalizer,
    PGMTDepthDynamics,
    PGMTDepthPolicy,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_perception import height_labels
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_replay import DepthReplayBuffer
from instinctlab.tasks.shadowing.pgmt.supermimic.pgmt_sim_context import ordinary

from .pgmt_runner import PGMTSuperMimicRunner


def sensor_tensor(value):
    return value.torch if hasattr(value, "torch") else value


class PGMTDepthRunner(PGMTSuperMimicRunner):
    policy_type = PGMTDepthPolicy
    imagination_type = PGMTDepthWorldModelEnv
    policy_learner_type = DepthPolicyLearner

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.perception = DepthCfg(**train_cfg["policy"]["perception"])
        if train_cfg["supermimic"]["execution"]["compile"] == "chain":
            raise ValueError("Depth supports selective compilation; full-chain mode is not validated")
        super().__init__(env, train_cfg, log_dir, device)

    def _policy_normalizer(self, shape, scan_width, **values):
        return DepthPolicyNormalizer(shape, scan_width, self.perception, **values)

    def _create_dynamics(self, *args):
        return PGMTDepthDynamics(*args, encoder=self.actor.encoder, perception=self.perception)

    def _build_replay_buffer(self):
        return DepthReplayBuffer(self.options.collection.replay_steps, self.options.collection.history_length)

    def _build_wm_learner(self):
        return DepthWorldModelLearner(
            self._dynamics, self.context, self.options.world_model, self.options.wm_optimizer, self.options.execution
        )

    def _compile_dynamics(self, dynamics):
        dynamics.predict_joint_delta = compile_function(dynamics.predict_joint_delta, True)

    def checkpoint_extra_contract(self):
        native = self.env.unwrapped.cfg
        return {
            "depth_sensor": native.depth_sensor,
            "scan": native.height_scan,
            "camera_mount": {
                "prim_path": native.scene.camera.prim_path,
                "position": native.scene.camera.offset.pos,
                "rotation": native.scene.camera.offset.rot,
                "convention": native.scene.camera.offset.convention,
            },
        }

    def camera(self):
        return self.env.unwrapped.scene["camera"].data

    def camera_valid(self):
        raw = sensor_tensor(self.camera().output["distance_to_image_plane"])
        return (torch.isfinite(raw) & (raw > 0) & (raw < self.perception.depth_far_m)).flatten(1).any(-1)

    def _collection_fields(self):
        valid = self.camera_valid()
        hits = sensor_tensor(self.env.unwrapped.scene["reconstruction_scanner"].data.ray_hits_w)
        target, mask = height_labels(
            self.rewards.state()[:, self._representation.layout.state["position"]][:, 2], hits, valid, self.perception
        )
        _, image = self.actor.inputs(self.observation)
        return {
            key: ordinary(value)
            for key, value in dict(
                depth_image=image, image_valid=valid, height_target=target, height_valid=mask
            ).items()
        }

    def _capture_transition(self, dt):
        super()._capture_transition(dt)
        image = sensor_tensor(self.camera().output["distance_to_image_plane_noised"]).permute(0, 3, 1, 2)
        self.terminal.update(next_depth_image=ordinary(image), next_image_valid=ordinary(self.camera_valid()))
