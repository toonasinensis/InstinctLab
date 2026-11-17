"""Latent recurrence reuses PGMT's physical observation, reward, and history semantics."""

import torch
from tensordict import TensorDict

from .pgmt_world_model_env import PGMTWorldModelEnv


class PGMTDepthWorldModelEnv(PGMTWorldModelEnv):
    def prepare(self, data, generator):
        inputs = super().prepare(data, generator)
        with torch.no_grad():
            image = self.replay_buffer.images_at(data["replay_age"], data["env_id"])
            inputs["latent"] = self.dynamics.visual.encoder(image)
        return inputs

    def _on_reset(self, inputs):
        super()._on_reset(inputs)
        self.latent = inputs["latent"]

    def get_observations(self):
        actor = TensorDict(dict(physical=self.observation, latent=self.latent), batch_size=[len(self.ids)])
        return TensorDict(dict(actor=actor, critic=self.feature_rows[-1]), batch_size=[len(self.ids)])

    def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
        step = self._rollout_step
        heights = self.observation[:, self.mdp.observation_layout["terrain"]]
        successor, logits, self.latent = self.dynamics.joint_transition(
            self.state,
            action,
            self.previous_action,
            heights,
            self.xi[self.ids],
            self.latent,
            recompute=self.execution.wm_recompute,
        )
        result = self.finish_transition(
            successor,
            logits,
            self.history,
            self.support,
            action,
            self.contexts[step + 1],
            self.ids,
            self.proprio_noise[step],
            self.observation_noise[step + 1],
        )
        return self.record_transition(result)
