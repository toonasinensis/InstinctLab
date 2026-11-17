"""v010 geometric AE and delta-z supervision on the shared PGMT WM loop."""

import torch
from typing import cast

from .pgmt_wm_learner import PGMTWorldModelLearner, Prediction


class DepthPrediction(Prediction):
    latent_mse: torch.Tensor


class DepthWorldModelLearner(PGMTWorldModelLearner):
    @property
    def visual(self):
        return self.dynamics.visual

    def prepare_latents(self, data, *, update_stats=False):
        data = dict(data)
        images = data.pop("image_window")
        with torch.no_grad():
            latent = self.visual.encoder(images.flatten(0, 1)).reshape(*images.shape[:2], -1)
        data["latent_initial"], data["latent_target"] = latent[:, 0], latent[:, 1:]
        image_valid = data["image_valid"][:, :1] & data["next_image_valid"].cumprod(1).bool()
        data["latent_mask"] = data["mask"] * image_valid
        if update_stats:
            self.visual.update_latent_stats(data["latent_initial"], data["latent_target"], data["latent_mask"])
        return data

    def prediction_initial(self, data):
        return data["state"][:, 0], data["latent_initial"]

    def prediction_step(self, state, carry, data, step, heights):
        return self.dynamics.joint_transition(
            state,
            data["action"][:, step],
            data["previous_action"][:, step],
            heights,
            self.context[data["env_id"]],
            carry,
        )

    def predict(self, data, horizon) -> DepthPrediction:
        if "latent_initial" not in data:
            data = self.prepare_latents(data)
        result = cast(DepthPrediction, super().predict(data, horizon))
        assert result["auxiliary_states"] is not None
        weight = result["mask"] * data["latent_mask"][:, :horizon]
        error = (result["auxiliary_states"] - data["latent_target"][:, :horizon]).square().mean(-1)
        mse = (error * weight).sum() / weight.sum().clamp_min(1)
        result["latent_mse"] = mse
        result["loss"] = result["loss"] + self.visual.cfg.latent_loss_weight * mse / self.visual.latent_variance()
        return result

    def loss(self, data):
        data = self.prepare_latents(data, update_stats=True)
        result = self.predict(data, self.cfg.horizon)
        physical_latent, bce = self.prediction_loss(result, data)
        geometry = self.visual.reconstruction_loss(
            data["reconstruction_image"], data["reconstruction_height"], data["reconstruction_valid"]
        )
        loss = physical_latent + self.visual.cfg.reconstruction_weight * geometry
        return loss, torch.stack(
            (
                bce,
                geometry.detach(),
                result["latent_mse"].detach(),
                self.visual.latent_std.mean(),
                self.visual.latent_delta_std.mean(),
                self.visual.latent_mean.mean(),
            )
        )

    def _batch_loss(self, td):
        return self.loss_forward(td.to_dict())

    def _summarize(self, stats, norms):
        row = stats[-1]
        return {
            "loss": row[0],
            "contact_bce": row[1],
            "grad_norm": torch.stack(norms).amax(),
            "Perception/height_reconstruction_mse": row[2],
            "Perception/latent_prediction_mse": row[3],
            "Perception/latent_std": row[4],
            "Perception/latent_delta_std": row[5],
            "Perception/latent_mean": row[6],
        }

    def sample(self, replay, env_ids, generator, *, validation=False, horizon=None):
        data = super().sample(replay, env_ids, generator, validation=validation, horizon=horizon)
        length = data["action"].shape[1]
        data["image_window"] = replay.image_windows(data, length)
        if not validation:
            eligible = env_ids[env_ids % self.cfg.holdout_stride != 0]
            count = self.visual.cfg.reconstruction_batch
            ids = eligible[torch.randint(len(eligible), (count,), device=eligible.device, generator=generator)]
            ages = torch.randint(len(replay), (count,), device=eligible.device, generator=generator)
            rows = replay.rows(ages)
            data["reconstruction_image"] = replay.images_at(ages, ids)
            data["reconstruction_height"] = replay.data["height_target"][rows, ids]
            data["reconstruction_valid"] = replay.data["height_valid"][rows, ids]
        return data
