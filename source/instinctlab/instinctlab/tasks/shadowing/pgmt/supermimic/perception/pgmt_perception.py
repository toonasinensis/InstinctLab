"""PGMT geometry labels and v010 latent scaling, extending copied visual code."""

import torch
from torch import nn

from .depth_latent import DepthOnlyEncoder, mlp


class PGMTDepthEncoder(DepthOnlyEncoder):
    def forward(self, depth):
        return super().forward(None, depth)


class GeometricPerception(nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder, self.cfg = encoder, cfg
        self.decoder = mlp(cfg.latent_width, cfg.decoder_hidden, cfg.height_points)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_width))
        self.register_buffer("latent_std", torch.ones(cfg.latent_width))
        self.register_buffer("latent_delta_std", torch.ones(cfg.latent_width))

    @torch.no_grad()
    def update_latent_stats(self, z0, targets, mask):
        # v010 masked moment/rms-delta update; never mutate scales inside an unroll.
        previous = torch.cat((z0[:, None], targets[:, :-1]), 1)
        weight = mask.to(targets.dtype)[..., None]
        count = weight.sum().clamp_min(1)
        mean = (targets * weight).sum((0, 1)) / count
        variance = ((targets - mean).square() * weight).sum((0, 1)) / count
        delta_rms = ((targets - previous).square() * weight).sum((0, 1)) / count
        valid = mask.sum() > 0
        for running, value in (
            (self.latent_mean, mean),
            (self.latent_std, variance.sqrt().clamp_min(self.cfg.latent_scale_floor)),
            (self.latent_delta_std, delta_rms.sqrt().clamp_min(self.cfg.latent_scale_floor)),
        ):
            running.lerp_(torch.where(valid, value, running), self.cfg.latent_stats_rate)

    def latent_variance(self):
        return self.latent_std.square().mean().clamp_min(self.cfg.latent_variance_floor)

    def reconstruction_loss(self, image, height, valid):
        prediction = self.decoder(self.encoder(image))
        # Mask before subtraction: 0 * NaN still gives NaN.
        error = torch.where(valid, prediction - torch.where(valid, height, 0), 0)
        return error.square().sum() / valid.sum().clamp_min(1)


def height_labels(root_height, ray_hits, image_valid, cfg):
    ground = ray_hits[..., 2]
    valid = torch.isfinite(ground) & image_valid[:, None]
    target = (root_height[:, None] - ground - cfg.height_offset_m).clamp(-cfg.height_clip_m, cfg.height_clip_m)
    return torch.where(valid, target, 0), valid
