"""Explicit visual tensor/loss contract, independent of PGMT simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthCfg:
    image_shape: tuple[int, int, int] = (1, 48, 64)
    latent_width: int = 64
    height_points: int = 187
    decoder_hidden: tuple[int, ...] = (128, 256, 512)
    actor_hidden: tuple[int, ...] = (1024, 512, 256)
    history_width: int = 930
    reference_width: int = 748
    actor_latent_detach: bool = True
    reconstruction_batch: int = 512
    reconstruction_weight: float = 1.0
    latent_loss_weight: float = 1.0
    latent_stats_rate: float = 0.01
    latent_scale_floor: float = 1e-3
    latent_variance_floor: float = 1e-6
    height_offset_m: float = 0.5
    height_clip_m: float = 1.0
    depth_far_m: float = 2.5

    def __post_init__(self):
        if tuple(self.image_shape) != (1, 48, 64) or self.latent_width != 64:
            raise ValueError("The copied v010 encoder requires one 48x64 image and z64")
        if min(self.height_points, self.reconstruction_batch, self.history_width, self.reference_width) < 1:
            raise ValueError("Visual dimensions and batch size must be positive")
        if not 0 < self.latent_stats_rate <= 1 or self.latent_scale_floor <= 0:
            raise ValueError("Invalid latent statistics settings")

    @property
    def image_width(self):
        channels, height, width = self.image_shape
        return channels * height * width

    @property
    def physical_width(self):
        return self.history_width + self.reference_width + self.height_points
