"""Depth encoders used by perceptive policies."""

from __future__ import annotations

import torch
from collections.abc import Sequence


class DepthConv2dEncoder(torch.nn.Module):
    """Spatial-size-independent depth encoder.

    The number of input channels (depth history frames) remains part of the
    interface, while adaptive pooling decouples the linear projection from the
    input image height and width.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        output_size: int,
        paddings: Sequence[int] | None = None,
        pooled_size: Sequence[int] = (4, 4),
        activation: str = "ELU",
    ):
        super().__init__()
        if paddings is None:
            paddings = [0] * len(channels)
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings)):
            raise ValueError("channels, kernel_sizes, strides, and paddings must have the same length")
        if len(channels) == 0:
            raise ValueError("DepthConv2dEncoder requires at least one convolution layer")
        if len(pooled_size) != 2:
            raise ValueError(f"pooled_size must contain (height, width), got {pooled_size}")

        activation_cls = getattr(torch.nn, activation)
        layers: list[torch.nn.Module] = []
        layer_in_channels = in_channels
        for layer_out_channels, kernel_size, stride, padding in zip(channels, kernel_sizes, strides, paddings):
            layers.extend(
                [
                    torch.nn.Conv2d(
                        layer_in_channels,
                        layer_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    activation_cls(),
                ]
            )
            layer_in_channels = layer_out_channels

        self.conv = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(tuple(pooled_size))
        self.projection = torch.nn.Linear(layer_in_channels * int(pooled_size[0]) * int(pooled_size[1]), output_size)
        self._output_size = output_size

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 4:
            raise ValueError(f"depth must have shape [batch, channels, height, width], got {tuple(depth.shape)}")
        feature = self.pool(self.conv(depth))
        return self.projection(feature.flatten(start_dim=1))

    @property
    def output_size(self) -> int:
        return self._output_size
