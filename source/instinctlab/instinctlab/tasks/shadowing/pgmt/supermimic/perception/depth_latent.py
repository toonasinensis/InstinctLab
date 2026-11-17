"""Generic v010 visual representation copied from Roban; see PROVENANCE.md."""

import torch
from torch import nn

from .depth_encoder import DepthConv2dEncoder


def mlp(inp, hidden, out, layer_norm=False):
    layers = []
    for h in hidden:
        layers.append(nn.Linear(inp, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.SiLU() if layer_norm else nn.ELU())
        inp = h
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


class DepthOnlyEncoder(nn.Module):
    """Visual latent has no proprio input; signature matches deployment interface."""

    def __init__(self):
        super().__init__()
        self.depth = DepthConv2dEncoder(1, (16, 32), (5, 3), (2, 2), 128, paddings=(2, 1), pooled_size=(6, 8))
        self.projection = mlp(128, (256,), 64)

    def forward(self, proprio, depth):
        return self.projection(self.depth(depth))
