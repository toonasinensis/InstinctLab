"""Differentiable terrain query. Real raycasts remain native simulator observations."""

from torch import nn

# Canonical PGMT family encoding, including families absent from a particular atlas.
TERRAIN_FAMILIES = ("flat", "slope", "stairs", "boxes", "rough")


class BilinearTerrain(nn.Module):
    def __init__(self, heightfield):
        super().__init__()
        self.heightfield = heightfield

    def query(self, xy):
        return self.heightfield.query(xy)

    def local(self, position, yaw):
        return self.heightfield.local(position, yaw)
