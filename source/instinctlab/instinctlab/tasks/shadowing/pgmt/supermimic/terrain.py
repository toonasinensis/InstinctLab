"""Differentiable terrain query. Real raycasts remain native simulator observations."""

from .interfaces import TerrainQuery

# Canonical PGMT family encoding, including families absent from a particular atlas.
TERRAIN_FAMILIES = ("flat", "slope", "stairs", "boxes", "rough")


class BilinearTerrain(TerrainQuery):
    def __init__(self, heightfield):
        super().__init__()
        self.heightfield = heightfield

    def query(self, xy):
        return self.heightfield.query(xy)

    def local(self, position, yaw):
        return self.heightfield.local(position, yaw)
