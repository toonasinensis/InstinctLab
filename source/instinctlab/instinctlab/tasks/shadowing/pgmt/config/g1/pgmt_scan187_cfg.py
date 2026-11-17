"""Heightscan policy specialization of the shared PGMT terrain recipe."""

from isaaclab.utils import configclass

from .pgmt_terrain_recipe_cfg import G1PGMTTerrainRecipeEnvCfg
from .pgmt_terrain_recipe_cfg import TerrainRecipeMotionCfg as Scan187MotionCfg  # noqa: F401


@configclass
class G1PGMTScan187EnvCfg(G1PGMTTerrainRecipeEnvCfg):
    """Keep the registered heightscan task name and its existing observation contract."""
