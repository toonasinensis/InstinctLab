"""Unitree dance2 and common bilinear terrain queries for four-glimpse SM."""

from isaaclab.utils import configclass

from instinctlab.tasks.shadowing.pgmt.mdp.terrain_observations import bilinear_elevation

from .pgmt_terrain_cfg import G1PGMTTerrainBPTTEnvCfg


@configclass
class G1PGMTGlimpseSMEnvCfg(G1PGMTTerrainBPTTEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference.motion_buffers["LAFAN"].motion_names = ["dance2_subject1.namedunitree.npz"]
        self.observations.policy.terrain.func = bilinear_elevation
        self.scene.height_scanner = None
        self.run_name = "G1PGMTUnitreeDance2GlimpseSM"
