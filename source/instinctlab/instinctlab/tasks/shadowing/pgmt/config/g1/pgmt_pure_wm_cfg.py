"""First pure-WM experiment: consistent noiseless observation reconstruction."""

from isaaclab.utils import configclass

from .pgmt_terrain_cfg import G1PGMTTerrainEnvCfg, G1PGMTTerrainEnvCfg_PLAY


@configclass
class G1PGMTPureWorldModelEnvCfg(G1PGMTTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Exploration is controlled by action noise in both collection/imagination.
        self.observations.policy.enable_corruption = False


@configclass
class G1PGMTPureWorldModelEnvCfg_PLAY(G1PGMTTerrainEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
