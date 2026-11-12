"""Flat pretraining terrain and original motion curriculum configuration."""

from isaaclab.utils import configclass

from ...mdp.terrain_commands import TerrainMotionCfg
from ...mdp.terrain_observations import bilinear_elevation
from .pgmt_terrain_cfg import G1PGMTTerrainSMEnvCfg


@configclass
class FlatWarmupMotionCfg(TerrainMotionCfg):
    class_type: str = "instinctlab.tasks.shadowing.pgmt.mdp.curriculum_commands:FlatWarmupTerrainMotion"
    flat_warmup: bool = True
    post_warmup_terrain_mode: str = "curriculum"


@configclass
class G1PGMTFlatWarmupEnvCfg(G1PGMTTerrainSMEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = FlatWarmupMotionCfg(**self.commands.motion.to_dict())
        self.commands.motion.class_type = (
            "instinctlab.tasks.shadowing.pgmt.mdp.curriculum_commands:FlatWarmupTerrainMotion"
        )
        self.scene.motion_reference.motion_buffers["LAFAN"].motion_names = ["dance1_subject2.named.npz"]
        self.observations.policy.terrain.func = bilinear_elevation
        self.scene.height_scanner = None
        self.run_name = "G1PGMTCNNFlatWarmupD1S2"
