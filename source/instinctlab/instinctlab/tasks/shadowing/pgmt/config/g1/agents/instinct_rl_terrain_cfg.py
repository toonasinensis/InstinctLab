"""Terrain adaptation PPO initialized from the native flat checkpoint."""

from isaaclab.utils import configclass

from .instinct_rl_ppo_cfg import G1PGMTShadowingPPORunnerCfg, NormalizersCfg, PolicyCfg


@configclass
class TerrainPolicyCfg(PolicyCfg):
    class_name = "instinctlab.tasks.shadowing.pgmt.models.terrain:TerrainActorCritic"


@configclass
class TerrainNormalizersCfg(NormalizersCfg):
    def __post_init__(self):
        self.policy.class_name = "instinctlab.tasks.shadowing.pgmt.terrain_runner:TerrainNormalization"
        self.critic.class_name = "instinctlab.tasks.shadowing.pgmt.terrain_runner:StableNormalization"


@configclass
class G1PGMTTerrainPPORunnerCfg(G1PGMTShadowingPPORunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.terrain_runner:TerrainRunner"
    policy: TerrainPolicyCfg = TerrainPolicyCfg()
    normalizers: TerrainNormalizersCfg = TerrainNormalizersCfg()
    warm_start: str | None = None
    terrain_gate_ramp: int = 200
    experiment_name = "g1_pgmt_lafan_terrain"
    run_name = "CrossAttentionTerrain"
