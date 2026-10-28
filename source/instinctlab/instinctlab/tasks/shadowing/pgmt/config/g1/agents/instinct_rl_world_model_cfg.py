"""Scheme one: preserve native PGMT PPO, add differentiable terrain imagination."""

from isaaclab.utils import configclass

from .instinct_rl_terrain_cfg import G1PGMTTerrainPPORunnerCfg


@configclass
class G1PGMTWorldModelRunnerCfg(G1PGMTTerrainPPORunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.world_model_runner:WorldModelRunner"
    stage2_initialization: str | None = None
    allow_surrogate_change: bool = False
    world_model: dict = dict(
        hidden=512,
        resolution=0.05,
        warmup=50,
        updates=16,
        batch_size=256,
        horizon=4,
        actor_lr=1e-5,
        anchor=1.0,
        teacher_ema=0.0,
        max_imagination_kl=0.0,
        replay_chunks=8,
        actor_updates=1,
        validation_threshold=1.0,
    )
    experiment_name = "g1_pgmt_world_model"
    run_name = "HeightmapImagination"
