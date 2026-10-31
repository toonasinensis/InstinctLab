"""Scheme A auxiliary WM; actor and grouped critic are still trained by PPO."""

from isaaclab.utils import configclass

from .instinct_rl_terrain_cfg import G1PGMTTerrainPPORunnerCfg


@configclass
class G1PGMTSupervisedWMRunnerCfg(G1PGMTTerrainPPORunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.wm_runner:SupervisedWMRunner"
    ppo_checkpoint: str | None = None
    world_model: dict = {
        "hidden": 512,
        "amp": True,
        "resolution": 0.05,
        "learning_rate": 3e-4,
        "updates": 8,
        "batch_size": 512,
        "train_horizon": 4,
        "validation_horizons": [1, 4, 8],
        "validation_batch": 256,
        "validation_interval": 25,
        "replay_chunks": 2,
    }
    experiment_name: str = "g1_pgmt_ppo_supervised_wm"
    run_name: str = "SchemeA_WMDiagnostics"
