import gymnasium as gym

from instinctlab.tasks.utils.backends import backend_entry_point

task_cfg = f"{__name__}.pgmt_heightscan_cfg"

for suffix in ("", "-Play"):
    gym.register(
        id=f"Instinct-PGMT-SuperMimic-T3_27-Newton{suffix}-v0",
        entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": backend_entry_point(f"{task_cfg}:T3_27PGMTHeightScanEnvCfg", "newton"),
            "instinct_rl_cfg_entry_point": f"{task_cfg}:T3_27PGMTSuperMimicCfg",
        },
    )
