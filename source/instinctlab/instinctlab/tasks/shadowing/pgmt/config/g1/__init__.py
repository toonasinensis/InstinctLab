from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.pgmt.config.g1"

register_backend_tasks(
    "Instinct-PGMT-Shadowing-G1",
    env_cfg_entry_point=f"{task_entry}.pgmt_shadowing_cfg:G1PGMTShadowingEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.pgmt_shadowing_cfg:G1PGMTShadowingEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1PGMTShadowingPPORunnerCfg",
)

register_backend_tasks(
    "Instinct-PGMT-Terrain-G1",
    env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_terrain_cfg:G1PGMTTerrainPPORunnerCfg",
)

register_backend_tasks(
    "Instinct-PGMT-TerrainFlat-G1",
    env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainFlatEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainFlatEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_terrain_cfg:G1PGMTTerrainPPORunnerCfg",
)
