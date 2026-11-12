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

register_backend_tasks(
    "Instinct-PGMT-TerrainPPOValidation-G1",
    env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainPPOValidationEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainPPOValidationEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_terrain_cfg:G1PGMTTerrainPPORunnerCfg",
)


for suffix in ("NoForce", "NoContactForce"):
    register_backend_tasks(
        f"Instinct-PGMT-TerrainPPO{suffix}-G1",
        env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainPPO{suffix}EnvCfg",
        play_env_cfg_entry_point=f"{task_entry}.pgmt_terrain_cfg:G1PGMTTerrainPPO{suffix}EnvCfg_PLAY",
        entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
        instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_terrain_cfg:G1PGMTTerrainPPORunnerCfg",
    )


for suffix, agent_class in (("", "G1PGMTSuperMimicCfg"), ("Baseline", "G1PGMTSuperMimicBaselineCfg")):
    register_backend_tasks(
        f"Instinct-PGMT-SuperMimic{suffix}-G1",
        env_cfg_entry_point=f"{task_entry}.pgmt_scan187_cfg:G1PGMTScan187EnvCfg",
        play_env_cfg_entry_point=f"{task_entry}.pgmt_scan187_cfg:G1PGMTScan187EnvCfg",
        entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
        instinct_rl_cfg_entry_point=f"{task_entry}.pgmt_supermimic_cfg:{agent_class}",
    )


register_backend_tasks(
    "Instinct-PGMT-SuperMimicTF-G1",
    env_cfg_entry_point=f"{task_entry}.pgmt_supermimic_cfg:G1PGMTSuperMimicTFEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.pgmt_supermimic_cfg:G1PGMTSuperMimicTFEnvCfg",
    entry_point="instinctlab.tasks.shadowing.pgmt.terrain_env:PGMTTerrainEnv",
    instinct_rl_cfg_entry_point=f"{task_entry}.pgmt_supermimic_cfg:G1PGMTSuperMimicTFCfg",
)
