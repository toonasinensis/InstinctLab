"""Original stage1 actor with standalone SuperMimic learning on PGMT terrain."""

from isaaclab.utils import configclass

from .instinct_rl_terrain_cfg import G1PGMTTerrainPPORunnerCfg


@configclass
class G1PGMTPureWorldModelRunnerCfg(G1PGMTTerrainPPORunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.pure_wm_runner:PureWorldModelRunner"
    pure_wm: dict = dict(
        hidden=2048,
        amp=True,
        resolution=0.05,
        replay_steps=1000,
        fill_steps=64,
        utd=0.4,
        wm_window=8,
        wm_updates=10,
        wm_batch=2048,
        wm_microbatch=256,
        actor_batch=4096,
        actor_microbatch=64,
        # H8 is the first prototype; H32 diagnostics are recorded from the start.
        # Increasing the training horizon requires a documented follow-up run.
        horizon=8,
        validation_horizon=32,
        validation_batch=128,
        validation_interval=25,
        physical_gate=dict(root_m=0.08, rotation_rad=0.3, joint_rad=0.2),
        warmup_iterations=500,
        terrain_gate_ramp=200,
        learning_rate=2e-4,
        action_noise=0.1,
        gamma=0.99,
        **{"lambda": 0.95},
        critic_epochs=5,
        critic_minibatches=4,
    )
    experiment_name = "g1_pgmt_pure_wm"
    run_name = "Stage1PureWM"
