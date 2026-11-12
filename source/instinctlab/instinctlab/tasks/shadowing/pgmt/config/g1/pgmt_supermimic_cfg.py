"""Readable training recipes for the composed SuperMimic backend."""

import math
from dataclasses import asdict, replace

from isaaclab.utils import configclass

from instinctlab.tasks.shadowing.pgmt.supermimic.config import (
    AnalyticFootRewardCfg,
    ContactRewardCfg,
    ExecutionCfg,
    RepresentationCfg,
    SuperMimicCfg,
)

from .agents.instinct_rl_ppo_cfg import NormalizersCfg
from .pgmt_scan187_cfg import G1PGMTScan187EnvCfg


@configclass
class HeightScanActorCfg:
    class_name: str = "instinctlab.tasks.shadowing.pgmt.models.terrain_cnn:TerrainCNNActor"
    scan_shape: tuple = (17, 11)


@configclass
class FourGlimpseActorCfg:
    class_name: str = "instinctlab.tasks.shadowing.pgmt.models.terrain:TerrainActor"


def full_recipe():
    """Current F recipe; algorithm choices are explicit and independent of execution."""
    cfg = SuperMimicCfg()
    lr = 2e-4 * math.sqrt(2)
    return replace(
        cfg,
        actor_optimizer=replace(cfg.actor_optimizer, lr=lr),
        wm_optimizer=replace(cfg.wm_optimizer, lr=lr),
        critic_optimizer=replace(cfg.critic_optimizer, lr=lr),
    )


def baseline_recipe():
    cfg = full_recipe()
    return replace(
        cfg,
        representation=RepresentationCfg(kind="delta", loss_scale="running"),
        execution=ExecutionCfg(compile="none", cuda_graph=False),
        actor_optimizer=replace(cfg.actor_optimizer, kind="adam"),
        wm_optimizer=replace(cfg.wm_optimizer, weight_decay=1e-5),
        critic_optimizer=replace(cfg.critic_optimizer, kind="adam"),
    )


@configclass
class G1PGMTSuperMimicCfg:
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.supermimic.runners.pgmt_runner:PGMTSuperMimicRunner"
    seed: int = 42
    device: str = "cuda:0"
    policy: HeightScanActorCfg = HeightScanActorCfg()
    normalizers: NormalizersCfg = NormalizersCfg()
    supermimic: dict = asdict(full_recipe())
    foot_contact: dict = asdict(ContactRewardCfg())
    analytic_foot: dict = asdict(AnalyticFootRewardCfg())
    termination_softness: dict = dict(height_m=0.03, tilt_cos=0.04, body_m=0.04)
    flat_warmup_updates: int = 2000
    terrain_gate_ramp: int = 1
    initialize_checkpoint: str | None = None
    max_iterations: int = 1000000000
    save_interval: int = 1000
    experiment_name: str = "pgmt_supermimic"
    run_name: str = "Full"
    init_at_random_ep_len: bool = False
    resume: bool = False
    load_run: str | None = ".*"
    load_checkpoint: str = "model_.*.pt"


@configclass
class G1PGMTSuperMimicBaselineCfg(G1PGMTSuperMimicCfg):
    supermimic: dict = asdict(baseline_recipe())
    run_name: str = "Baseline"


@configclass
class G1PGMTSuperMimicTFEnvCfg(G1PGMTScan187EnvCfg):
    """The original 21x21 scan with the unchanged four-glimpse TF policy."""

    height_scan: dict = dict(size=(2.0, 2.0), spacing=0.1, offset=(0.0, 0.0))


@configclass
class G1PGMTSuperMimicTFCfg(G1PGMTSuperMimicBaselineCfg):
    policy: FourGlimpseActorCfg = FourGlimpseActorCfg()
    run_name: str = "FourGlimpseTF"
