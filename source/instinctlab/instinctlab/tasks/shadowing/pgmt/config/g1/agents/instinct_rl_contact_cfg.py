"""Paired single-GPU SM runs: stage-one actor, fresh dynamics and value learners."""

from isaaclab.utils import configclass

from .instinct_rl_bptt_cfg import G1PGMTBPTTRunnerCfg


@configclass
class G1PGMTContactSMRunnerCfg(G1PGMTBPTTRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.contact_runner:ContactBPTTRunner"
    world_model: dict = {
        **G1PGMTBPTTRunnerCfg().world_model,
        "learning_rate": 2e-4,
        "predict_contact": True,
        "feet_only": True,
    }
    bptt: dict = {
        **G1PGMTBPTTRunnerCfg().bptt,
        "actor_batch": 2048,
        "actor_lr": 2e-4,
        "critic_lr": 2e-4,
    }
    foot_contact: dict = {
        "enabled": True,
        "match": 1.5,
        "support": 0.25,
        "penetration": 0.5,
        "slip": 1.0,
        "height_scale": 0.05,
        "penetration_scale": 0.02,
        "penetration_tolerance": 0.005,
        "warmup_updates": 500,
        "ramp_updates": 500,
    }
    experiment_name: str = "g1_pgmt_sm_contact"
