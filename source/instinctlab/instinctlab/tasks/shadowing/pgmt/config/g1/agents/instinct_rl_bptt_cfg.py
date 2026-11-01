"""Six-way BPTT experiment: value supervision source x policy horizon."""

from isaaclab.utils import configclass

from .instinct_rl_wm_cfg import G1PGMTSupervisedWMRunnerCfg


@configclass
class G1PGMTBPTTRunnerCfg(G1PGMTSupervisedWMRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.bptt_runner:BPTTRunner"
    wm_checkpoint: str | None = None
    max_iterations: int = 5000
    world_model: dict = {
        **G1PGMTSupervisedWMRunnerCfg().world_model,
        "amp": False,
        "predict_contact": False,
        "predict_force": False,
    }
    bptt: dict = {
        "critic_source": "imagined",
        "horizon": 8,
        "actor_batch": 256,
        "actor_lr": 1e-5,
        "critic_lr": 3e-4,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "gamma": 0.99,
        "lambda": 0.95,
        "action_noise": 0.1,
        "warmup_iterations": 32,
    }
    experiment_name: str = "g1_pgmt_bptt_critic_horizon"
    run_name: str = "NoContactForceBPTT"
