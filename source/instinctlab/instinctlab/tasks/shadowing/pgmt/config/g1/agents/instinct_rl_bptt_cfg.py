"""H32 model-gradient actor with real or imagined critic supervision."""

from isaaclab.utils import configclass

from .instinct_rl_wm_cfg import G1PGMTSupervisedWMRunnerCfg


@configclass
class G1PGMTBPTTRunnerCfg(G1PGMTSupervisedWMRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.bptt_runner:BPTTRunner"
    wm_checkpoint: str | None = None
    max_iterations: int = 5000
    world_model: dict = {
        **G1PGMTSupervisedWMRunnerCfg().world_model,
        "hidden": 2048,
        "batch_size": 2048,
        "updates": 10,
        "replay_steps": 1000,
        "amp": False,
        "predict_contact": False,
        "predict_force": False,
    }
    bptt: dict = {
        "critic_source": "imagined",
        "horizon": 32,
        "actor_batch": 4096,
        "actor_lr": 1e-5,
        "critic_lr": 3e-4,
        "lr_scaling": "none",
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "gamma": 0.99,
        "lambda": 0.95,
        "action_noise": 0.1,
        "warmup_iterations": 32,
        "initial_collect_steps": 32,
        "utd": 0.4,
        "checkpoint_activations": True,
    }
    bptt_execution: dict = {
        "bf16": False,
        "bf16_heads": False,
        "compile": False,
        "actor_checkpoint": True,
        "wm_checkpoint": True,
    }
    experiment_name: str = "g1_pgmt_bptt"
    run_name: str = "NoContactForceBPTT"
