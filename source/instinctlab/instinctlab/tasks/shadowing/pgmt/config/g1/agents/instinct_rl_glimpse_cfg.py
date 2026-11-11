"""Four-glimpse SM horizon and compiler experiments on native mesh terrain."""

from isaaclab.utils import configclass

from .instinct_rl_contact_cfg import G1PGMTContactSMRunnerCfg


@configclass
class G1PGMTGlimpseSMRunnerCfg(G1PGMTContactSMRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.glimpse_runner:GlimpseContactBPTTRunner"
    max_iterations: int = 1000000000
    save_interval: int = 1000
    bptt: dict = {**G1PGMTContactSMRunnerCfg().bptt, "horizon": 16, "train_collect_steps": 8}
    bptt_execution: dict = {**G1PGMTContactSMRunnerCfg().bptt_execution, "compile_mode": "none"}
    checkpoint_retention: dict = {"every": 1000, "latest": 3}
    experiment_name: str = "pgmt_glimpse_horizon"
