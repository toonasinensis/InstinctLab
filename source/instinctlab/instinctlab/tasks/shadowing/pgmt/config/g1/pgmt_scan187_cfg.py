"""187-point forward scan and explicitly controlled PartyMimic-inspired ablations."""

import math

from isaaclab.utils import configclass

from .pgmt_cnn_warmup_cfg import CNNPolicyCfg, FlatWarmupMotionCfg, G1PGMTCNNWarmupEnvCfg, G1PGMTCNNWarmupRunnerCfg


@configclass
class Scan187MotionCfg(FlatWarmupMotionCfg):
    class_type: str = "instinctlab.tasks.shadowing.pgmt.scan187_runner:Scan187Motion"
    terrain_sampling_weights: dict = {}


@configclass
class G1PGMTScan187EnvCfg(G1PGMTCNNWarmupEnvCfg):
    height_scan: dict = dict(size=(1.6, 1.0), spacing=0.1, offset=(0.4, 0.0))
    heightfield_resolution: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = Scan187MotionCfg(**self.commands.motion.to_dict())
        self.commands.motion.class_type = "instinctlab.tasks.shadowing.pgmt.scan187_runner:Scan187Motion"
        motion = self.commands.motion
        motion.randomize_platform_spawn = motion.spawn_root_only = True
        motion.spawn_center_region_width = 1.0
        motion.off_platform_half_width = 0.5
        motion.spawn_platform_margin = 0.2
        generator = self.scene.terrain.terrain_generator
        # 29 rings * 2 sides * 0.30 m + 1 m plateau + 2 * 1 m border.
        generator.size = (20.4, 20.4)
        proportions = dict(flat=0.2, slope_up=0.1, slope_down=0.1, stairs_up=0.2, stairs_down=0.2, boxes=0.0, rough=0.2)
        for name, cfg in generator.sub_terrains.items():
            cfg.proportion = proportions[name]
            if hasattr(cfg, "platform_width"):
                cfg.platform_width = 1.0
            if hasattr(cfg, "step_width"):
                # Native stair platform_width is an upper quantization bound.
                # 1.3 selects 29 rings robustly; the resulting plateau is 1.0 m.
                cfg.platform_width = 1.3


@configclass
class Scan187PolicyCfg(CNNPolicyCfg):
    scan_shape: tuple = (17, 11)


@configclass
class G1PGMTScan187RunnerCfg(G1PGMTCNNWarmupRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.scan187_runner:Scan187Runner"
    policy: Scan187PolicyCfg = Scan187PolicyCfg()
    max_iterations: int = 50000
    experiment_name: str = "pgmt_scan187_party"
    world_model: dict = {
        **G1PGMTCNNWarmupRunnerCfg().world_model,
        "predict_contact": True,
        "predict_force": False,
        "structured": False,
        "prediction_mode": "delta",
        "loss_scale": "running",
        "learning_rate": 2e-4 * math.sqrt(2),
    }
    bptt: dict = {
        **G1PGMTCNNWarmupRunnerCfg().bptt,
        "horizon": 32,
        "actor_lr": 2e-4 * math.sqrt(2),
        "critic_lr": 2e-4 * math.sqrt(2),
    }
    foot_contact: dict = {**G1PGMTCNNWarmupRunnerCfg().foot_contact, "enabled": True}
    analytic_foot: dict = {**G1PGMTCNNWarmupRunnerCfg().analytic_foot, "enabled": True}
    learner_backend: dict = dict(optimizers="baseline", capture=False)
    experiment: dict = dict(actor_optimizer="adam", wm_optimizer="adamw", ema_decay=0.995, value_gradient_weight=0.0)
