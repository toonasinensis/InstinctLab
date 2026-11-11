"""D1S2 CNN32 model-gradient training on flat tiles, then original terrain curriculum."""

from isaaclab.utils import configclass

from ...mdp.terrain_commands import TerrainMotionCfg
from ...mdp.terrain_observations import bilinear_elevation
from .agents.instinct_rl_contact_cfg import G1PGMTContactSMRunnerCfg
from .agents.instinct_rl_terrain_cfg import TerrainPolicyCfg
from .pgmt_terrain_cfg import G1PGMTTerrainBPTTEnvCfg


@configclass
class FlatWarmupMotionCfg(TerrainMotionCfg):
    class_type: str = "instinctlab.tasks.shadowing.pgmt.cnn_curriculum:FlatWarmupTerrainMotion"
    flat_warmup: bool = True
    post_warmup_terrain_mode: str = "curriculum"


@configclass
class G1PGMTCNNWarmupEnvCfg(G1PGMTTerrainBPTTEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = FlatWarmupMotionCfg(**self.commands.motion.to_dict())
        self.commands.motion.class_type = "instinctlab.tasks.shadowing.pgmt.cnn_curriculum:FlatWarmupTerrainMotion"
        self.scene.motion_reference.motion_buffers["LAFAN"].motion_names = ["dance1_subject2.named.npz"]
        self.observations.policy.terrain.func = bilinear_elevation
        self.scene.height_scanner = None
        self.run_name = "G1PGMTCNNFlatWarmupD1S2"


@configclass
class CNNPolicyCfg(TerrainPolicyCfg):
    class_name: str = "instinctlab.tasks.shadowing.pgmt.models.terrain_cnn:TerrainCNNActorCritic"


@configclass
class G1PGMTCNNWarmupRunnerCfg(G1PGMTContactSMRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.cnn_curriculum:CNNFlatWarmupRunner"
    policy: CNNPolicyCfg = CNNPolicyCfg()
    resume_terrain_change: bool = False
    random_initialization: bool = True
    flat_warmup_updates: int = 2000
    terrain_gate_ramp: int = 1
    warm_start: str | None = None
    ppo_checkpoint: str | None = None
    wm_checkpoint: str | None = None
    max_iterations: int = 5000
    save_interval: int = 1000
    world_model: dict = {**G1PGMTContactSMRunnerCfg().world_model, "predict_contact": False, "predict_force": False}
    analytic_foot: dict = {
        "enabled": False,
        "grid_x": 5,
        "grid_y": 3,
        "drop": 0.1,
        "drop_temperature": 0.01,
        "terrain_margin": 0.009,
        "acceleration_threshold": 30.0,
        "tau": 0.06,
        "illegal_weight": -1.0,
        "acceleration_weight": -0.01,
    }
    foot_contact: dict = {**G1PGMTContactSMRunnerCfg().foot_contact, "enabled": False}
    bptt: dict = {
        **G1PGMTContactSMRunnerCfg().bptt,
        "actor_batch": 4096,
        "train_collect_steps": None,
        "utd": 0.4,
        "actor_lr": 2e-4,
        "critic_lr": 2e-4,
        "lr_schedule": {"type": "cosine", "total_iterations": 250000, "min_factor": 0.03},
    }
    experiment_name: str = "pgmt_cnn_horizon_flatwarm"
