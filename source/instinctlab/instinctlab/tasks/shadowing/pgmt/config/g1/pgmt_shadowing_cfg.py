"""G1 configuration for flat LAFAN PGMT stage one."""

import os

from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.kit import KitBackendCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from instinctlab.assets.pgmt_g1 import BODY_NAMES, G1_ACTION_SCALE, G1_CYLINDER_CFG, JOINT_NAMES
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.tasks.shadowing.pgmt import pgmt_env_cfg as pgmt_cfg
from instinctlab.tasks.shadowing.pgmt.mdp import LafanNamedMotionCfg
from instinctlab.tasks.utils.backends import backend_viewers, physics_preset
from instinctlab.tasks.utils.newton import InstinctNewtonVisualizerCfg, newton_physics_cfg
from instinctlab.utils.config import set_cfg_joint_order

MOTION_FOLDER = os.environ.get("PGMT_LAFAN_DATA", os.path.expanduser("~/Datasets/pgmt_lafan_g1"))
G1_CFG = G1_CYLINDER_CFG.replace(joint_ordering=JOINT_NAMES)

motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    robot_model_path=G1_CFG.spawn.asset_path,
    link_of_interests=list(BODY_NAMES),
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    symmetric_augmentation_joint_names=list(JOINT_NAMES),
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=11,
    data_start_from="current_time",
    motion_buffers={
        "LAFAN": LafanNamedMotionCfg(path=f"{MOTION_FOLDER}/named", manifest=f"{MOTION_FOLDER}/manifest.json")
    },
    mp_split_method="None",
)


@configclass
class G1PGMTShadowingEnvCfg(pgmt_cfg.PGMTEnvCfg):
    scene: pgmt_cfg.PGMTSceneCfg = pgmt_cfg.PGMTSceneCfg(
        num_envs=4096, robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"), motion_reference=motion_reference_cfg
    )

    def __post_init__(self):
        super().__post_init__()
        order = list(JOINT_NAMES)
        set_cfg_joint_order(self.actions.joint_pos, order)
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        for name in ("joint_pos", "joint_vel"):
            getattr(self.observations.policy, name).params["asset_cfg"] = set_cfg_joint_order(
                SceneEntityCfg("robot"), order
            )
        set_cfg_joint_order(self.commands.joint_pos_ref_command.asset_cfg, order)
        set_cfg_joint_order(self.commands.joint_vel_ref_command.asset_cfg, order)
        self.sim.physics = physics_preset(PhysxCfg(), newton=newton_physics_cfg(njmax=384, nconmax=192))
        self.sim.visualizer_cfgs = backend_viewers(
            VisualizerCfg(backend=KitBackendCfg()), newton=InstinctNewtonVisualizerCfg()
        )
        self.run_name = "G1PGMTLafanStage1"


@configclass
class G1PGMTShadowingEnvCfg_PLAY(G1PGMTShadowingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
