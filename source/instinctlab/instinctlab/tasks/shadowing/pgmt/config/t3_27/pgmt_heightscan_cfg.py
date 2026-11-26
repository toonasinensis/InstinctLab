import os
from dataclasses import asdict

from isaaclab.utils import configclass

from instinctlab.assets.t3_27 import T3_27_ACTION_SCALE, T3_27_CFG, T3_27_JOINT_NAMES, T3_27_TRACKED_BODIES
from instinctlab.tasks.shadowing.pgmt.mdp.motion import PGMTGroupedH5MotionCfg
from instinctlab.tasks.shadowing.pgmt.mdp.terrain_observations import foothold_joint_reference
from instinctlab.tasks.shadowing.pgmt.pgmt_env_cfg import PGMTSceneCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.config import ContactRewardCfg

from ..g1.pgmt_scan187_cfg import G1PGMTScan187EnvCfg
from ..g1.pgmt_shadowing_cfg import configure_tracking_joints, motion_reference_cfg
from ..g1.pgmt_supermimic_cfg import G1PGMTSuperMimicCfg, HeightScanActorCfg

T3_27_MOTION_H5 = os.environ.get(
    "PGMT_T3_27_MOTION_H5",
    "~/project_2026/supermimic/data/motions/T3_27/soma_0.6/moving_upright_middlespeed.h5",
)


@configclass
class T3_27PGMTHeightScanEnvCfg(G1PGMTScan187EnvCfg):
    scene: PGMTSceneCfg = PGMTSceneCfg(
        num_envs=4096,
        robot=T3_27_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg.replace(
            robot_model_path=T3_27_CFG.spawn.asset_path,
            link_of_interests=list(T3_27_TRACKED_BODIES),
            symmetric_augmentation_joint_names=list(T3_27_JOINT_NAMES),
            motion_buffers={"SOMA": PGMTGroupedH5MotionCfg(path=T3_27_MOTION_H5)},
        ),
    )
    mdp_schema: str = "pgmt-t3-27-heightscan-v1"

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_buffer_name = "SOMA"
        self.commands.motion.hand_body_names = ("left_wrist_roll_link", "right_wrist_roll_link")
        self.commands.motion.impact_body_names = ("pelvis", "waist_yaw_link")
        self.commands.motion.curriculum_motion_end_min_s = 5.0
        self.observations.policy.joint_pos_ref.func = foothold_joint_reference
        self.observations.policy.joint_vel_ref.func = foothold_joint_reference
        self.observations.policy.joint_vel_ref.params["derivative"] = True
        self.run_name = "T3_27HeightScan"

    def _configure_tracking_joints(self):
        configure_tracking_joints(self, T3_27_JOINT_NAMES, T3_27_ACTION_SCALE)


@configclass
class T3_27HeightScanActorCfg(HeightScanActorCfg):
    non_terrain_width: int = 1574
    zero_height_scan: bool = False


@configclass
class T3_27PGMTSuperMimicCfg(G1PGMTSuperMimicCfg):
    policy: T3_27HeightScanActorCfg = T3_27HeightScanActorCfg()
    actor_height_scan_hz: float | None = None
    foot_contact: dict = dict(asdict(ContactRewardCfg()), max_foot_vertices=32)
    experiment_name: str = "t3_27_pgmt_supermimic"
    run_name: str = "T3_27HeightScan"
