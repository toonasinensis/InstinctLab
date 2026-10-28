"""Unpaired LAFAN terrain adaptation; inherits the validated stage-one robot and observations."""

import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from instinctlab.managers import MultiRewardCfg
from instinctlab.tasks.shadowing.pgmt.mdp import terrain_observations as obs
from instinctlab.tasks.shadowing.pgmt.mdp import terrain_rewards as rew
from instinctlab.tasks.shadowing.pgmt.mdp import terrain_terminations as done
from instinctlab.tasks.shadowing.pgmt.mdp.terrain_commands import TerrainMotionCfg, reset_on_terrain
from instinctlab.tasks.shadowing.pgmt.terrain_cfg import make_pgmt_terrain_cfg

from .pgmt_shadowing_cfg import MOTION_FOLDER, G1PGMTShadowingEnvCfg


@configclass
class RewardTerms:
    pass


@configclass
class TerrainRewardsCfg(MultiRewardCfg):
    upper = RewardTerms()
    lower = RewardTerms()
    auxiliary = RewardTerms()
    terrain = RewardTerms()


@configclass
class TerrainTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    motion_end = TerminationTermCfg(func=done.motion_end, time_out=True)
    tile_exit = TerminationTermCfg(func=done.tile_exit, time_out=True)
    root_height = TerminationTermCfg(func=done.height_failure, params={"threshold": 0.4})
    root_tilt = TerminationTermCfg(func=done.tilt_failure, params={"threshold": 1.0})
    body_error = TerminationTermCfg(func=done.body_failure, params={"threshold": 0.65})


@configclass
class G1PGMTTerrainEnvCfg(G1PGMTShadowingEnvCfg):
    flat_precheck: bool = False
    mdp_schema: str = "pgmt-lafan-terrain-v1"

    def __post_init__(self):
        super().__post_init__()
        if not self.flat_precheck:
            self.scene.terrain.terrain_type = "generator"
            self.scene.terrain.terrain_generator = make_pgmt_terrain_cfg()
            self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 20)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0], ordering="yx"),
            mesh_prim_paths=["/World/ground"],
            update_period=0.02,
            debug_vis=False,
        )
        self.commands.motion = TerrainMotionCfg(
            contact_labels_dir=os.environ.get("PGMT_CONTACT_LABELS", f"{MOTION_FOLDER}/contacts"),
            enable_terrain_curriculum=not self.flat_precheck,
            enable_tolerance=not self.flat_precheck,
        )
        self.observations.policy.motion_orientation.func = obs.orientation
        self.observations.policy.motion_height.func = obs.clearance
        self.observations.policy.corrected_velocity = ObservationTermCfg(func=obs.velocity)
        self.observations.policy.terrain = ObservationTermCfg(func=obs.elevation)
        self.observations.critic.state.func = obs.critic_state
        self.events.reset_robot.func = reset_on_terrain
        self.events.reset_robot.params = {}
        self.events.bin_fail_counter_smoothing = None
        self.curriculum.beyond_adaptive_sampling = None
        self.monitors = RewardTerms()
        self.rewards = TerrainRewardsCfg()
        stds = dict(
            position=0.3,
            orientation=0.4,
            linear_velocity=1.0,
            angular_velocity=3.14,
            joint_position=0.5,
            joint_velocity=5.0,
        )
        for group, weights in (("upper", (1, 1, 0.5, 0.5, 1, 0.5)), ("lower", (0.5, 2, 0.5, 0.5, 0.5, 0.5))):
            for (quantity, std), weight in zip(stds.items(), weights):
                setattr(
                    getattr(self.rewards, group),
                    quantity,
                    RewardTermCfg(
                        func=rew.tracking, weight=weight, params=dict(group=group, quantity=quantity, std=std)
                    ),
                )
        for quantity, weight, std in (
            ("root_orientation", 0.5, 0.4),
            ("root_velocity", 2.0, 1.0),
            ("floating_anchor", 1.0, 0.3),
            ("vertical_acceleration", -0.001, 1.0),
            ("ee_acceleration", -0.001, 1.0),
            ("action_rate", -0.05, 1.0),
            ("joint_limits", -15.0, 1.0),
            ("undesired_contact", -0.1, 1.0),
            ("impact", -1e-5, 1.0),
        ):
            setattr(
                self.rewards.auxiliary,
                quantity,
                RewardTermCfg(func=rew.auxiliary, weight=weight, params=dict(quantity=quantity, std=std)),
            )
        for quantity, weight in (
            ("touchdown", 10.0),
            ("match", 1.5),
            ("slip", -1.0),
            ("stumble", -20.0),
            ("switch", -30.0),
            ("force", -1e-6),
        ):
            setattr(
                self.rewards.terrain,
                quantity,
                RewardTermCfg(func=rew.terrain, weight=weight, params=dict(quantity=quantity)),
            )
        self.terminations = TerrainTerminationsCfg()
        self.run_name = "G1PGMTLafanTerrain"


@configclass
class G1PGMTTerrainEnvCfg_PLAY(G1PGMTTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.motion.enable_terrain_curriculum = False
        self.observations.policy.enable_corruption = False


@configclass
class G1PGMTTerrainFlatEnvCfg(G1PGMTTerrainEnvCfg):
    flat_precheck: bool = True


@configclass
class G1PGMTTerrainFlatEnvCfg_PLAY(G1PGMTTerrainFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
