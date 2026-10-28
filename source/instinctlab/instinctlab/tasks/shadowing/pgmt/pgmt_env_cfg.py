"""Flat PGMT stage one, composed from the perceptive-shadowing environment components."""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinctlab.managers import MultiRewardCfg
from instinctlab.tasks.shadowing.perceptive import perceptive_env_cfg as perceptive_cfg

from . import mdp as pgmt_mdp


@configclass
class PGMTSceneCfg(perceptive_cfg.PerceptiveShadowingSceneCfg):
    height_scanner = None
    camera = None

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_type = "plane"
        self.terrain.terrain_generator = None
        self.terrain.visual_material = None
        self.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        self.light = None
        self.sky_light = None


@configclass
class CommandsCfg:
    motion = pgmt_mdp.LafanMotionLifecycleCfg()
    joint_pos_ref_command = instinct_mdp.JointPosRefCommandCfg(current_state_command=False)
    joint_vel_ref_command = instinct_mdp.JointVelRefCommandCfg(current_state_command=False)


_policy_template = perceptive_cfg.ObservationsCfg.PolicyObsCfg()


@configclass
class PolicyObsCfg(ObsGroupCfg):
    projected_gravity = _policy_template.projected_gravity
    base_ang_vel = _policy_template.base_ang_vel
    joint_pos = _policy_template.joint_pos
    joint_vel = _policy_template.joint_vel
    last_action = _policy_template.last_action
    joint_pos_ref = _policy_template.joint_pos_ref
    joint_vel_ref = _policy_template.joint_vel_ref
    motion_orientation = ObsTermCfg(func=pgmt_mdp.reference_orientation, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
    motion_height = ObsTermCfg(func=pgmt_mdp.reference_height, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False
        self.joint_pos_ref.noise = UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        self.joint_vel_ref.noise = UniformNoiseCfg(n_min=-0.5, n_max=0.5)
        for term in (self.projected_gravity, self.base_ang_vel, self.joint_pos, self.joint_vel, self.last_action):
            term.history_length = 10


@configclass
class CriticObsCfg(ObsGroupCfg):
    state = ObsTermCfg(func=pgmt_mdp.privileged_state)
    enable_corruption = False
    concatenate_terms = False


@configclass
class ObservationsCfg:
    policy = PolicyObsCfg()
    critic = CriticObsCfg()


@configclass
class RewardsCfg(perceptive_cfg.RewardsCfg):
    anchor_height = RewardTermCfg(func=pgmt_mdp.anchor_height_reward, weight=1.0, params={"std": 0.3})
    applied_torque_limits_by_ratio = None

    def __post_init__(self):
        self.undesired_contacts.weight = -0.01


@configclass
class RewardGroupsCfg(MultiRewardCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class EventsCfg(perceptive_cfg.EventsCfg):
    randomize_ray_offsets = None
    randomize_actuator_gains = None
    randomize_rigid_body_mass = None

    def __post_init__(self):
        self.physics_material.params.update(
            static_friction_range=(0.3, 1.6), dynamic_friction_range=(0.3, 1.2), make_consistent=False
        )
        self.reset_robot.params.update(
            position_offset=[0.0, 0.0, 0.0],
            randomize_pose_range={},
            randomize_velocity_range=pgmt_mdp.RESET_VELOCITY,
            randomize_joint_pos_range=(0.0, 0.0),
        )


@configclass
class CurriculumCfg:
    beyond_adaptive_sampling = CurriculumTermCfg(
        func=pgmt_mdp.PGMTAdaptiveWeighting,
        params={
            "adaptive_uniform_ratio": 0.2,
            "adaptive_kernel_size": 3,
            "adaptive_alpha": 0.001,
            "adaptive_lambda": 0.8,
        },
    )


@configclass
class TerminationsCfg(perceptive_cfg.TerminationsCfg):
    illegal_reset_contact = None
    dataset_exhausted = None
    out_of_border = None

    def __post_init__(self):
        self.base_pos_too_far.params["distance_threshold"] = 0.4
        self.base_pg_too_far.params.update(projected_gravity_threshold=1.0, z_only=True)
        self.link_pos_too_far.params["distance_threshold"] = 0.4


@configclass
class PGMTEnvCfg(InstinctLabRLEnvCfg):
    scene: PGMTSceneCfg = PGMTSceneCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: perceptive_cfg.ActionsCfg = perceptive_cfg.ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardGroupsCfg = RewardGroupsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    monitors: perceptive_cfg.MonitorCfg = perceptive_cfg.MonitorCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
