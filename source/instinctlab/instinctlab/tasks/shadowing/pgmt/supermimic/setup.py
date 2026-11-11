"""Construct PGMT components once; no PPO algorithm, storage, or runner is created."""

import copy
import torch
from dataclasses import dataclass, replace

import warp as wp
from instinct_rl import modules
from instinct_rl.modules.normalizer import EmpiricalNormalization
from instinct_rl.utils.utils import get_subobs_size

from instinctlab.tasks.shadowing.pgmt.analytic_foot_rewards import AnalyticFootRewards
from instinctlab.tasks.shadowing.pgmt.foot_rewards import FootContactRewards
from instinctlab.tasks.shadowing.pgmt.heightfield import HeightField
from instinctlab.tasks.shadowing.pgmt.wm_rewards import AnalyticRewards

from .collector import Collector, ordinary
from .config import SuperMimicCfg
from .execution import compile_function
from .layouts import RobotLayout
from .learners.policy_learner import PolicyLearner
from .learners.wm_learner import WorldModelLearner
from .mdp import PGMTMDP
from .models import Dynamics, PGMTPolicy, Value
from .replay_buffer import ReplayBuffer
from .representations import REPRESENTATIONS, DeltaRepresentation
from .terrain import BilinearTerrain
from .world_model_env import PGMTWorldModelEnv


class PolicyNormalizer(EmpiricalNormalization):
    def __init__(self, shape, scan_width, **kwargs):
        super().__init__(shape, **kwargs)
        self.scan_width = scan_width

    def update(self, x):
        with torch.inference_mode(False), torch.no_grad():
            super().update(x)

    def forward(self, x):
        normalized = super().forward(x)
        return torch.cat((normalized[..., : -self.scan_width], x[..., -self.scan_width :]), -1)


def simulator_context(native):
    names = ("physics_material", "add_joint_default_pos", "base_com")
    if any(native.event_manager.get_term_cfg(name).mode != "startup" for name in names):
        raise ValueError("SM context currently requires startup-only physical randomization")
    material = native.event_manager.get_term_cfg("physics_material").func._impl
    event = native.event_manager.get_term_cfg("base_com").func
    robot = native.scene["robot"].data
    return ordinary(
        torch.cat(
            (
                wp.to_torch(material._friction_binding),
                robot.default_joint_pos.torch - robot.default_joint_pos_nominal,
                (robot.body_com_pos_b.torch - event.default_com[..., :3])[:, event.asset_cfg.body_ids].flatten(1),
            ),
            -1,
        )
    )


@dataclass
class Components:
    """Explicit ownership of the trainable models, simulator adapter and learners."""

    networks: torch.nn.Module
    normalizers: dict[str, EmpiricalNormalization]
    terrain: BilinearTerrain
    rewards: AnalyticRewards
    context: torch.Tensor
    dynamics: Dynamics
    policy: PGMTPolicy
    mdp: PGMTMDP
    value: Value
    foot_rewards: FootContactRewards
    analytic_foot: AnalyticFootRewards | None
    replay: ReplayBuffer
    generators: dict[str, torch.Generator]
    collector: Collector
    world_env: PGMTWorldModelEnv
    wm_learner: WorldModelLearner
    policy_learner: PolicyLearner


def build_components(env, cfg: dict, options: SuperMimicCfg, device: torch.device) -> Components:
    native = env.unwrapped
    scan = native.cfg.height_scan
    scan_width = 1
    for size in scan["size"]:
        scan_width *= round(size / scan["spacing"]) + 1
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(cfg["seed"])
        policy_cfg = copy.deepcopy(cfg["policy"])
        # Reuse the existing network constructor and checkpoint key layout.
        # The unused PPO critic is retained only as a portable inference export.
        networks = modules.build_actor_critic(
            policy_cfg.pop("class_name"),
            policy_cfg,
            env.get_obs_format(),
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(device)
        networks.critic.requires_grad_(False)
        networks.std.requires_grad_(False)
        normalizers = {}
        for group, values in cfg["normalizers"].items():
            values = {key: value for key, value in values.items() if key != "class_name"}
            shape = get_subobs_size(env.get_obs_format()[group])
            normalizer = (
                PolicyNormalizer(shape, scan_width, **values)
                if group == "policy"
                else EmpiricalNormalization(shape, **values)
            )
            normalizers[group] = normalizer.to(device).eval()
        env.reset()
    layout = RobotLayout(env.num_actions)
    command = native.command_manager.get_term("motion")
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(cfg["seed"] + options.seeds.world_model)
        field = (
            native.pgmt_heightfield
            if hasattr(native, "pgmt_heightfield")
            else HeightField.from_context(command.terrain, native.cfg.heightfield_resolution)
        )
        terrain = BilinearTerrain(field.to(device))
        rewards = AnalyticRewards(native, field)
        context = simulator_context(native)
        representation = REPRESENTATIONS[options.representation.kind](
            layout, options.representation, native.step_dt, native.cfg.decimation, rewards.root_com[0]
        ).to(device)
        dynamics = Dynamics(
            representation,
            terrain,
            options.dynamics,
            context.shape[-1],
            command.foot_ids,
            scan_width,
            options.execution,
        ).to(device)
    if options.representation.kind == "velocity":
        # Keep shared hidden/contact initialization identical across output widths.
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(cfg["seed"] + options.seeds.world_model)
            rep = DeltaRepresentation(
                layout,
                replace(options.representation, kind="delta"),
                native.step_dt,
                native.cfg.decimation,
                rewards.root_com[0],
            ).to(device)
            reference = Dynamics(
                rep,
                terrain,
                options.dynamics,
                context.shape[-1],
                command.foot_ids,
                scan_width,
                options.execution,
            ).to(device)
        source = dict(reference.named_parameters())
        with torch.no_grad():
            for name, value in dynamics.named_parameters():
                if source[name].shape == value.shape:
                    value.copy_(source[name])
    policy = PGMTPolicy(networks.actor, normalizers["policy"], options.execution)
    reference_frames = command.reference.cfg.num_frames
    mdp = PGMTMDP(
        rewards,
        normalizers["policy"],
        dynamics,
        options.collection.history_length,
        reference_frames,
        cfg["termination_softness"],
    )
    width = mdp.observation_layout.width + layout.feature_width + context.shape[-1] + mdp.critic_context_width
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(cfg["seed"] + options.seeds.value)
        value = Value(width, options.policy.critic_hidden).to(device)
    foot_rewards = FootContactRewards(rewards, cfg["foot_contact"])
    analytic_foot = AnalyticFootRewards(rewards, cfg["analytic_foot"]) if cfg["analytic_foot"]["enabled"] else None
    if (cfg["foot_contact"]["enabled"] or analytic_foot is not None) and not options.dynamics.predict_contact:
        raise ValueError("Enabled contact rewards require the foot probability head")
    replay = ReplayBuffer(options.collection.replay_steps, options.collection.history_length)
    generators = {
        name: torch.Generator(device=device).manual_seed(cfg["seed"] + getattr(options.seeds, name))
        for name in ("replay", "actor", "collection", "critic", "observation", "gradient")
    }
    collector = Collector(
        env,
        policy,
        mdp,
        replay,
        foot_rewards,
        analytic_foot,
        options.collection,
        generators["collection"],
        cfg["flat_warmup_updates"],
        cfg["terrain_gate_ramp"],
    )
    world_env = PGMTWorldModelEnv(
        policy,
        dynamics,
        value,
        mdp,
        context,
        foot_rewards,
        analytic_foot,
        options.policy,
        options.execution,
    )
    wm_learner = WorldModelLearner(dynamics, context, options.world_model, options.wm_optimizer, options.execution)
    policy_learner = PolicyLearner(
        policy,
        value,
        world_env,
        options.policy,
        options.actor_optimizer,
        options.critic_optimizer,
        options.execution,
        generators["critic"],
        generators["gradient"],
    )
    if options.execution.compile != "none":
        policy.network.action_from_inputs = compile_function(policy.network.action_from_inputs, True)
        dynamics.predict_delta = compile_function(dynamics.predict_delta, True)

    return Components(
        networks=networks,
        normalizers=normalizers,
        terrain=terrain,
        rewards=rewards,
        context=context,
        dynamics=dynamics,
        policy=policy,
        mdp=mdp,
        value=value,
        foot_rewards=foot_rewards,
        analytic_foot=analytic_foot,
        replay=replay,
        generators=generators,
        collector=collector,
        world_env=world_env,
        wm_learner=wm_learner,
        policy_learner=policy_learner,
    )
