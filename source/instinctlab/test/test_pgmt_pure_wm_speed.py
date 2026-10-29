"""Reward/BPTT equivalence when moving independent work out of the time loop."""

import torch
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytorch_kinematics as pk

from instinctlab.assets.pgmt_g1 import ASSET_DIR, BODY_NAMES, JOINT_NAMES
from instinctlab.tasks.shadowing.pgmt.config.g1.pgmt_terrain_cfg import G1PGMTTerrainEnvCfg
from instinctlab.tasks.shadowing.pgmt.heightfield import HeightField
from instinctlab.tasks.shadowing.pgmt.models.terrain import TerrainActor
from instinctlab.tasks.shadowing.pgmt.pure_wm_mdp import PureImaginationMDP, RobotKinematics
from instinctlab.tasks.shadowing.pgmt.pure_wm_model import ImaginedValue, TerrainDynamics, lambda_returns
from instinctlab.tasks.shadowing.pgmt.pure_wm_runner import PureWorldModelRunner
from instinctlab.tasks.shadowing.pgmt.terrain_runner import TerrainNormalization


def fixture():
    torch.manual_seed(48)
    runner = PureWorldModelRunner.__new__(PureWorldModelRunner)
    runner.device = torch.device("cpu")
    runner.options = dict(horizon=3, gamma=0.99, action_noise=0.1, **{"lambda": 0.95})
    mdp = PureImaginationMDP.__new__(PureImaginationMDP)
    cfg = G1PGMTTerrainEnvCfg()
    mdp.dt = 0.02
    mdp.command = SimpleNamespace(
        cfg=cfg.commands.motion,
        upper_ids=[0, 7, 8, 9, 10, 11, 12, 13],
        lower_ids=[1, 2, 3, 4, 5, 6],
        upper_joint_ids=list(range(11, 29)),
        lower_joint_ids=list(range(11)),
        foot_ids=[3, 6],
        hand_ids=[10, 13],
        motion_anchor_body_index=7,
    )
    mdp.terms = {
        group + "/" + name: term
        for group in ("upper", "lower", "auxiliary", "terrain")
        for name, term in vars(getattr(cfg.rewards, group)).items()
        if not name.startswith("_")
    }
    mdp.thresholds = dict(root_height=0.4, root_tilt=1.0, body_error=0.65)
    mdp.tile_half = torch.tensor([4.0, 4.0])
    mdp.limits = torch.tensor([[-2.0, 2.0]]).expand(29, -1)
    xy = torch.linspace(-4, 4, 81)
    yy, xx = torch.meshgrid(xy, xy, indexing="ij")
    mdp.heightfield = HeightField(0.02 * xx + 0.03 * yy, [-4, -4], 0.1)
    urdf = Path(ASSET_DIR) / "pgmt_g1/urdf/g1/main.urdf"
    mdp.kinematics = RobotKinematics(str(urdf), JOINT_NAMES, BODY_NAMES, "cpu")
    chain = pk.build_chain_from_urdf(urdf.read_text())
    mdp.reference = SimpleNamespace(
        _robot_kinematics_chain=chain,
        _joint_order_isaac_to_pk=[JOINT_NAMES.index(name) for name in chain.get_joint_parameter_names()],
        _link_indices_pk=chain.get_frame_indices(*BODY_NAMES),
        cfg=SimpleNamespace(link_of_interests=BODY_NAMES),
    )
    runner.mdp = mdp
    runner.critic = ImaginedValue(296 + 14 + 4 + 441).requires_grad_(False)
    torch.nn.init.normal_(runner.critic.net[-1].weight, std=0.02)
    return runner


class SpeedContractTest(unittest.TestCase):
    def test_reference_windows_are_batched_without_changing_time_or_environment(self):
        mdp = PureImaginationMDP.__new__(PureImaginationMDP)
        calls = []

        def context(start, step):
            calls.append(len(start["frame"]))
            return dict(frame=(start["frame"] + step)[:, None], bias=start["bias"], env=start["env_id"])

        mdp.reference_context = context
        start = dict(frame=torch.tensor([11, 51, 91]), env_id=torch.tensor([2, 9, 1]), bias=torch.randn(3, 29))
        actual = mdp.reference_sequence(start, 3)
        self.assertEqual(calls, [12])
        for step in range(4):
            expected = context(start, step)
            for key in actual:
                torch.testing.assert_close(actual[key][step], expected[key], atol=0, rtol=0)

    def test_cached_kinematics_match_original_pose_path(self):
        runner = fixture()
        state = torch.randn(5, 71) * 0.1
        state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
        root_com, body_com = torch.randn(5, 3), torch.randn(5, 14, 3)
        actual = runner.mdp.kinematics.forward(state, root_com, body_com)
        expected = runner.mdp.bodies(state)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=3e-7, rtol=3e-6)

    def test_batched_reward_value_and_gradients_match_sequential_with_absorbing_mask(self):
        runner = fixture()
        horizon, batch = 3, 4
        states = torch.randn(horizon + 1, batch, 71) * 0.03
        states[..., 2] += 0.8
        states[..., 6] += 1
        states[..., 3:7] = torch.nn.functional.normalize(states[..., 3:7], dim=-1)
        states.requires_grad_()
        actions = (torch.randn(horizon + 1, batch, 29) * 0.2).requires_grad_()
        contacts = torch.rand(horizon + 1, batch, 14)
        supports = (torch.randn(horizon + 1, batch) * 0.01).requires_grad_()
        identity = torch.zeros(horizon + 1, batch, 11, 4)
        identity[..., 3] = 1
        contexts = dict(
            root_com=torch.randn(horizon + 1, batch, 3) * 0.01,
            body_com=torch.randn(horizon + 1, batch, 14, 3) * 0.01,
            ref_rotation=identity,
            ref_clearance=torch.full((horizon + 1, batch, 11), 0.8),
            ref_position=torch.randn(horizon + 1, batch, 3) * 0.02,
            ref_velocity=torch.randn(horizon + 1, batch, 11, 3) * 0.1,
            ref_body_position=torch.randn(horizon + 1, batch, 14, 3) * 0.03,
            ref_body_rotation=identity[:, :, :1].expand(-1, -1, 14, -1),
            ref_body_velocity=torch.randn(horizon + 1, batch, 14, 3) * 0.1,
            ref_body_angular=torch.randn(horizon + 1, batch, 14, 3) * 0.1,
            ref_q=torch.randn(horizon + 1, batch, 29) * 0.1,
            ref_qd=torch.randn(horizon + 1, batch, 29) * 0.1,
            ref_contact=torch.rand(horizon + 1, batch, 14),
            ref_valid=torch.ones(horizon + 1, batch, 14),
            level=torch.full((horizon + 1, batch), 3),
            family=torch.zeros(horizon + 1, batch, dtype=torch.long),
            tile_origin=torch.zeros(horizon + 1, batch, 3),
            default_q=torch.zeros(horizon + 1, batch, 29),
            default_qd=torch.zeros(horizon + 1, batch, 29),
            critic_command=torch.randn(horizon + 1, batch, 68),
            ref_joints=torch.randn(horizon + 1, batch, 638),
        )
        # Startup COM bindings are invariant across a trajectory.
        contexts["root_com"][:] = contexts["root_com"][0].clone()
        contexts["body_com"][:] = contexts["body_com"][0].clone()
        xi = torch.randn(batch, 4)
        age = torch.tensor([0, 1, 8, 2])
        validity = torch.ones(horizon, batch)
        validity[0, 0], validity[1, 1] = 0, 0
        kin = runner.mdp.kinematics.forward(
            states.flatten(0, 1), contexts["root_com"].flatten(0, 1), contexts["body_com"].flatten(0, 1)
        )
        kin = tuple(value.reshape(horizon + 1, batch, *value.shape[1:]) for value in kin)
        actual = runner.imagined_objective(states, actions, contacts, supports, contexts, kin, validity, age, xi)
        # fullgraph makes a Python/GPU synchronization regression fail explicitly.
        compiled = torch.compile(runner.imagined_objective, backend="eager", fullgraph=True)
        captured = compiled(states, actions, contacts, supports, contexts, kin, validity, age, xi)
        for left, right in zip(captured, actual):
            torch.testing.assert_close(left, right)
        discount, objective = torch.ones(batch), torch.zeros(batch)
        features, rewards, survivals = [], [], []
        for step in range(horizon + 1):
            context = {key: value[step] for key, value in contexts.items()}
            features.append(
                runner.critic_features(states[step], actions[step], supports[step], contacts[step], context, xi)
            )
            if step == 0:
                continue
            previous = {key: value[step - 1] for key, value in contexts.items()}
            reward, survival, _ = runner.mdp.objective(
                states[step],
                actions[step],
                actions[step - 1],
                supports[step],
                context,
                states[step - 1],
                contacts[step],
                contacts[step - 1],
                previous,
                (age + step - 1 > 0).float(),
            )
            objective = objective + discount * reward
            survival = survival * validity[step - 1]
            discount = discount * 0.99 * survival
            rewards.append(reward.detach())
            survivals.append(survival.detach())
        objective = objective + discount * runner.critic(features[-1])
        expected_loss = -objective.mean()
        torch.testing.assert_close(actual[0], expected_loss, atol=2e-6, rtol=1e-5)
        original_grad = torch.autograd.grad(expected_loss, (states, actions, supports), retain_graph=True)
        batched_grad = torch.autograd.grad(actual[0], (states, actions, supports))
        for left, right in zip(batched_grad, original_grad):
            torch.testing.assert_close(left, right, atol=3e-5, rtol=2e-4)
        expected_features = torch.stack(features).detach()
        torch.testing.assert_close(actual[1], expected_features[:-1].flatten(0, 1), atol=2e-6, rtol=1e-5)
        values = runner.critic(expected_features.flatten(0, 1)).reshape(horizon + 1, batch)
        expected_targets = lambda_returns(torch.stack(rewards), torch.stack(survivals), values)
        torch.testing.assert_close(actual[2], expected_targets.flatten(), atol=2e-6, rtol=1e-5)
        self.assertFalse(actual[1].requires_grad)
        self.assertFalse(actual[2].requires_grad)
        runner.actor = TerrainActor()
        runner.policy_normalizer = TerrainNormalization(2119).eval()
        runner.wm = TerrainDynamics(4, hidden=16, amp=False).requires_grad_(False)
        context = {key: value[0] for key, value in contexts.items()}
        next_context = {key: value[1] for key, value in contexts.items()}
        history, noise = torch.randn(batch, 10, 93), torch.randn(batch, 29)
        step_args = (
            states[0].detach(),
            actions[0].detach(),
            contacts[0],
            supports[0].detach(),
            history,
            context,
            next_context,
            xi,
            noise,
        )
        expected_step = runner.imagined_step(*step_args)
        captured_step = torch.compile(runner.imagined_step, backend="eager", fullgraph=True)(*step_args)
        for left, right in zip(captured_step[:-1], expected_step[:-1]):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
