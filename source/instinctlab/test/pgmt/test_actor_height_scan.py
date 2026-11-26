import torch
from types import SimpleNamespace

from instinctlab.tasks.shadowing.pgmt.supermimic.actor_height_scan import ActorHeightScan
from instinctlab.tasks.shadowing.pgmt.supermimic.envs.pgmt_world_model_env import PGMTWorldModelEnv


def test_actor_height_scan_holds_across_steps_and_refreshes_on_reset():
    sampler = ActorHeightScan(10, 0.02, slice(1, 2))
    held = torch.zeros(2, 1)
    for first_step in range(6):
        episode_step = torch.tensor([first_step, (first_step + 2) % 4])
        fresh = torch.full_like(held, first_step + 10)
        held = sampler.hold(held, fresh, episode_step)
        expected = ([10, 0], [10, 0], [10, 12], [10, 12], [10, 12], [15, 12])[first_step]
        torch.testing.assert_close(held[:, 0], torch.tensor(expected, dtype=held.dtype))


def test_imagined_actor_uses_replay_phase_and_critic_keeps_the_fresh_scan():
    env = PGMTWorldModelEnv.__new__(PGMTWorldModelEnv)
    env.actor_height_scan = ActorHeightScan(10, 0.02, slice(1, 2))
    env.mdp = SimpleNamespace(
        heightfield=SimpleNamespace(local=lambda position, yaw: (position[:, :1], torch.ones(2, dtype=torch.bool))),
        process_observation=lambda scan, name: scan,
    )
    current = torch.tensor([[101.0, 5.0], [201.0, 10.0]])
    env.features_forward = lambda *args: (current, current, torch.ones(2, dtype=torch.bool))
    from tensordict import TensorDict

    rows = dict(
        state=torch.zeros(2, 1, 3),
        history=torch.zeros(2, 1, 3),
        support=torch.zeros(2, 1),
        previous_action=torch.zeros(2, 1, 1),
        episode_step=torch.tensor([[4], [0]]),
        actor_scan_pose=torch.tensor([[[2.0, 0.0, 0.0, 0.0]], [[7.0, 0.0, 0.0, 0.0]]]),
        env_id=torch.arange(2),
    )
    noise = torch.zeros(1, 2, 1)
    inputs = TensorDict(
        dict(
            data=TensorDict(rows, batch_size=[]),
            **{
                name: TensorDict(dict(k=noise), batch_size=[1, 2])
                for name in ("contexts", "proprio_noise", "observation_noise")
            }
        ),
        batch_size=[],
    )
    env._on_reset(inputs)
    observed = env.get_observations()
    torch.testing.assert_close(observed["actor"], torch.tensor([[101.0, 2.0], [201.0, 10.0]]))
    torch.testing.assert_close(observed["critic"], current)
    env.observation = fresh = torch.tensor([[102.0, 6.0], [202.0, 11.0]], requires_grad=True)
    env.feature_rows.append(fresh)
    env._compute_commands()
    observed = env.get_observations()
    torch.testing.assert_close(observed["actor"], torch.tensor([[102.0, 6.0], [202.0, 10.0]]))
    torch.testing.assert_close(observed["critic"], fresh)
    torch.testing.assert_close(env.observation, fresh)
    observed["actor"][:, 1].sum().backward()
    torch.testing.assert_close(fresh.grad, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
