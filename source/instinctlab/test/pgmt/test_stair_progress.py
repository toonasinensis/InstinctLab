import torch

from instinctlab.tasks.shadowing.pgmt.supermimic.utils.pgmt_progress import StairProgressReward


def test_stair_progress_gates_positive_rewards_and_provides_gradient_at_zero():
    progress = StairProgressReward(0.02, 0.32, 0.2, 0.05, 0.5, 0.5)
    history = torch.zeros(6, progress.steps, 4)
    future = torch.tensor(
        [
            [0.0, 0, 0.16, 0],
            [0.08, 0, 0.16, 0],
            [-0.08, 0, 0.16, 0],
            [0, 0.08, 0.16, 0],
            [0, 0, 0.01, 0],
            [0, 0, 0.16, 0],
        ],
        requires_grad=True,
    )
    family = torch.tensor([2, 2, 2, 2, 2, 0])[:, None]
    rewards = torch.tensor([2.0, -1.0]).expand(6, 1, -1)
    actual = progress.apply(rewards, history, future[:, None], family)[:, 0]
    expected = torch.tensor([[0, -1, 0], [2, -1, 0.005], [0, -1, -0.005], [0, -1, 0], [2, -1, 0], [2, -1, 0]])
    torch.testing.assert_close(actual, expected)
    actual[0].sum().backward()
    assert future.grad[0, 0] > 0
    torch.testing.assert_close(future.grad[0, 1], torch.zeros(()))


def test_stair_progress_rollout_matches_native_history_and_resets():
    progress = StairProgressReward(0.02, 0.32, 0.2, 0.05, 0.5, 0.5)
    step = torch.arange(1, 36) * 0.02
    future = torch.stack((step * 0.18, step * 0, step * 0.5, step * 0), dim=-1)[None]
    position = torch.zeros(1, 4)
    reset = torch.tensor([True])
    history = progress.record(None, position, reset)
    rewards = torch.ones(1, 35, 2)
    family = torch.full((1, 35), 2)
    imagined = progress.apply(rewards, history, future, family)
    native = []
    for i in range(future.shape[1]):
        point = future[:, i : i + 1]
        native.append(progress.apply(rewards[:, i : i + 1], history, point, family[:, i : i + 1]))
        history = progress.record(history, point[:, 0], ~reset)
    torch.testing.assert_close(torch.cat(native, dim=1), imagined)
    new = progress.record(history, position + 20, reset)
    torch.testing.assert_close(new, torch.full_like(history, 20))
