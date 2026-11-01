"""CPU regression: compact noisy histories survive reset, overwrite and refill."""

import importlib.util
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


@torch.jit.script
def scripted_function(value):
    return value.sin() * value.cos()
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[2] / "source/instinctlab/instinctlab/tasks/shadowing/pgmt/wm_replay.py"
    spec = importlib.util.spec_from_file_location("wm_replay", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.WMReplay(17)
    generator = torch.Generator().manual_seed(42)
    ids = torch.arange(4)
    for refill in range(2):
        histories, expected = torch.randn(4, 10, 93, generator=generator), []
        steps = torch.full((4,), 20, dtype=torch.long)
        for t in range(57):
            frame = torch.randn(4, 93, generator=generator)
            reset = (t + ids) % (11 + ids) == 0
            histories = torch.cat((histories[:, 1:], frame[:, None]), 1)
            histories[reset] = frame[reset, None]
            steps = torch.where(reset, 0, steps + 1)
            replay.add(dict(history=histories.clone(), episode_step=steps.clone(), done=reset.clone()))
            expected.append(histories.clone())
            ages = torch.arange(len(replay))[:, None].expand(-1, 4)
            actual = replay.histories(ages, ids[None])
            torch.testing.assert_close(actual, torch.stack(expected[-len(replay) :]), rtol=0, atol=0)
            data = replay.sequences(ids, min(8, len(replay)), generator)
            target = torch.stack(expected[-len(replay) :])
            sampled = target[data["replay_age"][:, None] + torch.arange(data["history"].shape[1]), ids[:, None]]
            torch.testing.assert_close(data["history"], sampled, rtol=0, atol=0)
        assert "history" not in replay.data and replay.data["proprio"].shape == (17, 4, 93)
        replay.clear()
        assert replay.added == 0
    print("PASS: exact compact history, native reset backfill, ring overwrite, sequence sampling and refill")
    value = torch.linspace(-1, 1, 19, requires_grad=True)
    direct = torch.autograd.grad(scripted_function(value).sum(), value)[0]
    with set_checkpoint_early_stop(False):
        result = checkpoint(scripted_function, value, use_reentrant=False, preserve_rng_state=False)
    recomputed = torch.autograd.grad(result.sum(), value)[0]
    torch.testing.assert_close(direct, recomputed, rtol=0, atol=0)
    print("PASS: full non-reentrant checkpoint recomputation through TorchScript")


if __name__ == "__main__":
    main()
