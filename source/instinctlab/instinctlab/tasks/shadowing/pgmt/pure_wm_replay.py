"""Chronological replay independent of PPO rollout chunks and episode resets."""

import torch
import torch.distributed as dist


class ReferenceCoverageError(RuntimeError):
    """The bounded sampler could not find enough complete reference windows."""


class TransitionReplay:
    def __init__(self, steps, envs, device):
        if steps < 2 or envs < 8:
            raise ValueError("Replay needs at least two steps and eight environments for a held-out split")
        self.capacity, self.envs, self.device = steps, envs, device
        self.data = {}
        self.pointer = self.size = 0

    @torch.no_grad()
    def add(self, record):
        if not self.data:
            self.data = {
                name: torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=self.device)
                for name, value in record.items()
            }
        if record.keys() != self.data.keys():
            raise ValueError("Replay transition schema changed")
        for name, value in record.items():
            if value.shape[0] != self.envs or value.shape != self.data[name].shape[1:]:
                raise ValueError(f"Replay shape mismatch for {name}")
            self.data[name][self.pointer].copy_(value.detach())
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def rows(self, ages):
        return (self.pointer - self.size + ages) % self.capacity

    def sample_envs(self, batch, validation):
        indices = torch.arange(self.envs, device=self.device)
        eligible = indices[indices % 8 == 0] if validation else indices[indices % 8 != 0]
        return eligible[torch.randint(len(eligible), (batch,), device=self.device)]

    def sequences(self, batch, length, validation=False):
        if not 1 <= length <= self.size:
            raise ValueError("Replay does not contain a complete requested time window")
        age = torch.randint(self.size - length + 1, (batch,), device=self.device)
        env = self.sample_envs(batch, validation)
        rows = self.rows(age[:, None] + torch.arange(length, device=self.device))
        result = {name: value[rows, env[:, None]] for name, value in self.data.items()}
        # Include the terminating transition; exclude every successor in a new episode.
        alive = (~result["done"]).float()
        result["mask"] = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1).cumprod(1)
        result["env_id"] = env
        return result

    def starts(self, batch, horizon, motion_lengths, history_length=10):
        if self.size < history_length:
            raise ValueError("Fill replay before sampling actor histories")
        records, histories, count = [], [], 0
        # A bounded sampler reports bad reference coverage instead of hanging a rank.
        for _ in range(32):
            remaining = batch - count
            age = torch.randint(history_length - 1, self.size, (2 * remaining,), device=self.device)
            env = self.sample_envs(len(age), validation=False)
            row = self.rows(age)
            frame, motion = self.data["frame"][row, env], self.data["motion"][row, env]
            # PGMT observes the current frame and ten future frames at each step.
            valid = frame + horizon + 10 < motion_lengths[motion]
            chosen = valid.nonzero().flatten()[:remaining]
            if not len(chosen):
                continue
            age, env, row = age[chosen], env[chosen], row[chosen]
            sample = {name: value[row, env] for name, value in self.data.items()}
            sample["env_id"] = env
            back = torch.arange(history_length - 1, -1, -1, device=self.device)
            back = torch.minimum(back[None], sample["episode_step"][:, None])
            history_rows = self.rows(age[:, None] - back)
            history = self.data["proprioception"][history_rows, env[:, None]]
            # IsaacLab's CircularBuffer backfills every slot with the first
            # post-reset observation. Never include a previous episode's data.
            records.append(sample)
            histories.append(history)
            count += len(chosen)
            if count == batch:
                return {name: torch.cat([x[name] for x in records]) for name in records[0]}, torch.cat(histories)
        raise ReferenceCoverageError("Insufficient motion-reference coverage for the requested imagination horizon")

    def collective_starts(self, batch, horizon, motion_lengths, history_length=10):
        """Decide reference-coverage failure together before any rank enters BPTT."""
        sample, error = None, None
        try:
            sample = self.starts(batch, horizon, motion_lengths, history_length)
        except ReferenceCoverageError as exc:
            error = exc
        if not dist.is_initialized():
            if error is not None:
                raise error
            return sample
        failed = torch.tensor(int(error is not None), device=self.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise ReferenceCoverageError(
                "Insufficient motion-reference coverage on at least one rank; actor update aborted on all ranks"
            ) from error
        return sample

    def clear(self):
        self.pointer = self.size = 0
