"""Terminal-inclusive replay with causal history, including ring wraparound.

PGMT retains actual terminal successor states, unlike the desktop state-only
buffer. A sampled sequence includes its terminal step and masks later resets.
"""

import torch
from tensordict import TensorDict

from .replay_buffer import FieldSpec, ReplayBuffer


class PGMTReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, history_length):
        self.history_length = history_length
        self.capacity = capacity
        self._storage = None
        self._step = self.num_transitions = 0
        self.added = 0
        self.history_prefix = None

    def __len__(self):
        return self.size

    @property
    def pointer(self):
        return self._step

    @property
    def size(self):
        return self.num_transitions

    @property
    def data(self):
        return {} if self._storage is None else {key: value.transpose(0, 1) for key, value in self._storage.items()}

    def rows(self, ages):
        return (self.pointer - self.size + ages) % self.capacity

    @torch.no_grad()
    def insert(self, record):
        record = dict(record)
        if "history" in record:
            record = dict(record)
            history = record.pop("history")
            record["proprio"] = history[:, -1]
            if self.size == 0:
                with torch.inference_mode(False):
                    self.history_prefix = history[:, :-1].clone()
            elif self.size == self.capacity:
                ids = torch.arange(history.shape[0], device=history.device)
                oldest = self.histories(torch.zeros_like(ids), ids)
                assert self.history_prefix is not None
                self.history_prefix.copy_(oldest[:, 1:])
        record["terminated"] = record.get("failure", record["done"])
        record["truncated"] = record["done"] & ~record["terminated"]
        if self._storage is None:
            schema = {key: FieldSpec(dim=tuple(value.shape[1:]), dtype=value.dtype) for key, value in record.items()}
            with torch.inference_mode(False):
                super().__init__(schema, self.capacity, record["state"].shape[0], str(record["state"].device))
        super().insert(TensorDict(record, batch_size=[self.num_envs]))
        self.added = min(self.added + 1, self.size)

    def histories(self, ages, env_ids):
        """Recover exactly the noisy native history, including resets and overwritten rows."""
        assert self.history_prefix is not None, "History was not stored in this replay"
        steps = self.data["episode_step"][self.rows(ages), env_ids]
        offsets = torch.arange(1 - self.history_length, 1, device=ages.device)
        history_ages = ages[..., None] + offsets
        history_ages = torch.maximum(history_ages, (ages - steps)[..., None])
        values = self.data["proprio"][self.rows(history_ages.clamp_min(0)), env_ids[..., None]]
        prefix = self.history_prefix[
            env_ids[..., None], (history_ages + self.history_length - 1).clamp(0, self.history_length - 2)
        ]
        return torch.where((history_ages < 0)[..., None], prefix, values)

    def sequences(self, env_ids, horizon, generator, history=True, ages=None):
        if not 1 <= horizon <= self.size:
            raise ValueError("WM replay does not yet contain the requested horizon")
        device = env_ids.device
        age = (
            ages
            if ages is not None
            else torch.randint(self.size - horizon + 1, env_ids.shape, device=device, generator=generator)
        )
        rows = self.rows(age[:, None] + torch.arange(horizon, device=device))
        data = {key: value[rows, env_ids[:, None]] for key, value in self.data.items()}
        if history and self.history_prefix is not None:
            data["history"] = self.histories(age[:, None] + torch.arange(horizon, device=device), env_ids[:, None])
        alive = (~data["done"]).float().cumprod(1)
        data["mask"] = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1)
        data["env_id"] = env_ids
        data["replay_age"] = age
        return data

    def clear(self):
        self._step = self.num_transitions = 0
        self.added = 0
