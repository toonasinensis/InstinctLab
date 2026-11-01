"""Chronological supervised replay spanning PPO rollouts, with episode masks."""

import torch


class WMReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.pointer = self.size = 0
        self.added = 0
        self.history_prefix = None

    def __len__(self):
        return self.size

    def rows(self, ages):
        return (self.pointer - self.size + ages) % self.capacity

    @torch.no_grad()
    def add(self, record):
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
                self.history_prefix.copy_(oldest[:, 1:])
        if not self.data:
            # First add happens inside the PPO inference rollout.
            with torch.inference_mode(False):
                self.data = {
                    key: torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=value.device)
                    for key, value in record.items()
                }
        for key, value in record.items():
            self.data[key][self.pointer].copy_(value)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.added = min(self.added + 1, self.size)

    def histories(self, ages, env_ids):
        """Recover exactly the noisy native history, including resets and overwritten rows."""
        steps = self.data["episode_step"][self.rows(ages), env_ids]
        offsets = torch.arange(-9, 1, device=ages.device)
        history_ages = ages[..., None] + offsets
        history_ages = torch.maximum(history_ages, (ages - steps)[..., None])
        values = self.data["proprio"][self.rows(history_ages.clamp_min(0)), env_ids[..., None]]
        prefix = self.history_prefix[env_ids[..., None], (history_ages + 9).clamp(0, 8)]
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
        self.pointer = self.size = 0
        self.added = 0
