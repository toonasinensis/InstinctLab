"""Chronological supervised replay spanning PPO rollouts, with episode masks."""

import torch


class WMReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.pointer = self.size = 0

    def __len__(self):
        return self.size

    def rows(self, ages):
        return (self.pointer - self.size + ages) % self.capacity

    @torch.no_grad()
    def add(self, record):
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

    def sequences(self, env_ids, horizon, generator):
        if not 1 <= horizon <= self.size:
            raise ValueError("WM replay does not yet contain the requested horizon")
        device = env_ids.device
        age = torch.randint(self.size - horizon + 1, env_ids.shape, device=device, generator=generator)
        rows = self.rows(age[:, None] + torch.arange(horizon, device=device))
        data = {key: value[rows, env_ids[:, None]] for key, value in self.data.items()}
        alive = (~data["done"]).float().cumprod(1)
        data["mask"] = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1)
        data["env_id"] = env_ids
        data["replay_age"] = age
        return data

    def clear(self):
        self.pointer = self.size = 0
