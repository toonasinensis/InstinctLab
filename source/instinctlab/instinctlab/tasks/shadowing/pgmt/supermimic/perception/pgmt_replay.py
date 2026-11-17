"""Host image ring plus the existing terminal-inclusive PGMT physical replay."""

import torch

from instinctlab.tasks.shadowing.pgmt.supermimic.pgmt_replay_buffer import PGMTReplayBuffer


class DepthReplayBuffer(PGMTReplayBuffer):
    image_fields = ("depth_image", "next_depth_image")

    def __init__(self, capacity, history_length):
        super().__init__(capacity, history_length)
        self.images = {}

    @torch.no_grad()
    def insert(self, record):
        record = dict(record)
        for name in self.image_fields:
            image = record.pop(name)
            if name not in self.images:
                self.images[name] = torch.empty((self.capacity, *image.shape), dtype=torch.float16, device="cpu")
            self.images[name][self.pointer].copy_(image)
        super().insert(record)

    def images_at(self, ages, ids, *, successor=False):
        name = "next_depth_image" if successor else "depth_image"
        return self.images[name][self.rows(ages).cpu(), ids.cpu()].to(device=ids.device, dtype=torch.float32)

    def image_windows(self, data, horizon):
        ages, ids = data["replay_age"], data["env_id"]
        initial = self.images_at(ages, ids)[:, None]
        offsets = torch.arange(horizon, device=ids.device)
        following = self.images_at(ages[:, None] + offsets, ids[:, None], successor=True)
        return torch.cat((initial, following), 1)
