"""Four-glimpse SM with bounded checkpoint storage for horizon comparisons."""

from .checkpoint_retention import retain_checkpoints
from .contact_runner import ContactBPTTRunner


class GlimpseContactBPTTRunner(ContactBPTTRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if train_cfg["bptt_execution"].get("compile_mode") == "chain" and train_cfg["bptt"]["horizon"] != 16:
            raise ValueError("Four-glimpse chain compilation is accepted only at H16; H32 failed the gradient gate")
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path, infos=None):
        super().save(path, infos)
        if self.rank == 0:
            retain_checkpoints(path, **self.cfg["checkpoint_retention"])
