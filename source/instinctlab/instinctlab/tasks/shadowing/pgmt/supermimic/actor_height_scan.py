import math
import torch

from ..selection import TERRAIN_FAMILIES


class ActorHeightScan:
    def __init__(self, rate_hz, step_dt, section, noise_std_m=0.0, random_phase=False, flat_zero_probability=0.0):
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("Actor heightscan frequency must be positive and finite")
        ratio = 1.0 / (step_dt * rate_hz)
        period = round(ratio)
        if period < 1 or not math.isclose(ratio, period, rel_tol=1e-7):
            raise ValueError("Actor heightscan frequency must divide the policy control frequency")
        if not math.isfinite(noise_std_m) or noise_std_m < 0:
            raise ValueError("Actor heightscan noise standard deviation must be finite and nonnegative")
        if not 0 <= flat_zero_probability <= 1:
            raise ValueError("Actor heightscan flat zero probability must be in [0, 1]")
        self.period = period
        self.section = section
        self.noise_std_m = noise_std_m
        self.random_phase = random_phase
        self.flat_zero_probability = flat_zero_probability

    def hold(self, previous, current, episode_step, phase=0):
        refresh = (episode_step == 0) | (episode_step.remainder(self.period) == phase)
        return torch.where(refresh[:, None], current, previous)

    def draw_noise(self, shape, device, generator):
        noise = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        return (noise * self.noise_std_m).to(torch.float16)

    @staticmethod
    def add_noise(scan, noise):
        return scan if noise is None else (scan + noise.to(scan.dtype)).clamp(-5, 5)

    def flat_zero_mask(self, previous, family, episode_step, generator):
        flat = family == TERRAIN_FAMILIES["flat"]
        draw = torch.rand(family.shape, device=family.device, generator=generator)
        selected = flat & (draw < self.flat_zero_probability)
        return selected if previous is None else torch.where(episode_step == 0, selected, previous & flat)

    def observation(self, observation, held_scan, zero_mask=None):
        if zero_mask is not None:
            held_scan = held_scan.masked_fill(zero_mask[:, None], 0)
        return torch.cat((observation[:, : self.section.start], held_scan), dim=-1)
