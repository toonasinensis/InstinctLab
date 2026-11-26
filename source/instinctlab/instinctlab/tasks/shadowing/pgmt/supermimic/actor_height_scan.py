import math
import torch


class ActorHeightScan:
    def __init__(self, rate_hz, step_dt, section):
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("Actor heightscan frequency must be positive and finite")
        ratio = 1.0 / (step_dt * rate_hz)
        period = round(ratio)
        if period < 1 or not math.isclose(ratio, period, rel_tol=1e-7):
            raise ValueError("Actor heightscan frequency must divide the policy control frequency")
        self.period = period
        self.section = section

    def hold(self, previous, current, episode_step):
        refresh = episode_step.remainder(self.period) == 0
        return torch.where(refresh[:, None], current, previous)

    def observation(self, observation, held_scan):
        return torch.cat((observation[:, : self.section.start], held_scan), dim=-1)
