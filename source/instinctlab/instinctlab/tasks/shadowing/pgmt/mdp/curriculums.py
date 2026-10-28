"""Keep the source failure-mixture distribution on native motion-reference bins."""

import torch
from torch.nn import functional as F

from instinctlab.envs.mdp.curriculums.motion_reference import BeyondConcatMotionAdaptiveWeighting


class PGMTAdaptiveWeighting(BeyondConcatMotionAdaptiveWeighting):
    def _compute_smoothed_probs(self):
        counts = self.motion_bin_fail_counter
        widths = counts._batch_sizes
        size = int(widths.max())
        mask = torch.arange(size, device=self.device)[None] < widths[:, None]
        padded = torch.zeros(len(widths), size, device=self.device)
        padded[mask] = counts._concatenated_tensor
        padded = padded.clamp(max=10000.0 * padded[mask].mean())
        smooth = F.conv1d(F.pad(padded[:, None], (0, self.adaptive_kernel_size - 1)), self.kernel.view(1, 1, -1))[:, 0]
        values = smooth[mask]
        uniform = torch.ones_like(values) / len(values)
        if values.sum() > 1e-12:
            probability = values / values.sum()
            middle = probability.clamp(max=200.0 * probability.mean())
            middle /= middle.sum()
            hard = probability.clamp(max=10000.0 * probability.mean())
            hard /= hard.sum()
            return 0.2 * uniform + 0.6 * middle + 0.2 * hard
        return uniform
