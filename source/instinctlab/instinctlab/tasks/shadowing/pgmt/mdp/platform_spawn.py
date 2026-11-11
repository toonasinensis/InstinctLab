"""Sample the original central spawn platform without moving terrain origins."""

import torch


def platform_half_extents(generator, column_names, center_region_width):
    """Return safe central rectangles; flat/rough use an explicit central region."""
    extents = []
    for name in column_names:
        cfg = generator.sub_terrains[name]
        if hasattr(cfg, "step_width"):
            inner = [size - 2 * cfg.border_width for size in generator.size]
            count = int(min((size - cfg.platform_width) // (2 * cfg.step_width) + 1 for size in inner))
            half = [(size - 2 * count * cfg.step_width) / 2 for size in inner]
        else:
            width = getattr(cfg, "platform_width", center_region_width)
            inset = generator.horizontal_scale if hasattr(cfg, "slope_range") else 0.0
            half = [min(width, size) / 2 - inset for size in generator.size]
        if min(half) <= 0:
            raise ValueError(f"Invalid central spawn platform for {name}: {half}")
        extents.append(half)
    return extents


def sample_platform_offsets(body_xy, half_extents, margin):
    """Uniform root XY within the platform after accounting for the rotated pose."""
    footprint = torch.cat((body_xy, body_xy.new_zeros(len(body_xy), 1, 2)), dim=1)
    lower = -half_extents + margin - footprint.amin(dim=1)
    upper = half_extents - margin - footprint.amax(dim=1)
    feasible = (upper >= lower).all(dim=-1)
    if not feasible.all():
        indices = (~feasible).nonzero().flatten().tolist()
        raise ValueError(f"Reference pose does not fit the central spawn platform; reset rows {indices[:16]}")
    return lower + torch.rand_like(lower) * (upper - lower)
