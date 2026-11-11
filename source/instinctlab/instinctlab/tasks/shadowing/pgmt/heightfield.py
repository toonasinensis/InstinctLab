"""Fixed world terrain with differentiable, yaw-aligned sampling coordinates.

The mesh is rasterized once. Autograd differentiates coordinates, not the mesh,
so imagination needs neither a differentiable simulator nor ray-query adjoints.
"""

import torch
from torch import nn
from torch.nn import functional as F


class HeightField(nn.Module):
    def __init__(self, heights, origin_xy, resolution, scan=None):
        super().__init__()
        if heights.ndim != 2 or min(heights.shape) < 2 or not torch.isfinite(heights).all():
            raise ValueError("Height field must be a finite [y,x] raster with at least two samples per axis")
        if resolution <= 0:
            raise ValueError("Height field resolution must be positive")
        self.register_buffer("heights", heights.detach()[None, None])
        self.register_buffer("origin", heights.new_tensor(origin_xy).detach())
        self.resolution = float(resolution)
        self.register_buffer("extent", heights.new_tensor([heights.shape[1] - 1, heights.shape[0] - 1]) * resolution)
        # Native GridPatternCfg(ordering="yx") uses meshgrid(x,y,indexing="ij"):
        # X is the slow index. Keep checkpoint ordering, even though raster rows are Y.
        scan = scan or dict(size=(2.0, 2.0), spacing=0.1, offset=(0.0, 0.0))
        axes = [
            torch.linspace(
                -size / 2, size / 2, round(size / scan["spacing"]) + 1, device=heights.device, dtype=heights.dtype
            )
            + offset
            for size, offset in zip(scan["size"], scan["offset"])
        ]
        x, y = torch.meshgrid(
            *axes,
            indexing="ij",
        )
        self.register_buffer("local_xy", torch.stack((x, y), -1).reshape(-1, 2))

    def query(self, xy):
        coordinates = 2 * (xy - self.origin) / self.extent - 1
        valid = (coordinates.abs() <= 1).all(-1)
        # One shared input raster, not B duplicated copies of the full terrain.
        sampled = F.grid_sample(
            self.heights, coordinates.reshape(1, 1, -1, 2), align_corners=True, padding_mode="zeros"
        ).reshape(xy.shape[:-1])
        return sampled, valid

    def local(self, position, yaw):
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        x, y = self.local_xy.unbind(-1)
        xy = torch.stack((c * x - s * y, s * x + c * y), -1) + position[:, None, :2]
        heights, valid = self.query(xy)
        return (heights - position[:, 2:3]).clamp(-5, 5), valid.all(-1)

    @classmethod
    @torch.no_grad()
    def from_context(cls, context, resolution=0.05):
        terrain = context.env.scene.terrain
        if terrain.cfg.terrain_type == "plane":
            # The flat task also needs explicit finite bounds for rollout validity.
            origins = context.env.scene.env_origins[:, :2]
            lo, hi = origins.amin(0) - 100, origins.amax(0) + 100
        else:
            origins = terrain.terrain_origins[..., :2].reshape(-1, 2)
            generator = terrain.cfg.terrain_generator
            half_tile = origins.new_tensor(generator.size) / 2
            # Cover the existing mesh border, excluding its outermost ray edge.
            border = max(float(generator.border_width) - resolution, 0)
            lo, hi = origins.amin(0) - half_tile - border, origins.amax(0) + half_tile + border
        counts = ((hi - lo) / resolution).ceil().long() + 1
        x = torch.arange(int(counts[0]), device=lo.device) * resolution + lo[0]
        y = torch.arange(int(counts[1]), device=lo.device) * resolution + lo[1]
        # Bound temporary memory independently of raster resolution (1 cm maps
        # have 25 times as many samples as 5 cm maps).
        values = torch.empty(len(y) * len(x), device=lo.device)
        for begin in range(0, values.numel(), 65536):
            indices = torch.arange(begin, min(begin + 65536, values.numel()), device=lo.device)
            chunk = torch.stack((x[indices % len(x)], y[indices // len(x)]), -1)
            height, _, valid = context.query(chunk)
            if not valid.all():
                raise ValueError("Terrain bake contains ray misses; refusing to invent a floor")
            values[begin : begin + len(indices)] = height
        return cls(values.reshape(len(y), len(x)), lo, resolution, getattr(context.env.cfg, "height_scan", None))
