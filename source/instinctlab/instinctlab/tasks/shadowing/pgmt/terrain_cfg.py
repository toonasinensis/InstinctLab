"""PGMT-inspired terrain draft; geometric values are ours, not paper-reported.

Explicit import only: existing training tasks are unchanged.
Rows map exactly to L0--L9 despite Isaac Lab's within-row difficulty jitter.
"""

import math
import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.trimesh import mesh_terrains


def level_fraction(difficulty: float) -> float:
    return min(9, max(0, int(math.floor(difficulty * 10)))) / 9.0


def level_slope(difficulty, cfg):
    return hf_terrains.pyramid_sloped_terrain(level_fraction(difficulty), cfg)


def level_stairs(difficulty, cfg):
    return mesh_terrains.pyramid_stairs_terrain(level_fraction(difficulty), cfg)


def level_inverted_stairs(difficulty, cfg):
    """Concave stairs: the central platform is below the surrounding rim."""
    return mesh_terrains.inverted_pyramid_stairs_terrain(level_fraction(difficulty), cfg)


def level_boxes(difficulty, cfg):
    meshes, origin = mesh_terrains.random_grid_terrain(level_fraction(difficulty), cfg)
    # Isaac Lab computes box offsets in float32, then adds float64 widths.
    # Adjacent boxes can consequently have sub-micrometre gaps. A vertical
    # height ray exactly on a seam then misses even though holes=False.
    # Reconstruct shared XY edges from the same integer grid coordinate in
    # float64. Heights, topology, sampling RNG and the central platform stay
    # unchanged. The pinned generator returns border meshes, grid, platform.
    grid = meshes[-2]
    border = cfg.size[0] - int(cfg.size[0] / cfg.grid_width) * cfg.grid_width
    xy = grid.vertices[:, :2]
    exact = np.rint((xy - border / 2) / cfg.grid_width) * cfg.grid_width + border / 2
    if np.max(np.abs(exact - xy)) > 2e-6:
        raise RuntimeError("Unexpected random-grid mesh layout while repairing shared edges")
    vertices = grid.vertices.copy()
    vertices[:, :2] = exact
    grid.vertices = vertices
    return meshes, origin


def level_rough(difficulty, cfg):
    # Isaac Lab's uniform rough terrain otherwise ignores difficulty.
    amplitude = 0.005 + 0.045 * level_fraction(difficulty)
    local_cfg = cfg.copy()
    local_cfg.noise_range = (-amplitude, amplitude)
    return hf_terrains.random_uniform_terrain(difficulty, local_cfg)


def make_pgmt_terrain_cfg(seed: int = 42):
    """100 tiles: 10 levels x 10 columns, each of five families has 20%."""
    return terrain_gen.TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=(8.0, 8.0),
        border_width=10.0,
        num_rows=10,
        num_cols=10,
        difficulty_range=(0.0, 1.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
            "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
                function=level_slope, proportion=0.1, slope_range=(0.03, 0.30), platform_width=2.0, border_width=0.25
            ),
            "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
                function=level_slope,
                proportion=0.1,
                slope_range=(0.03, 0.30),
                platform_width=2.0,
                border_width=0.25,
                inverted=True,
            ),
            "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
                function=level_stairs,
                proportion=0.1,
                step_height_range=(0.02, 0.20),
                step_width=0.30,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
            "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                function=level_inverted_stairs,
                proportion=0.1,
                step_height_range=(0.02, 0.20),
                step_width=0.30,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                function=level_boxes,
                proportion=0.2,
                grid_width=0.45,
                grid_height_range=(0.02, 0.25),
                platform_width=2.0,
                holes=False,
            ),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                function=level_rough,
                proportion=0.2,
                noise_range=(-0.005, 0.005),
                noise_step=0.005,
                downsampled_scale=0.20,
                border_width=0.25,
            ),
        },
    )
