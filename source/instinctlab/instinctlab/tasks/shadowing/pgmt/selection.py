"""Motion selection and the pinned terrain generator's column-to-family mapping."""

import math
from pathlib import Path

TERRAIN_FAMILIES = {"flat": 0, "slope_up": 1, "slope_down": 1, "stairs_up": 2, "stairs_down": 2, "boxes": 3, "rough": 4}


def select_motion_files(files, names):
    """Select exact motion names in manifest order; None preserves the full manifest."""
    if names is None:
        return list(files)

    def stem(name):
        return Path(name).name.removesuffix(".npz").removesuffix(".named")

    selected = [stem(name) for name in names]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("motion_names must be nonempty and unique, or null for all motions")
    missing = set(selected) - {stem(name) for name in files}
    if missing:
        raise ValueError(f"Motion names absent from the manifest: {sorted(missing)}")
    return [name for name in files if stem(name) in selected]


def terrain_column_names(sub_terrains, num_cols):
    """Match Isaac Lab's curriculum columns, including zero-proportion terrains."""
    names = list(sub_terrains)
    weights = [float(sub_terrains[name].proportion) for name in names]
    if num_cols <= 0 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Terrain proportions must be finite, nonnegative and have a positive sum")
    if set(names) - TERRAIN_FAMILIES.keys():
        raise ValueError("PGMT terrain families must have an explicit name-to-family mapping")
    cumulative, total, denominator = [], 0.0, sum(weights)
    for weight in weights:
        total += weight / denominator
        cumulative.append(total)
    # This is the exact selection rule in the pinned TerrainGenerator.
    columns = [
        names[next(i for i, end in enumerate(cumulative) if col / num_cols + 0.001 < end)] for col in range(num_cols)
    ]
    absent = {name for name, weight in zip(names, weights) if weight > 0} - set(columns)
    if absent:
        raise ValueError(f"Increase num_cols to represent these nonzero terrain proportions: {sorted(absent)}")
    return columns
