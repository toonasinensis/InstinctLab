"""G1 scene with analytic box terrain and genuine MuJoCo terrain-only ray casts.

Stair dimensions follow Isaac Lab's pyramid_stairs_terrain (BSD-3-Clause,
Copyright 2022-2026 Isaac Lab Project Developers). Boxes follow the PGMT
level_boxes settings. Random realizations are local, not the remote training mesh.
"""

from __future__ import annotations

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .mujoco_robot import G1_MJCF


def terrain_layout(kind="stairs", level=2, seed=42):
    """One geometry description shared by MuJoCo and the placement preview."""
    if kind not in ("flat", "stairs", "stairs_down", "boxes", "box_test") or not 0 <= level <= 9:
        raise ValueError("Terrain must be flat/stairs/stairs_down/boxes/box_test and level must be 0..9")
    if kind == "box_test":
        # Geometry is fixed; retain L9 lower-body tracking tolerance for this obstacle test.
        level = 9
    boxes = []
    plane_height = 0.0

    def box(name, center, dimensions, color=(0.44, 0.49, 0.55, 1)):
        if min(dimensions) <= 0:
            raise ValueError(f"Invalid box dimensions: {dimensions}")
        boxes.append(dict(name=name, center=list(center), dimensions=list(dimensions), rgba=list(color)))

    center_height = 0.0
    info = dict(
        kind=kind,
        level=level,
        seed=seed,
        tile_size_m=8.0,
        geometry="primitive boxes; no convexification of nonconvex terrain",
        random_realization="local seed, not the remote training mesh",
    )
    if kind == "box_test":
        test_boxes = []
        for height, (x, y), color in zip(
            (0.10, 0.20, 0.30, 0.40),
            ((-1.25, -1.25), (1.25, -1.25), (-1.25, 1.25), (1.25, 1.25)),
            ((0.22, 0.69, 0.52, 1), (0.23, 0.53, 0.87, 1), (0.94, 0.64, 0.20, 1), (0.84, 0.32, 0.39, 1)),
        ):
            label = f"{round(height*100)} cm"
            box(f"test_box_{round(height*100)}cm", (x, y, height / 2), (1.0, 1.0, height), color)
            test_boxes.append(dict(x=x, y=y, height_m=height, label=label, rgba=list(color)))
        info.update(
            test_boxes=test_boxes,
            box_heights_m=[0.10, 0.20, 0.30, 0.40],
            box_footprint_m=[1.0, 1.0],
            box_center_spacing_m=2.5,
            layout="2x2 separated boxes on a flat floor; central 1.5 m corridors",
            fixed_geometry=True,
            random_realization="deterministic, seed-independent test layout",
            tolerance_level=9,
        )
    elif kind.startswith("stairs"):
        h, width, border, platform = 0.02 + 0.18 * level / 9, 0.30, 1.0, 2.0
        n = int((8 - 2 * border - platform) // (2 * width) + 1)
        downward = kind == "stairs_down"
        center_height = (n + 1) * h * (-1 if downward else 1)
        plane_height = -2.0 if downward else 0.0
        if downward:
            # Tile border at zero; the underlying plane must not fill the pit.
            for i, (p, d) in enumerate(
                [
                    ((0, 3.5, -0.1), (8, 1, 0.2)),
                    ((0, -3.5, -0.1), (8, 1, 0.2)),
                    ((3.5, 0, -0.1), (1, 6, 0.2)),
                    ((-3.5, 0, -0.1), (1, 6, 0.2)),
                ]
            ):
                box(f"border_{i}", p, d)
        for k in range(n):
            side = 6 - 2 * k * width
            offset = 3 - (k + 0.5) * width
            top = -(k + 1) * h if downward else (k + 1) * h
            bottom = -(n + 1) * h if downward else -h
            for j, (x, y, sx, sy) in enumerate(
                [
                    (0, offset, side, width),
                    (0, -offset, side, width),
                    (offset, 0, width, side - 2 * width),
                    (-offset, 0, width, side - 2 * width),
                ]
            ):
                box(
                    f"step_{k}_{j}",
                    (x, y, (top + bottom) / 2),
                    (sx, sy, top - bottom),
                    (0.36 + 0.035 * (k % 2), 0.42 + 0.035 * (k % 2), 0.48 + 0.035 * (k % 2), 1),
                )
        bottom = center_height - h if downward else -h
        box(
            "platform",
            (0, 0, (center_height + bottom) / 2),
            (6 - 2 * n * width, 6 - 2 * n * width, center_height - bottom),
        )
        info.update(
            step_height_m=h,
            step_width_m=width,
            nominal_platform_width_m=platform,
            actual_platform_width_m=6 - 2 * n * width,
            num_steps=n,
        )
    elif kind == "boxes":
        width, amplitude = 0.45, 0.02 + 0.23 * level / 9
        n = int(8 / width)
        rng = np.random.default_rng(seed)
        plane_height = -1.0
        for i in range(n):
            for j in range(n):
                x, y = (np.array([i, j]) + 0.5 - n / 2) * width
                top = float(rng.uniform(-amplitude, amplitude))
                box(
                    f"block_{i}_{j}",
                    (x, y, (top - 1) / 2),
                    (width, width, top + 1),
                    (0.37 + top * 0.5, 0.44 + top * 0.5, 0.48 + top * 0.5, 1),
                )
        border = (8 - n * width) / 2
        for i, (p, d) in enumerate(
            [
                ((0, 4 - border / 2, -0.5), (8, border, 1)),
                ((0, -4 + border / 2, -0.5), (8, border, 1)),
                ((4 - border / 2, 0, -0.5), (border, 8 - 2 * border, 1)),
                ((-4 + border / 2, 0, -0.5), (border, 8 - 2 * border, 1)),
            ]
        ):
            box(f"border_{i}", p, d)
        center_height = amplitude
        box("platform", (0, 0, (amplitude - 1) / 2), (2, 2, 1 + amplitude))
        info.update(grid_width_m=width, grid_height_range_m=[-amplitude, amplitude], platform_width_m=2.0)
    if kind in ("boxes", "stairs_down"):
        # Negative cells/pit need a lower base plane, but rays beyond the tile
        # must see a zero-height apron, not an artificial one/two-metre cliff.
        for i, (p, d) in enumerate(
            [
                ((0, 12, -0.1), (40, 16, 0.2)),
                ((0, -12, -0.1), (40, 16, 0.2)),
                ((12, 0, -0.1), (16, 8, 0.2)),
                ((-12, 0, -0.1), (16, 8, 0.2)),
            ]
        ):
            box(f"outside_apron_{i}", p, d)
        info["outside_tile"] = "flat zero-height apron; episode terminates when pelvis leaves the 8x8 tile"
    info.update(center_height_m=center_height, boxes=boxes, plane_height_m=plane_height)
    return info


def layout_heights(info, xy):
    """Exact vertical intersection with the same axis-aligned boxes as the sim."""
    xy = np.asarray(xy, dtype=float)
    height = np.full(xy.shape[:-1], info["plane_height_m"])
    for box in info["boxes"]:
        center, size = np.array(box["center"]), np.array(box["dimensions"])
        inside = (np.abs(xy - center[:2]) <= size[:2] / 2).all(-1)
        height[inside] = np.maximum(height[inside], center[2] + size[2] / 2)
    return height


def build_scene(kind="stairs", level=2, seed=42, dt=0.005, mjcf=G1_MJCF):
    info = terrain_layout(kind, level, seed)
    root = ET.parse(mjcf).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((Path(mjcf).parent / compiler.get("meshdir")).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(dt))
    option.set("integrator", "implicitfast")
    for contact in root.findall("contact"):
        for pair in list(contact):
            if pair.tag == "pair" and pair.get("geom2") == "floor":
                contact.remove(pair)
    for default in root.iter("default"):
        if default.get("class") == "collision":
            default.find("geom").set("friction", "1.0 0.005 0.0001")
    floor = next(g for g in root.iter("geom") if g.get("name") == "floor")
    for key, value in dict(
        group="0", contype="1", friction="1.0 0.005 0.0001", pos=f"0 0 {info['plane_height_m']}"
    ).items():
        floor.set(key, value)
    world = root.findall("worldbody")[-1]
    for box in info["boxes"]:
        ET.SubElement(
            world,
            "geom",
            name=box["name"],
            type="box",
            group="0",
            contype="1",
            conaffinity="1",
            pos=" ".join(map(str, box["center"])),
            size=" ".join(str(s / 2) for s in box["dimensions"]),
            rgba=" ".join(map(str, box["rgba"])),
            friction="1.0 0.005 0.0001",
        )
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    # Nominal, unrandomized joints; Isaac default joint friction is zero.
    model.dof_frictionloss[:] = 0
    model.dof_damping[:] = 0
    return model, xml, info


def yaw_rotation(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class TerrainScanner:
    def __init__(self, model):
        self.model = model
        x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21), indexing="ij")
        self.xy = np.stack((x, y), -1).reshape(441, 2)
        self.groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.direction = np.array([0.0, 0.0, -1.0])

    def heights(self, data, xy, ray_top=20.0):
        xy = np.asarray(xy, dtype=np.float64)
        result = np.empty(xy.shape[:-1], dtype=np.float64)
        hit_id = np.zeros(1, dtype=np.int32)
        for point, out in zip(xy.reshape(-1, 2), result.reshape(-1, 1)):
            start = np.array([point[0], point[1], ray_top], dtype=np.float64)
            distance = mujoco.mj_ray(self.model, data, start, self.direction, self.groups, True, -1, hit_id)
            if distance < 0 or self.model.geom_bodyid[hit_id[0]] != 0:
                raise RuntimeError(f"Terrain ray missed or hit robot at {point}: {hit_id[0]}")
            out[0] = ray_top - distance
        return result

    def scan(self, data, root_pos, root_quat):
        xy = self.xy @ yaw_rotation(root_quat).T + root_pos[:2]
        height = self.heights(data, xy, ray_top=float(root_pos[2] + 20))
        points = np.column_stack((xy, height))
        return np.clip(height - root_pos[2], -5, 5).astype(np.float32)[None], points
