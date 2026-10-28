"""Validated motion/terrain choices and repeatable interactive robot placement."""

from __future__ import annotations

import json
import numpy as np
import threading
from pathlib import Path

import mujoco

from .pgmt_runtime import PGMTMuJoCo
from .pgmt_terrain import build_scene, layout_heights, terrain_layout

TERRAINS = {
    "flat": "平地",
    "stairs": "金字塔台阶",
    "stairs_down": "下凹台阶",
    "boxes": "随机方块",
    "box_test": "箱台测试：10/20/30/40 cm",
}


class SceneCatalog:
    def __init__(self, motion_dir, contacts, seed=42, clearance=0.075):
        self.motion_dir, self.contacts = Path(motion_dir), Path(contacts)
        self.seed, self.clearance = seed, clearance
        manifest = json.loads((self.contacts / "manifest.json").read_text())
        self.motions = {}
        for entry in manifest["files"]:
            name = entry["name"]
            if Path(name).name != name or not (self.motion_dir / name).is_file():
                continue
            category = next(
                (
                    label
                    for prefix, label in [
                        ("dance", "舞蹈"),
                        ("fight", "搏击 / 运动"),
                        ("ground", "地面动作"),
                        ("jumps", "跳跃"),
                        ("run", "跑步"),
                        ("sprint", "冲刺"),
                        ("walk", "步行"),
                    ]
                    if name.startswith(prefix)
                ),
                "其他",
            )
            self.motions[name] = dict(
                name=name,
                frames=int(entry["frames"]),
                fps=float(entry["fps"]),
                seconds=entry["frames"] / entry["fps"],
                category=category,
                ground=entry["behavior"] == "ground",
                max_frame=int(entry["frames"]) - 12,
            )
        self.cache = {}
        self.lock = threading.Lock()

    def public(self):
        return dict(
            motions=list(self.motions.values()),
            terrains=[
                dict(value=k, label=v, fixed_level=9 if k == "box_test" else 0 if k == "flat" else None)
                for k, v in TERRAINS.items()
            ],
            levels=list(range(10)),
            xy_bounds=[-3.9, 3.9],
            seed=self.seed,
        )

    @staticmethod
    def _number(value, name, lower, upper, integer=False):
        if isinstance(value, bool):
            raise ValueError(f"{name} 必须是数值")
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{name} 必须是数值") from None
        if not np.isfinite(number) or not lower <= number <= upper or (integer and not number.is_integer()):
            raise ValueError(f"{name} 必须在 {lower} 到 {upper} 之间" + ("，且为整数" if integer else ""))
        return int(number) if integer else number

    def validate(self, request):
        if not isinstance(request, dict):
            raise ValueError("场景设置必须是一个对象")
        name = request.get("motion")
        if not isinstance(name, str) or name not in self.motions:
            raise ValueError("请从动作列表中选择动作")
        kind = request.get("terrain")
        if not isinstance(kind, str) or kind not in TERRAINS:
            raise ValueError("请从地形列表中选择有效地形")
        if not isinstance(request.get("paused"), bool):
            raise ValueError("需要指定放置后暂停或播放")
        level = self._number(request.get("level"), "难度", 0, 9, True)
        return dict(
            motion=name,
            terrain=kind,
            level=9 if kind == "box_test" else level,
            start_frame=self._number(request.get("start_frame"), "起始帧", 0, self.motions[name]["max_frame"], True),
            x=self._number(request.get("x"), "X", -3.9, 3.9),
            y=self._number(request.get("y"), "Y", -3.9, 3.9),
            heading_deg=self._number(request.get("heading_deg"), "朝向", -180, 180),
            paused=request["paused"],
        )

    def preview(self, kind, level):
        if kind not in TERRAINS:
            raise ValueError("未知地形")
        level = self._number(level, "难度", 0, 9, True)
        if kind == "box_test":
            level = 9
        key = (kind, level)
        with self.lock:
            if key not in self.cache:
                info = terrain_layout(kind, level, self.seed)
                axis = np.linspace(-4, 4, 81)
                x, y = np.meshgrid(axis, axis, indexing="xy")
                height = layout_heights(info, np.stack((x, y), -1))
                self.cache[key] = dict(
                    terrain=kind,
                    level=level,
                    seed=self.seed,
                    extent=[-4, 4],
                    heights=height.tolist(),
                    plane_height=info["plane_height_m"],
                    boxes=[
                        dict(
                            x=b["center"][0],
                            y=b["center"][1],
                            sx=b["dimensions"][0],
                            sy=b["dimensions"][1],
                            top=b["center"][2] + b["dimensions"][2] / 2,
                        )
                        for b in info["boxes"]
                    ],
                    detail={
                        k: v
                        for k, v in info.items()
                        if k
                        in (
                            "step_height_m",
                            "step_width_m",
                            "grid_width_m",
                            "grid_height_range_m",
                            "test_boxes",
                            "box_heights_m",
                            "box_footprint_m",
                            "box_center_spacing_m",
                            "fixed_geometry",
                        )
                    },
                )
            return self.cache[key]

    def describe(self, sim, revision):
        name = Path(sim.motion.path).name
        return dict(
            revision=revision,
            motion=name,
            **{k: v for k, v in self.motions[name].items() if k != "name"},
            terrain=sim.terrain_info["kind"],
            level=sim.terrain_info["level"],
            start_frame=sim.start_frame,
            seed=self.seed,
            **sim.spawn_info,
        )

    def build(self, request, cfg, previous):
        settings = self.validate(request)
        changed = (
            settings["terrain"] != previous.terrain_info["kind"] or settings["level"] != previous.terrain_info["level"]
        )
        if changed:
            model, xml, info = build_scene(settings["terrain"], settings["level"], self.seed, cfg["sim"]["dt"])
        else:
            model, xml, info = previous.model, None, previous.terrain_info
        sim = PGMTMuJoCo(
            model,
            cfg,
            self.motion_dir / settings["motion"],
            self.contacts,
            info,
            settings["start_frame"],
            np.deg2rad(settings["heading_deg"]),
            (settings["x"], settings["y"]),
            self.clearance,
            absolute_heading=True,
            surface_placement=True,
            reset_seconds=previous.reset_seconds,
        )
        return sim, xml, changed


class ViewerWindow:
    """Use public passive-viewer lifecycle; join its renderer before opening another."""

    def __init__(self, model, data, callback, root_pos, camera=None, lookat=None):
        from mujoco import viewer

        before = set(threading.enumerate())
        self.handle = viewer.launch_passive(model, data, key_callback=callback, show_left_ui=False, show_right_ui=False)
        self.threads = [t for t in threading.enumerate() if t not in before and "_launch_internal" in t.name]
        with self.handle.lock():
            self.handle.cam.azimuth, self.handle.cam.elevation, self.handle.cam.distance = camera or (135, -33, 4.3)
            self.handle.cam.lookat[:] = np.asarray(root_pos) + [0, 0, -0.15] if lookat is None else lookat
            self.handle.opt.geomgroup[3] = 0
            self.handle.opt.label = mujoco.mjtLabel.mjLABEL_SELECTION

    def close(self):
        self.handle.close()
        for thread in self.threads:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("MuJoCo renderer did not close before scene replacement")

    def camera(self):
        with self.handle.lock():
            return (self.handle.cam.azimuth, self.handle.cam.elevation, self.handle.cam.distance)
