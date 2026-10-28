"""Actual attention tensors, terrain scan overlays, and a local live dashboard."""

from __future__ import annotations

import json
import numpy as np
import threading
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco

from .pgmt_runtime import REFERENCE_DISPLAYS, validate_reset_seconds
from .pgmt_terrain import yaw_rotation

COLORS = np.array([[1.0, 0.69, 0.20], [0.21, 0.85, 0.65], [0.66, 0.52, 1.0], [1.0, 0.39, 0.52]])


class Dashboard:
    def __init__(self, port, catalog=None):
        self.snapshot = {}
        self.commands = deque()
        self.lock = threading.Lock()
        dashboard = self
        page = (Path(__file__).with_name("pgmt_dashboard.html")).read_bytes()
        controls = (Path(__file__).with_name("pgmt_controls.js")).read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def reply(self, value, code=200):
                payload = json.dumps(value, allow_nan=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path.split("?")[0] == "/catalog" and catalog:
                    self.reply(catalog.public())
                    return
                if self.path.split("?")[0] == "/preview" and catalog:
                    try:
                        query = parse_qs(urlparse(self.path).query)
                        self.reply(catalog.preview(query.get("terrain", [""])[0], query.get("level", [""])[0]))
                    except ValueError as error:
                        self.reply(dict(error=str(error)), 400)
                    return
                if self.path.split("?")[0] == "/state":
                    with dashboard.lock:
                        payload = json.dumps(dashboard.snapshot, allow_nan=False).encode()
                    mime = "application/json"
                elif self.path.split("?")[0] == "/":
                    payload, mime = page, "text/html; charset=utf-8"
                elif self.path.split("?")[0] == "/controls.js":
                    payload, mime = controls, "text/javascript; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if self.path != "/control" or not 0 < length <= 4096:
                        raise ValueError("无效控制请求")
                    request = json.loads(self.rfile.read(length))
                    value = request["command"]
                    if value == "scene" and catalog:
                        value = dict(
                            command="scene",
                            settings=catalog.validate(request.get("settings")),
                            request_id=uuid.uuid4().hex,
                        )
                    elif value == "reset_time":
                        value = dict(
                            command="reset_time",
                            seconds=validate_reset_seconds(request.get("seconds")),
                            request_id=uuid.uuid4().hex,
                        )
                    elif value == "reference_display":
                        mode = request.get("mode")
                        if not isinstance(mode, str) or mode not in REFERENCE_DISPLAYS:
                            raise ValueError("参考显示必须是 terrain（地形适配）或 source（原始 NPZ）")
                        value = dict(command="reference_display", mode=mode, request_id=uuid.uuid4().hex)
                    elif value not in (
                        "pause",
                        "reset",
                        "real",
                        "flat",
                        "off",
                        "shuffled",
                        "ghost",
                        "scan",
                        "reference_toggle",
                    ):
                        raise ValueError("未知控制命令")
                    with dashboard.lock:
                        dashboard.commands.append(value)
                except (KeyError, ValueError, TypeError) as error:
                    self.reply(dict(error=str(error)), 400)
                    return
                if isinstance(value, dict):
                    self.reply(dict(request_id=value["request_id"], status="queued"), 202)
                    return
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def update(self, snapshot):
        with self.lock:
            self.snapshot = snapshot

    def drain(self):
        with self.lock:
            commands = list(self.commands)
            self.commands.clear()
        return commands

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class SceneOverlay:
    def __init__(self, simulation):
        self.sim = simulation
        self.reference = mujoco.MjData(simulation.model)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[2] = 1
        self.perturbation = mujoco.MjvPerturb()

    def draw(self, scene, result, show_scan=True, show_ghost=True, reference_display="terrain"):
        sim = self.sim
        scene.ngeom = 0
        if show_ghost:
            pos, quat = sim.display_reference(reference_display)
            self.reference.qpos[:] = sim.model.qpos0
            self.reference.qpos[:3] = pos[sim.pelvis]
            self.reference.qpos[3:7] = quat[sim.pelvis]
            self.reference.qpos[sim.qpos_ids] = sim.motion["joint_pos"][sim.frame]
            mujoco.mj_forward(sim.model, self.reference)
            mujoco.mjv_addGeoms(
                sim.model, self.reference, self.options, self.perturbation, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
            )
            for i in range(scene.ngeom):
                scene.geoms[i].rgba[:] = [0.25, 0.95, 0.85, 0.24]
        for box in sim.terrain_info.get("test_boxes", []):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("MuJoCo overlay capacity exceeded")
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                np.array([box["x"], box["y"], box["height_m"] + 0.10]),
                np.eye(3).flatten(),
                np.ones(4, dtype=np.float32),
            )
            geom.label = box["label"]
            scene.ngeom += 1
        if not show_scan:
            return

        def sphere(pos, radius, color, label=""):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("MuJoCo overlay capacity exceeded")
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius),
                np.asarray(pos),
                np.eye(3).flatten(),
                np.asarray(color, dtype=np.float32),
            )
            geom.label = label
            scene.ngeom += 1

        def line(a, b, color, width=0.007):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.asarray(color, dtype=np.float32),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a), np.asarray(b))
            scene.ngeom += 1

        points = sim.scan_points.copy()
        points[:, 2] += 0.014
        span = max(0.02, np.ptp(points[:, 2]))
        h = (points[:, 2] - points[:, 2].min()) / span
        for p, value in zip(points, h):
            # Scan colors encode ground height, NOT attention.
            sphere(p, 0.012, [0.18 + 0.30 * value, 0.52 + 0.30 * value, 0.85 - 0.25 * value, 0.64])
        weights = result["intent_weights"][:, 0, -4:].mean(0)
        rotation = yaw_rotation(sim.root_quat)
        centers = result["glimpse_centers"][
            ..., ::-1
        ]  # Native scan is x-major; grid_sample coordinates are reversed in physical XY.
        offsets = np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2], [-0.2, -0.2]])
        for k, (center, mass) in enumerate(zip(centers, weights)):
            # Query the visible terrain at the patch edges. The policy itself uses
            # grid_sample on the 21x21 scan; its actual bilinear values are in the dashboard.
            local = center + offsets
            world_xy = local @ rotation.T + sim.root_pos[:2]
            corners = np.column_stack((world_xy, sim.scanner.heights(sim.data, world_xy) + 0.034 + k * 0.003))
            rgba = np.r_[COLORS[k], 0.95]
            for a, b in zip(corners[:-1], corners[1:]):
                line(a, b, rgba, 0.006 + 0.025 * min(float(mass), 0.2))
            xy = center @ rotation.T + sim.root_pos[:2]
            z = sim.scanner.heights(sim.data, xy[None])[0]
            sphere([*xy, z + 0.045], 0.027, rgba, f"T{k} {mass*100:.2f}%")
        root = sim.root_pos
        arrow_start = np.array([root[0], root[1], sim.flat_height + root[2] + 0.025])
        arrow_end = arrow_start.copy()
        arrow_end[:2] += rotation[:, 0] * 0.45
        line(arrow_start, arrow_end, [0.98, 0.98, 0.98, 1.0], 0.009)


def reference_display_snapshot(sim, mode, visible):
    pos, _ = sim.display_reference(mode)
    return dict(
        reference_display=mode,
        show_ghost=visible,
        reference_ghost=dict(
            pelvis_z_m=float(pos[sim.pelvis, 2]),
            npz_pelvis_z_m=float(sim.motion["body_pos_w"][sim.frame, sim.pelvis, 2]),
            support_z_m=float(sim.support_z),
        ),
    )


def make_snapshot(sim, policy, result, observations, metrics, mode, paused, step):
    weights = result["intent_weights"][:, 0]
    rotation = yaw_rotation(sim.root_quat)
    return dict(
        step=step,
        checkpoint_update=policy.update,
        mode=mode,
        paused=paused,
        reset_seconds=sim.reset_seconds,
        terrain=sim.terrain_info["kind"],
        level=sim.terrain_info["level"],
        height=observations["terrain"].reshape(21, 21).T.tolist(),
        centers=result["glimpse_centers"][..., ::-1].tolist(),
        patches=result["glimpse_patches"].reshape(4, 5, 5).transpose(0, 2, 1).reshape(4, 25).tolist(),
        weights=weights.tolist(),
        history=result["history_weights"][:, 0].tolist(),
        terrain_mass=float(weights[:, -4:].sum(-1).mean()),
        metrics=metrics,
        counterfactual=result["counterfactual"],
        root_pos=sim.root_pos.tolist(),
        colors=COLORS.tolist(),
        root_heading_deg=float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        scan_definition="terrain world z minus pelvis z; yaw aligned; [-5,5] m",
        patch_definition="4 learned 5x5 patches, side 0.4 m; NOT 441 attention weights",
    )
