"""Single-robot PGMT MuJoCo playback and interactive terrain/spawn selection."""

import argparse
import numpy as np
import signal
import time
import torch
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import mujoco

from .pgmt_runtime import REFERENCE_DISPLAYS, PGMTMuJoCo, validate_reset_seconds
from .pgmt_scene import TERRAINS, SceneCatalog, ViewerWindow
from .pgmt_terrain import build_scene
from .pgmt_visualization import Dashboard, SceneOverlay, make_snapshot, reference_display_snapshot
from .policy import TerrainPolicy

REPO = Path(__file__).resolve().parents[7]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--motion", type=Path, default=REPO / "artifacts/pgmt_lafan_native_play/data/named/walk2_subject1.named.npz"
    )
    p.add_argument("--contacts", type=Path, default=REPO / "artifacts/pgmt_lafan_native_play/data/contacts")
    p.add_argument("--terrain", choices=list(TERRAINS), default="box_test")
    p.add_argument("--level", type=int, choices=range(10), default=9)
    p.add_argument("--start-frame", type=int, default=1500)
    p.add_argument("--start-xy", type=float, nargs=2, default=(0.0, 0.0))
    p.add_argument("--heading-deg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=6020)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument(
        "--reset-seconds",
        type=validate_reset_seconds,
        default=0.0,
        help="0 disables timed reset; failure/motion-end/tile-exit resets remain active",
    )
    p.add_argument("--paused", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=0, help="0 plays until closed")
    return p


def main():
    args = parser().parse_args()
    if args.speed <= 0 or args.steps < 0:
        raise ValueError("Speed must be positive and steps nonnegative")
    torch.set_num_threads(1)
    policy = TerrainPolicy(args.checkpoint)
    model, _, info = build_scene(args.terrain, args.level, args.seed, policy.env_cfg["sim"]["dt"])
    sim = PGMTMuJoCo(
        model,
        policy.env_cfg,
        args.motion,
        args.contacts,
        info,
        args.start_frame,
        np.deg2rad(args.heading_deg),
        args.start_xy,
        absolute_heading=True,
        surface_placement=True,
        reset_seconds=args.reset_seconds,
    )
    catalog = SceneCatalog(args.motion.parent, args.contacts, args.seed)
    dashboard = Dashboard(args.port, catalog)
    print(f"[POLICY] Native PGMT checkpoint {policy.update}: {policy.path}", flush=True)
    print(f"[DASHBOARD] {dashboard.url} · click the map to choose a spawn point", flush=True)
    print("[VIEWER] Single robot; free camera; G reference, H scan, Space pause, R reset", flush=True)
    commands = deque()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def key(code):
        command = {
            32: "pause",
            82: "reset",
            71: "ghost",
            72: "scan",
            86: "reference_toggle",
            49: "real",
            50: "flat",
            51: "off",
            52: "shuffled",
        }.get(code)
        if command:
            commands.append(command)

    window = None if args.headless else ViewerWindow(model, sim.data, key, sim.root_pos)
    viewer = window.handle if window else None
    overlay = SceneOverlay(sim)
    mode, paused = "real", args.paused
    reference_display = "terrain"
    show_scan = show_ghost = True
    scene_revision = step = 0
    scene_settings = catalog.describe(sim, scene_revision)
    control_status = timer_status = reference_status = last_reset = None
    needs_observation = True
    snapshot = None
    metrics = sim.metrics()
    try:
        while running and (not args.steps or step < args.steps) and (viewer is None or viewer.is_running()):
            started = time.monotonic()
            commands.extend(dashboard.drain())
            while commands:
                command = commands.popleft()
                if isinstance(command, dict):
                    kind = command["command"]
                    if kind == "scene":
                        try:
                            candidate, _, terrain_changed = catalog.build(command["settings"], policy.env_cfg, sim)
                        except (ValueError, OSError, StopIteration) as e:
                            control_status = dict(request_id=command["request_id"], status="error", message=str(e))
                            continue
                        candidate.episodes = sim.episodes + 1
                        if terrain_changed:
                            camera = window.camera() if window else None
                            lookat = viewer.cam.lookat.copy() if viewer else None
                            if window:
                                window.close()
                            window = (
                                None
                                if args.headless
                                else ViewerWindow(
                                    candidate.model, candidate.data, key, candidate.root_pos, camera, lookat
                                )
                            )
                            viewer = window.handle if window else None
                        else:
                            with viewer.lock() if viewer else nullcontext():
                                mujoco.mj_copyData(sim.data, sim.model, candidate.data)
                                candidate.data = sim.data
                        sim, model = candidate, candidate.model
                        overlay = SceneOverlay(sim)
                        scene_revision += 1
                        scene_settings = catalog.describe(sim, scene_revision)
                        paused = command["settings"]["paused"]
                        metrics = sim.metrics()
                        needs_observation = True
                        last_reset = None
                        control_status = dict(
                            request_id=command["request_id"],
                            status="applied",
                            revision=scene_revision,
                            message="已放置并暂停" if paused else "已放置并开始播放",
                        )
                        print(f"[PLACEMENT] {scene_settings}", flush=True)
                    elif kind == "reset_time":
                        sim.reset_seconds = command["seconds"]
                        timer_status = dict(
                            request_id=command["request_id"],
                            status="applied",
                            seconds=sim.reset_seconds,
                            message=f"定时重置：{sim.reset_seconds:g} 秒" if sim.reset_seconds else "定时重置已关闭",
                        )
                    elif kind == "reference_display":
                        reference_display = command["mode"]
                        reference_status = dict(
                            request_id=command["request_id"],
                            status="applied",
                            mode=reference_display,
                            message=REFERENCE_DISPLAYS[reference_display],
                        )
                elif command == "pause":
                    paused = not paused
                elif command == "ghost":
                    show_ghost = not show_ghost
                elif command == "scan":
                    show_scan = not show_scan
                elif command == "reference_toggle":
                    reference_display = "source" if reference_display == "terrain" else "terrain"
                elif command in ("reset", "real", "flat", "off", "shuffled"):
                    if command != "reset":
                        mode = command
                    sim.reset()
                    needs_observation = True
                    metrics = sim.metrics()
                    last_reset = None
            if needs_observation:
                obs = sim.observations()
                result = policy.evaluate(obs, sim.flat_height, sim.action_scale, mode=mode)
                snapshot = make_snapshot(sim, policy, result, obs, metrics, mode, paused, step)
                snapshot["scene"] = scene_settings
                needs_observation = False
            snapshot.update(
                paused=paused,
                control_status=control_status,
                timer_status=timer_status,
                last_reset=last_reset,
                reference_display_status=reference_status,
                reset_seconds=sim.reset_seconds,
            )
            snapshot.update(reference_display_snapshot(sim, reference_display, show_ghost))
            dashboard.update(snapshot)
            if viewer:
                with viewer.lock():
                    overlay.draw(viewer.user_scn, result, show_scan, show_ghost, reference_display)
                viewer.sync()
            if not paused:
                metrics = sim.step(result["actions"])
                step += 1
                needs_observation = True
                if metrics["failures"] or metrics["timeouts"]:
                    last_reset = dict(
                        step=step,
                        episode=sim.episodes,
                        episode_time_s=metrics["episode_time_s"],
                        reasons=metrics["failures"] + metrics["timeouts"],
                    )
                    sim.reset()
                    metrics = sim.metrics()
            time.sleep(max(0, sim.dt / args.speed - (time.monotonic() - started)))
    finally:
        dashboard.close()
        if window:
            window.close()


if __name__ == "__main__":
    main()
