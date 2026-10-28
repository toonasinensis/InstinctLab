"""Check native actor parity, actual scan ordering and selectable MuJoCo placements."""

import json
import numpy as np
import torch
from pathlib import Path

from isaaclab.sensors import patterns

from instinctlab.tasks.shadowing.pgmt.msic.pgmt_runtime import PGMTMuJoCo
from instinctlab.tasks.shadowing.pgmt.msic.pgmt_scene import SceneCatalog
from instinctlab.tasks.shadowing.pgmt.msic.pgmt_terrain import TerrainScanner, build_scene, layout_heights
from instinctlab.tasks.shadowing.pgmt.msic.policy import TerrainPolicy


def main():
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "artifacts/pgmt_lafan_terrain_play/model_9600/model_9600.pt"
    p = TerrainPolicy(checkpoint)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    obs = dict(prop=torch.randn(16, 930), rbt_cmd_mf=torch.randn(16, 748), terrain=torch.randn(16, 441))
    with torch.inference_mode():
        expected = p.actor(p.normalize(obs))
        actual = p._forward(obs)["actions"]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
    report = dict(checkpoint=str(checkpoint), native_actor_max_action_error=float((actual - expected).abs().max()))
    model, _, info = build_scene("flat", 0)
    pattern = p.env_cfg["scene"]["height_scanner"]["pattern_cfg"]
    starts, _ = patterns.grid_pattern(
        patterns.GridPatternCfg(resolution=pattern["resolution"], size=pattern["size"], ordering=pattern["ordering"]),
        "cpu",
    )
    scanner = TerrainScanner(model)
    np.testing.assert_allclose(scanner.xy, starts.numpy()[:, :2], atol=2e-7)
    report["scan_order_matches_native"] = True
    contacts = root / "artifacts/pgmt_lafan_native_play/data/contacts"
    motion = root / "artifacts/pgmt_lafan_native_play/data/named/walk2_subject1.named.npz"
    sim = PGMTMuJoCo(
        model,
        p.env_cfg,
        motion,
        contacts,
        info,
        1500,
        0,
        (0, 0),
        surface_placement=True,
        absolute_heading=True,
        reset_seconds=0,
    )
    catalog = SceneCatalog(motion.parent, contacts)
    report["catalog_motions"] = len(catalog.motions)
    placements = []
    for kind, level, x, y, yaw in [
        ("flat", 0, 1.2, -0.7, 90),
        ("stairs", 3, 0, 0, -90),
        ("stairs_down", 5, 0, 0, 30),
        ("boxes", 2, 0, 0, 0),
        ("box_test", 9, 1.25, 1.25, 45),
    ]:
        request = dict(
            motion=motion.name, terrain=kind, level=level, start_frame=1500, x=x, y=y, heading_deg=yaw, paused=True
        )
        candidate, _, _ = catalog.build(request, p.env_cfg, sim)
        np.testing.assert_allclose(candidate.root_pos[:2], (x, y), atol=1e-8)
        np.testing.assert_allclose(candidate.spawn_info["heading_deg"], yaw, atol=1e-4)
        obs = candidate.observations()
        result = p.evaluate(obs, candidate.flat_height, candidate.action_scale)
        assert np.isfinite(result["actions"]).all()
        xy = np.array([[0.0, 0.0], [1.25, 1.25], [-1.25, -1.25], [2.5, 0.0]])
        np.testing.assert_allclose(
            candidate.scanner.heights(candidate.data, xy), layout_heights(candidate.terrain_info, xy), atol=1e-6
        )
        for _ in range(10):
            candidate.step(result["actions"])
            obs = candidate.observations()
            result = p.evaluate(obs, candidate.flat_height, candidate.action_scale)
        placements.append(dict(terrain=kind, placement="pass", control_steps=10, xy=[x, y], heading_deg=yaw))
        sim = candidate
    report["placements"] = placements
    output = root / "agents/pgmt_mujoco_validation_20261028/report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
