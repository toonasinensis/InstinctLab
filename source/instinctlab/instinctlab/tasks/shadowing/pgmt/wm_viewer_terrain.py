"""CPU-only, exact-geometry repair of legacy centroid-cropped viewer terrain."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGenerator

from instinctlab.utils import instinct_rl_cli as cli

from .wm_visualization import (
    ArrayFile,
    digest,
    terrain_component_centers,
    terrain_description,
    terrain_tile_faces,
)


def read_array(raw, description):
    return np.frombuffer(
        raw, dtype="<" + description["dtype"], count=int(np.prod(description["shape"])),
        offset=description["offset"],
    ).reshape(description["shape"])


def triangle_keys(points, faces):
    # Preserve vertex order/winding. Both meshes originate from the same generator.
    return np.ascontiguousarray(points[faces], dtype="<f4").reshape(-1, 9).view("V36").ravel()


def boundary_edges(points, faces):
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    return int(np.count_nonzero(np.bincount(mesh.edges_unique_inverse) == 1))


def regression_check():
    """The old crop opens a closed slab; selecting its whole component must not."""
    box = trimesh.creation.box(extents=[8, 8, 1])
    neighbor = box.copy()
    neighbor.apply_translation([8, 0, 0])
    mesh = trimesh.util.concatenate([box, neighbor])
    owners = terrain_component_centers(mesh.vertices, mesh.faces)
    selected = terrain_tile_faces(mesh.faces, owners, [0, 0, 0], [8, 8])
    centers = mesh.vertices[mesh.faces].mean(1)
    broken = mesh.faces[(np.abs(centers[:, :2]) <= 2.5).all(1)]
    assert len(selected) == 12 and boundary_edges(mesh.vertices, selected) == 0
    assert boundary_edges(mesh.vertices, broken) > 0
    return dict(old_boundary_edges=boundary_edges(mesh.vertices, broken), new_boundary_edges=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory; preserve the original capture")
    original = args.capture / "manifest.json"
    manifest = json.loads(original.read_text())
    checkpoint = Path(manifest["checkpoint"])
    env_yaml = cli.load_saved_yaml(str(checkpoint.parent / "params/env.yaml"))
    agent_yaml = cli.load_saved_yaml(str(checkpoint.parent / "params/agent.yaml"))
    cfg, _ = cli.resolve_play_cfg(
        "Instinct-PGMT-TerrainWMValidation-G1-Newton-v0", [], env_yaml=env_yaml, agent_yaml=agent_yaml,
    )
    np.random.seed(manifest["scene_seed"])
    generator = TerrainGenerator(cfg.scene.terrain.terrain_generator, device="cpu")
    points = np.asarray(generator.terrain_mesh.vertices, dtype="<f4")
    faces = np.asarray(generator.terrain_mesh.faces, dtype="<u4")
    keys = np.unique(triangle_keys(points, faces))
    old_bytes = (args.capture / "meshes.bin").read_bytes()
    old_meshes = []
    for case in manifest["cases"]:
        previous = case["terrain"]
        old_points = read_array(old_bytes, previous["vertices"])
        old_faces = read_array(old_bytes, previous["faces"])
        # Exact float32 positions and face winding, not a visual approximation.
        match = np.isin(triangle_keys(old_points, old_faces), keys)
        if not match.all():
            raise ValueError(f"CPU terrain differs from original capture: {case['name']}, match={match.mean()}")
        old_meshes.append((old_points, old_faces))
    owners = terrain_component_centers(points, faces)
    regression = regression_check()
    shutil.copytree(args.capture, args.output)
    mesh_path = args.output / "meshes.bin"
    # Robot geometry is the unchanged prefix; only terrain arrays are replaced.
    robot_end = min(case["terrain"]["vertices"]["offset"] for case in manifest["cases"])
    temporary = args.output / "meshes-repaired.bin"
    arrays = ArrayFile(temporary)
    arrays.stream.write(old_bytes[:robot_end])
    results = []
    for case, (old_points, old_faces) in zip(manifest["cases"], old_meshes):
        chosen = terrain_tile_faces(faces, owners, case["origin"], generator.cfg.size)
        vertices, remap = np.unique(chosen, return_inverse=True)
        new_points, new_faces = points[vertices], remap.reshape(-1, 3)
        before, after = boundary_edges(old_points, old_faces), boundary_edges(new_points, new_faces)
        if case["family"] in ("stairs_up", "stairs_down") and after != 0:
            raise AssertionError(f"Stair boxes are still torn: {case['name']} has {after} boundary edges")
        case["terrain"] = dict(vertices=arrays.add(new_points), faces=arrays.add(new_faces, "u4"))
        case["display_terrain"] = terrain_description(case["family"], case["level"], generator.cfg)
        results.append(dict(
            case=case["name"], old_triangles=len(old_faces), new_triangles=len(new_faces),
            old_boundary_edges=before, new_boundary_edges=after, exact_old_triangle_match_fraction=1.0,
        ))
    arrays.stream.close()
    temporary.replace(mesh_path)
    if mesh_path.read_bytes()[:robot_end] != old_bytes[:robot_end]:
        raise AssertionError("Terrain repair changed robot geometry")
    before_data, after_data = digest(args.capture / "data.bin"), digest(args.output / "data.bin")
    if before_data != after_data:
        raise AssertionError("Terrain repair changed trajectory data")
    manifest["display_terrain_revision"] = 2
    manifest["display_terrain_repair"] = dict(
        original_manifest_sha256=digest(original), repair_source_sha256=digest(__file__),
        method="complete connected terrain components by assigned tile; no per-triangle crop",
        cpu_only=True, data_sha256=after_data, exact_original_triangles_matched=True,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for path in (Path(__file__).parent / "wm_viewer").iterdir():
        if path.is_file():
            shutil.copy2(path, args.output / path.name)
    report = dict(
        status="PASS", cpu_only=True, simulation_steps=0, optimizer_updates=0,
        original_capture=str(args.capture), regression=regression, cases=results,
        trajectory_data_unchanged=True, data_sha256=after_data, robot_mesh_prefix_unchanged=True,
        exact_original_triangle_positions_and_winding=True,
    )
    (args.output / "terrain-repair-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep existing numeric capture checks but update their file identities.
    verification_path = args.output / "verification.json"
    verification = json.loads(verification_path.read_text())
    verification["display_terrain_repair"] = report
    for name in verification["sha256"]:
        verification["sha256"][name] = digest(args.output / name)
    verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    print("WM_TERRAIN_REPAIR_COMPLETE " + json.dumps(report), flush=True)
