"""Offline reference contacts against source flat z=0, using the selected URDF collisions.

Labels are engineering supervision, NOT author-supplied PGMT annotations. Source
motions are never modified. Mesh/collision vertices are queried against a plane;
confidence is masked near thresholds, during penetration, or with nonfinite data.
"""

import argparse
import json
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path


def collision_vertices(urdf):
    tree = ET.parse(urdf)
    output = {}
    for link in tree.findall("link"):
        vertices = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is not None:
                path = (urdf.parent / mesh.attrib["filename"]).resolve()
                shape = trimesh.load(path, force="mesh", process=False)
                points = shape.convex_hull.vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            elif geometry.find("box") is not None:
                points = trimesh.creation.box(np.fromstring(geometry.find("box").attrib["size"], sep=" ")).vertices
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                points = trimesh.creation.cylinder(
                    float(cylinder.attrib["radius"]),
                    float(cylinder.attrib["length"]),
                    sections=16,
                ).vertices
            elif geometry.find("sphere") is not None:
                points = trimesh.creation.icosphere(
                    subdivisions=2,
                    radius=float(geometry.find("sphere").attrib["radius"]),
                ).vertices
            elif geometry.find("capsule") is not None:
                # Roban uses URDF capsules: length is the cylindrical segment,
                # centred at the link collision origin, with spherical end caps.
                capsule = geometry.find("capsule")
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(capsule.attrib["radius"])).vertices
                half = np.array([0.0, 0.0, float(capsule.attrib["length"]) / 2])
                points = np.concatenate((sphere - half, sphere + half))
            else:
                raise ValueError(f"Unsupported collision geometry on {link.attrib['name']}: {ET.tostring(geometry)}")
            origin = collision.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rot = trimesh.transformations.euler_matrix(*rpy)[:3, :3]
            vertices.append(points @ rot.T + xyz)
        if vertices:
            output[link.attrib["name"]] = np.concatenate(vertices).astype(np.float32)
    # Isaac merges fixed descendants (feet, rubber hands, head) into dynamic bodies.
    fixed = {}
    for joint in tree.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        transform = trimesh.transformations.euler_matrix(*rpy)
        transform[:3, 3] = xyz
        fixed[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            transform,
        )
    merged = {name: [points] for name, points in output.items() if name not in fixed}
    for name, points in output.items():
        if name not in fixed:
            continue
        parent, transform = fixed[name]
        while parent in fixed:
            parent, next_transform = fixed[parent]
            transform = next_transform @ transform
        transformed = points @ transform[:3, :3].T + transform[:3, 3]
        merged.setdefault(parent, []).append(transformed.astype(np.float32))
    return {name: np.concatenate(parts) for name, parts in merged.items()}


def make_labels(path, vertices):
    with np.load(path, allow_pickle=False) as data:
        names = data["body_names"].astype(str).tolist()
        p, q, v = (data[key] for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w"))
        count = len(p)
        bottom = np.full((count, len(names)), np.nan, np.float32)
        for index, name in enumerate(names):
            if name not in vertices:
                continue
            # Last row of rotation(wxyz): maps collision local coordinates to world height.
            w, x, y, z = q[:, index].T
            row = np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
            bottom[:, index] = (row @ vertices[name].T).min(-1) + p[:, index, 2]
        speed = np.linalg.norm(v, axis=-1)
        contact = (bottom < 0.04) & (speed < 0.4)
        # Ambiguous phase transitions and invalid source geometry are NOT negative labels.
        valid = np.isfinite(bottom) & np.isfinite(speed) & (bottom >= -0.05)
        valid &= (np.abs(bottom - 0.04) > 0.01) & (np.abs(speed - 0.4) > 0.1)
        fps = float(data["fps"].reshape(-1)[0])
        return dict(
            contact=contact,
            valid=valid,
            bottom=bottom,
            body_names=np.asarray(names),
            fps=np.asarray([fps]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--urdf",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vertices = collision_vertices(args.urdf)
    entries = []
    for entry in json.loads(args.manifest.read_text())["files"]:
        path = args.data_dir / entry["name"]
        labels = make_labels(path, vertices)
        destination = args.output / (path.name + ".contacts.npz")
        np.savez_compressed(destination, **labels)
        # Conservative metadata compatibility: all ground routines remain represented on flat/rough L0.
        behavior = "ground" if path.name.startswith("ground") else "general"
        entries.append(
            dict(
                name=path.name,
                frames=len(labels["contact"]),
                fps=float(labels["fps"][0]),
                behavior=behavior,
                valid_fraction=float(labels["valid"].mean()),
                contact_fraction=float(labels["contact"].mean()),
            )
        )
    meta = dict(
        schema=1,
        algorithm="collision-plane-v1",
        source_surface="infinite plane z=0 m (source data assumption)",
        contact_height_m=0.04,
        contact_speed_m_s=0.4,
        files=entries,
    )
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            dict(
                files=len(entries),
                frames=sum(e["frames"] for e in entries),
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
