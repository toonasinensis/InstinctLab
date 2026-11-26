import argparse
import hashlib
import json
import math
import numpy as np
import tempfile
import torch
from importlib.resources import files
from pathlib import Path
from torch import nn
from xml.etree import ElementTree

from isaaclab.utils.string import resolve_matching_names_values

from instinctlab.tasks.shadowing.pgmt.models.terrain_cnn import TerrainCNNActor
from instinctlab.utils.instinct_rl_cli import load_saved_yaml

from .layouts import TensorLayout
from .pgmt_models import PolicyNormalizer
from .runners.pgmt_checkpoint import SCHEMA

T3_SCHEMA = "pgmt-t3-27-heightscan-v1"
ACTOR_CLASS = "instinctlab.tasks.shadowing.pgmt.models.terrain_cnn:TerrainCNNActor"
MODEL_NAME = "policy.onnx"
OPSET = 18


def joint_values(config, names, default=None):
    if not isinstance(config, dict):
        return [default if config is None else config] * len(names)
    indices, _, values = resolve_matching_names_values(config, names)
    result = [default] * len(names)
    for index, value in zip(indices, values, strict=True):
        result[index] = value
    return result


def robot_metadata(env):
    robot = env["scene"]["robot"]
    action = env["actions"]["joint_pos"]
    names = list(action["joint_names"])
    if len(names) != 27 or len(set(names)) != 27 or names != list(robot["joint_ordering"]):
        raise ValueError("The saved action order must explicitly match the 27 robot joints")
    if (
        action["class_type"].rsplit(":", 1)[-1] != "JointPositionAction"
        or action["asset_name"] != "robot"
        or not action["preserve_order"]
        or any(term is not None for name, term in env["actions"].items() if name != "joint_pos")
    ):
        raise ValueError("Expected one ordered joint position action term")

    asset_name = Path(robot["spawn"]["asset_path"]).name
    urdf = files("instinctlab").joinpath("assets", "resources", "t3_27", asset_name).read_bytes()
    limits = {joint.attrib["name"]: joint.find("limit") for joint in ElementTree.fromstring(urdf).findall("joint")}
    position_limits = [[float(limits[name].attrib[key]) for key in ("lower", "upper")] for name in names]

    actuator_keys = dict(
        kp="stiffness", kd="damping", effort_limit_nm="effort_limit_sim", velocity_limit_rad_s="velocity_limit_sim"
    )
    actuator_values = {key: {} for key in actuator_keys}
    assigned = set()
    for group in robot["actuators"].values():
        _, members, _ = resolve_matching_names_values(dict.fromkeys(group["joint_names_expr"]), names)
        if assigned.intersection(members):
            raise ValueError("Overlapping actuator groups in the saved robot config")
        assigned.update(members)
        for key, source in actuator_keys.items():
            values = joint_values(group.get(source), members)
            if any(value is None or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Missing or invalid {source} for actuator group {members}")
            actuator_values[key].update(zip(members, values, strict=True))
    if assigned != set(names):
        raise ValueError("The saved actuator groups must cover every action joint")

    default_pos = joint_values(robot["init_state"]["joint_pos"], names, 0.0)
    default_vel = joint_values(robot["init_state"]["joint_vel"], names, 0.0)
    action_offset = default_pos if action["use_default_offset"] else joint_values(action["offset"], names, 0.0)
    control = {
        "type": "joint_position",
        "delta_action": False,
        "action_scale": joint_values(action["scale"], names, 1.0),
        "action_offset_rad": action_offset,
        "uses_default_joint_pos_as_offset": action["use_default_offset"],
        "target_clip_rad": None if action["clip"] is None else joint_values(action["clip"], names),
        "target_expression": "q_target = action_offset_rad + action_scale * action; then apply target_clip_rad",
        "pd_target_velocity_rad_s": [0.0] * len(names),
    }
    model = {
        "name": "T3_27",
        "root_link": "pelvis",
        "asset_name": asset_name,
        "repository_urdf_sha256": hashlib.sha256(urdf).hexdigest(),
        "joint_names": names,
        "default_joint_pos_rad": default_pos,
        "default_joint_vel_rad_s": default_vel,
        "joint_position_limits_rad": position_limits,
        "joint_position_limits_source": "repository_urdf",
        "pd": {
            **{key: [values[name] for name in names] for key, values in actuator_values.items()},
            "kp_unit": "Nm/rad",
            "kd_unit": "Nm/(rad/s)",
            "source": "saved_simulation_actuator_config",
        },
    }
    return model, control


def observation_metadata(env, names, scan_shape, non_terrain_width, step_s):
    policy = env["observations"]["policy"]
    motion = env["scene"]["motion_reference"]
    motion_cfg = env["commands"]["motion"]
    n = len(names)
    history = policy["projected_gravity"]["history_length"]
    future = motion["num_frames"]
    if (
        policy["concatenate_terms"]
        or policy["history_length"] is not None
        or policy["history_order"] != "term"
        or history < 1
        or motion["data_start_from"] != "current_time"
        or future != 11
        or motion_cfg.get("foothold_adaptation", False)
    ):
        raise ValueError("Unsupported history or reference settings for T3_27 heightscan export")

    terms = {
        "projected_gravity": ((history, 3), "unit_vector", "World down [0,0,-1] expressed in the pelvis frame"),
        "base_ang_vel": ((history, 3), "rad/s", "Pelvis angular velocity in the pelvis frame, XYZ"),
        "joint_pos": ((history, n), "rad", "Measured joint position minus default_joint_pos_rad"),
        "joint_vel": ((history, n), "rad/s", "Measured joint velocity minus default_joint_vel_rad_s"),
        "last_action": ((history, n), "unitless", "Previous raw policy action, zero at reset"),
        "joint_pos_ref": ((future, n), "rad", "Reference joint position minus default_joint_pos_rad, zero when masked"),
        "joint_vel_ref": (
            (future, n),
            "rad/s",
            "Reference joint velocity minus default_joint_vel_rad_s, zero when masked",
        ),
        "motion_orientation": (
            (future, 3, 2),
            "unitless",
            "First two columns of R_robot_pelvis^T @ R_placed_reference_pelvis, flattened by row",
        ),
        "motion_height": (
            (future,),
            "m",
            "Reference pelvis Z above the reference motion origin, before terrain support adjustment",
        ),
        "corrected_velocity": (
            (future, 3),
            "m/s",
            "Corrected reference velocity in the current reference yaw frame, XYZ with Z zero",
        ),
        "terrain": (scan_shape, "m", "World terrain Z minus robot pelvis Z, clipped to [-5,5]"),
    }
    saved_order = tuple(key for key, value in policy.items() if isinstance(value, dict) and "func" in value)
    if saved_order != tuple(terms):
        raise ValueError(f"Unexpected policy observation order: {saved_order}")
    functions = {
        "joint_pos": ("joint_pos_rel",),
        "joint_vel": ("joint_vel_rel",),
        "joint_pos_ref": ("foothold_joint_reference", "generated_commands"),
        "joint_vel_ref": ("foothold_joint_reference", "generated_commands"),
        "motion_orientation": ("orientation",),
        "motion_height": ("clearance",),
        "corrected_velocity": ("velocity",),
        "terrain": ("bilinear_elevation",),
    }
    for index, name in enumerate(terms):
        term = policy[name]
        if (
            term["func"].rsplit(":", 1)[-1] not in functions.get(name, (name,))
            or term.get("modifiers")
            or term.get("delay_max_lag", 0)
            or term["history_length"] != (history if index < 5 else 0)
        ):
            raise ValueError(f"Unsupported saved observation definition for {name}")
    for name in ("joint_pos", "joint_vel"):
        asset = policy[name]["params"]["asset_cfg"]
        if asset["name"] != "robot" or list(asset["joint_names"]) != names or not asset["preserve_order"]:
            raise ValueError(f"Unexpected joint order for {name}")
    for name in ("joint_pos_ref", "joint_vel_ref"):
        command_name = policy[name]["params"]["command_name"]
        command = env["commands"][command_name]
        if command_name != name + "_command" or command["current_state_command"]:
            raise ValueError(f"Unexpected motion reference command for {name}")
        if policy[name]["func"].endswith(":foothold_joint_reference") and policy[name]["params"].get(
            "derivative", False
        ) != (name == "joint_vel_ref"):
            raise ValueError(f"Unexpected reference derivative for {name}")
        asset = command["asset_cfg"]
        if asset["name"] != "robot" or list(asset["joint_names"]) != names or not asset["preserve_order"]:
            raise ValueError(f"Unexpected joint order for {name}")

    layout = TensorLayout(tuple((name, math.prod(shape)) for name, (shape, _, _) in terms.items()))
    scan = env["height_scan"]
    if layout["terrain"].start != non_terrain_width or scan_shape != tuple(
        round(size / scan["spacing"]) + 1 for size in scan["size"]
    ):
        raise ValueError("Observation layout does not match the checkpoint's CNN input")
    return {
        "layout": [
            {
                "name": name,
                "slice": [layout[name].start, layout[name].stop],
                "shape": shape,
                "unit": unit,
                "description": description,
                "clip": policy[name]["clip"],
                "scale": policy[name]["scale"],
            }
            for name, (shape, unit, description) in terms.items()
        ],
        "flatten_order": "C",
        "preprocessing_order": ["compute_term", "clip_if_set", "scale_if_set", "flatten", "concatenate"],
        "history": {
            "frames": history,
            "order": "oldest_to_newest",
            "frame_interval_s": step_s,
            "reset_padding": "repeat_first_sample_per_term",
        },
        "motion_reference": {
            "frames": future,
            "frame_interval_s": motion["frame_interval_s"],
            "frame_offsets": list(range(future)),
            "foot_placement_adaptation": False,
            "motion_asset": Path(motion["motion_buffers"][motion_cfg["motion_buffer_name"]]["path"]).name,
            "corrected_velocity": {
                "position_error": (
                    "Rotate current placed reference pelvis XY minus robot pelvis XY into current reference yaw"
                ),
                "reference_velocity": "Rotate each future placed reference velocity into current reference yaw",
                "gain_s_inv": 1.0,
                "gate_speed_m_s": 0.2,
                "correction_clip_m_s_per_axis": 0.5,
                "expression": "v_xy + clip(clip(norm(v_xy)/0.2,0,1) * position_error_xy,-0.5,0.5)",
            },
        },
        "terrain_scan": {
            "frame": "robot pelvis yaw only",
            "axes": ["forward_x", "left_y"],
            "shape": scan_shape,
            "size_xy_m": scan["size"],
            "center_offset_xy_m": scan["offset"],
            "spacing_m": scan["spacing"],
            "ordering": "x_outer_y_inner_increasing",
            "sampling": "bilinear",
            "training_map_resolution_m": env["heightfield_resolution"],
            "height_clip_m": [-5.0, 5.0],
        },
    }


def verify_onnx(path, model, sample, scan_width):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    scans = sample.repeat(9, 1)
    generator = torch.Generator().manual_seed(0)
    scans[1:, :-scan_width] = 0.3 * torch.randn(scans[1:, :-scan_width].shape, generator=generator)
    scans[1:, -scan_width:] = -0.9 + 0.3 * torch.rand(scans[1:, -scan_width:].shape, generator=generator)
    error = 0.0
    with torch.inference_mode():
        for observation in scans.split(1):
            expected = model(observation).numpy()
            (actual,) = session.run(["action"], {"observation": observation.numpy()})
            if not np.isfinite(actual).all():
                raise ValueError("The exported policy produced a nonfinite action")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
            error = max(error, float(np.max(np.abs(actual - expected))))
    return error


def export_policy(checkpoint, output_dir, params_dir, weights):
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    env = load_saved_yaml(str(params_dir / "env.yaml"))
    agent = load_saved_yaml(str(params_dir / "agent.yaml"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("infos", {}).get("schema") != T3_SCHEMA or env.get("mdp_schema") != T3_SCHEMA:
        raise ValueError("Expected matching T3_27 heightscan checkpoint and environment schema")
    payload = saved["supermimic"]
    policy_cfg = dict(payload["contract"]["policy"])
    if payload["schema"] != SCHEMA or policy_cfg != agent["policy"] or policy_cfg.pop("class_name") != ACTOR_CLASS:
        raise ValueError("Expected the saved SuperMimic TerrainCNNActor configuration")
    if saved["scan187"]["scan"] != env["height_scan"]:
        raise ValueError("Saved heightscan geometry does not match params/env.yaml")
    robot, control = robot_metadata(env)
    names = robot["joint_names"]
    sim_dt = float(env["sim"]["dt"])
    decimation = int(env["decimation"])
    step_s = sim_dt * decimation
    scan_shape = tuple(policy_cfg["scan_shape"])
    non_terrain = policy_cfg["non_terrain_width"]
    scan_width = math.prod(scan_shape)
    width = non_terrain + scan_width
    observations = observation_metadata(env, names, scan_shape, non_terrain, step_s)
    actor = TerrainCNNActor(len(names), **policy_cfg)
    state = (
        saved["scan187"]["actor_ema"]
        if weights == "ema"
        else {
            name.removeprefix("actor."): value
            for name, value in saved["model_state_dict"].items()
            if name.startswith("actor.")
        }
    )
    actor.load_state_dict(state, strict=True)
    norm_cfg = dict(agent["normalizers"]["policy"])
    if norm_cfg.pop("class_name") != "EmpiricalNormalization":
        raise ValueError("Expected the saved empirical policy normalizer")
    normalizer = PolicyNormalizer(width, scan_width, **norm_cfg)
    normalizer.load_state_dict(saved["policy_normalizer_state_dict"], strict=True)
    model = nn.Sequential(normalizer, actor).eval().requires_grad_(False)
    metadata = {
        "schema_version": 1,
        "checkpoint": {
            "name": checkpoint.name,
            "training_schema": T3_SCHEMA,
            "iteration": saved["iter"],
            "weights": weights,
        },
        "model": {
            "file": MODEL_NAME,
            "opset": OPSET,
            "input": {"name": "observation", "shape": [1, width], "dtype": "float32"},
            "output": {"name": "action", "shape": [1, len(names)], "dtype": "float32", "type": "raw_policy_action"},
        },
        "robot": robot,
        "control": {
            **control,
            "policy_period_s": step_s,
            "policy_frequency_hz": 1.0 / step_s,
            "training_sim_dt_s": sim_dt,
            "training_decimation": decimation,
        },
        "observations": {
            **observations,
            "normalization": {
                "embedded_in_onnx": True,
                "normalized_slice": [0, non_terrain],
                "passthrough_slice": [non_terrain, width],
                "epsilon": normalizer.eps,
            },
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        onnx_path = bundle / MODEL_NAME
        example = torch.zeros(1, width)
        torch.onnx.export(
            model,
            (example,),
            onnx_path,
            input_names=["observation"],
            output_names=["action"],
            opset_version=OPSET,
            dynamo=False,
        )
        error = verify_onnx(onnx_path, model, example, scan_width)
        (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        bundle.rename(output_dir)
    return error


def main(default_export_root):
    parser = argparse.ArgumentParser(description="Export a T3_27 PGMT heightscan policy and deployment metadata.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Override the default tmp/export/<run>/<checkpoint>_<weights> path."
    )
    parser.add_argument(
        "--params-dir", type=Path, help="Directory containing the checkpoint's saved env.yaml and agent.yaml."
    )
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    args = parser.parse_args()
    checkpoint = args.checkpoint.absolute()
    params_dir = args.params_dir or checkpoint.parent / "params"
    output_dir = args.output_dir or default_export_root / checkpoint.parent.name / f"{checkpoint.stem}_{args.weights}"
    error = export_policy(checkpoint, output_dir, params_dir, args.weights)
    print(f"Exported {args.weights} policy to {output_dir} (ONNX Runtime max absolute error: {error:.3g})")
