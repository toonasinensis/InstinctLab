"""Matched native-physics benchmark: fixed motion/frame/yaw/terrain cases."""

import hashlib
import json
import numpy as np
import torch
from pathlib import Path
from torch.nn import functional as F

from isaaclab.utils.math import quat_apply

from .models.terrain import TerrainActor
from .world_model import yaw_angle


def load_actor(path, device, terrain_gate=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    actor = TerrainActor().to(device).eval()
    pure_wm = checkpoint.get("schema") == "pgmt-purewm-v1"
    if pure_wm:
        state, normalizer = checkpoint["actor"], checkpoint["policy_normalizer"]
        schema = checkpoint["schema"]
    else:
        state = {
            key.removeprefix("actor."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("actor.")
        }
        normalizer = checkpoint["policy_normalizer_state_dict"]
        schema = checkpoint["infos"]["schema"]
    mean, std = normalizer["_mean"], normalizer["_std"]
    if schema == "pgmt-lafan-native-v1":
        expanded = actor.state_dict()
        for name, value in state.items():
            if name == "command_embedding.weight":
                expanded[name].zero_()
                expanded[name][:, :65].copy_(value)
            else:
                expanded[name].copy_(value)
        state = expanded
        mean = F.pad(mean, (0, 2119 - 1645))
        std = F.pad(std, (0, 2119 - 1645), value=1)
    elif schema != "pgmt-lafan-terrain-v1" and not pure_wm:
        raise ValueError("Unsupported PGMT checkpoint schema")
    actor.load_state_dict(state, strict=True)
    if terrain_gate is not None:
        actor.terrain_gate.fill_(terrain_gate)

    def policy(obs):
        normalized = (obs - mean) / (std + 0.01)
        action = actor(torch.cat((normalized[:, :1678], obs[:, 1678:]), -1))
        return action.clamp(-20, 20) if pure_wm else action

    return policy


def make_cases(command, levels, seed, steps, motion_ids=None):
    random = np.random.default_rng(seed)
    cases = []
    for motion in range(len(command.label_lengths)) if motion_ids is None else motion_ids:
        for level in levels:
            for family in range(5):
                length = int(command.label_lengths[motion])
                frame = int(random.integers(0, max(1, length - steps - 12)))
                column = 2 * family + int(random.integers(2))
                cases.append(
                    dict(
                        motion=motion,
                        name=Path(command.motion._all_motion_files[motion]).name,
                        frame=frame,
                        yaw=float(random.uniform(-np.pi, np.pi)),
                        level=level,
                        family=family,
                        column=column,
                    )
                )
    return cases


@torch.inference_mode()
def evaluate(env, checkpoints, output, *, levels=(0, 3, 6), seed=17, steps=500, motion_ids=None, terrain_gate=None):
    """Evaluate first episodes only; reset frames never count as successful tracking."""
    native = env.unwrapped
    c = native.command_manager.get_term("motion")
    terrain = native.scene.terrain
    if terrain.cfg.terrain_type != "generator" or c.cfg.enable_terrain_curriculum:
        raise ValueError("Benchmark requires generated terrain with curriculum disabled")
    if terrain_gate not in (None, 0.0, 1.0):
        raise ValueError("The perception ablation gate must be zero or one")
    cases = make_cases(c, levels, seed, steps, motion_ids)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    original_reward = native.reward_manager.compute
    measurement = {}
    measurement_names = (
        "joint_mse",
        "body_error",
        "slip",
        "root_xy_mse",
        "corrected_planar_velocity_mse",
        "raw_planar_velocity_mse",
        "actual_planar_speed",
        "reference_planar_speed",
        "root_yaw_mse",
    )

    def compute_reward(dt):
        result = original_reward(dt)
        measurement["joint_mse"] = (c.joint_pos - c.robot_joint_pos).square().mean(-1).clone()
        measurement["body_error"] = (c.body_pos_relative_w - c.robot_body_pos_w).norm(dim=-1).mean(-1).clone()
        measurement["slip"] = c.contacts.terrain_term("slip").clone()
        measurement["root_xy_mse"] = (c.anchor_pos_w[:, :2] - c.robot_anchor_pos_w[:, :2]).square().sum(-1)
        actual_velocity = c.robot_anchor_lin_vel_w[:, :2]
        reference_velocity = c.body_lin_vel_w[:, c.motion_anchor_body_index, :2]
        measurement["corrected_planar_velocity_mse"] = (
            (c.corrected_velocity_w[:, :2] - actual_velocity).square().sum(-1)
        )
        measurement["raw_planar_velocity_mse"] = (reference_velocity - actual_velocity).square().sum(-1)
        measurement["actual_planar_speed"] = actual_velocity.norm(dim=-1)
        measurement["reference_planar_speed"] = reference_velocity.norm(dim=-1)
        actual_yaw, reference_yaw = yaw_angle(c.robot_anchor_quat_w), yaw_angle(c.anchor_quat_w)
        yaw_difference = actual_yaw - reference_yaw
        measurement["root_yaw_mse"] = torch.atan2(yaw_difference.sin(), yaw_difference.cos()).square()
        measurement["actual_xy"] = c.robot_anchor_pos_w[:, :2].clone()
        measurement["reference_xy"] = c.anchor_pos_w[:, :2].clone()
        measurement["actual_yaw"], measurement["reference_yaw"] = actual_yaw, reference_yaw
        measurement["failed"] = native.reset_terminated.clone()
        measurement["exited"] = native.termination_manager.get_term("tile_exit").clone()
        return result

    native.reward_manager.compute = compute_reward
    reports = []
    try:
        for checkpoint in checkpoints:
            policy = load_actor(checkpoint, native.device, terrain_gate)
            results = []
            for offset in range(0, len(cases), native.num_envs):
                batch = cases[offset : offset + native.num_envs]
                padded = batch + [batch[-1]] * (native.num_envs - len(batch))
                c.evaluation_cases = {
                    key: torch.tensor([case[key] for case in padded], device=native.device)
                    for key in ("motion", "frame", "yaw")
                }
                terrain.terrain_levels[:] = torch.tensor([case["level"] for case in padded], device=native.device)
                terrain.terrain_types[:] = torch.tensor([case["column"] for case in padded], device=native.device)
                terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                c.started.zero_()
                native.seed(seed)
                obs, _ = env.reset()
                # Applicability and denominator come from the entire requested
                # reference, independent of when this policy falls or resets.
                local_frame = c.frame_ids - c.label_starts[c.motion_ids]
                future_frames = local_frame[:, None] + torch.arange(steps + 1, device=native.device)
                future_frames = torch.minimum(future_frames, c.label_lengths[c.motion_ids, None] - 11)
                future_motions = c.motion_ids[:, None].expand_as(future_frames)
                raw_path = c.motion._all_motion_sequences.base_pos_w[
                    future_motions.flatten(), future_frames.flatten()
                ].reshape(native.num_envs, steps + 1, 3)
                planned = quat_apply(c.rotation[:, None].expand(-1, steps + 1, -1), raw_path) + c.offset[:, None]
                planned_path_length = (planned[:, 1:, :2] - planned[:, :-1, :2]).norm(dim=-1).sum(-1)
                planned_terrain_steps = (
                    (planned[:, 1:, :2] - native.scene.env_origins[:, None, :2]).abs().amax(-1) > 1.1
                ).sum(-1)
                alive = torch.arange(native.num_envs, device=native.device) < len(batch)
                count = torch.zeros(native.num_envs, device=native.device)
                totals = {key: torch.zeros_like(count) for key in ("return", *measurement_names)}
                failed, exited = torch.zeros_like(alive), torch.zeros_like(alive)
                previous_actual = c.robot_anchor_pos_w[:, :2].clone()
                previous_reference = c.anchor_pos_w[:, :2].clone()
                path_actual, path_reference, progress = (torch.zeros_like(count) for _ in range(3))
                reference_terrain_steps, matched_terrain_steps, stalled_steps = (
                    torch.zeros_like(count) for _ in range(3)
                )
                traces = []
                for _ in range(steps):
                    obs, reward, done, _ = env.step(policy(obs))
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                        raise FloatingPointError("Nonfinite native evaluation")
                    count += alive
                    totals["return"] += reward.sum(-1) * alive
                    for key in measurement_names:
                        totals[key] += measurement[key] * alive
                    actual, reference = measurement["actual_xy"], measurement["reference_xy"]
                    actual_delta, reference_delta = actual - previous_actual, reference - previous_reference
                    distance = reference_delta.norm(dim=-1)
                    along = (actual_delta * reference_delta / distance[:, None].clamp_min(1e-6)).sum(-1)
                    path_actual += actual_delta.norm(dim=-1) * alive
                    path_reference += distance * alive
                    progress += along * alive
                    reference_terrain = (reference - native.scene.env_origins[:, :2]).abs().amax(-1) > 1.1
                    actual_terrain = (actual - native.scene.env_origins[:, :2]).abs().amax(-1) > 1.1
                    reference_terrain_steps += reference_terrain * alive
                    matched_terrain_steps += reference_terrain * actual_terrain * alive
                    stalled_steps += (distance / native.step_dt > 0.2) * (along / native.step_dt < 0.05) * alive
                    traces.append(
                        torch.cat(
                            (
                                actual,
                                reference,
                                measurement["actual_yaw"][:, None],
                                measurement["reference_yaw"][:, None],
                                measurement["joint_mse"][:, None],
                                alive[:, None].float(),
                            ),
                            -1,
                        ).cpu()
                    )
                    previous_actual, previous_reference = actual, reference
                    failed |= measurement["failed"] & alive
                    exited |= measurement["exited"] & alive
                    alive &= ~done.bool()
                    if not alive.any():
                        break
                # Pose samples are captured before reset, with explicit first-episode validity.
                np.savez_compressed(
                    output / f"trajectory-{len(reports) + 1}-{offset}.npz",
                    samples=torch.stack(traces).numpy()[:, : len(batch)],
                    dt=native.step_dt,
                    columns=np.asarray(
                        [
                            "actual_x",
                            "actual_y",
                            "reference_x",
                            "reference_y",
                            "actual_yaw",
                            "reference_yaw",
                            "joint_mse",
                            "valid",
                        ]
                    ),
                    case_indices=np.arange(offset, offset + len(batch)),
                )
                for i, case in enumerate(batch):
                    n = max(1, int(count[i]))
                    results.append(
                        dict(
                            **case,
                            steps=n,
                            failed=bool(failed[i]),
                            tile_exit=bool(exited[i]),
                            survived_horizon=n == steps and not bool(failed[i]) and not bool(exited[i]),
                            episode_return=float(totals["return"][i]),
                            joint_rmse=float((totals["joint_mse"][i] / n).sqrt()),
                            body_error=float(totals["body_error"][i] / n),
                            slip=float(totals["slip"][i] / n),
                            root_xy_rmse=float((totals["root_xy_mse"][i] / n).sqrt()),
                            corrected_planar_velocity_rmse=float(
                                (totals["corrected_planar_velocity_mse"][i] / n).sqrt()
                            ),
                            raw_planar_velocity_rmse=float((totals["raw_planar_velocity_mse"][i] / n).sqrt()),
                            actual_planar_speed=float(totals["actual_planar_speed"][i] / n),
                            reference_planar_speed=float(totals["reference_planar_speed"][i] / n),
                            root_yaw_rmse=float((totals["root_yaw_mse"][i] / n).sqrt()),
                            actual_path_m=float(path_actual[i]),
                            reference_path_m=float(path_reference[i]),
                            reference_aligned_progress_m=float(progress[i]),
                            planned_reference_path_m=float(planned_path_length[i]),
                            progress_ratio=float(progress[i] / planned_path_length[i].clamp_min(1e-6)),
                            locomotion_applicable=bool(planned_path_length[i] >= 1.0),
                            terrain_follow_fraction=float(
                                matched_terrain_steps[i] / planned_terrain_steps[i].clamp_min(1)
                            ),
                            reference_terrain_steps=int(reference_terrain_steps[i]),
                            planned_reference_terrain_steps=int(planned_terrain_steps[i]),
                            stalled_seconds=float(stalled_steps[i] * native.step_dt),
                        )
                    )
                print(f"[Benchmark] {Path(checkpoint).name}: {len(results)}/{len(cases)} cases", flush=True)
            report = dict(
                checkpoint=str(Path(checkpoint).resolve()),
                sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                protocol=dict(
                    seed=seed,
                    levels=list(levels),
                    horizon=steps,
                    cases=len(cases),
                    simulator="PGMT native Newton/MJWarp; terrain-only rays",
                    terrain_gate_override=terrain_gate,
                    trajectory_columns="XY/yaw and joint error; pre-reset samples with first-episode validity",
                    progress_definition="sum actual XY displacement projected onto each same-time reference tangent",
                    progress_limits=(
                        "Numerator stops at first termination; denominator/applicability use the full planned reference"
                    ),
                ),
                aggregate={
                    key: float(np.mean([r[key] for r in results]))
                    for key in (
                        "survived_horizon",
                        "failed",
                        "tile_exit",
                        "steps",
                        "episode_return",
                        "joint_rmse",
                        "body_error",
                        "slip",
                        "root_xy_rmse",
                        "corrected_planar_velocity_rmse",
                        "raw_planar_velocity_rmse",
                        "actual_planar_speed",
                        "reference_planar_speed",
                        "root_yaw_rmse",
                        "stalled_seconds",
                    )
                },
                cases=results,
            )
            reports.append(report)
            (output / f"result-{len(reports)}.json").write_text(json.dumps(report, indent=2) + "\n")
            print("[Benchmark]", json.dumps(report["aggregate"]), flush=True)
    finally:
        native.reward_manager.compute = original_reward
        c.evaluation_cases = None
    (output / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports
