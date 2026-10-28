"""Matched native-physics benchmark: fixed motion/frame/yaw/terrain cases."""

import hashlib
import json
import numpy as np
import torch
from pathlib import Path
from torch.nn import functional as F

from .models.terrain import TerrainActor


def load_actor(path, device, terrain_gate=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    actor = TerrainActor().to(device).eval()
    state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    normalizer = checkpoint["policy_normalizer_state_dict"]
    mean, std = normalizer["_mean"], normalizer["_std"]
    if checkpoint["infos"]["schema"] == "pgmt-lafan-native-v1":
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
    elif checkpoint["infos"]["schema"] != "pgmt-lafan-terrain-v1":
        raise ValueError("Unsupported PGMT checkpoint schema")
    actor.load_state_dict(state, strict=True)
    if terrain_gate is not None:
        actor.terrain_gate.fill_(terrain_gate)

    def policy(obs):
        normalized = (obs - mean) / (std + 0.01)
        return actor(torch.cat((normalized[:, :1678], obs[:, 1678:]), -1))

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

    def compute_reward(dt):
        result = original_reward(dt)
        measurement["joint_mse"] = (c.joint_pos - c.robot_joint_pos).square().mean(-1).clone()
        measurement["body_error"] = (c.body_pos_relative_w - c.robot_body_pos_w).norm(dim=-1).mean(-1).clone()
        measurement["slip"] = c.contacts.terrain_term("slip").clone()
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
                alive = torch.arange(native.num_envs, device=native.device) < len(batch)
                count = torch.zeros(native.num_envs, device=native.device)
                totals = {key: torch.zeros_like(count) for key in ("return", "joint_mse", "body_error", "slip")}
                failed, exited = torch.zeros_like(alive), torch.zeros_like(alive)
                for _ in range(steps):
                    obs, reward, done, _ = env.step(policy(obs))
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                        raise FloatingPointError("Nonfinite native evaluation")
                    count += alive
                    totals["return"] += reward.sum(-1) * alive
                    for key in ("joint_mse", "body_error", "slip"):
                        totals[key] += measurement[key] * alive
                    failed |= measurement["failed"] & alive
                    exited |= measurement["exited"] & alive
                    alive &= ~done.bool()
                    if not alive.any():
                        break
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
