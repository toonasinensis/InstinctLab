"""Fixed-weight rollout capture and portable, simulator-free browser data export."""

import argparse
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.utils.math import quat_apply

from instinctlab.utils import instinct_rl_cli as cli

from .wm_dynamics import yaw_angle
from .wm_runner import SupervisedWMRunner, ordinary


CASES = [
    ("flat", 0, False),
    ("stairs_up", 5, False),
    ("stairs_up", 9, True),
    ("stairs_down", 9, True),
    ("rough", 9, False),
    ("rough", 9, True),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArrayFile:
    """Little-endian, four-byte-aligned arrays with explicit browser-side shapes."""

    def __init__(self, path):
        self.stream = path.open("xb")

    def add(self, values, dtype="f4"):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype="<" + dtype)
        if not np.isfinite(values).all():
            raise FloatingPointError("Refusing nonfinite viewer data")
        description = dict(offset=self.stream.tell(), shape=list(values.shape), dtype=dtype)
        self.stream.write(values.tobytes())
        return description


def robot_meshes(urdf, joint_names, arrays):
    """URDF visual origins and joint order, without MuJoCo recompilation/retargeting."""
    import trimesh

    root = ET.parse(urdf).getroot()

    def vector(element, attribute, default):
        return [float(v) for v in (element.get(attribute, default) if element is not None else default).split()]

    links, cached = [], {}
    for link in root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("PGMT browser export expects URDF mesh visuals")
            path = (urdf.parent / mesh.get("filename")).resolve()
            if path not in cached:
                geometry = trimesh.load_mesh(path, process=False)
                cached[path] = dict(vertices=arrays.add(geometry.vertices), faces=arrays.add(geometry.faces, "u4"))
            origin, color = visual.find("origin"), visual.find("material/color")
            visuals.append(
                dict(
                    **cached[path],
                    position=vector(origin, "xyz", "0 0 0"),
                    rpy=vector(origin, "rpy", "0 0 0"),
                    scale=vector(mesh, "scale", "1 1 1"),
                    color=vector(color, "rgba", "0.65 0.68 0.72 1"),
                )
            )
        links.append(dict(name=link.get("name"), visuals=visuals))
    joints = []
    for joint in root.findall("joint"):
        kind = joint.get("type")
        if kind not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"Unsupported visual joint: {kind}")
        origin = joint.find("origin")
        joints.append(
            dict(
                parent=joint.find("parent").get("link"),
                child=joint.find("child").get("link"),
                index=-1 if kind == "fixed" else joint_names.index(joint.get("name")),
                position=vector(origin, "xyz", "0 0 0"),
                rpy=vector(origin, "rpy", "0 0 0"),
                axis=vector(joint.find("axis"), "xyz", "0 0 1"),
            )
        )
    return dict(links=links, joints=joints, root="pelvis", quaternion="xyzw", urdf_sha256=digest(urdf))


@torch.no_grad()
def export_windows(runner, rows, stride, arrays):
    """Every regular anchor is exported, including terminal/invalid prefixes."""
    horizon = 32
    c, rewards = runner.rewards.command, runner.rewards
    device = runner.device
    starts = torch.arange(0, len(rows) - horizon + 1, stride, device=device)
    lanes = torch.arange(runner.env.num_envs, device=device)
    env_ids = lanes.repeat_interleave(len(starts))
    ages = starts.repeat(len(lanes))
    chronological = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    indices = ages[:, None] + torch.arange(horizon, device=device)
    data = {key: value[indices, env_ids[:, None]] for key, value in chronological.items()}
    alive = (~data["done"]).float().cumprod(1)
    data.update(mask=torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1), env_id=env_ids)
    adjacent = data["mask"][:, 1:].bool()
    torch.testing.assert_close(data["next_state"][:, :-1][adjacent], data["state"][:, 1:][adjacent], atol=1e-5, rtol=0)
    prediction = runner.predict(data, horizon)
    _, _, masks, states, contacts, forces, _ = prediction
    supports = runner.prediction_supports(data, states)
    keys = list(rewards.terms)
    supported = rewards.supported_keys(runner.wm.predict_contact, runner.wm.predict_force)
    native, oracle, predicted = data["native_reward"], [], []
    true_feet, pred_feet, true_maps, pred_maps, map_valid = [], [], [], [], []
    native_fk_error = []
    initial = data["state"][:, 0]
    initial_feet = rewards.bodies(initial, env_ids)[0][:, c.foot_ids]
    for step in range(horizon):
        state, real = states[:, step], data["next_state"][:, step]
        context = {key[4:]: value[:, step] for key, value in data.items() if key.startswith("ref_")}
        common = dict(
            action=data["action"][:, step], previous_action=data["previous_action"][:, step],
            context=context, previous_ref_velocity=data["previous_ref_velocity"][:, step],
            derivative_valid=data["derivative_valid"][:, step], ids=env_ids,
        )
        oracle.append(rewards.evaluate(
            state=real, previous_state=data["state"][:, step], contact=data["next_contact"][:, step],
            previous_contact=data["contact"][:, step], force=data["next_force"][:, step],
            support=data["next_support"][:, step], exact_surface=True, **common,
        )[0])
        contact = (contacts[:, step] > 0.5).float() if contacts is not None else None
        previous_contact = None if contacts is None else (
            (contacts[:, step - 1] > 0.5).float() if step else data["contact"][:, 0]
        )
        force = None if forces is None else quat_apply(
            state[:, None, 3:7].expand(-1, 14, -1), forces[:, step].reshape(-1, 14, 3)
        )
        predicted.append(rewards.evaluate(
            state=state, previous_state=states[:, step - 1] if step else initial,
            contact=contact, previous_contact=previous_contact, force=force, support=supports[:, step], **common,
        )[0])
        real_bodies = rewards.bodies(real, env_ids)[0]
        native_fk_error.append((real_bodies - data["native_body_position"][:, step]).norm(dim=-1).amax(-1))
        true_feet.append(data["native_body_position"][:, step, c.foot_ids])
        pred_feet.append(rewards.bodies(state, env_ids)[0][:, c.foot_ids])
        real_map, rv = runner.heightfield.local(real[:, :3], yaw_angle(real[:, 3:7]))
        pred_map, pv = runner.heightfield.local(state[:, :3], yaw_angle(state[:, 3:7]))
        true_maps.append(real_map)
        pred_maps.append(pred_map)
        map_valid.append(rv & pv)
    oracle, predicted = torch.stack(oracle, 1), torch.stack(predicted, 1)
    truth = torch.cat((initial[:, None], data["next_state"]), 1)
    guesses = torch.cat((initial[:, None], states), 1)
    true_feet = torch.cat((initial_feet[:, None], torch.stack(true_feet, 1)), 1)
    pred_feet = torch.cat((initial_feet[:, None], torch.stack(pred_feet, 1)), 1)
    errors = (guesses[..., :3] - truth[..., :3]).norm(dim=-1)
    feet_error = (pred_feet - true_feet).norm(dim=-1).amax(-1)
    # Same fixed initial frame; no root/yaw alignment or corrective physics.
    torch.testing.assert_close(truth[:, 0], guesses[:, 0], atol=0, rtol=0)
    tensors = dict(
        actual=truth, predicted=guesses, actual_feet=true_feet, predicted_feet=pred_feet,
        root_error=errors, foot_error=feet_error, mask=masks, map_valid=torch.stack(map_valid, 1),
        actual_map=torch.stack(true_maps, 1), predicted_map=torch.stack(pred_maps, 1),
        native_reward=native, oracle_reward=oracle, predicted_reward=predicted,
        actual_contact=data["next_contact"][:, :, c.foot_ids],
        actions=data["action"],
    )
    if contacts is not None:
        tensors["predicted_contact"] = contacts[:, :, c.foot_ids]
    descriptions = {key: arrays.add(value) for key, value in tensors.items()}
    windows = []
    for index in range(len(env_ids)):
        lane, start = int(env_ids[index]), int(ages[index])
        episode_valid = int(data["mask"][index].sum())
        usable = masks[index].bool().long().cumprod(0)
        valid = int(usable.sum())
        terminal = bool(data["done"][index, episode_valid - 1])
        windows.append(dict(
            index=index, case=lane, start_step=start, episode=int(data["episode"][index, 0]),
            episode_step=int(data["episode_step"][index, 0]), valid_steps=valid,
            episode_valid_steps=episode_valid, terminal=terminal,
            root_end_cm=float(errors[index, valid] * 100),
            foot_end_cm=float(feet_error[index, valid] * 100),
        ))
    valid = masks.bool()
    parity = (oracle - native).abs()[valid]
    fk_error = torch.stack(native_fk_error, 1)[valid]
    if float(fk_error.max()) > 3e-4:
        raise AssertionError(f"URDF visual FK disagrees with native link poses: {float(fk_error.max())}m")
    # These are capture checks, not a benchmark of tracking or model effectiveness.
    summary = dict(
        windows=len(windows), valid_h32=sum(w["valid_steps"] == 32 for w in windows),
        terminal_windows=sum(w["terminal"] for w in windows),
        oracle_reward_mae=float(parity.mean()), oracle_reward_max=float(parity.max()),
        native_fk_max_error_m=float(fk_error.max()),
        initial_states_identical=True, reset_prefix_continuity=True, no_per_frame_alignment=True,
        shared_recorded_actions=True,
    )
    return descriptions, windows, keys, supported, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7311)
    args = parser.parse_args()
    if args.steps < 32 or args.stride < 1:
        parser.error("steps must cover H32 and stride must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = digest(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    wm = saved["supervised_wm"]
    if wm["schema"] != SupervisedWMRunner.schema:
        raise ValueError("Unsupported WM checkpoint schema")
    env_yaml = cli.load_saved_yaml(str(args.checkpoint.parent / "params/env.yaml"))
    agent = cli.load_saved_yaml(str(args.checkpoint.parent / "params/agent.yaml"))
    task = "Instinct-PGMT-TerrainWMValidation-G1-Newton-v0"
    cfg, _ = cli.resolve_play_cfg(task, [], env_yaml=env_yaml, agent_yaml=agent)
    cfg.scene.num_envs = len(CASES)
    cfg.seed = int(env_yaml["seed"])
    cfg.sim.device = "cuda:0"
    cfg.commands.motion.enable_terrain_curriculum = False
    cfg.observations.policy.enable_corruption = False
    cfg.log_dir = str(args.output)
    agent.update(warm_start=None, ppo_checkpoint=None, device="cuda:0", seed=args.seed, world_model=wm["options"])
    started = time.monotonic()
    with launch_simulation(cfg, dict(device="cuda:0", visualizer="none")):
        from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg))
        try:
            runner = SupervisedWMRunner(env, agent, log_dir=None, device="cuda:0")
            runner.alg.actor_critic.load_state_dict(saved["model_state_dict"], strict=True)
            for key, normalizer in runner.normalizers.items():
                normalizer.load_state_dict(saved[key + "_normalizer_state_dict"])
            runner.wm.load_state_dict(wm["model"], strict=True)
            runner.wm.eval().requires_grad_(False)
            runner.eval_mode()
            modules = [runner.alg.actor_critic, runner.wm, *runner.normalizers.values()]
            baseline = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for module in modules]
            if runner.rewards.contract != wm["reward_contract"]:
                raise ValueError("Evaluation reward contract differs from checkpoint")
            native, c = runner.native, runner.rewards.command
            capture = native.reward_manager.compute

            def capture_visual_state(dt):
                result = capture(dt)
                runner.terminal["native_body_position"] = ordinary(c.robot_body_pos_w)
                return result

            native.reward_manager.compute = capture_visual_state
            terrain = native.scene.terrain
            ids = torch.arange(env.num_envs, device=runner.device)
            levels = ids.new_tensor([case[1] for case in CASES])
            columns = ids.new_tensor([c.terrain.column_names.index(case[0]) for case in CASES])
            terrain.terrain_levels.copy_(levels)
            terrain.terrain_types.copy_(columns)
            terrain.env_origins.copy_(terrain.terrain_origins[levels, columns])
            c.started.zero_()
            original_reset = c.reset_placement

            def reset_placement(reset_ids):
                if isinstance(reset_ids, slice):
                    reset_ids = ids[reset_ids]
                original_reset(reset_ids)
                selected = reset_ids[reset_ids.new_tensor([CASES[int(i)][2] for i in reset_ids], dtype=torch.bool)]
                if not len(selected):
                    return
                position, rotation = c.robot_anchor_pos_w[selected].clone(), c.robot_anchor_quat_w[selected].clone()
                old_support = c.terrain.support_z[selected].clone()
                position[:, 0] += 1.6
                ground, _, valid = c.terrain.query(position[:, :2])
                if not valid.all():
                    raise ValueError("Off-platform capture spawn missed the terrain")
                position[:, 2] += ground - old_support
                c.offset[selected, 0] += 1.6
                c.terrain.support_z[selected] = ground
                c.robot.write_root_pose_to_sim(torch.cat((position, rotation), -1), env_ids=selected)
                c.terrain.update(reset_ids=selected)
                c.contacts.reset(selected)

            c.reset_placement = reset_placement
            native.seed(args.seed)
            obs, _ = env.reset()
            policy = runner.get_inference_policy(device=runner.device)
            episode = ids * 0
            rows, endings = [], []
            with torch.inference_mode():
                for step in range(args.steps):
                    record = dict(
                        state=ordinary(runner.rewards.state()), contact=ordinary(c.contacts.contact.float()),
                        previous_action=ordinary(native.action_manager.action), support=ordinary(c.terrain.support_z),
                        previous_ref_velocity=ordinary(c.contacts.previous_ref_velocity),
                        episode=ordinary(episode), episode_step=ordinary(native.episode_length_buf),
                    )
                    obs, reward, done, info = env.step(policy(obs))
                    record.update(runner.terminal, done=ordinary(done.bool()))
                    rows.append(record)
                    for lane in done.nonzero().flatten().cpu().tolist():
                        endings.append(dict(case=lane, step=step + 1, episode=int(episode[lane])))
                    episode += done.long()
                    if (step + 1) % 100 == 0:
                        print(f"WM_CAPTURE step={step + 1}/{args.steps} endings={len(endings)}", flush=True)
            arrays = ArrayFile(args.output / "data.bin")
            descriptions, windows, keys, supported, summary = export_windows(runner, rows, args.stride, arrays)
            arrays.stream.close()
            mesh_arrays = ArrayFile(args.output / "meshes.bin")
            robot = robot_meshes(Path(c.reference.cfg.robot_model_path), c.robot.joint_names, mesh_arrays)
            # Use the exact mesh used by native terrain queries, not the baked WM raster.
            mesh = c.terrain._mesh
            points = wp.to_torch(mesh.points).cpu().numpy()
            faces = wp.to_torch(mesh.indices).cpu().numpy().reshape(-1, 3)
            centers = points[faces].mean(1)
            cases = []
            for lane, (family, level, shifted) in enumerate(CASES):
                trajectory = torch.stack([r["state"][lane, :3] for r in rows]).cpu().numpy()
                lo, hi = trajectory[:, :2].min(0) - 2.5, trajectory[:, :2].max(0) + 2.5
                chosen = faces[((centers[:, :2] >= lo) & (centers[:, :2] <= hi)).all(1)]
                vertices, remap = np.unique(chosen, return_inverse=True)
                cases.append(dict(
                    name=f"{family} · L{level} · {'off-platform spawn' if shifted else 'native center'}",
                    family=family, level=level, off_platform_spawn=shifted,
                    origin=terrain.env_origins[lane].cpu().tolist(),
                    terrain=dict(vertices=mesh_arrays.add(points[vertices]), faces=mesh_arrays.add(remap.reshape(-1, 3), "u4")),
                ))
            mesh_arrays.stream.close()
            for module, before in zip(modules, baseline):
                for key, value in module.state_dict().items():
                    if not torch.equal(value.detach().cpu(), before[key]):
                        raise AssertionError(f"Frozen capture changed weights/normalizer: {key}")
            if digest(args.checkpoint) != checkpoint_sha:
                raise AssertionError("Checkpoint file changed during capture")
            summary.update(frozen_weights_and_normalizers=True, checkpoint_unchanged=True, optimizer_updates=0)
            manifest = dict(
                schema="pgmt-wm-viewer-v1", dt=native.step_dt, horizon=32, train_horizon=wm["options"]["train_horizon"],
                motion="dance1_subject2", checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha,
                checkpoint_iteration=saved.get("iter"), wm_updates=wm["updates"], scene_seed=cfg.seed, trajectory_seed=args.seed,
                source_sha256=digest(__file__), state="p3, xyzw4, world COM v3, world w3, q29, qd29",
                policy="frozen deterministic PPO actor; observation noise disabled", terrain_sha256=runner.terrain_digest,
                training_terrain_sha256=wm["ranks"][0]["terrain_sha256"],
                map_convention="21x21, x slow/y fast; relative to pelvis; same raster sampled at true/predicted pose",
                robot=robot, cases=cases, windows=windows, arrays=descriptions, reward_keys=keys, predicted_reward_keys=supported,
                summary=summary, endings=endings, elapsed_seconds=time.monotonic() - started,
            )
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            for path in (Path(__file__).parent / "wm_viewer").iterdir():
                if path.is_file():
                    shutil.copy2(path, args.output / path.name)
            print("WM_VIEWER_COMPLETE " + json.dumps(dict(output=str(args.output), **summary)), flush=True)
        finally:
            env.close()
