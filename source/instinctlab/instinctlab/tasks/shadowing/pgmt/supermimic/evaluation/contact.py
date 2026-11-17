"""Terrain-only physical contact quality, independent of policy rewards and BCE.

The trace is sampled at the native control-step boundary, before terminal auto-reset. Raw
forces, speeds and labels are retained so thresholds can be changed offline.
"""

import csv
import json
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContactEvaluationCfg:
    force_on_n: float = 5.0
    force_off_n: float = 2.0
    nonfoot_force_n: float = 5.0
    overhang_drop_m: float = 0.1
    terrain_margin_m: float = 0.009
    landing_window_s: float = 0.12
    minimum_flight_s: float = 0.04
    grid_x: int = 5
    grid_y: int = 3
    sensitivity_thresholds_n: tuple = (1.0, 5.0, 10.0)


def hysteresis(force, active, on, off):
    contact = np.zeros_like(force, dtype=bool)
    previous = np.zeros(force.shape[1:], dtype=bool)
    for t in range(len(force)):
        previous = np.where(previous, force[t] > off, force[t] > on) & active[t, :, None]
        contact[t] = previous
    return contact


def mean_valid(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if len(values) else None


def contact_point_velocity(com_velocity, com_position, point):
    offset = torch.nan_to_num(point - com_position)
    return com_velocity[..., :3] + torch.cross(com_velocity[..., 3:], offset, dim=-1)


class ContactEvaluation:
    """Read a separate filtered sensor; never modify the training contact sensor."""

    def __init__(self, env, finished, cfg=ContactEvaluationCfg()):
        from isaaclab.utils.math import quat_apply

        from instinctlab.tasks.shadowing.pgmt.mdp.contact_labels import collision_vertices

        self.env, self.finished, self.cfg = env, finished, cfg
        self.command = c = env.command_manager.get_term("motion")
        self.sensor = env.scene["evaluation_contacts"]
        if self.sensor.num_filter_objects != 1:
            raise ValueError("Contact benchmark requires exactly one ground mesh filter")
        self.foot_names = [c.reference.cfg.link_of_interests[i] for i in c.foot_ids]
        self.sensor_feet = [self.sensor.body_names.index(n) for n in self.foot_names]
        self.robot_feet = [c.robot.body_names.index(n) for n in self.foot_names]
        nonfeet = [i for i, n in enumerate(self.sensor.body_names) if n not in self.foot_names]
        self.groups = {"all": nonfeet}
        for group, parts in {
            "torso": ("pelvis", "torso"),
            "arms": ("shoulder", "elbow", "wrist"),
            "legs": ("hip", "knee", "ankle_pitch"),
        }.items():
            self.groups[group] = [i for i in nonfeet if any(part in self.sensor.body_names[i] for part in parts)]
        if any(not ids for ids in self.groups.values()):
            raise ValueError("Nonfoot reporting groups must be nonempty")
        vertices = collision_vertices(c.reference.cfg.robot_model_path)
        points = [torch.as_tensor(vertices[n], device=env.device, dtype=torch.float32) for n in self.foot_names]
        lo, hi = torch.stack([v.amin(0) for v in points]), torch.stack([v.amax(0) for v in points])
        u, v = torch.meshgrid(
            torch.linspace(0, 1, cfg.grid_x, device=env.device),
            torch.linspace(0, 1, cfg.grid_y, device=env.device),
            indexing="ij",
        )
        self.grid = lo[:, None].expand(-1, u.numel(), -1).clone()
        self.grid[..., 0] += u.flatten() * (hi[:, 0] - lo[:, 0])[:, None]
        self.grid[..., 1] += v.flatten() * (hi[:, 1] - lo[:, 1])[:, None]
        self.quat_apply = quat_apply
        mass = c.robot.data.body_mass.torch.sum(-1)
        if mass.numel() == 1:
            mass = mass.expand(env.num_envs)
        self.weight = (mass * np.linalg.norm(env.cfg.sim.gravity)).cpu().numpy()
        self.samples = []
        self.dt = env.step_dt

    def __call__(self, dt):
        self.sample()

    def sample(self):
        c, sensor = self.command, self.sensor
        force = sensor.data.normal_force_matrix_w.torch[:, :, 0]
        foot_force = force[:, self.sensor_feet]
        magnitude = foot_force.norm(dim=-1)
        normal = foot_force / magnitude.clamp_min(1e-8)[..., None]
        point = sensor.data.contact_pos_w.torch[:, self.sensor_feet, 0]
        if not torch.isfinite(point[magnitude > self.cfg.force_off_n]).all():
            raise FloatingPointError("Loaded foot has no finite terrain contact position")
        data = c.robot.data
        com = data.body_com_pose_w.torch[:, self.robot_feet, :3]
        velocity = data.body_com_vel_w.torch[:, self.robot_feet]
        # No-contact positions are NaN; these values never enter contact metrics.
        velocity_at_contact = contact_point_velocity(velocity, com, point)
        tangent = velocity_at_contact - (velocity_at_contact * normal).sum(-1, keepdim=True) * normal
        pose = data.body_link_pose_w.torch[:, self.robot_feet]
        grid = pose[:, :, None, :3] + self.quat_apply(
            pose[:, :, None, 3:].expand(-1, -1, self.grid.shape[1], -1),
            self.grid[None].expand(self.env.num_envs, -1, -1, -1),
        )
        height, _, valid = c.terrain.query(grid[..., :2])
        drop = grid[..., 2] - height - self.cfg.terrain_margin_m
        fraction = ((drop > self.cfg.overhang_drop_m) & valid).sum(-1) / valid.sum(-1).clamp_min(1)
        fraction = fraction.masked_fill(~valid.all(-1), float("nan"))
        nonfoot = torch.stack([force[:, ids].norm(dim=-1).amax(-1) for ids in self.groups.values()], -1)
        reference = c.reference_contacts[:, c.foot_ids]
        reference_valid = c.reference_contact_valid[:, c.foot_ids]
        # T,N,15 float32; reference/valid/active are exact 0/1 in this trace.
        values = torch.cat(
            (
                magnitude,
                tangent.norm(dim=-1),
                fraction,
                nonfoot,
                reference.float(),
                reference_valid.float(),
                (~self.finished)[:, None].float(),
            ),
            -1,
        )
        if not torch.isfinite(values[:, :4]).all():
            raise FloatingPointError("Nonfinite ground force or contact-point velocity")
        self.samples.append(values.cpu())

    def finish(self, cases, output):
        trace = torch.stack(self.samples)
        self.samples.clear()  # Release duplicate CPU history before long-episode analysis.
        metadata = dict(
            config=asdict(self.cfg),
            sample_dt=self.dt,
            sampling="native control boundary; physics/decimation unchanged",
            body_weight_n=self.weight.tolist(),
            foot_names=self.foot_names,
            nonfoot_groups=self.groups,
            sensor_body_names=self.sensor.body_names,
            filters=self.sensor.filter_object_names,
            fields=[
                "left_force_n",
                "right_force_n",
                "left_slip_m_s",
                "right_slip_m_s",
                "left_overhang",
                "right_overhang",
                "nonfoot_all_n",
                "nonfoot_torso_n",
                "nonfoot_arms_n",
                "nonfoot_legs_n",
                "left_reference",
                "right_reference",
                "left_reference_valid",
                "right_reference_valid",
                "active",
            ],
            initial_contacts="Landing requires at least 40ms observed flight before contact",
            landing_window_s=self.cfg.landing_window_s,
            minimum_flight_s=self.cfg.minimum_flight_s,
            landing_censoring="Terminal-truncated event peaks retained and explicitly counted",
            overhang="5x3 rotated foot collision AABB sole grid; true terrain rays; subtract 9mm terrain margin",
        )
        assert trace.shape[-1] == len(metadata["fields"])
        np.savez_compressed(
            output / "contact_trace.npz", samples=trace.numpy(), metadata=np.asarray(json.dumps(metadata))
        )
        result = analyze_contacts(trace.numpy(), cases, self.weight, self.dt, self.cfg)
        (output / "contact_report.json").write_text(json.dumps(dict(metadata=metadata, **result), indent=2) + "\n")
        for name, rows in [("contact_episodes", result["episodes"]), ("landing_events", result["landing_events"])]:
            if rows:
                with (output / (name + ".csv")).open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        return result


def analyze_contacts(trace, cases, body_weight, dt, cfg=ContactEvaluationCfg()):
    force, speed, overhang = trace[..., :2], trace[..., 2:4], trace[..., 4:6]
    nonfoot = trace[..., 6:10]
    reference, valid, active = (
        trace[..., 10:12].astype(bool),
        trace[..., 12:14].astype(bool),
        trace[..., 14].astype(bool),
    )
    contact = hysteresis(force, active, cfg.force_on_n, cfg.force_off_n)
    starts = np.zeros_like(contact)
    flight = np.zeros(contact.shape[1:], dtype=np.int32)
    minimum_flight = int(round(cfg.minimum_flight_s / dt))
    for t in range(len(contact)):
        starts[t] = contact[t] & (flight >= minimum_flight)
        flight = np.where(contact[t], 0, flight + 1)
        flight[~active[t]] = 0
    events, per_episode_peaks = [], [[] for _ in cases]
    window = int(round(cfg.landing_window_s / dt))
    for i in range(len(cases)):
        episode_end = int(active[:, i].sum())
        for foot in range(2):
            times = np.flatnonzero(starts[:, i, foot])
            for j, t in enumerate(times):
                next_landing = int(times[j + 1]) if j + 1 < len(times) else len(trace)
                end = min(episode_end, int(t) + window, next_landing)
                peak = float(force[t:end, i, foot].max() / body_weight[i])
                censored = bool(end == episode_end and end < int(t) + window)
                event = dict(
                    case_id=int(i),
                    terrain=cases[i]["terrain"],
                    level=cases[i]["level"],
                    foot=foot,
                    time_s=float((t + 1) * dt),
                    peak_body_weight=peak,
                    observed_window_s=(end - int(t)) * dt,
                    right_censored=censored,
                )
                events.append(event)
                per_episode_peaks[i].append(peak)
    rows = []
    for i, case in enumerate(cases):
        mask = contact[:, i]
        v = valid[:, i] & active[:, i, None]
        ref = reference[:, i]
        counts = dict(
            support_foot_seconds=float(mask.sum() * dt),
            landing_events=len(per_episode_peaks[i]),
            reference_stance_foot_seconds=float((v & ref).sum() * dt),
            reference_swing_foot_seconds=float((v & ~ref).sum() * dt),
        )
        over = overhang[:, i][mask]
        row = dict(
            **case,
            **counts,
            slip_mean_cm_s=mean_valid(speed[:, i][mask] * 100),
            slip_p95_cm_s=percentile(speed[:, i][mask] * 100, 95),
            overhang_percent=mean_valid(over * 100),
            overhang_query_valid_percent=100 * float(np.isfinite(over).mean()) if len(over) else None,
            landing_peak_bw_mean=mean_valid(per_episode_peaks[i]),
            nonfoot_episode_percent=100 * float((nonfoot[:, i, 0][active[:, i]] > cfg.nonfoot_force_n).any()),
            missing_stance_percent=(
                100 * float((v & ref & ~mask).sum()) / max(int((v & ref).sum()), 1) if (v & ref).any() else None
            ),
            unexpected_contact_percent=(
                100 * float((v & ~ref & mask).sum()) / max(int((v & ~ref).sum()), 1) if (v & ~ref).any() else None
            )
        )
        for j, name in enumerate(("all", "torso", "arms", "legs")):
            row["nonfoot_" + name + "_time_percent"] = 100 * float(
                (nonfoot[:, i, j][active[:, i]] > cfg.nonfoot_force_n).mean()
            )
        rows.append(row)
    keys = [key for key in rows[0] if key not in cases[0]]

    def aggregate(selected):
        return {key: mean_valid([r[key] if r[key] is not None else np.nan for r in selected]) for key in keys}

    cells = []
    for terrain in dict.fromkeys(c["terrain"] for c in cases):
        for level in range(10):
            selected = [r for r in rows if r["terrain"] == terrain and r["level"] == level]
            if selected:
                cells.append(dict(terrain=terrain, level=level, episodes=len(selected), **aggregate(selected)))
    macro = aggregate(cells)

    def event_score(selected):
        peaks = [e["peak_body_weight"] for e in selected]
        return dict(
            events=len(selected),
            median_body_weight=percentile(peaks, 50),
            p95_body_weight=percentile(peaks, 95),
            right_censored=sum(e["right_censored"] for e in selected),
        )

    sensitivity = {}
    for on in cfg.sensitivity_thresholds_n:
        mask = hysteresis(force, active, on, on * cfg.force_off_n / cfg.force_on_n)
        sensitivity[str(on)] = dict(
            support_foot_seconds=float(mask.sum() * dt),
            pooled_slip_cm_s=mean_valid(speed[mask] * 100),
            pooled_overhang_percent=mean_valid(overhang[mask] * 100),
        )
    return dict(
        macro=macro,
        cells=cells,
        terrains={
            name: aggregate([r for r in rows if r["terrain"] == name])
            for name in dict.fromkeys(c["terrain"] for c in cases)
        },
        landing=event_score(events),
        landing_by_terrain={
            name: event_score([e for e in events if e["terrain"] == name])
            for name in dict.fromkeys(c["terrain"] for c in cases)
        },
        sensitivity=sensitivity,
        episodes=rows,
        landing_events=events,
    )
