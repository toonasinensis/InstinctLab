"""Aggregate recorded episode rows without accessing or changing the simulator."""

import csv
import hashlib
import json
import numpy as np
import torch

from .metrics import ERROR_KEYS


def summarize(rows):
    def average(group):
        return dict(
            episodes=len(group),
            fail_percent=100 * np.mean([r["fail"] for r in group]),
            duration_s=np.mean([r["duration_s"] for r in group]),
            **{k: np.mean([r[k] for r in group]) for k in (*ERROR_KEYS, "native_env_return")},
        )

    terrains = {
        name: average([r for r in rows if r["terrain"] == name]) for name in dict.fromkeys(r["terrain"] for r in rows)
    }
    keys = (*ERROR_KEYS, "fail_percent", "duration_s", "native_env_return")
    return dict(
        episode_weighted=average(rows),
        terrains=terrains,
        macro={k: float(np.mean([v[k] for v in terrains.values()])) for k in keys},
        failure_counts={
            n: sum(n in r["termination_flags"] for r in rows)
            for n in dict.fromkeys(n for r in rows for n in r["termination_flags"])
        },
    )


def write_report(args, cfg, saved, native, metric, cases, elapsed):
    """Publish the unchanged report schema and per-episode/error-trace artifacts."""
    rows = metric.rows(cases)
    complete = bool(metric.finished.all())
    # Publish tracking/termination results before expensive contact trace compression.
    report = dict(
        protocol="saved-training-mdp-v1",
        checkpoint=str(args.checkpoint),
        iteration=int(saved["iter"]),
        weights=args.weights,
        seed=cfg.seed,
        envs=native.num_envs,
        episode_length_s=cfg.episode_length_s,
        fixed_level=args.environment.terrain_level,
        complete=complete,
        env_config=str(args.env_config),
        task=args.task,
        error_target="training XY/yaw-aligned target; raw error",
        termination="unchanged native training terminations with native lower-body tolerance",
        elapsed_seconds=elapsed,
        score=summarize(rows),
        episodes=rows,
        contacts_sampling="control boundary; native physics decimation unchanged",
        physics=dict(
            dt=native.physics_dt,
            control_dt=native.step_dt,
            decimation=native.cfg.decimation,
            physics_handles_decimation=getattr(native, "_physics_handles_decimation", None),
        ),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
        phase=dict(flat_warmup_active=native.command_manager.get_term("motion").flat_warmup_active),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        dict(
            errors_cm=torch.stack(metric.trace) if metric.trace else None,
            active=torch.stack(metric.active_trace) if metric.active_trace else None,
        ),
        args.output / "error_trace.pt",
    )
    return report
