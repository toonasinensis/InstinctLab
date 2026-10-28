"""Compare matched-step TensorBoard traces without mixing reward logging conventions."""

import argparse
import json
import numpy as np
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

METRICS = {
    "reward_per_step": "Train/mean_reward_each_timestep_0",
    "episode_length": "Train/mean_episode_length",
    "body_error_m": "Episode/Metrics/motion/error_body_pos",
    "joint_error_rad": "Episode/Metrics/motion/error_joint_pos",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-updates", type=int, default=300)
    args = parser.parse_args()
    baseline = EventAccumulator(args.baseline, size_guidance={"scalars": 0}).Reload()
    candidate = EventAccumulator(args.candidate, size_guidance={"scalars": 0}).Reload()
    report = {"baseline": args.baseline, "candidate": args.candidate, "metrics": {}}
    for name, tag in METRICS.items():
        a = {v.step: v.value for v in baseline.Scalars(tag)}
        b = {v.step: v.value for v in candidate.Scalars(tag)}
        steps = sorted(set(a) & set(b))
        av, bv = np.array([a[s] for s in steps]), np.array([b[s] for s in steps])
        window = min(25, len(steps))
        smooth_a = np.convolve(av, np.ones(window) / window, mode="valid")
        smooth_b = np.convolve(bv, np.ones(window) / window, mode="valid")
        tail_a, tail_b = float(av[-50:].mean()), float(bv[-50:].mean())
        scale = max(float(np.mean(np.abs(smooth_a))), 1e-3)
        tail_relative = abs(tail_a - tail_b) / max(abs(tail_a), 1e-3)
        # Near zero, reward percentages are unstable; use its baseline training range.
        reward_scale = max(float(np.ptp(smooth_a)), abs(tail_a), 0.01)
        tolerance = 0.25
        tail_pass = abs(tail_a - tail_b) <= tolerance * (
            reward_scale if name == "reward_per_step" else max(abs(tail_a), 1e-3)
        )
        curve_scale = reward_scale if name == "reward_per_step" else scale
        curve_pass = float(np.abs(smooth_a - smooth_b).mean()) <= tolerance * curve_scale
        report["metrics"][name] = {
            "updates": len(steps),
            "baseline_tail50": tail_a,
            "candidate_tail50": tail_b,
            "tail_relative_difference": tail_relative,
            "smoothed_mae_relative": float(np.abs(smooth_a - smooth_b).mean() / scale),
            "tail_within_25_percent": bool(tail_pass),
            "smoothed_curve_within_25_percent": bool(curve_pass),
            "finite": bool(np.isfinite(av).all() and np.isfinite(bv).all()),
        }
    report["enough_updates"] = all(v["updates"] >= args.min_updates for v in report["metrics"].values())
    report["similar_short_run"] = report["enough_updates"] and all(
        v["tail_within_25_percent"] and v["smoothed_curve_within_25_percent"] and v["finite"]
        for v in report["metrics"].values()
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
