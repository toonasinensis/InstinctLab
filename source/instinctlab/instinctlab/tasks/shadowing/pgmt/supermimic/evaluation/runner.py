"""Run frozen policies in the native training environment and attach passive metrics."""

import json
import math
import time
import torch

from .metrics import EpisodeMetrics
from .observer import TransitionObserver
from .policy import load_policy
from .report import write_report


def run(args, cfg, saved):
    import gymnasium as gym

    from isaaclab.sensors import ContactSensorCfg

    from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

    from .contact import ContactEvaluation

    if args.contacts:
        cfg.scene.evaluation_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            filter_shape_prim_expr=[cfg.scene.terrain.prim_path + "/.*"],
            track_contact_points=True,
            history_length=0,
            update_period=0.0,
        )
    env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
    native = env.unwrapped
    try:
        c = native.command_manager.get_term("motion")
        if args.environment.finish_flat_warmup and c.flat_warmup_active:
            c.finish_flat_warmup()  # Restore the native post-2000 phase, as the training runner does.
        terrain = native.scene.terrain
        if args.environment.terrain_level is not None:
            terrain.terrain_levels.fill_(args.environment.terrain_level)
            terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        c.started.zero_()
        if cfg.seed is not None:
            native.seed(cfg.seed)
        observation, _ = env.reset()
        policy, norm = load_policy(saved, env.num_actions, native.device, args.weights)
        normalizer_before = {k: v.clone() for k, v in norm.state_dict().items()}
        metric = EpisodeMetrics(native)
        contact = ContactEvaluation(native, metric.finished) if args.contacts else None
        observer = TransitionObserver(native, metric, contact=contact, check_purity=args.smoke)
        if not args.observe_check:
            native.add_transition_observer(observer)
        cases = [
            dict(
                case_id=i,
                terrain=c.terrain.column_names[int(terrain.terrain_types[i])],
                level=int(terrain.terrain_levels[i]),
                start_frame=int(c.frame_ids[i]),
                x=float(c.spawn_offsets[i, 0]),
                y=float(c.spawn_offsets[i, 1]),
            )
            for i in range(env.num_envs)
        ]
        initial = dict(
            root=native.scene["robot"].data.root_state_w.torch.cpu().clone(),
            joint_pos=c.robot_joint_pos.cpu().clone(),
            joint_vel=c.robot_joint_vel.cpu().clone(),
            observation=observation.cpu().clone(),
            rotation=c.rotation.cpu().clone(),
            offset=c.offset.cpu().clone(),
            support=c.terrain.support_z.cpu().clone(),
            cases=cases,
        )
        torch.save(initial, args.output / "initial_state.pt")
        fingerprints = []
        started = time.monotonic()
        with torch.inference_mode():
            for step in range(args.steps or math.ceil(cfg.episode_length_s / native.step_dt)):
                action = policy.collect(observation)
                if not torch.isfinite(action).all():
                    raise FloatingPointError("Nonfinite action; no clipping or replacement in evaluation")
                observation, _, _, _ = env.step(action)
                if args.smoke:
                    fingerprints.append(
                        torch.cat((action.flatten(), native.scene["robot"].data.root_state_w.torch.flatten())).cpu()
                    )
                if (step + 1) % 100 == 0:
                    print(f"EVAL step={step+1} finished={int(metric.finished.sum())}/{env.num_envs}", flush=True)
                if metric.finished.all():
                    break
        if any(not torch.equal(value, normalizer_before[key]) for key, value in norm.state_dict().items()):
            raise AssertionError("Evaluation updated the frozen policy normalizer")
        (args.output / "observer_purity.json").write_text(json.dumps(observer.audit, indent=2) + "\n")
        report = write_report(args, cfg, saved, native, metric, cases, time.monotonic() - started)
        complete = report["complete"]
        if fingerprints:
            torch.save(torch.stack(fingerprints), args.output / "fingerprints.pt")
        if contact and not args.observe_check:
            contact.finish(cases, args.output)
        print(
            "EVAL_RESULT " + json.dumps({k: v for k, v in report.items() if k not in ("episodes", "score")}), flush=True
        )
        if not complete and not args.smoke:
            raise RuntimeError("Evaluation cohort did not finish")
    finally:
        env.close()
