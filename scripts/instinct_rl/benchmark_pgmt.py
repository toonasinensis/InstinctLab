"""Compare PGMT terrain checkpoints in identical, explicit native simulation cases."""

import argparse
import gymnasium as gym

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils.instinct_rl_cli import resolve_train_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 3, 6])
    parser.add_argument("--motions", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--terrain-gate", type=float, choices=[0.0, 1.0], help="Override terrain attention for ablation"
    )
    args = parser.parse_args()
    task = "Instinct-PGMT-Terrain-G1-Newton-v0"
    cfg, _ = resolve_train_cfg(task, [])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.commands.motion.enable_terrain_curriculum = False
    cfg.observations.policy.enable_corruption = False
    with launch_simulation(cfg, dict(device=args.device, visualizer="none")):
        from instinctlab.tasks.shadowing.pgmt.benchmark import evaluate
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg))
        try:
            evaluate(
                env,
                args.checkpoint,
                args.output,
                levels=args.levels,
                seed=args.seed,
                steps=args.steps,
                motion_ids=args.motions,
                terrain_gate=args.terrain_gate,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
