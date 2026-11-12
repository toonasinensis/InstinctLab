"""PGMT metrics on SuperMimic's logger boundary; reward units are unchanged."""

import json
import math
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from instinctlab.tasks.shadowing.pgmt.episode_rewards import EpisodeRewards

from .logger import Logger


class PGMTLogger(Logger):
    def __init__(self, env, mdp, foot_rewards, analytic_foot, log_root, **kwargs):
        super().__init__(**kwargs)
        self.env, self.native, self.command = env, env.unwrapped, mdp.command
        self.rewards, self.foot_rewards, self.analytic_foot = mdp.rewards, foot_rewards, analytic_foot
        self.log_root, self.writer, self.stream = log_root, None, None
        self.term_groups = dict(
            base19=tuple(self.rewards.terms), contact4=tuple("contact/" + key for key in foot_rewards.names)
        )
        if analytic_foot is not None:
            self.term_groups["analytic2"] = tuple(analytic_foot.names)
        self.terms = sum(self.term_groups.values(), ())
        self.episode_rewards = EpisodeRewards(
            env.num_envs, self.terms, len(self.rewards.terms), self.native.max_episode_length_s, self.device
        )

    def open(self):
        self.writer = SummaryWriter(str(self.log_root / "tensorboard"))
        self.stream = (self.log_root / "metrics.jsonl").open("a", buffering=1)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.stream.close()
            self.writer = self.stream = None

    def diagnostics(self, metrics, iteration):
        with (self.log_root / "wm-metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(iteration=iteration, **metrics)) + "\n")

    def record(self, metrics, phase, iteration, learner_round, actor_updates):
        scalars = {self.scalar_tag(key): float(value) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in scalars.values()):
            raise FloatingPointError("Nonfinite training metrics")
        self.stream.write(
            json.dumps(
                dict(
                    phase=phase,
                    iteration=iteration,
                    learner_round=learner_round,
                    actor_updates=actor_updates,
                    time=time.time(),
                    metrics=scalars,
                )
            )
            + "\n"
        )
        if phase == "train":
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, iteration)
        # Only task aggregates are retained; do not accumulate reset extras forever.
        self.ep_extras.clear()
        print(
            f"[SuperMimic] {phase} actor={actor_updates} seconds={scalars['Performance/iteration_seconds']:.3f}",
            flush=True,
        )

    def reward_metrics(self, prefix: str, means) -> dict[str, float]:
        result: dict[str, float] = {
            prefix + "/" + name: float(value) for name, value in zip(self.terms, means, strict=True)
        }
        offset = 0
        for group, terms in self.term_groups.items():
            result[prefix + "/" + group] = float(means[offset : offset + len(terms)].sum())
            offset += len(terms)
        result[prefix + "/total"] = float(means.sum())
        return result

    def metrics(self, replay, last_steps, imagined_means):
        rows = replay.rows(torch.arange(len(replay) - last_steps, len(replay), device=self.device))
        rewards = [replay.data[key][rows] for key in ("native_reward", "native_foot_reward")]
        if self.analytic_foot is not None:
            rewards.append(replay.data["native_analytic_reward"][rows])
        result = self.reward_metrics("RealReward", torch.cat(rewards, -1).mean((0, 1)))
        result.update(self.reward_metrics("ImaginedReward", imagined_means))
        result.update(self.episode_rewards.take())
        result["Contact/scale"] = float(self.foot_rewards.scale)
        done = replay.data["done"][rows].bool()
        if done.any():
            lengths = replay.data["episode_step"][rows][done].float() + 1
            result["Episode/mean_episode_length"] = float(lengths.mean())
            result["Episode/median_episode_length"] = float(lengths.median())
        command = self.command
        result["Terrain/mean_level"] = float(command.terrain.level.float().mean())
        away = (command.robot_anchor_pos_w[:, :2] - self.native.scene.env_origins[:, :2]).abs().amax(
            -1
        ) > command.cfg.off_platform_half_width
        result["Terrain/off_platform_fraction"] = float(away.float().mean())
        for family in {name.split("_")[0] for name in command.terrain.column_names}:
            columns = [i for i, name in enumerate(command.terrain.column_names) if name.startswith(family)]
            selected = torch.stack([self.native.scene.terrain.terrain_types == i for i in columns]).any(0)
            if selected.any():
                result["Terrain/" + family + "_level"] = float(command.terrain.level[selected].float().mean())
        result.update(
            {
                key: float(value)
                for key, value in self.native.extras.get("log", {}).items()
                if key.startswith("Episode_Termination/")
            }
        )
        return result

    @staticmethod
    def scalar_tag(key):
        if key.startswith("PGMT/raw/"):
            return "Tracking/raw/" + key.removeprefix("PGMT/raw/")
        if key.startswith("PGMT/effective/"):
            return "Tracking/effective/" + key.removeprefix("PGMT/effective/")
        if key.startswith("PGMT/terrain_"):
            return "Terrain/" + key.removeprefix("PGMT/terrain_")
        if key.startswith("PGMT/"):
            return "Episode/" + key.removeprefix("PGMT/")
        return key
