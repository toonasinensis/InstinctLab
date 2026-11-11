"""One training loop composed from the simulator collector and two SM learners."""

import json
import math
import subprocess
import time
import torch
import torch.distributed as dist
from dataclasses import asdict
from pathlib import Path
from tensordict import TensorDict

from tensorboardX import SummaryWriter

from . import checkpoint
from .config import from_dict
from .learners.optimizers import set_learning_rate
from .setup import build_components


class SuperMimicRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("This backend currently validates single-GPU learners only")
        self.env, self.cfg, self.device = env, train_cfg, torch.device(device)
        self.options = from_dict(train_cfg["supermimic"])
        self.log_root = Path(log_dir) if log_dir else None
        self.current_learning_iteration = self.learner_round = 0
        self.warmup_complete = False
        self.latest_metrics = {}
        self.git_status_repos = []
        self.components = build_components(env, train_cfg, self.options, self.device)
        if train_cfg.get("initialize_checkpoint"):
            checkpoint.initialize_legacy_weights(self, train_cfg["initialize_checkpoint"])
        if self.log_root is not None:
            self.log_root.mkdir(parents=True, exist_ok=True)
            (self.log_root / "params").mkdir(exist_ok=True)
            (self.log_root / "params/supermimic.json").write_text(json.dumps(asdict(self.options), indent=2) + "\n")
            if not train_cfg.get("resume", False):
                self.save(self.log_root / "initial.pt")

    def add_git_repo_to_log(self, path):
        source = Path(path).resolve()
        result = subprocess.run(["git", "-C", str(source.parent), "rev-parse", "HEAD"], capture_output=True, text=True)
        self.git_status_repos.append(dict(path=str(source), commit=result.stdout.strip(), error=result.stderr.strip()))
        if self.log_root is not None:
            (self.log_root / "params/source.json").write_text(json.dumps(self.git_status_repos, indent=2) + "\n")

    def save(self, path):
        checkpoint.save(self, path)

    def load(self, path):
        checkpoint.load(self, path)
        if self.log_root is not None:
            self.save(self.log_root / "resumed.pt")

    def get_inference_policy(self, device=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError("Create the runner on the intended inference device")
        self.components.policy.eval()
        return self.components.policy

    def update_learning_rates(self):
        pairs = dict(
            actor=(self.components.policy_learner.actor_optimizer, self.options.actor_optimizer),
            critic=(self.components.policy_learner.value_optimizer, self.options.critic_optimizer),
            world_model=(self.components.wm_learner.optimizer, self.options.wm_optimizer),
        )
        return {
            name: set_learning_rate(optimizer, cfg.lr, self.options.schedule, self.learner_round)
            for name, (optimizer, cfg) in pairs.items()
        }

    def iteration(self, train_actor: bool) -> dict:
        started = time.perf_counter()
        rates = self.update_learning_rates()
        metrics = self.components.collector.collect(
            None, self.components.policy_learner.actor_updates, self.warmup_complete
        )
        torch.cuda.synchronize(self.device)
        collected = time.perf_counter()
        self.components.wm_learner.update_normalization(self.components.replay, self.components.collector.ids)
        batches = torch.stack(
            [
                TensorDict(
                    self.components.wm_learner.sample(
                        self.components.replay, self.components.collector.ids, self.components.generators["replay"]
                    ),
                    batch_size=[],
                )
                for _ in range(self.options.world_model.num_batches)
            ]
        )
        wm = self.components.wm_learner.update(batches)
        torch.cuda.synchronize(self.device)
        fitted = time.perf_counter()
        data = self.components.collector.sample_actor(
            self.options.policy, self.options.world_model, self.components.generators["actor"]
        )
        inputs = self.components.world_env.prepare(data, self.components.generators["observation"])
        policy = self.components.policy_learner.update(inputs, train_actor=train_actor)
        torch.cuda.synchronize(self.device)
        updated = time.perf_counter()
        metrics.update(self.components.collector.metrics(policy.pop("reward_means")))
        metrics.update({key if "/" in key else "Train/" + key: value for key, value in policy.items()})
        metrics.update({"WorldModel/" + key: value for key, value in wm.items()})
        metrics.update({"Train/" + key + "_learning_rate": value for key, value in rates.items()})
        metrics.update(
            {
                "Performance/collection_seconds": collected - started,
                "Performance/wm_seconds": fitted - collected,
                "Performance/policy_critic_seconds": updated - fitted,
                "Performance/collection_steps": self.components.collector.steps,
                "Performance/collected_this_round": self.components.collector.last_steps,
                "EMA/updates": self.components.policy_learner.actor_updates,
            }
        )
        captures = self.components.wm_learner.capture_status | self.components.policy_learner.capture_status
        metrics.update({"Performance/cuda_graph_" + name: float(active) for name, active in captures.items()})
        return metrics

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            raise ValueError("SM collection requires native reset/history semantics")
        if self.log_root is None:
            raise ValueError("Training requires an explicit output directory")
        writer = SummaryWriter(str(self.log_root / "tensorboard"))
        phases = [] if self.warmup_complete else [("warmup", self.options.collection.warmup_iterations)]
        phases.append(("train", num_learning_iterations))
        try:
            if len(self.components.replay) < self.options.collection.initial_steps:
                self.components.collector.collect(
                    self.options.collection.initial_steps,
                    self.components.policy_learner.actor_updates,
                    self.warmup_complete,
                )
            previous = time.perf_counter()
            with (self.log_root / "metrics.jsonl").open("a", buffering=1) as stream:
                for phase, count in phases:
                    for index in range(count):
                        metrics = self.iteration(train_actor=phase == "train")
                        if index % self.options.world_model.validation_interval == 0:
                            self.latest_metrics = self.components.wm_learner.validate(
                                self.components.replay,
                                self.components.collector.ids,
                                self.components.generators["replay"],
                            )
                            with (self.log_root / "wm-metrics.jsonl").open("a") as diagnostic:
                                diagnostic.write(
                                    json.dumps(dict(iteration=self.current_learning_iteration, **self.latest_metrics))
                                    + "\n"
                                )
                            for key, value in self.latest_metrics.items():
                                if key.startswith("current/") and (
                                    key.startswith(f"current/h{self.options.world_model.horizon}/")
                                    or key.endswith("/root_m")
                                ):
                                    metrics["WorldModel/" + key] = value
                        scalars = {self.scalar_tag(key): float(value) for key, value in metrics.items()}
                        if not all(math.isfinite(value) for value in scalars.values()):
                            raise FloatingPointError("Nonfinite training metrics")
                        now = time.perf_counter()
                        scalars["Performance/iteration_seconds"] = now - previous
                        scalars["Performance/peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
                        previous = now
                        stream.write(
                            json.dumps(
                                dict(
                                    phase=phase,
                                    iteration=self.current_learning_iteration,
                                    learner_round=self.learner_round,
                                    actor_updates=self.components.policy_learner.actor_updates,
                                    time=time.time(),
                                    metrics=scalars,
                                )
                            )
                            + "\n"
                        )
                        if phase == "train":
                            for key, value in scalars.items():
                                writer.add_scalar(key, value, self.current_learning_iteration)
                        print(
                            "[SuperMimic]"
                            f" {phase} {index + 1}/{count} actor={self.components.policy_learner.actor_updates} "
                            f"seconds={scalars['Performance/iteration_seconds']:.3f}",
                            flush=True,
                        )
                        self.learner_round += 1
                        if phase == "train":
                            self.current_learning_iteration += 1
                            if self.current_learning_iteration % self.cfg["save_interval"] == 0:
                                self.save(self.log_root / f"model_{self.current_learning_iteration}.pt")
                    if phase == "warmup":
                        self.warmup_complete = True
                        self.save(self.log_root / "warmup.pt")
            self.save(self.log_root / f"model_{self.current_learning_iteration}.pt")
        finally:
            writer.close()
            self.components.collector.close()

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
