"""Stage-one initialized SM contact ablation, with identical learners in both arms."""

import torch
import torch.distributed as dist

from .analytic_foot_rewards import AnalyticFootRewards
from .bptt_runner import BPTTRunner
from .episode_rewards import EpisodeRewards
from .foot_rewards import FootContactRewards

# Detailed WM validation remains in wm-metrics.jsonl; the training panel shows essentials.
WM_OVERVIEW_METRICS = {
    "current/h8/root_m",
    "current/h8/rotation_rad",
    "current/h8/joint_rad",
    "current/h8/heightmap_mae_m",
    "current/h8/foot_contact_f1",
    "current/h16/root_m",
    "current/h32/root_m",
}


class ContactBPTTRunner(BPTTRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        if self.wm.predict_force or (self.wm.predict_contact and not self.options.get("feet_only")):
            raise ValueError("Contact ablation supports only a two-foot head or no prediction head")
        if not self.wm.predict_contact and train_cfg["foot_contact"]["enabled"]:
            raise ValueError("Contact rewards require the two-foot prediction head")
        if train_cfg.get("analytic_foot", {}).get("enabled") and not self.wm.predict_contact:
            raise ValueError("Analytic footstep rewards require learned two-foot contact probabilities")
        self.foot_rewards = FootContactRewards(self.rewards, train_cfg["foot_contact"])
        self.analytic_foot = (
            AnalyticFootRewards(self.rewards, train_cfg["analytic_foot"])
            if train_cfg.get("analytic_foot", {}).get("enabled")
            else None
        )
        self.episode_rewards = EpisodeRewards(
            self.env.num_envs,
            [
                *self.rewards.terms,
                *("contact/" + name for name in self.foot_rewards.names),
                *(self.analytic_foot.names if self.analytic_foot else ()),
            ],
            len(self.rewards.terms),
            self.native.max_episode_length_s,
            self.device,
        )
        self.imagined_metrics = {}
        self.real_metrics = {}
        self.centers = None
        self.record_attention = False
        if hasattr(self.actor, "terrain_selector"):
            self.actor.terrain_selector.register_forward_hook(self.record_centers)
        if self.log_root:
            self.save(str(self.log_root / "initial.pt"))

    def record_centers(self, module, inputs, output):
        # Log real collection consistently across eager and compiled learners.
        # Mutating diagnostics inside an activation-checkpointed graph is invalid.
        if self.record_attention:
            self.centers = 0.8 * output.detach().tanh().reshape(-1, 4, 2)

    def save(self, path, infos=None):
        super().save(
            path,
            infos={
                "foot_contact": self.cfg["foot_contact"],
                "analytic_foot": self.cfg.get("analytic_foot"),
                "stage1_checkpoint": self.cfg.get("warm_start"),
                "terrain_collision": "native_mesh_ground_margin_0.009",
                "episode_reward_semantics": "completed_episode_sum_div_max_seconds",
            },
        )

    def collect(self, steps=None):
        self.foot_rewards.set_updates(self.actor_updates)
        self.initial_gate = (
            min(1.0, (self.actor_updates + 1) / self.cfg["terrain_gate_ramp"]) if self.warmup_complete else 0.0
        )
        self.record_attention = True
        try:
            result = super().collect(steps)
        finally:
            self.record_attention = False
        data = self.replay.data
        rows = self.replay.rows(
            torch.arange(len(self.replay) - self.last_collection_steps, len(self.replay), device=self.device)
        )
        base = data["native_reward"][rows].mean((0, 1))
        self.real_metrics = {"RealReward/" + key: float(value) for key, value in zip(self.rewards.terms, base)}
        self.real_metrics["RealReward/base19"] = float(base.sum())
        self.real_metrics["Contact/scale"] = float(self.foot_rewards.scale)
        done = data["done"][rows].bool()
        if done.any():
            lengths = data["episode_step"][rows][done].float() + 1
            self.real_metrics["Episode/mean_episode_length"] = float(lengths.mean())
            self.real_metrics["Episode/median_episode_length"] = float(lengths.median())
        return result

    def capture_terminal(self):
        super().capture_terminal()
        original = self.native.reward_manager.compute

        def compute(dt):
            result = original(dt)
            if hasattr(self, "foot_rewards"):
                terminal = self.terminal
                ids = torch.arange(self.env.num_envs, device=self.device)
                terms = self.foot_rewards(
                    terminal["next_state"],
                    terminal["next_contact"][:, self.rewards.command.foot_ids],
                    terminal["ref_ref_contact"],
                    terminal["ref_ref_valid"],
                    ids,
                )
                terminal["native_foot_reward"] = terms.detach().clone()
                episode_terms = [terminal["native_reward"], terminal["native_foot_reward"]]
                if self.analytic_foot is not None:
                    c = self.rewards.command
                    # Invalid derivatives mark the first step after any kind of reset.
                    self.analytic_foot.memory.mul_(terminal["derivative_valid"])
                    terminal["foot_acceleration_memory"] = self.analytic_foot.memory.clone()
                    bodies = self.rewards.bodies(terminal["next_state"], ids)
                    terms, memory = self.analytic_foot.evaluate(
                        bodies,
                        c.contacts.acceleration,
                        terminal["derivative_valid"],
                        ids,
                        self.analytic_foot.memory,
                        1,
                        contact=terminal["next_contact"][:, c.foot_ids],
                        hard=True,
                    )
                    terminal["native_analytic_reward"] = terms.detach().clone()
                    self.analytic_foot.memory.copy_(memory)
                    self.analytic_foot.memory[self.native.reset_buf.bool()] = 0
                    episode_terms.append(terminal["native_analytic_reward"])
                self.episode_rewards.add(
                    torch.cat(episode_terms, -1),
                    self.native.reset_buf.bool(),
                )
            return result

        self.native.reward_manager.compute = compute

    def augment_imagined_rewards(
        self, rewards, state, context, ids, horizon, bodies, logits, previous_bodies=None, derivative_valid=None
    ):
        rows = ids[:, None].expand(-1, horizon).flatten()
        contact = (
            self.foot_rewards(state, logits.sigmoid(), context["ref_contact"], context["ref_valid"], rows, bodies)
            if self.wm.predict_contact
            else state.new_zeros(len(state), 4)
        )
        terms = [rewards, contact]
        if self.analytic_foot is not None:
            acceleration = (bodies[2] - previous_bodies[2]) / self.rewards.dt
            analytic, _ = self.analytic_foot.evaluate(
                bodies,
                acceleration,
                derivative_valid,
                rows,
                context["foot_acceleration_memory"],
                horizon,
                contact=logits.sigmoid(),
            )
            terms.append(analytic)
        combined = torch.cat(terms, -1)
        return combined, combined.detach().mean(0)

    def record_imagined_rewards(self, means):
        analytic = means[23:]
        means, contact = means[:19], means[19:23]
        self.imagined_metrics = {"ImaginedReward/" + key: float(value) for key, value in zip(self.rewards.terms, means)}
        self.imagined_metrics.update(
            {"ImaginedReward/contact/" + key: float(value) for key, value in zip(self.foot_rewards.names, contact)}
        )
        if self.analytic_foot is not None:
            self.imagined_metrics.update(
                {"ImaginedReward/" + key: float(value) for key, value in zip(self.analytic_foot.names, analytic)}
            )
            self.imagined_metrics["ImaginedReward/analytic2"] = float(analytic.sum())
        self.imagined_metrics["ImaginedReward/base19"] = float(means.sum())
        self.imagined_metrics["ImaginedReward/total"] = float(means.sum() + contact.sum() + analytic.sum())

    def world_model_loss(self, data):
        prediction = self.predict(data, self.options["train_horizon"])
        if not self.wm.predict_contact:
            return prediction[0], prediction[0].new_zeros(())
        probability, mask = prediction[4], prediction[2]
        with torch.no_grad():
            target = self.contact_target(data["next_contact"][:, : probability.shape[1]])
            bce = torch.nn.functional.binary_cross_entropy(
                probability.clamp(1e-6, 1 - 1e-6), target, reduction="none"
            ).mean(-1)
            metric = (bce * mask).sum() / mask.sum().clamp_min(1)
        return prediction[0], metric

    def world_model_training_loss(self, data):
        loss, metric = self._world_model_loss(data)
        self.contact_bce = float(metric.detach())
        return loss

    def update_bptt(self, actor=True):
        result = super().update_bptt(actor=actor)
        for name in ("terrain_selector", "terrain_encoder", "encoder"):
            if not hasattr(self.actor, name):
                continue
            gradients = [p.grad.norm().square() for p in getattr(self.actor, name).parameters() if p.grad is not None]
            prefix = "Encoder/" if name == "encoder" else "Attention/"
            result[prefix + name.removeprefix("terrain_") + "_gradient_norm"] = (
                float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
            )
        return result

    def scalar_tag(self, key):
        if "/" in key:
            if key.startswith("PGMT/raw/"):
                return "Tracking/raw/" + key.removeprefix("PGMT/raw/")
            if key.startswith("PGMT/effective/"):
                return "Tracking/effective/" + key.removeprefix("PGMT/effective/")
            if key.startswith("PGMT/terrain_"):
                return "Terrain/" + key.removeprefix("PGMT/terrain_")
            if key.startswith("PGMT/"):
                return "Episode/" + key.removeprefix("PGMT/")
            return key
        if key.startswith("wm_"):
            return "WorldModel/" + key.removeprefix("wm_")
        if "seconds" in key or "bytes" in key or key in ("real_steps", "collection_steps", "seconds"):
            return "Performance/" + key
        return "Train/" + key

    def write_extra_scalars(self, writer):
        # Partial episodes stay in the accumulator; diagnostics below retain
        # their original dt-scaled per-transition mean semantics.
        episode = self.episode_rewards.take()
        data = self.replay.data
        rows = self.replay.rows(
            torch.arange(len(self.replay) - self.last_collection_steps, len(self.replay), device=self.device)
        )
        contact = data["native_foot_reward"][rows].mean((0, 1))
        self.real_metrics.update(
            {"RealReward/contact/" + key: float(value) for key, value in zip(self.foot_rewards.names, contact)}
        )
        self.real_metrics["RealReward/total"] = self.real_metrics["RealReward/base19"] + float(contact.sum())
        if self.analytic_foot is not None:
            analytic = data["native_analytic_reward"][rows].mean((0, 1))
            self.real_metrics.update(
                {"RealReward/" + key: float(value) for key, value in zip(self.analytic_foot.names, analytic)}
            )
            self.real_metrics["RealReward/analytic2"] = float(analytic.sum())
            self.real_metrics["RealReward/total"] += float(analytic.sum())
        if self.wm.predict_contact:
            self.real_metrics["WorldModel/contact_bce"] = self.contact_bce
        c = self.rewards.command
        terrain = {"Terrain/mean_level": c.terrain.level.float().mean()}
        away = (c.robot_anchor_pos_w[:, :2] - self.native.scene.env_origins[:, :2]).abs().amax(
            -1
        ) > c.cfg.off_platform_half_width
        terrain["Terrain/off_platform_fraction"] = away.float().mean()
        if self.centers is not None:
            pairs = torch.triu_indices(4, 4, 1, device=self.device)
            distance = torch.cdist(self.centers, self.centers)[:, pairs[0], pairs[1]]
            terrain["Attention/center_pair_distance_mean"] = distance.mean()
            moments = torch.stack((self.centers.sum(0), self.centers.square().sum(0)))
            if dist.is_initialized():
                dist.all_reduce(moments)
            count = len(self.centers) * self.world
            center_mean = moments[0] / count
            center_std = ((moments[1] / count - center_mean.square()).clamp_min(0) * count / (count - 1)).sqrt()
            for index in range(4):
                for axis, name in enumerate(("x", "y")):
                    terrain[f"Attention/center{index}_{name}_mean"] = center_mean[index, axis]
                    terrain[f"Attention/center{index}_{name}_std"] = center_std[index, axis]
        for family in ("flat", "slope", "stairs", "boxes", "rough"):
            selected = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
            for column, name in enumerate(c.terrain.column_names):
                if name.startswith(family):
                    selected |= self.native.scene.terrain.terrain_types == column
            totals = torch.stack((c.terrain.level[selected].float().sum(), selected.sum().float()))
            if dist.is_initialized():
                dist.all_reduce(totals)
            if totals[1] > 0:
                terrain["Terrain/" + family + "_level"] = totals[0] / totals[1]
        metrics = {**self.real_metrics, **self.imagined_metrics, **terrain}
        # Equal local environment/batch counts make rank means valid here.
        # Completed episodes and per-family levels use counts above instead.
        keys = [key for key in metrics if not key.startswith("Episode/")]
        values = torch.tensor([metrics[key] for key in keys], device=self.device)
        if dist.is_initialized():
            dist.all_reduce(values)
            values /= self.world
        metrics.update(zip(keys, values.cpu().tolist()))
        if writer is None:
            return
        for key, value in {**episode, **metrics}.items():
            writer.add_scalar(key, value, self.current_learning_iteration)
        for key, value in self.latest_metrics.items():
            if key in WM_OVERVIEW_METRICS:
                writer.add_scalar("WorldModel/" + key, value, self.current_learning_iteration)
