"""Newton collection and terminal snapshots, independent of either learner."""

import torch

from instinctlab.tasks.shadowing.pgmt.episode_rewards import EpisodeRewards


def ordinary(value):
    with torch.inference_mode(False):
        return value.detach().clone()


class Collector:
    def __init__(
        self,
        env,
        policy,
        mdp,
        replay,
        foot_rewards,
        analytic_foot,
        cfg,
        generator,
        flat_warmup_updates,
        terrain_gate_ramp,
    ):
        self.env, self.native = env, env.unwrapped
        self.policy, self.mdp, self.replay, self.cfg, self.generator = policy, mdp, replay, cfg, generator
        self.rewards, self.foot_rewards, self.analytic_foot = mdp.rewards, foot_rewards, analytic_foot
        self.command = self.rewards.command
        self.flat_warmup_updates, self.terrain_gate_ramp = flat_warmup_updates, terrain_gate_ramp
        self.device = mdp.default_q.device
        self.ids = torch.arange(env.num_envs, device=self.device)
        self.observation, _ = env.get_observations()
        self.steps = self.last_steps = 0
        self.step_accumulator = 0.0
        self.terminal = {}
        self.term_groups = dict(
            base19=tuple(self.rewards.terms), contact4=tuple("contact/" + key for key in foot_rewards.names)
        )
        if analytic_foot is not None:
            self.term_groups["analytic2"] = tuple(analytic_foot.names)
        self.terms = sum(self.term_groups.values(), ())
        self.episode_rewards = EpisodeRewards(
            env.num_envs, self.terms, len(self.rewards.terms), self.native.max_episode_length_s, self.device
        )
        self.original_reward_compute = self.native.reward_manager.compute
        self.native.reward_manager.compute = self.capture_terminal
        self.normalize_during_warmup = cfg.update_normalizer_during_warmup

    def close(self):
        self.native.reward_manager.compute = self.original_reward_compute

    def capture_terminal(self, dt):
        result = self.original_reward_compute(dt)
        command = self.command
        terminal = {"ref_" + key: ordinary(value) for key, value in self.rewards.context().items()}
        terminal.update(
            next_state=ordinary(self.rewards.state()),
            next_contact=ordinary(command.contacts.contact.float()),
            next_force=ordinary(command.contacts.force),
            next_support=ordinary(command.terrain.support_z),
            derivative_valid=ordinary(command.contacts.valid_derivative.float()),
            action=ordinary(self.native.action_manager.action),
            native_reward=ordinary(
                torch.stack(
                    [
                        self.native.reward_manager._termwise_reward_buf[group][name] * dt
                        for key in self.rewards.terms
                        for group, name in [key.split("/")]
                    ],
                    -1,
                )
            ),
        )
        terminal["native_foot_reward"] = ordinary(
            self.foot_rewards(
                terminal["next_state"],
                terminal["next_contact"][:, command.foot_ids],
                terminal["ref_ref_contact"],
                terminal["ref_ref_valid"],
                self.ids,
            )
        )
        episode_terms = [terminal["native_reward"], terminal["native_foot_reward"]]
        if self.analytic_foot is not None:
            self.analytic_foot.memory.mul_(terminal["derivative_valid"])
            terminal["foot_acceleration_memory"] = self.analytic_foot.memory.clone()
            bodies = self.rewards.bodies(terminal["next_state"], self.ids)
            terms, memory = self.analytic_foot.evaluate(
                bodies,
                command.contacts.acceleration,
                terminal["derivative_valid"],
                self.ids,
                self.analytic_foot.memory,
                1,
                contact=terminal["next_contact"][:, command.foot_ids],
                hard=True,
            )
            terminal["native_analytic_reward"] = ordinary(terms)
            self.analytic_foot.memory.copy_(memory)
            self.analytic_foot.memory[self.native.reset_buf.bool()] = 0
            episode_terms.append(terminal["native_analytic_reward"])
        self.episode_rewards.add(torch.cat(episode_terms, -1), self.native.reset_buf.bool())
        self.terminal = terminal
        return result

    def finish_flat_warmup(self, actor_updates):
        if not self.command.flat_warmup_active or actor_updates < self.flat_warmup_updates:
            return
        if len(self.replay):
            self.replay.data["done"][(self.replay.pointer - 1) % self.replay.capacity].fill_(True)
        self.command.finish_flat_warmup()
        self.observation, _ = self.env.reset()
        self.episode_rewards.running.zero_()
        if self.native.episode_length_buf.any() or self.command.terrain.outside.any():
            raise AssertionError("Terrain transition did not reset native episode state")

    @torch.inference_mode()
    def collect(self, steps, actor_updates, warmup_complete):
        self.finish_flat_warmup(actor_updates)
        self.foot_rewards.set_updates(actor_updates)
        gate = min(1.0, (actor_updates + 1) / self.terrain_gate_ramp) if warmup_complete else 0.0
        self.policy.network.eval()
        self.policy.network.terrain_gate.fill_(gate)
        self.command.metrics["terrain_gate"][:] = gate
        if steps is None:
            self.step_accumulator += self.cfg.steps_per_update
            steps = int(self.step_accumulator)
            self.step_accumulator -= steps
        self.last_steps, rows = steps, []
        for _ in range(steps):
            if not warmup_complete and self.normalize_during_warmup:
                self.policy.normalizer.update(self.observation)
            record = dict(
                state=ordinary(self.rewards.state()),
                contact=ordinary(self.command.contacts.contact.float()),
                previous_action=ordinary(self.native.action_manager.action),
                support=ordinary(self.command.terrain.support_z),
                previous_ref_velocity=ordinary(self.command.contacts.previous_ref_velocity),
                history=ordinary(self.mdp.raw_history(self.observation)),
                episode_step=ordinary(self.native.episode_length_buf),
                **{key: ordinary(value) for key, value in self.mdp.placement().items()},
            )
            noise = (
                torch.randn(self.env.num_envs, self.env.num_actions, device=self.device, generator=self.generator)
                * self.cfg.action_noise
            )
            action = self.policy.collect(self.observation) + noise
            self.observation, _, done, info = self.env.step(action)
            record.update(
                self.terminal,
                done=ordinary(done.bool()),
                action_noise=ordinary(noise),
                failure=ordinary(self.native.reset_terminated.bool()),
            )
            self.replay.add(record)
            rows.append({key: ordinary(value) for key, value in info["step"].items()})
            self.steps += 1
        self.policy.network.intent = self.policy.network.intent.detach()
        return {key: torch.stack([row[key] for row in rows]).mean() for key in rows[0]} | {
            "Curriculum/flat_warmup": float(self.command.flat_warmup_active),
            "Curriculum/actor_updates_before_collection": float(actor_updates),
        }

    def sample_actor(self, cfg, wm_cfg, generator):
        horizon = max(cfg.horizon, wm_cfg.horizon, *wm_cfg.validation_horizons)
        eligible = self.ids[self.ids % wm_cfg.holdout_stride != 0]
        pieces, total = [], 0
        for _ in range(self.cfg.sampling_attempts):
            ids = eligible[
                torch.randint(
                    len(eligible),
                    (self.cfg.sampling_oversample * cfg.batch_size,),
                    device=self.device,
                    generator=generator,
                )
            ]
            ages = torch.randint(len(self.replay) - horizon + 1, ids.shape, device=self.device, generator=generator)
            rows = self.replay.rows(ages)
            frame, motion = (self.replay.data[key][rows, ids] for key in ("frame", "motion"))
            valid = frame + horizon + self.mdp.reference_frames < self.command.label_lengths[motion]
            selected = valid.nonzero().flatten()[: cfg.batch_size - total]
            pieces.append((ids[selected], ages[selected]))
            total += len(selected)
            if total == cfg.batch_size:
                return self.replay.sequences(
                    torch.cat([part[0] for part in pieces]),
                    horizon,
                    generator,
                    ages=torch.cat([part[1] for part in pieces]),
                )
        raise RuntimeError("Insufficient motion windows for the configured actor batch")

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

    def metrics(self, imagined_means):
        rows = self.replay.rows(torch.arange(len(self.replay) - self.last_steps, len(self.replay), device=self.device))
        rewards = [self.replay.data[key][rows] for key in ("native_reward", "native_foot_reward")]
        if self.analytic_foot is not None:
            rewards.append(self.replay.data["native_analytic_reward"][rows])
        result = self.reward_metrics("RealReward", torch.cat(rewards, -1).mean((0, 1)))
        result.update(self.reward_metrics("ImaginedReward", imagined_means))
        result.update(self.episode_rewards.take())
        result["Contact/scale"] = float(self.foot_rewards.scale)
        done = self.replay.data["done"][rows].bool()
        if done.any():
            lengths = self.replay.data["episode_step"][rows][done].float() + 1
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
