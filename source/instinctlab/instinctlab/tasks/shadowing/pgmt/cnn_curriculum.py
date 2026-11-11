"""Exact CNN actor with an actor-update-clocked flat terrain pretraining phase."""

import json
import os
import torch

from .bptt_mdp import raw_history
from .contact_runner import ContactBPTTRunner
from .mdp.terrain_commands import TerrainMotion


class FlatWarmupTerrainMotion(TerrainMotion):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        self.original_types = terrain.terrain_types.clone()
        self.original_levels = terrain.terrain_levels.clone()
        if cfg.post_warmup_terrain_mode not in ("curriculum", "uniform"):
            raise ValueError("Post-warmup terrain mode must be curriculum or uniform")
        self.original_curriculum = cfg.enable_terrain_curriculum and cfg.post_warmup_terrain_mode == "curriculum"
        self.original_randomize = cfg.post_warmup_terrain_mode == "uniform"
        cfg.enable_terrain_curriculum = self.original_curriculum
        cfg.randomize_terrain_on_reset = self.original_randomize
        self.flat_warmup_active = cfg.flat_warmup
        if self.flat_warmup_active:
            flat_columns = (self.terrain.column_families == 0).nonzero().flatten()
            if not len(flat_columns):
                raise ValueError("Flat warmup requires physical flat terrain columns")
            ids = torch.arange(env.num_envs, device=self.device)
            terrain.terrain_types.copy_(flat_columns[ids % len(flat_columns)])
            terrain.terrain_levels.zero_()
            terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
            self.cfg.enable_terrain_curriculum = False
            self.cfg.randomize_terrain_on_reset = False
        self.sampling_contract["flat_warmup"] = cfg.flat_warmup
        self.sampling_contract["post_warmup_terrain_mode"] = cfg.post_warmup_terrain_mode
        self.metrics["flat_warmup"] = torch.full((env.num_envs,), float(self.flat_warmup_active), device=self.device)

    def finish_flat_warmup(self):
        """Change assignments only immediately before the runner resets all environments."""
        if not self.flat_warmup_active:
            raise RuntimeError("Flat warmup can finish only once")
        terrain = self.env.scene.terrain
        terrain.terrain_types.copy_(self.original_types)
        terrain.terrain_levels.copy_(self.original_levels)
        terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        self.cfg.enable_terrain_curriculum = self.original_curriculum
        self.cfg.randomize_terrain_on_reset = self.original_randomize
        self.flat_warmup_active = False
        self.metrics["flat_warmup"].zero_()
        # The forced boundary is not a success or a failure of the old flat episode.
        self.started.zero_()


class CNNFlatWarmupRunner(ContactBPTTRunner):
    @torch.inference_mode()
    def collect(self, steps=None):
        command = self.rewards.command
        reset_history = None
        if command.flat_warmup_active and self.actor_updates >= self.cfg["flat_warmup_updates"]:
            # Keep real successor records intact, but prohibit a WM sequence across the reset.
            if len(self.replay):
                self.replay.data["done"][(self.replay.pointer - 1) % self.replay.capacity].fill_(True)
            command.finish_flat_warmup()
            self.observation, _ = self.env.reset()
            self.episode_rewards.running.zero_()
            if self.native.episode_length_buf.any() or command.terrain.outside.any():
                raise AssertionError("Terrain transition did not produce clean initial episodes")
            reset_history = raw_history(self.observation).clone()
            if not torch.isfinite(self.observation).all():
                raise FloatingPointError("Nonfinite observation after terrain transition")
            record = dict(
                actor_updates=self.actor_updates,
                collection_steps=self.collection_steps,
                phase="terrain",
                forced_reset=True,
                reset_is_failure=False,
                replay_retained=True,
                previous_transition_marked_done=True,
                fixed_wm_validation="original flat cohort retained",
            )
            if self.rank == 0:
                (self.log_root / "terrain_transition.json").write_text(json.dumps(record, indent=2) + "\n")
            print("TERRAIN_TRANSITION " + json.dumps(record), flush=True)
        result = super().collect(steps)
        if reset_history is not None:
            ids = torch.arange(self.env.num_envs, device=self.device)
            ages = torch.full_like(ids, len(self.replay) - self.last_collection_steps)
            reconstructed = self.replay.histories(ages, ids)
            if not torch.equal(reset_history, reconstructed):
                raise AssertionError("Replay history crossed the forced terrain reset")
            if self.rank == 0:
                record["reset_episode_steps_zero"] = True
                record["reset_outside_false"] = True
                record["replay_initial_history_exact"] = True
                (self.log_root / "terrain_transition.json").write_text(json.dumps(record, indent=2) + "\n")
        result["Curriculum/flat_warmup"] = float(command.flat_warmup_active)
        result["Curriculum/actor_updates_before_collection"] = float(self.actor_updates)
        return result

    def write_extra_scalars(self, writer):
        super().write_extra_scalars(writer)
        if writer is None:
            return
        # Preserve the native last-completed-episode activation mean exactly.
        # The BPTT runner otherwise drops extras[log] and these termination tags.
        for key, value in self.native.extras.get("log", {}).items():
            if key.startswith("Episode_Termination/"):
                writer.add_scalar(key, value, self.current_learning_iteration)

    def save(self, path, infos=None):
        super().save(path, infos)
        if self.rank == 0:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            command = self.rewards.command
            saved["cnn_curriculum"] = dict(
                flat_warmup_active=command.flat_warmup_active,
                flat_warmup_updates=self.cfg["flat_warmup_updates"],
                original_types=command.original_types.cpu(),
                original_levels=command.original_levels.cpu(),
                original_curriculum=command.original_curriculum,
                original_randomize=command.original_randomize,
                actor_type="terrain_cnn32",
                source_architecture="20261105_cnn_dance2",
                frozen_validation=(
                    None
                    if self.frozen_validation is None
                    else {key: value.cpu() for key, value in self.frozen_validation.items()}
                ),
            )
            torch.save(saved, str(path) + ".cnn-tmp")
            os.replace(str(path) + ".cnn-tmp", path)

    def load(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved["cnn_curriculum"]
        if state["flat_warmup_updates"] != self.cfg["flat_warmup_updates"]:
            raise ValueError("Resume cannot change the flat warmup update threshold")
        command = self.rewards.command
        command.original_types.copy_(state["original_types"])
        command.original_levels.copy_(state["original_levels"])
        command.original_curriculum = state["original_curriculum"]
        command.original_randomize = state.get("original_randomize", False)
        command.flat_warmup_active = state["flat_warmup_active"]
        command.cfg.enable_terrain_curriculum = False if command.flat_warmup_active else command.original_curriculum
        command.cfg.randomize_terrain_on_reset = False if command.flat_warmup_active else command.original_randomize
        command.metrics["flat_warmup"].fill_(float(command.flat_warmup_active))
        # TerrainRunner.load restores saved live assignments and resets before restoring BPTT counters.
        result = super().load(path)
        cohort = None if self.cfg.get("resume_terrain_change", False) else state["frozen_validation"]
        self.frozen_validation = (
            None if cohort is None else {key: value.to(self.device) for key, value in cohort.items()}
        )
        if self.cfg.get("resume_terrain_change", False):
            print(
                "TERRAIN_TRANSFER "
                + json.dumps(
                    dict(
                        parent_checkpoint=str(path),
                        actor_updates=self.actor_updates,
                        wm_updates=self.wm_updates,
                        learner_round=self.learner_round,
                        replay_size=len(self.replay),
                        frozen_validation_cleared=self.frozen_validation is None,
                        flat_warmup_active=command.flat_warmup_active,
                        platform_width=command.cfg.spawn_center_region_width,
                        off_platform_half_width=command.cfg.off_platform_half_width,
                        root_only=command.cfg.spawn_root_only,
                    )
                ),
                flush=True,
            )
            if self.log_root:
                self.save(str(self.log_root / "resumed.pt"))
        return result
