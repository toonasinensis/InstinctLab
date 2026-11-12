"""Flat pretraining and weighted terrain assignment, independent of training algorithms."""

import torch

from .terrain_commands import TerrainMotion


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


class Scan187Motion(FlatWarmupTerrainMotion):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.terrain_sampling_weights:
            # Geometry columns are discrete; environment assignment probabilities
            # retain the requested mixture without enlarging the terrain atlas.
            names = self.terrain.column_names
            weights = torch.tensor(
                [cfg.terrain_sampling_weights[name] / names.count(name) for name in names], device=self.device
            )
            self.original_types.copy_(torch.multinomial(weights, env.num_envs, replacement=True))
