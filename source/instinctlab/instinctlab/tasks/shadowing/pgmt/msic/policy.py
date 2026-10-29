"""Load a native InstinctLab PGMT terrain actor and its frozen observation statistics."""

import numpy as np
import torch
from pathlib import Path
from torch.nn import functional as F

from instinctlab.assets.pgmt_g1 import BODY_NAMES
from instinctlab.tasks.shadowing.pgmt.models.cross_attention import unpack_terms
from instinctlab.tasks.shadowing.pgmt.models.terrain import TerrainActor
from instinctlab.utils.instinct_rl_cli import load_saved_yaml


class TerrainPolicy:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint).resolve()
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        schema = saved.get("schema", saved.get("infos", {}).get("schema"))
        self.pure_wm = schema == "pgmt-purewm-v1"
        if self.pure_wm:
            state, normalizer = saved["actor"], saved["policy_normalizer"]
            self.update = saved["iteration"]
        elif schema == "pgmt-lafan-terrain-v1":
            state = {
                k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")
            }
            normalizer = saved["policy_normalizer_state_dict"]
            self.update = saved["iter"]
        else:
            raise ValueError("Select a native PGMT terrain or pure-WM checkpoint")
        self.actor = TerrainActor().eval()
        self.actor.load_state_dict(state, strict=True)
        self.mean, self.std = normalizer["_mean"], normalizer["_std"]
        self.gate = float(self.actor.terrain_gate)
        self.env_cfg = load_saved_yaml(str(self.path.parent / "params/env.yaml"))
        command = self.env_cfg["commands"]["motion"]
        command.update(
            body_names=list(BODY_NAMES),
            future_step_num=list(range(11)),
            velocity_gain=1.0,
            velocity_cap=0.5,
            velocity_gate_speed=0.2,
        )
        self.provenance = dict(
            checkpoint=str(self.path),
            schema=schema,
            update=self.update,
            actor="native TerrainActor",
            normalization="frozen; raw height map bypass",
            parameters=sum(p.numel() for p in self.actor.parameters()),
        )

    def normalize(self, obs):
        x = torch.cat([obs[key] for key in ("prop", "rbt_cmd_mf", "terrain")], -1)
        normalized = (x - self.mean) / (self.std + 0.01)
        return torch.cat((normalized[:, :1678], x[:, 1678:]), -1)

    def _forward(self, obs):
        x = self.normalize(obs)
        actor = self.actor
        history = unpack_terms(x[:, :930], actor.prop_widths, 10)
        reference = unpack_terms(x[:, 930:1678], actor.reference_widths, 11)
        heights = x[:, 1678:]
        tokens = actor.state_embedding(history)
        state, history_weights = actor.history_attention(tokens[:, -1:], tokens, return_weights=True)
        centers = 0.8 * actor.terrain_selector(
            torch.cat((heights, state[:, 0], reference.flatten(1)), -1)
        ).tanh().reshape(-1, 4, 2)
        grid = centers[:, :, None, None] + actor.patch_offsets[None, None]
        patches = F.grid_sample(
            heights.reshape(-1, 1, 21, 21), grid.reshape(-1, 20, 5, 2), padding_mode="border", align_corners=True
        ).reshape(-1, 4, 25)
        terrain = actor.terrain_encoder(torch.cat((patches, centers), -1))
        source = torch.cat((state, actor.command_embedding(reference), terrain), 1)
        bias = x.new_zeros(1, 16)
        bias[:, -4:] = actor.terrain_gate.log()
        positions = torch.cat((actor.key_positions, actor.key_positions.new_zeros(4)))
        intent, weights = actor.intent_attention(
            state, source, actor.query_positions, positions, attention_bias=bias, return_weights=True
        )
        actions = actor.action_head(torch.cat((history[:, -1], intent[:, 0]), -1))
        if self.pure_wm:
            actions = actions.clamp(-20, 20)
        return dict(
            actions=actions,
            history_weights=history_weights,
            intent_weights=weights,
            glimpse_centers=centers,
            glimpse_patches=patches,
        )

    @torch.inference_mode()
    def evaluate(self, observations, flat_height, scale, diagnose=False, mode="real"):
        obs = {k: torch.as_tensor(v, dtype=torch.float32).reshape(1, -1) for k, v in observations.items()}
        if {k: v.shape[-1] for k, v in obs.items()} != {"prop": 930, "rbt_cmd_mf": 748, "terrain": 441}:
            raise ValueError("Expected PGMT observations with widths 930/748/441")
        if not all(torch.isfinite(v).all() for v in obs.values()):
            raise FloatingPointError("Nonfinite PGMT observations")
        outputs = {}
        real = self._forward(obs)
        outputs["real"] = real
        if mode in ("flat", "shuffled") or diagnose:
            for name in ("flat", "shuffled") if diagnose else (mode,):
                modified = {k: v.clone() for k, v in obs.items()}
                if name == "flat":
                    modified["terrain"].fill_(float(np.clip(flat_height, -5, 5)))
                else:
                    modified["terrain"] = modified["terrain"][
                        :, torch.randperm(441, generator=torch.Generator().manual_seed(42))
                    ]
                outputs[name] = self._forward(modified)
        if mode == "off" or diagnose:
            try:
                self.actor.terrain_gate.zero_()
                outputs["off"] = self._forward(obs)
            finally:
                self.actor.terrain_gate.fill_(self.gate)
        result = {k: v[0].numpy() for k, v in outputs[mode].items()}
        result["counterfactual"] = {}
        for name, output in outputs.items():
            if name == "real":
                continue
            diff = (output["actions"] - real["actions"])[0].numpy() * scale
            result["counterfactual"][name] = dict(
                target_rmse_rad=float(np.sqrt(np.mean(diff**2))), target_max_rad=float(np.abs(diff).max())
            )
        if not np.isfinite(result["actions"]).all():
            raise FloatingPointError("Nonfinite action")
        return result
