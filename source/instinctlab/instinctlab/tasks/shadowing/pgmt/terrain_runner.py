"""Explicit stage-one warm-start and stateful terrain PPO checkpoints."""

import os
import torch
import torch.distributed as dist

from instinct_rl.modules.normalizer import EmpiricalNormalization
from instinct_rl.runners import OnPolicyRunner

from .terrain_ppo import TerrainPPO


class StableNormalization(EmpiricalNormalization):
    def update(self, x):
        # The pinned normalizer replaces std/count buffers. Keep replacements writable on reload.
        with torch.inference_mode(False), torch.no_grad():
            super().update(x)

    def sync_across_processes(self):
        with torch.inference_mode(False), torch.no_grad():
            super().sync_across_processes()


class TerrainNormalization(StableNormalization):
    def forward(self, x):
        normalized = super().forward(x)
        return torch.cat((normalized[:, :1678], x[:, 1678:]), -1)



class TerrainRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        # The pinned runner only resolves built-in algorithm names. Keep the specialization task-local.
        self.alg = TerrainPPO(self.alg.actor_critic, device=device, **self.alg_cfg)
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env, env.get_obs_format(), env.num_actions, num_rewards=4
        )
        self.alg.normalizers = self.normalizers
        self.source_checkpoint = None
        if train_cfg.get("warm_start"):
            self.warm_start(train_cfg["warm_start"])

    def warm_start(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved["infos"]["schema"] != "pgmt-lafan-native-v1":
            raise ValueError("Warm-start requires a native PGMT stage-one checkpoint")
        model = self.alg.actor_critic
        source = saved["model_state_dict"]
        actor = model.actor.state_dict()
        for name, value in actor.items():
            key = "actor." + name
            if key not in source:
                if name.startswith(("terrain_", "patch_offsets")):
                    continue
                raise ValueError(f"Stage-one actor tensor missing: {key}")
            old = source[key]
            if name == "command_embedding.weight":
                value.zero_()
                value[:, :65].copy_(old)
            else:
                value.copy_(old)
        model.actor.load_state_dict(actor, strict=True)
        model.std.data.copy_(source["std"])
        for group, normalizer in self.normalizers.items():
            state = normalizer.state_dict()
            old = saved[group + "_normalizer_state_dict"]
            for name, value in state.items():
                if group == "policy" and value.ndim > 0 and value.shape[-1] == 2119:
                    value[..., :1645].copy_(old[name])
                else:
                    value.copy_(old[name])
            normalizer.load_state_dict(state)
        self.source_checkpoint = str(path)
        self.alg.learning_rate = saved["infos"]["rank_states"][0]["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        print(f"[PGMT] Warm-started actor and normalizers from {path}; critic/optimizer are new")

    def rollout_step(self, obs, extra_obs):
        gate = min(1.0, self.current_learning_iteration / max(1, self.cfg.get("terrain_gate_ramp", 200)))
        self.alg.actor_critic.actor.terrain_gate.fill_(gate)
        self.env.unwrapped.command_manager.get_term("motion").metrics["terrain_gate"][:] = gate
        return super().rollout_step(obs, extra_obs)

    def save(self, path, infos=None):
        env = self.env.unwrapped
        c = env.command_manager.get_term("motion")
        t = env.scene.terrain
        state = dict(
            failures=c.stratum_failure.cpu(),
            visits=c.stratum_visits.cpu(),
            levels=t.terrain_levels.cpu() if t.cfg.terrain_type == "generator" else None,
            types=t.terrain_types.cpu() if t.cfg.terrain_type == "generator" else None,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device),
            learning_rate=self.alg.learning_rate,
        )
        states = [None] * dist.get_world_size() if dist.is_initialized() else [state]
        if dist.is_initialized():
            dist.all_gather_object(states, state)
        temporary = str(path) + ".tmp"
        super().save(
            temporary,
            infos=dict(
                schema=env.cfg.mdp_schema,
                num_envs=env.num_envs,
                rank_states=states,
                source_checkpoint=self.source_checkpoint,
                terrain_gate_ramp=self.cfg.get("terrain_gate_ramp", 200),
                user=infos,
            ),
        )

        if not self.is_mp_rank_other_process():
            os.replace(temporary, path)
        if dist.is_initialized():
            dist.barrier()

    def load(self, path):
        info = torch.load(path, map_location="cpu", weights_only=True)["infos"]
        env = self.env.unwrapped
        c = env.command_manager.get_term("motion")
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if (
            info["schema"] != env.cfg.mdp_schema
            or info["num_envs"] != env.num_envs
            or len(info["rank_states"]) != world
            or info["terrain_gate_ramp"] != self.cfg.get("terrain_gate_ramp", 200)
        ):
            raise ValueError("Terrain resume changed schema, rank count, environment count or gate ramp")
        super().load(path)
        state = info["rank_states"][rank]
        c.stratum_failure.copy_(state["failures"])
        c.stratum_visits.copy_(state["visits"])
        t = env.scene.terrain
        if state["levels"] is not None:
            t.terrain_levels.copy_(state["levels"])
            t.terrain_types.copy_(state["types"])
            t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
        self.alg.learning_rate = state["learning_rate"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        self.source_checkpoint = info["source_checkpoint"]
        c.started.zero_()
        env.termination_manager.terminated.zero_()
        self.env.reset()
        return info["user"]
