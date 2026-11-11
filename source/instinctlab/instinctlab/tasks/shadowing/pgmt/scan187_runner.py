"""Controlled scan-layout experiments, passive actor EMA and value-gradient fitting."""

import copy
import os
import torch
from torch import nn

from .bptt_distributed import backward
from .cnn_curriculum import CNNFlatWarmupRunner, FlatWarmupTerrainMotion
from .training_optim import MuonAdam


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


class Scan187Runner(CNNFlatWarmupRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.actor_ema = None
        self.gradient_data = None
        self.gradient_metrics = {}
        with torch.random.fork_rng(devices=[torch.device(device)]):
            torch.manual_seed(train_cfg["seed"])
            super().__init__(env, train_cfg, log_dir, device)
        options = train_cfg["experiment"]
        rates = self.learning_rates["effective"]
        if options["actor_optimizer"] == "muon":
            self.actor_optimizer = MuonAdam(
                self.actor, [self.actor.encoder, self.actor.action_head], rates["actor"], (0.9, 0.95)
            )
        if options["wm_optimizer"] == "muon":
            # Keep first/last trunk matrices, heads, terrain encoder and vectors
            # on AdamW; orthogonalize only the two hidden transition matrices.
            linear = [layer for layer in self.wm.trunk if isinstance(layer, nn.Linear)]
            self.wm_optimizer = MuonAdam(self.wm, linear[1:], rates["world_model"], (0.9, 0.999), weight_decay=1e-5)
        self.actor_ema = copy.deepcopy(self.actor).requires_grad_(False)
        self.gradient_generator = torch.Generator(device=self.device).manual_seed(train_cfg["seed"] + 15013)
        if self.log_root:
            self.save(str(self.log_root / "initial.pt"))

    def imagine_rollout(self, data, contexts, proprio_noise, observation_noise):
        if self.cfg["experiment"]["value_gradient_weight"] > 0 and torch.is_grad_enabled():
            self.gradient_context = (data, contexts[0], observation_noise[0])
        return super().imagine_rollout(data, contexts, proprio_noise, observation_noise)

    def actor_loss(self, data):
        if self.cfg["experiment"]["value_gradient_weight"] <= 0:
            return super().actor_loss(data)
        # Supervise at imagination START states. This avoids confusing incoming
        # rewards/contact memory with the return from an interior state. It is a
        # start-state adaptation of PartyMimic, not its all-interior-state target.
        initial = data["state"][:, 0].detach().requires_grad_()
        data = {**data, "state": torch.cat((initial[:, None], data["state"][:, 1:]), 1)}
        result = super().actor_loss(data)
        target = torch.autograd.grad(-len(initial) * result[0], initial, retain_graph=True)[0]
        context_data, context, noise = self.gradient_context
        self.gradient_data = dict(
            state=initial.detach(),
            target=target.detach(),
            history=data["history"][:, 0].detach(),
            support=data["support"][:, 0].detach(),
            context=context,
            noise=noise,
            ids=data["env_id"],
            scale=(data["next_state"][:, 0] - initial.detach()).std(0).clamp_min(1e-3),
        )
        return result

    def gradient_loss(self, indices):
        data = self.gradient_data
        state = data["state"][indices].detach().requires_grad_()
        context = {key: value[indices] for key, value in data["context"].items()}
        noise = {key: value[indices] for key, value in data["noise"].items()}

        def features(value):
            return self.features(
                value, data["history"][indices], data["support"][indices], context, data["ids"][indices], noise
            )[0]

        x = features(state)
        leaf = x.detach().requires_grad_()
        # Double backward touches the critic only. The terrain/FK input Jacobian
        # is evaluated by a centered directional difference, as in PartyMimic.
        u = torch.autograd.grad(self.value(leaf).sum(), leaf, create_graph=True)[0]
        prediction = torch.autograd.grad(x, state, grad_outputs=u.detach())[0].detach().requires_grad_()
        target = data["target"][indices]
        scale = data["scale"]

        def tangent(value):
            rotation = state[:, 3:7].detach()
            angular = value[:, 3:7] - (value[:, 3:7] * rotation).sum(-1, keepdim=True) * rotation
            return torch.cat((value[:, :3], angular, value[:, 7:]), -1)

        predicted, expected = tangent(prediction) * scale, tangent(target) * scale
        error = (predicted - expected).square().sum() / expected.square().sum().clamp_min(1e-12)
        direction = tangent(torch.autograd.grad(error, prediction)[0])
        with torch.no_grad():
            epsilon = 0.05 / (direction / scale).norm(dim=-1).clamp_min(1e-12)

            def shifted(sign):
                value = state + sign * epsilon[:, None] * direction
                return torch.cat((value[:, :3], nn.functional.normalize(value[:, 3:7], dim=-1), value[:, 7:]), -1)

            jacobian = (features(shifted(1)) - features(shifted(-1))) / (2 * epsilon[:, None])
            cosine = nn.functional.cosine_similarity(predicted, expected, dim=-1).mean()
        return (u * jacobian).sum(), error.detach(), cosine

    def fit_value(self, features, targets, mask):
        if self.gradient_data is None:
            return super().fit_value(features, targets, mask)
        features, targets = features.flatten(0, 1), targets.flatten()
        selected = mask.flatten().bool()
        features, targets = features[selected], targets[selected]
        losses, errors, cosines = [], [], []
        for epoch in range(self.bptt["critic_epochs"]):
            order = torch.randperm(len(targets), device=self.device, generator=self.critic_generator)
            for indices in order.tensor_split(self.bptt["critic_minibatches"]):
                loss = self.value_objective(features[indices], targets[indices])
                value_loss = loss.detach()
                if epoch == 0:
                    count = min(4096, len(self.gradient_data["state"]))
                    ids = torch.randperm(
                        len(self.gradient_data["state"]), device=self.device, generator=self.gradient_generator
                    )[:count]
                    surrogate, error, cosine = self.gradient_loss(ids)
                    loss = loss + self.cfg["experiment"]["value_gradient_weight"] * surrogate
                    errors.append(error)
                    cosines.append(cosine)
                self.value_optimizer.zero_grad(set_to_none=True)
                backward(loss, self.value)
                nn.utils.clip_grad_norm_(self.value.parameters(), 1.0, error_if_nonfinite=True)
                self.value_optimizer.step()
                self.critic_updates += 1
                losses.append(value_loss)
        self.gradient_metrics = dict(
            relative_error=float(torch.stack(errors).mean()), cosine=float(torch.stack(cosines).mean())
        )
        self.gradient_data = None
        return float(torch.stack(losses).mean())

    def update_bptt(self, actor=True):
        stats = super().update_bptt(actor)
        if actor:
            decay = self.cfg["experiment"]["ema_decay"]
            with torch.no_grad():
                torch._foreach_lerp_(list(self.actor_ema.parameters()), list(self.actor.parameters()), 1 - decay)
                for averaged, current in zip(self.actor_ema.buffers(), self.actor.buffers()):
                    averaged.copy_(current)
        return stats

    def write_extra_scalars(self, writer):
        super().write_extra_scalars(writer)
        if writer is not None:
            writer.add_scalar("EMA/updates", self.actor_updates, self.current_learning_iteration)
            for key, value in self.gradient_metrics.items():
                writer.add_scalar("ValueGradient/" + key, value, self.current_learning_iteration)

    def save(self, path, infos=None):
        super().save(path, infos)
        if self.actor_ema is not None:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state["scan187"] = dict(
                gradient_rng=self.gradient_generator.get_state().cpu(),
                options=self.cfg["experiment"],
                actor_ema={k: v.cpu() for k, v in self.actor_ema.state_dict().items()},
                ema_updates=self.actor_updates,
                scan=self.native.cfg.height_scan,
                heightfield_resolution=self.options["resolution"],
                gradient_target="imagined H32 return gradient at start states; history/support held fixed",
            )
            torch.save(state, str(path) + ".ema-tmp")
            os.replace(str(path) + ".ema-tmp", path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)["scan187"]
        if state["options"] != self.cfg["experiment"]:
            raise ValueError("Scan187 experiment options changed on resume")
        result = super().load(path)
        self.actor_ema.load_state_dict(state["actor_ema"])
        self.gradient_generator.set_state(state["gradient_rng"])
        return result
