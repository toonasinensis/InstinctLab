"""PGMT simulation/replay coordinator with separate desktop-style SM learners.

Environment collection, analytic MDP, diagnostics and checkpoint compatibility
stay shared with PGMT. Optimizer work belongs to WorldModelLearner/PolicyLearner;
only those updates may be captured. Contexts, random draws and logging stay here.
"""

import json
import torch
from tensordict import TensorDict
from torch import nn

from .bptt_distributed import scheduled_learning_rates
from .bptt_mdp import PROPRIO_WIDTHS
from .contact_runner import ContactBPTTRunner
from .learners.learner_utils import Optimizers
from .learners.policy_learner import PolicyLearner
from .learners.wm_learner import WorldModelLearner
from .scan187_runner import Scan187Runner
from .wm_dynamics import TerrainDynamics


def middle_weights(module):
    return [layer.weight for layer in module if isinstance(layer, nn.Linear)][1:-1]


class DesktopRunner(Scan187Runner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.backend = train_cfg["learner_backend"]
        super().__init__(env, train_cfg, log_dir, device)
        if self.world != 1 or self.cfg["experiment"]["value_gradient_weight"]:
            raise ValueError("Desktop learner ablations use one GPU and ordinary imagined critic regression")
        if self.wm.prediction_mode == "velocity":
            # Head width must not perturb initialization of the shared contact head.
            with torch.random.fork_rng(devices=[self.device]):
                torch.manual_seed(train_cfg["seed"] + 1741)
                reference = TerrainDynamics(
                    self.xi.shape[-1],
                    hidden=self.options["hidden"],
                    amp=self.options["amp"],
                    predict_contact=True,
                    predict_force=False,
                    contact_indices=self.rewards.command.foot_ids,
                    scan_points=len(self.heightfield.local_xy),
                    ground_query=self.heightfield.query,
                ).to(self.device)
            source = reference.state_dict()
            self.wm.load_state_dict(
                {k: source[k] for k, v in self.wm.state_dict().items() if k in source and source[k].shape == v.shape},
                strict=False,
            )
            del reference
        if self.backend["capture"]:
            self.actor.action_from_inputs = torch.compile(
                self.actor.action_from_inputs, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
            self.wm.predict_delta = torch.compile(
                self.wm.predict_delta, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
        rates = self.learning_rates["effective"]
        desktop = self.backend["optimizers"] == "desktop"
        actor_matrices = middle_weights(self.actor.action_head) if desktop else []
        value_matrices = middle_weights(self.value.network) if desktop else []
        # The heads are outside trunk: all trunk Linear layers after its input
        # layer are hidden-to-hidden matrices, not output matrices.
        wm_matrices = [layer.weight for layer in self.wm.trunk if isinstance(layer, nn.Linear)][1:]
        self.actor_optimizer = Optimizers(self.actor, actor_matrices, rates["actor"], (0.9, 0.95))
        self.wm_optimizer = Optimizers(self.wm, wm_matrices, rates["world_model"], weight_decay=0 if desktop else 1e-5)
        self.value_optimizer = Optimizers(self.value, value_matrices, rates["critic"])
        self.wm_learner = WorldModelLearner(self.wm, self.wm_optimizer, self._world_model_loss, self.backend["capture"])
        self.policy_learner = PolicyLearner(
            self.actor,
            self.wm,
            self.value,
            self.actor_optimizer,
            self.value_optimizer,
            self.prepared_imagination,
            self.value_loss,
            self.backend["capture"],
            state_modules=(self.normalizers["policy"], self.heightfield),
            state_tensors=(self.foot_rewards.scale, self.xi),
        )
        if self.log_root:
            (self.log_root / "params").mkdir(exist_ok=True)
            (self.log_root / "params" / "learner_backend.json").write_text(json.dumps(self.backend, indent=2) + "\n")
            self.save(str(self.log_root / "initial.pt"))

    def prepare_imagination(self, data):
        start = self.start(data)
        batch, horizon = len(data["env_id"]), self.bptt["horizon"]
        contexts = [TensorDict(self.mdp.context(start, step), batch_size=[batch]) for step in range(horizon + 1)]
        observation_noise = [self.mdp.draw_noise(batch, self.mdp.observation_widths, self.noise_generator)]
        proprio_noise = []
        for _ in range(horizon):
            proprio_noise.append(
                TensorDict(self.mdp.draw_noise(batch, PROPRIO_WIDTHS, self.noise_generator), batch_size=[batch])
            )
            observation_noise.append(self.mdp.draw_noise(batch, self.mdp.observation_widths, self.noise_generator))
        # Scalar batch_size permits differently shaped data, time-major contexts
        # and noise in the same captured input structure.
        return TensorDict(
            dict(
                data=TensorDict(data, batch_size=[]),
                contexts=torch.stack(contexts),
                proprio_noise=torch.stack(proprio_noise),
                observation_noise=torch.stack([TensorDict(n, batch_size=[batch]) for n in observation_noise]),
            ),
            batch_size=[],
        )

    def prepared_imagination(self, inputs):
        loss, features, targets, alive, valid, _, _ = self.imagine_rollout(
            inputs["data"].to_dict(),
            [row.to_dict() for row in inputs["contexts"].unbind(0)],
            [row.to_dict() for row in inputs["proprio_noise"].unbind(0)],
            [row.to_dict() for row in inputs["observation_noise"].unbind(0)],
        )
        return loss, features, targets, alive, valid, self.imagined_reward_means

    def record_imagined_rewards(self, means):
        self.imagined_reward_means = means

    def fit_world_model(self, update_moments=True):
        if update_moments:
            self.update_moments()
        self.wm.train()
        batches = torch.stack(
            [
                TensorDict(self.sample(self.options["train_horizon"]), batch_size=[])
                for _ in range(self.options["updates"])
            ]
        )
        result = self.wm_learner.update(batches)
        if not torch.isfinite(result["grad_norm"]):
            raise FloatingPointError("Nonfinite WM gradient")
        self.wm_updates += self.options["updates"]
        self.training_loss = float(result["loss"])
        self.contact_bce = float(result["contact_bce"])

    def update_bptt(self, actor=True):
        inputs = self.prepare_imagination(self.sample_actor())
        self.wm_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        self.wm.requires_grad_(False)
        self.value.requires_grad_(False)
        try:
            if actor:
                result = self.policy_learner.update_actor(inputs)
                self.actor_updates += 1
            else:
                with torch.no_grad():
                    loss, features, targets, alive, valid, means = self.prepared_imagination(inputs)
                result = dict(
                    loss=loss,
                    features=features,
                    targets=targets,
                    survival=alive,
                    valid=valid,
                    reward_means=means,
                    grad_norm=loss.new_zeros(()),
                    encoder_grad_norm=loss.new_zeros(()),
                )
            if not result["valid"].all():
                raise RuntimeError("Invalid imagined domain; training aborted")
            if any(p.grad is not None for p in self.wm.parameters()):
                raise AssertionError("Actor wrote world-model parameter gradients")
        finally:
            self.wm.requires_grad_(True)
            self.value.requires_grad_(True)
        features, targets = result["features"].flatten(0, 1), result["targets"].flatten()
        orders = torch.stack(
            [
                torch.randperm(len(targets), device=self.device, generator=self.critic_generator)
                for _ in range(self.bptt["critic_epochs"])
            ]
        )
        critic_input = TensorDict(dict(features=features, targets=targets, orders=orders), batch_size=[])
        # Record graphs only after warmup; the shared pool replay order is WM,
        # actor, critic throughout formal training.
        value = (self.policy_learner.update_value if actor else self.policy_learner._value)(critic_input)
        self.critic_updates += self.bptt["critic_epochs"] * self.bptt["critic_minibatches"]
        if actor:
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(self.actor_ema.parameters()),
                    list(self.actor.parameters()),
                    1 - self.cfg["experiment"]["ema_decay"],
                )
                for averaged, current in zip(self.actor_ema.buffers(), self.actor.buffers()):
                    averaged.copy_(current)
        ContactBPTTRunner.record_imagined_rewards(self, result["reward_means"])
        return dict(
            actor_loss=float(result["loss"]),
            critic_loss=float(value["loss"]),
            actor_grad_norm=float(result["grad_norm"]),
            imagination_survival=float(result["survival"]),
            critic_valid_fraction=1.0,
            **{"Encoder/encoder_gradient_norm": float(result["encoder_grad_norm"])}
        )

    def update_learning_rates(self, iteration):
        rates = scheduled_learning_rates(self.learning_rates["effective"], self.bptt["lr_schedule"], iteration)
        for name, optimizer in (
            ("actor", self.actor_optimizer),
            ("critic", self.value_optimizer),
            ("world_model", self.wm_optimizer),
        ):
            for group in optimizer.param_groups:
                group["lr"].fill_(rates[name])

    def write_extra_scalars(self, writer):
        super().write_extra_scalars(writer)
        if writer is not None and self.backend["capture"]:
            for name, update in (
                ("wm", self.wm_learner.update),
                ("actor", self.policy_learner.update_actor),
                ("critic", self.policy_learner.update_value),
            ):
                writer.add_scalar(
                    "Performance/cuda_graph_" + name, float(update._graph is not None), self.current_learning_iteration
                )

    def save(self, path, infos=None):
        super().save(path, infos)
        if self.actor_ema is not None:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state["desktop_backend"] = dict(schema=1, options=self.backend)
            torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("desktop_backend", {}).get("options") != self.backend:
            raise ValueError("Desktop learner settings differ from the checkpoint")
        return super().load(path)
