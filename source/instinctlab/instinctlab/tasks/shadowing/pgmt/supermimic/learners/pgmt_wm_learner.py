"""SuperMimic-style autoregressive WM learner with PGMT terminal masks and BCE.

Sampling is external to captured updates. Loss is evaluated on integrated states;
normalization updates consume only newly collected training environments.
"""

import torch
from torch.nn import functional as F
from typing import TypedDict

from instinctlab.tasks.shadowing.pgmt.supermimic.learners.pgmt_learner_utils import (
    CapturedUpdate,
    compile_function,
    graph_state,
)
from instinctlab.tasks.shadowing.pgmt.supermimic.learners.wm_learner import (
    WorldModelLearner as SuperMimicWorldModelLearner,
)
from instinctlab.tasks.shadowing.pgmt.tensor_math import yaw_angle

from .pgmt_learner_utils import build_optimizers, hidden_matrices, optimize


class Prediction(TypedDict):
    loss: torch.Tensor
    states: torch.Tensor
    errors: torch.Tensor
    mask: torch.Tensor
    probabilities: torch.Tensor | None


class PGMTWorldModelLearner(SuperMimicWorldModelLearner):
    def __init__(self, dynamics, context, cfg, optimizer_cfg, execution):
        self.context, self.optimizer_cfg, self.execution = context, optimizer_cfg, execution
        super().__init__(dynamics, dynamics.representation, cfg, device=str(context.device))
        self.optimizer = self.optimizers[0]
        self.loss_forward = compile_function(self.loss, execution.compile != "none")
        self.updates = 0
        self.frozen_validation = None

    def _build_optimizers(self):
        return [
            build_optimizers(
                self.dynamics, hidden_matrices(self.dynamics.trunk, output_in_sequence=False), self.optimizer_cfg
            )
        ]

    def _loss_weights(self):
        # PGMT's integrated-state error includes quaternion and terminal masks.
        return {}

    def _compile(self, fn):
        return compile_function(fn, self.execution.compile != "none")

    def _capture(self, fn, state):
        return CapturedUpdate(fn, self.graph_state, self.execution.capture_warmup) if self.execution.cuda_graph else fn

    def _prepare_batches(self, batches):
        return batches

    def _batch_loss(self, td):
        loss, contact = self.loss_forward(td.to_dict())
        return loss, contact.reshape(1)

    def _optimize(self, loss):
        return optimize(self.optimizer, loss, self._all_params, self.optimizer_cfg)

    def _summarize(self, stats, norms):
        # Preserve PGMT last-minibatch loss and maximum gradient norm.
        return dict(loss=stats[-1][0], contact_bce=stats[-1][1], grad_norm=torch.stack(norms).amax())

    @property
    def capture_status(self):
        return {"wm": isinstance(self._captured_update, CapturedUpdate) and self._captured_update.captured}

    def graph_state(self):
        return graph_state([self.dynamics, self.dynamics.terrain], [self.optimizer]) + [self.context]

    def update(self, batches):
        result = super().update(batches)
        self.updates += len(batches)
        return result

    def predict(self, data, horizon: int) -> Prediction:
        model = self.dynamics
        rep = model.representation
        state = data["state"][:, 0]
        norm = model.delta_norm if model.error_norm is None else model.error_norm
        scale = rep.nominal_error if rep.cfg.loss_scale == "bryson" else torch.maximum(norm.std(), rep.error_floor)
        states, probabilities, losses, errors, masks = [], [], [], [], []
        valid_prefix = state.new_ones(len(state))
        for step in range(horizon):
            heights, valid = model.terrain.local(
                state[:, rep.layout.state["position"]], yaw_angle(state[:, rep.layout.state["rotation"]])
            )
            state, logits = model.predict_transition(
                state, data["action"][:, step], data["previous_action"][:, step], heights, self.context[data["env_id"]]
            )
            error = rep.state_error(state, data["next_state"][:, step])
            loss = (error / scale).square().mean(-1)
            if logits is not None:
                target = data["next_contact"][:, step, model.contact_indices]
                loss = loss + model.cfg.contact_bce_weight * F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ).mean(-1)
                probabilities.append(logits.sigmoid())
            valid_prefix = valid_prefix * valid
            masks.append(data["mask"][:, step] * valid_prefix)
            states.append(state)
            losses.append(loss)
            errors.append(error)
        mask = torch.stack(masks, 1)
        loss = (torch.stack(losses, 1) * mask).sum() / mask.sum().clamp_min(1)
        return Prediction(
            loss=loss,
            states=torch.stack(states, 1),
            errors=torch.stack(errors, 1),
            mask=mask,
            probabilities=torch.stack(probabilities, 1) if probabilities else None,
        )

    def loss(self, data):
        result = self.predict(data, self.cfg.horizon)
        p, mask = result["probabilities"], result["mask"]
        metric = result["loss"].new_zeros(())
        if p is not None:
            with torch.no_grad():
                target = data["next_contact"][:, : p.shape[1], self.dynamics.contact_indices]
                epsilon = self.dynamics.cfg.probability_epsilon
                bce = F.binary_cross_entropy(p.clamp(epsilon, 1 - epsilon), target, reduction="none").mean(-1)
                metric = (bce * mask).sum() / mask.sum().clamp_min(1)
        return result["loss"], metric

    @torch.no_grad()
    def update_normalization(self, replay, env_ids):
        model, rep = self.dynamics, self.dynamics.representation
        ids = env_ids[env_ids % self.cfg.holdout_stride != 0]
        inputs, deltas, errors = [], [], []
        ages = torch.arange(len(replay) - replay.added, len(replay), device=ids.device)
        for row in replay.rows(ages):
            data = {key: replay.data[key][row, ids] for key in ("state", "action", "previous_action", "next_state")}
            heights, valid = model.terrain.local(
                data["state"][:, rep.layout.state["position"]],
                yaw_angle(data["state"][:, rep.layout.state["rotation"]]),
            )
            inputs.append(
                model.inputs(data["state"], data["action"], data["previous_action"], heights, self.context[ids])[valid]
            )
            deltas.append(rep.target(data["state"], data["next_state"])[valid])
            if model.error_norm is not None:
                errors.append(rep.state_error(data["next_state"], data["state"])[valid])
        model.input_norm.update(torch.cat(inputs))
        model.delta_norm.update(torch.cat(deltas))
        if model.error_norm is not None:
            model.error_norm.update(torch.cat(errors))
        replay.added = 0

    def sample(self, replay, env_ids, generator, *, validation=False, horizon=None):
        selected = env_ids % self.cfg.holdout_stride == 0
        eligible = env_ids[selected if validation else ~selected]
        count = self.cfg.validation_batch if validation else self.cfg.batch_size
        ids = eligible[torch.randint(len(eligible), (count,), device=eligible.device, generator=generator)]
        return replay.sequences(ids, horizon or self.cfg.horizon, generator, history=False)

    @torch.no_grad()
    def validate(self, replay, env_ids, generator):
        horizon = max(self.cfg.validation_horizons)
        if self.frozen_validation is None:
            self.frozen_validation = self.sample(replay, env_ids, generator, validation=True, horizon=horizon)
        current = self.sample(replay, env_ids, generator, validation=True, horizon=horizon)
        metrics = {}
        layout = self.dynamics.representation.layout
        for name, data in (("fixed", self.frozen_validation), ("current", current)):
            result = self.predict(data, horizon)
            for length in self.cfg.validation_horizons:
                weight = result["mask"][:, length - 1]
                mean = lambda value: (value * weight).sum() / weight.sum().clamp_min(1)
                prefix = f"{name}/h{length}/"
                for field, tag in (
                    ("position", "root_m"),
                    ("rotation", "rotation_rad"),
                    ("joint_position", "joint_rad"),
                ):
                    err = result["errors"][:, length - 1, layout.delta[field]]
                    metrics[prefix + tag] = float(mean(err.square().mean(-1)).sqrt())
                state, target = result["states"][:, length - 1], data["next_state"][:, length - 1]
                scan, valid = self.dynamics.terrain.local(
                    state[:, layout.state["position"]], yaw_angle(state[:, layout.state["rotation"]])
                )
                real_scan, real_valid = self.dynamics.terrain.local(
                    target[:, layout.state["position"]], yaw_angle(target[:, layout.state["rotation"]])
                )
                scan_weight = weight * valid * real_valid
                metrics[prefix + "heightmap_mae_m"] = float(
                    ((scan - real_scan).abs().mean(-1) * scan_weight).sum() / scan_weight.sum().clamp_min(1)
                )
                if result["probabilities"] is not None:
                    predicted = result["probabilities"][:, length - 1] > self.dynamics.cfg.contact_threshold
                    actual = data["next_contact"][:, length - 1, self.dynamics.contact_indices].bool()
                    denominator = mean(predicted.sum(-1) + actual.sum(-1))
                    metrics[prefix + "foot_contact_f1"] = float(
                        2 * mean((predicted & actual).sum(-1)) / denominator.clamp_min(torch.finfo(weight.dtype).eps)
                    )
                metrics[prefix + "valid_fraction"] = float(weight.mean())
        return metrics

    def state_dict(self):
        return dict(
            model=self.dynamics.state_dict(),
            optimizer=self.optimizer.state_dict(),
            updates=self.updates,
            frozen_validation=self.frozen_validation,
        )

    def load_state_dict(self, state):
        if isinstance(self._captured_update, CapturedUpdate):
            self._captured_update.reset()

        self.dynamics.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.frozen_validation = state["frozen_validation"]
