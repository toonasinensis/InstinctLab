"""SuperMimic Muon/Adam grouping with a device learning-rate schedule.

The desktop implementation freezes Muon's Python LR in a captured graph. This
implementation uses the same PyTorch Muon update with a mutable device scalar.
"""

import math
import torch
from torch import nn

from instinctlab.tasks.shadowing.pgmt.supermimic.config import OptimizerCfg, ScheduleCfg


class CapturableMuon(torch.optim.Muon):
    """PyTorch Muon math with a device LR instead of a capture-time float."""

    @torch.no_grad()
    def step(self, closure=None):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        if closure is not None:
            raise ValueError("PGMT Muon updates do not use closures")
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
                ratio = _adjust_lr(1.0, group["adjust_lr_fn"], parameter.shape)
                parameter.mul_(1 - lr * group["weight_decay"])
                parameter.addcmul_(update.to(parameter.dtype), (-lr * ratio).to(parameter.dtype))


class OptimizerGroup:
    """One checkpointable optimizer interface for disjoint Muon and Adam groups."""

    def __init__(self, module, matrices, lr, betas=(0.9, 0.999), weight_decay=0.0, decoupled_decay=False):
        selected = {id(parameter) for parameter in matrices}
        self.optimizers = []
        if selected:
            self.optimizers.append(
                CapturableMuon(
                    [p for p in module.parameters() if id(p) in selected],
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                )
            )
        adam = torch.optim.AdamW if decoupled_decay else torch.optim.Adam
        self.optimizers.append(
            adam(
                [p for p in module.parameters() if id(p) not in selected],
                lr=torch.tensor(lr, device=next(module.parameters()).device),
                betas=betas,
                weight_decay=weight_decay,
                fused=next(module.parameters()).is_cuda,
                capturable=next(module.parameters()).is_cuda,
            )
        )
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                if not torch.is_tensor(group["lr"]):
                    group["lr"] = torch.tensor(group["lr"], device=next(module.parameters()).device)

    @property
    def state(self):
        return {p: state for optimizer in self.optimizers for p, state in optimizer.state.items()}

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {str(i): optimizer.state_dict() for i, optimizer in enumerate(self.optimizers)}

    def load_state_dict(self, state):
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state[str(i)])
            for group in optimizer.param_groups:
                group["lr"] = group["lr"].to(group["params"][0].device)


def hidden_matrices(module: nn.Sequential, *, output_in_sequence: bool = True):
    """Desktop rule: first and output Linear weights use Adam; hidden matrices use Muon."""
    weights = [layer.weight for layer in module if isinstance(layer, nn.Linear)]
    return weights[1:-1] if output_in_sequence else weights[1:]


def build_optimizers(module, matrices, cfg: OptimizerCfg):
    if cfg.kind not in ("adam", "adamw", "muon"):
        raise ValueError(f"Unknown optimizer: {cfg.kind}")
    return OptimizerGroup(
        module,
        matrices if cfg.kind == "muon" else [],
        cfg.lr,
        cfg.betas,
        cfg.weight_decay,
        cfg.kind == "adamw" or (cfg.kind == "muon" and cfg.weight_decay != 0),
    )


def optimize(optimizers, loss, parameters, cfg: OptimizerCfg):
    # Adapted from SuperMimic _optimize_step; async finite check is capture-safe.
    optimizers.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
    torch._assert_async(torch.isfinite(norm), "Nonfinite learner gradient")
    if cfg.clip_grad_value:
        nn.utils.clip_grad_value_(parameters, cfg.clip_grad_value)
    optimizers.step()
    return norm.detach()


def set_learning_rate(optimizers, base_lr: float, schedule: ScheduleCfg, iteration: int):
    progress = min(max(iteration, 0) / schedule.duration, 1.0)
    factor = schedule.min_factor + (1 - schedule.min_factor) * (1 + math.cos(math.pi * progress)) / 2
    for group in optimizers.param_groups:
        group["lr"].fill_(base_lr * factor)
    return base_lr * factor
