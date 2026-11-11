"""DDP training entries and explicit learning rates for PGMT BPTT."""

import math
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def learning_rates(bptt, world_model, world):
    mode = bptt.get("lr_scaling", "none")
    if mode not in ("none", "sqrt"):
        raise ValueError(f"Unknown BPTT learning-rate scaling: {mode}")
    base = dict(actor=bptt["actor_lr"], critic=bptt["critic_lr"], world_model=world_model["learning_rate"])
    factor = math.sqrt(world) if mode == "sqrt" else 1.0
    return dict(
        scaling=mode, world_size=world, base=base, effective={key: value * factor for key, value in base.items()}
    )


def scheduled_learning_rates(rates, schedule, iteration):
    """Schedule the already scaled rates by outer learner round, including warmup."""
    if schedule is None:
        return rates.copy()
    if schedule["type"] != "cosine" or schedule["total_iterations"] <= 0:
        raise ValueError(f"Invalid BPTT learning-rate schedule: {schedule}")
    floor = schedule["min_factor"]
    if not 0 <= floor <= 1:
        raise ValueError("BPTT cosine min_factor must be between zero and one")
    progress = min(iteration / schedule["total_iterations"], 1.0)
    factor = floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    return {name: value * factor for name, value in rates.items()}


class TrainingObjective(nn.Module):
    """Register only the trainable network, including its complete loss forward.

    The callback's runner is not a Module. Frozen WM/critic input gradients in
    the actor objective therefore do not register a second parameter reducer.
    """

    def __init__(self, network, forward):
        super().__init__()
        self.network = network
        self.objective = forward

    def forward(self, *inputs):
        return self.objective(*inputs)


def distributed_objective(network, forward, device):
    objective = TrainingObjective(network, forward)
    if dist.is_initialized():
        objective = DistributedDataParallel(
            objective,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    return objective


def backward(loss, network):
    loss.backward()
    if dist.is_initialized():
        # The previous averaging path supplied zeros for globally unused
        # parameters. Preserve Adam/AdamW moments and decay in that case;
        # locally unused parameters already receive their DDP-reduced gradient.
        for parameter in network.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
