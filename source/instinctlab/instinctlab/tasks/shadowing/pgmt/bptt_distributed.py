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
