"""Run with torchrun on a verified runtime; no simulator is required."""

import copy
import importlib.util
import math
import os
import torch
import torch.distributed as dist
import unittest
from pathlib import Path
from torch import nn
from torch.utils.checkpoint import checkpoint

spec = importlib.util.spec_from_file_location(
    "bptt_distributed", Path(__file__).parents[1] / "instinctlab/tasks/shadowing/pgmt/bptt_distributed.py"
)
bptt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bptt)


class ConditionalNetwork(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.main = nn.Linear(8, 8)
        self.local = nn.Linear(8, 8)
        self.unused = nn.Linear(8, 8)
        self.register_buffer("local_statistic", torch.tensor(float(rank)))

    def forward(self, x, branch):
        x = self.main(x)
        if branch != "none":
            x = x + self.local(x)
        if branch == "all":
            x = x + self.unused(x)
        return x


def manual_average(network):
    parameters = list(network.parameters())
    flat = torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten() for p in parameters])
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    offset = 0
    for parameter in parameters:
        parameter.grad = flat[offset : offset + parameter.numel()].view_as(parameter)
        offset += parameter.numel()


class TestBPTTDistributed(unittest.TestCase):
    def test_learning_rates_are_derived_once_from_base(self):
        options = dict(actor_lr=1e-5, critic_lr=3e-4)
        wm = dict(learning_rate=3e-4)
        for world in (1, 2, 4):
            for mode in ("none", "sqrt"):
                configured = dict(options, lr_scaling=mode)
                rates = bptt.learning_rates(configured, wm, world)
                scale = math.sqrt(world) if mode == "sqrt" else 1
                self.assertEqual(rates["base"], dict(actor=1e-5, critic=3e-4, world_model=3e-4))
                self.assertEqual(rates["effective"]["actor"], 1e-5 * scale)
                self.assertEqual(rates["effective"]["world_model"], 3e-4 * scale)
                self.assertEqual(bptt.learning_rates(configured, wm, world), rates)
                self.assertEqual(configured, dict(options, lr_scaling=mode))
        with self.assertRaises(ValueError):
            bptt.learning_rates(dict(options, lr_scaling="linear"), wm, 2)

    @unittest.skipUnless("RANK" in os.environ, "requires torchrun")
    def test_ddp_matches_manual_average_with_masks_and_unused_parameters(self):
        rank = dist.get_rank()
        device = (
            torch.device("cuda", int(os.environ["LOCAL_RANK"])) if torch.cuda.is_available() else torch.device("cpu")
        )
        torch.manual_seed(42)
        network = ConditionalNetwork(rank).to(device)
        reference = copy.deepcopy(network)
        frozen = nn.Linear(8, 8).to(device).requires_grad_(False)
        branch = "all"

        def objective(module, x):
            # Multiple uses and nonreentrant recomputation, one complete DDP forward.
            for _ in range(32):
                x = torch.tanh(checkpoint(module, x, branch, use_reentrant=False) + 0.1 * frozen(x))
            return x.square().mean(), x.detach()

        entry = bptt.distributed_objective(network, lambda x: objective(network, x), device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=1e-5)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-5)
        generator = torch.Generator(device=device).manual_seed(100 + rank)
        for step in range(3):
            # Unequal valid counts must still give each rank equal weight.
            x = torch.randn(rank + 2, 8, device=device, generator=generator)
            branch = "all" if step == 0 else "local" if step == 1 and rank % 2 else "none"
            reference_optimizer.zero_grad(set_to_none=True)
            expected, _ = objective(reference, x)
            expected.backward()
            manual_average(reference)
            optimizer.zero_grad(set_to_none=True)
            actual, _ = entry(x)
            bptt.backward(actual, network)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            for parameter, ref in zip(network.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, ref.grad, rtol=1e-5, atol=1e-7)
            for module, opt in ((network, optimizer), (reference, reference_optimizer)):
                nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
            torch.testing.assert_close(list(network.parameters()), list(reference.parameters()), rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(optimizer.state_dict(), reference_optimizer.state_dict(), rtol=1e-5, atol=1e-7)
            self.assertEqual(network.local_statistic.item(), rank)
            self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))
            for parameter in network.parameters():
                first = parameter.detach().clone()
                dist.broadcast(first, src=0)
                torch.testing.assert_close(parameter, first, rtol=0, atol=0)
        self.assertFalse(any(key.startswith("module.") for key in network.state_dict()))


if __name__ == "__main__":
    if "RANK" in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    try:
        unittest.main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
