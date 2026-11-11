"""Update capture adapted from HybridRobotics SuperMimic (Apache-2.0).

CapturedUpdate and graph_state follow the desktop learner implementation.
Tensor learning rates also drive Muon, so graph replay retains the PGMT schedule.
"""

import functools
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable, Sequence
from tensordict import TensorDict
from typing import Literal


def graph_state(modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]) -> list[torch.Tensor]:
    """What a captured update reads that Python can rebind: parameters, buffers,
    optimizer state and learning-rate tensors."""
    state: list[torch.Tensor] = []
    for module in modules:
        state += [*module.parameters(), *module.buffers()]
    for optimizer in optimizers:
        state += [t for s in optimizer.state.values() for t in s.values()]
        state += [group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])]
    return state


def _identity(state: list[torch.Tensor]) -> list[int]:
    return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
    return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
    """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as
    a CUDA graph and replays it on the device of ``state()``. ``fn`` must be
    capture-safe: no host syncs, fixed shapes, and every tensor it reads besides
    its inputs updated in place. Python values it reads are fixed at capture, and
    each replay overwrites the outputs it returns. All CapturedUpdates on a device
    record on one stream into one memory pool and must replay in the order they
    were recorded. A replay raises if a tensor in ``state()`` moved; a learning rate
    that changes is a device tensor, changed in place with ``group["lr"].fill_(lr)``."""

    def __init__(
        self,
        fn: Callable[[TensorDict], dict[str, torch.Tensor]],
        state: Callable[[], list[torch.Tensor]],
        warmup: int = 3,
    ):
        self._fn = fn
        self._state = state
        self._warmup = warmup
        self._inputs: TensorDict | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
        state = self._state()
        if not state[0].is_cuda:
            return self._fn(inputs)
        with torch.cuda.device(state[0].device):
            stream, pool = _capture_stream_and_pool(state[0].device)
            if self._inputs is None:
                self._inputs = inputs.clone()
            else:
                self._inputs.update_(inputs)
            if self._warmup > 0:
                self._warmup -= 1
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs = self._fn(self._inputs)
                torch.cuda.current_stream().wait_stream(stream)
                return outputs
            if self._graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._outputs = self._fn(self._inputs)
                self._graph, self._identity = graph, _identity(self._state())
            elif _identity(state) != self._identity:
                raise RuntimeError("a captured tensor moved")
            self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs


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


class Optimizers:
    """One checkpointable optimizer interface for disjoint Muon and Adam groups."""

    def __init__(self, module, matrices, lr, betas=(0.9, 0.999), weight_decay=0.0):
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
        adam = torch.optim.AdamW if weight_decay else torch.optim.Adam
        self.optimizers.append(
            adam(
                [p for p in module.parameters() if id(p) not in selected],
                lr=torch.tensor(lr, device=next(module.parameters()).device),
                betas=betas,
                weight_decay=weight_decay,
                fused=True,
                capturable=True,
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


def optimize(optimizer, loss, parameters, clip_value=0.0):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=False)
    torch._assert_async(torch.isfinite(norm), "Nonfinite learner gradient")
    if clip_value:
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
    optimizer.step()
    return norm
