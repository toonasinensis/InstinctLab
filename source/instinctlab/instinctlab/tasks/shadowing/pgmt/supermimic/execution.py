"""CUDA update capture adapted from HybridRobotics SuperMimic (Apache-2.0).

Compilation and optimizer capture are independent, explicit execution choices.
The algorithm owns parameters; compiled callables never own checkpoint state.
"""

import functools
import torch
from collections.abc import Callable, Iterable
from tensordict import TensorDict
from torch import nn


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
        self._warmup = self._initial_warmup = warmup
        self._inputs: TensorDict | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def reset(self):
        """Discard captured optimizer state after a checkpoint load."""
        self._inputs, self._graph = None, None
        self._identity, self._outputs = [], {}
        self._warmup = self._initial_warmup

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
            assert self._graph is not None
            self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs


def compile_function(function, enabled: bool):
    if not enabled:
        return function
    return torch.compile(function, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
