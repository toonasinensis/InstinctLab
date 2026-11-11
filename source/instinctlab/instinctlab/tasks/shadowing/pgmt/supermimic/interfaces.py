"""Algorithm boundaries: concrete tasks provide semantics, learners provide updates."""

import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict
from torch import nn


class StateRepresentation(nn.Module, ABC):
    @property
    @abstractmethod
    def prediction_dim(self) -> int: ...

    @abstractmethod
    def represent(self, state: torch.Tensor, ground: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def target(self, state: torch.Tensor, successor: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def apply_prediction(self, state: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def state_error(self, predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor: ...


class TerrainQuery(nn.Module, ABC):
    @abstractmethod
    def query(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def local(self, position: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class DynamicsModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, state, action, previous_action, heights, context, *, recompute=False):
        """Return successor state and optional contact logits."""


class Policy(nn.Module, ABC):
    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor: ...


class WorldModelEnv(ABC):
    @abstractmethod
    def prepare(self, data: dict, generator: torch.Generator) -> TensorDict:
        """Prepare external randomness/reference data before graph capture."""

    @abstractmethod
    def reset(self, inputs: TensorDict) -> None: ...

    @abstractmethod
    def step(self, step: int) -> None: ...

    @abstractmethod
    def rollout(self, inputs: TensorDict) -> dict[str, torch.Tensor]: ...


class Learner(ABC):
    @abstractmethod
    def update(self, data: TensorDict) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None: ...
