"""Named tensor layouts. Sizes are derived once, never repeated at call sites."""

from dataclasses import dataclass
from functools import cached_property
from typing import Mapping


@dataclass(frozen=True)
class TensorLayout:
    fields: tuple[tuple[str, int], ...]

    @cached_property
    def slices(self) -> Mapping[str, slice]:
        offset, result = 0, {}
        for name, width in self.fields:
            if name in result or width <= 0:
                raise ValueError(f"Invalid tensor field {name}: {width}")
            result[name] = slice(offset, offset + width)
            offset += width
        return result

    @property
    def width(self) -> int:
        return sum(width for _, width in self.fields)

    def __getitem__(self, name: str) -> slice:
        return self.slices[name]

    def unpack(self, tensor):
        return {name: tensor[..., section] for name, section in self.slices.items()}


@dataclass(frozen=True)
class RobotLayout:
    joints: int
    spatial: int = 3
    quaternion: int = 4

    @cached_property
    def state(self) -> TensorLayout:
        return TensorLayout(
            (
                ("position", self.spatial),
                ("rotation", self.quaternion),
                ("linear_velocity", self.spatial),
                ("angular_velocity", self.spatial),
                ("joint_position", self.joints),
                ("joint_velocity", self.joints),
            )
        )

    @cached_property
    def delta(self) -> TensorLayout:
        return TensorLayout(
            tuple((name, self.spatial if name == "rotation" else width) for name, width in self.state.fields)
        )

    @cached_property
    def velocity(self) -> TensorLayout:
        return TensorLayout(tuple((name, width) for name, width in self.delta.fields if "velocity" in name))

    @property
    def feature_width(self) -> int:
        # Height, projected gravity, two root velocities, joint positions/velocities.
        return 1 + self.spatial * 3 + self.joints * 2
