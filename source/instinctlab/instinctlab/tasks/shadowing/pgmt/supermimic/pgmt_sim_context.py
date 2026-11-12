"""IsaacLab/Newton physical context with the established PGMT column semantics."""

import torch

import warp as wp

from .sim_context import SimContext


def ordinary(value):
    with torch.inference_mode(False):
        return value.detach().clone()


def simulator_context(native):
    names = ("physics_material", "add_joint_default_pos", "base_com")
    if any(native.event_manager.get_term_cfg(name).mode != "startup" for name in names):
        raise ValueError("SM context currently requires startup-only physical randomization")
    material = native.event_manager.get_term_cfg("physics_material").func._impl
    event = native.event_manager.get_term_cfg("base_com").func
    robot = native.scene["robot"].data
    return ordinary(
        torch.cat(
            (
                wp.to_torch(material._friction_binding),
                robot.default_joint_pos.torch - robot.default_joint_pos_nominal,
                (robot.body_com_pos_b.torch - event.default_com[..., :3])[:, event.asset_cfg.body_ids].flatten(1),
            ),
            -1,
        )
    )


class PGMTSimContext(SimContext):
    def __init__(self, env):
        # PGMT normalizes the concatenated context in Dynamics.input_norm.
        # Preserve that convention rather than applying the mjlab range transform twice.
        torch.nn.Module.__init__(self)
        self.register_buffer("values", simulator_context(env))

    @property
    def output_dim(self):
        return self.values.shape[-1]

    def read(self, env):
        return self.values
