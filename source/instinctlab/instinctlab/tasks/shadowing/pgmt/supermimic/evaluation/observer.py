"""Compose passive metric collectors at the native pre-reset boundary."""

import torch
from typing import Protocol


class MetricCollector(Protocol):
    def __call__(self, dt: float) -> None: ...


class TransitionObserver:
    def __init__(self, env, episodes: MetricCollector, *, contact=None, check_purity=False):
        self.env = env
        # Contacts read the active mask before EpisodeMetrics marks terminal episodes finished.
        self.collectors = (contact, episodes) if contact is not None else (episodes,)
        self.check_purity = check_purity
        self.audit = {"calls": 0, "state_unchanged": True, "rng_unchanged": True}

    def snapshot(self):
        native = self.env
        c = native.command_manager.get_term("motion")
        terrain = native.scene.terrain
        tensors = dict(
            root=c.robot.data.root_state_w.torch,
            q=c.robot_joint_pos,
            qd=c.robot_joint_vel,
            reference_time=c.reference._timestamp_torch,
            rotation=c.rotation,
            offset=c.offset,
            support=c.terrain.support_z,
            levels=terrain.terrain_levels,
            types=terrain.terrain_types,
            episode_length=native.episode_length_buf,
            reset=native.reset_buf,
            terminated=native.termination_manager.terminated,
        )
        tensors.update({"contact_" + k: v for k, v in vars(c.contacts).items() if isinstance(v, torch.Tensor)})
        return (
            {k: v.clone() for k, v in tensors.items()},
            torch.get_rng_state(),
            torch.cuda.get_rng_state(native.device),
            native._sim_step_counter,
        )

    def __call__(self, dt):
        before = self.snapshot() if self.check_purity else None
        for collector in self.collectors:
            collector(dt)
        if before is not None:
            after = self.snapshot()
            differences = [k for k in before[0] if not torch.equal(before[0][k], after[0][k])]
            if differences or before[3] != after[3]:
                raise AssertionError(f"Observer modified native state: {differences}")
            if not torch.equal(before[1], after[1]) or not torch.equal(before[2], after[2]):
                raise AssertionError("Observer consumed random numbers")
            self.audit["calls"] += 1
