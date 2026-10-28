"""Simulator-independent PGMT MDP primitives (SI units, explicit engineering choices)."""

import torch

GROUPS = ("upper", "lower", "auxiliary", "terrain")
FAMILIES = ("flat", "slope", "stairs", "boxes", "rough")


def tolerance_error(error, level, family, maximum, alpha=1.0):
    enabled = ((family == 1) | (family == 2) | (family == 3)).to(error.dtype)
    tolerance = maximum * (level.to(error.dtype) / 9).clamp(0, 1) * enabled * alpha
    while tolerance.ndim < error.ndim:
        tolerance = tolerance.unsqueeze(-1)
    return (error - tolerance).clamp_min(0)


def corrected_planar_velocity(reference_velocity, position_error, gain=1.0, cap=0.5, gate_speed=0.2):
    """Inputs/outputs in the SAME reference yaw frame; per-axis correction clipping."""
    speed = reference_velocity[..., :2].norm(dim=-1, keepdim=True)
    gate = (speed / gate_speed).clamp(0, 1)
    result = torch.zeros_like(reference_velocity)
    result[..., :2] = reference_velocity[..., :2] + (gate * gain * position_error[..., :2]).clamp(-cap, cap)
    return result


def tangent_velocity(velocity, normal):
    normal = torch.nn.functional.normalize(normal, dim=-1)
    return velocity - (velocity * normal).sum(-1, keepdim=True) * normal


def grouped_gae(rewards, values, dones, last_values, gamma, lam):
    """Four returns; sum the unnormalized group advantages BEFORE normalization."""
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_values if step == len(rewards) - 1 else values[step + 1]
        alive = 1 - dones[step].to(values.dtype).reshape(-1, 1)
        delta = rewards[step] + gamma * alive * next_value - values[step]
        advantage = delta + gamma * lam * alive * advantage
        returns[step] = values[step] + advantage
    return returns, (returns - values).sum(-1, keepdim=True)


def reference_column_map(frames=11, old_width=65, new_width=68):
    """Selector uses FRAME-major channels; observation normalizer is TERM-major."""
    return (torch.arange(frames)[:, None] * new_width + torch.arange(old_width)).flatten()
