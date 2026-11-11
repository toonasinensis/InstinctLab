"""Completed-episode reward statistics with the native RewardManager denominator."""

import torch
import torch.distributed as dist


class EpisodeRewards:
    """Accumulate actual transitions across collection rounds, including terminal steps."""

    def __init__(self, num_envs, terms, base_terms, max_seconds, device):
        self.terms, self.base_terms, self.max_seconds = tuple(terms), base_terms, max_seconds
        self.running = torch.zeros(num_envs, len(terms), device=device)
        self.completed = self.running.new_zeros(len(terms))
        self.count = torch.zeros((), dtype=torch.long, device=device)

    @torch.no_grad()
    def add(self, reward, done):
        self.running.add_(reward.detach())
        self.completed.add_(self.running[done].sum(0))
        self.count.add_(done.sum())
        self.running[done] = 0

    @torch.no_grad()
    def take(self):
        count, completed = self.count.clone(), self.completed.clone()
        if dist.is_initialized():
            dist.all_reduce(count)
            dist.all_reduce(completed)
        count = int(count)
        if not count:
            return {}
        means = completed / (count * self.max_seconds)
        metrics = {"Episode_Reward/" + key.replace("/", "_"): float(value) for key, value in zip(self.terms, means)}
        metrics["Episode_Reward/base19"] = float(means[: self.base_terms].sum())
        metrics["Episode_Reward/contact4"] = float(means[self.base_terms : self.base_terms + 4].sum())
        if len(self.terms) > self.base_terms + 4:
            metrics["Episode_Reward/analytic2"] = float(means[self.base_terms + 4 :].sum())
        metrics["Episode_Reward/total"] = float(means.sum())
        metrics["Episode/completed_count"] = count
        self.completed.zero_()
        self.count.zero_()
        return metrics
