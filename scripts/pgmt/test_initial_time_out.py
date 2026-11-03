"""CPU regression checks for the initial timeout lifecycle and its RNG isolation."""

import torch
import unittest
from types import SimpleNamespace

from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg

from instinctlab.tasks.shadowing.pgmt.config.g1.pgmt_terrain_cfg import (
    G1PGMTTerrainBPTTEnvCfg,
    G1PGMTTerrainBPTTEnvCfg_PLAY,
    G1PGMTTerrainPPOValidationEnvCfg,
)
from instinctlab.tasks.shadowing.pgmt.mdp.terrain_terminations import InitialTimeOut


class InitialTimeOutTests(unittest.TestCase):
    def make_term(self, seed=42, count=4096):
        env = SimpleNamespace(
            num_envs=count,
            device="cpu",
            max_episode_length=500,
            cfg=SimpleNamespace(seed=seed),
            episode_length_buf=torch.zeros(count, dtype=torch.long),
        )
        term = InitialTimeOut(TerminationTermCfg(func=InitialTimeOut, time_out=True), env)
        return env, term

    def test_rng_isolation_and_rank_seed(self):
        state = torch.get_rng_state().clone()
        _, first = self.make_term()
        _, repeat = self.make_term()
        _, other_rank = self.make_term(seed=43)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(first.limits, repeat.limits))
        self.assertFalse(torch.equal(first.limits, other_rank.limits))

    def test_startup_resets_preserve_deadlines_and_true_age(self):
        env, term = self.make_term()
        limits = term.limits.clone()
        for ids in (None, slice(None), torch.arange(env.num_envs)):
            term.reset(ids)
        self.assertTrue(torch.equal(term.limits, limits))
        self.assertEqual(int(env.episode_length_buf.sum()), 0)
        self.assertFalse(term(env).any())

    def test_three_cycles_keep_full_later_episodes(self):
        env, term = self.make_term()
        limits = term.limits.clone()
        first_end = torch.zeros_like(limits)
        full_counts = torch.zeros_like(limits)
        partial_counts = torch.zeros_like(limits)
        maximum_reset_batch = 0
        for step in range(1, 1501):
            env.episode_length_buf += 1
            partial, full = term(env), time_out(env)
            self.assertFalse((partial & full).any())
            done = partial | full
            first_end[(first_end == 0) & done] = step
            full_counts += full
            partial_counts += partial
            ids = done.nonzero().flatten()
            maximum_reset_batch = max(maximum_reset_batch, len(ids))
            age = env.episode_length_buf.clone()
            term.reset(ids)
            self.assertTrue(torch.equal(age, env.episode_length_buf))
            env.episode_length_buf[ids] = 0
        self.assertTrue(torch.equal(first_end, limits))
        self.assertTrue(torch.equal(partial_counts, (limits < 500).long()))
        self.assertTrue(torch.equal(full_counts, 2 + (limits == 500).long()))
        self.assertLess(maximum_reset_batch, env.num_envs // 16)

    def test_early_failure_consumes_only_that_environments_deadline(self):
        env, term = self.make_term(count=4)
        limits = term.limits.clone()
        env.episode_length_buf[:] = 1
        term.reset(torch.tensor([0, 2]))
        self.assertEqual(term.limits[[0, 2]].tolist(), [500, 500])
        self.assertTrue(torch.equal(term.limits[[1, 3]], limits[[1, 3]]))
        env.episode_length_buf[:] = 500
        self.assertFalse(term(env)[[0, 2]].any())
        self.assertTrue(time_out(env)[[0, 2]].all())

    def test_only_bptt_training_staggers(self):
        train = G1PGMTTerrainBPTTEnvCfg()
        self.assertIs(train.terminations.initial_time_out.func, InitialTimeOut)
        self.assertTrue(train.terminations.initial_time_out.time_out)
        self.assertIs(train.terminations.time_out.func, time_out)
        self.assertIsNone(G1PGMTTerrainBPTTEnvCfg_PLAY().terminations.initial_time_out)
        self.assertIsNone(G1PGMTTerrainPPOValidationEnvCfg().terminations.initial_time_out)


if __name__ == "__main__":
    unittest.main()
