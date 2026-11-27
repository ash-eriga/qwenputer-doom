#!/usr/bin/env python3
"""CPU-only correctness tests for speculative.py (no torch, no GPU)."""
from __future__ import annotations

import unittest

from speculative import CausalSpeculator, CopyDrafter, OracleDrafter

M32 = 0xFFFFFFFF


class ToyLM:
    """Input-sensitive deterministic recurrent model: wrong suffixes poison later states."""

    def __init__(self, seed: int = 0xC0FFEE, vocab: int = 16):
        self.vocab = vocab
        self.state = seed

    def initial_next(self) -> int:
        return ((self.state >> 13) ^ (self.state >> 3)) % self.vocab

    def step(self, token: int) -> int:
        self.state = (self.state * 1664525 + 1013904223 + token * 2654435761) & M32
        self.state ^= (self.state >> 11) & M32
        return ((self.state >> 17) ^ (self.state >> 5) ^ token) % self.vocab


class ToySession:
    """Minimal object with the Session surface used by CausalSpeculator."""

    def __init__(self):
        self.model = ToyLM()
        self.next = self.model.initial_next()
        self.last = None
        self.n = 0
        self.waiting = 0

    def verify(self, toks: list[int]) -> dict:
        if not toks:
            raise ValueError("empty verify")
        preds = [self.model.step(t) for t in toks]
        self.next = preds[-1]
        self.last = toks[-1]
        self.n += len(toks)
        return {"predicted": preds}

    def _state(self):
        return self.model.state

    def _set_state(self, state) -> None:
        self.model.state = state


def reference_stream(n: int) -> list[int]:
    m = ToyLM()
    nxt = m.initial_next()
    out = []
    while len(out) < n:
        out.append(nxt)
        nxt = m.step(nxt)
    return out


class SpeculativeTests(unittest.TestCase):
    def test_oracle_with_forced_rejections_matches_reference(self):
        n, draft = 128, 16
        ref = reference_stream(n + 4 * draft)
        sess = ToySession()
        corrupt = {19, 23, 41, 42, 70, 99}
        spec = CausalSpeculator(sess, OracleDrafter(ref, corrupt, vocab=16), draft)
        got = spec.run(n)
        self.assertEqual(got, ref[:n])
        self.assertGreater(spec.stats.rollback_rounds, 0)
        self.assertGreater(spec.stats.rollback_replay_tokens, 0)
        self.assertEqual(spec.stats.replay_shape_mismatches, 0)
        # Position n-1 is known as lookahead after consuming exactly n-1 input tokens.
        self.assertEqual(sess.n, n - 1)
        self.assertEqual(spec.pos, n - 1)

    def test_copy_drafter_is_past_only_and_still_exact(self):
        n = 96
        ref = reference_stream(n + 64)
        sess = ToySession()
        spec = CausalSpeculator(sess, CopyDrafter(ngram=5), 12)
        got = spec.run(n)
        self.assertEqual(got, ref[:n])
        self.assertEqual(sess.n, n - 1)
        self.assertEqual(spec.pos, n - 1)
        self.assertEqual(spec.stats.replay_shape_mismatches, 0)

    def test_full_oracle_acceptance_avoids_rollback(self):
        n, draft = 64, 16
        ref = reference_stream(n + draft)
        sess = ToySession()
        spec = CausalSpeculator(sess, OracleDrafter(ref), draft)
        got = spec.run(n)
        self.assertEqual(got, ref[:n])
        self.assertEqual(spec.stats.rollback_rounds, 0)
        self.assertEqual(spec.stats.full_accept_rounds, spec.stats.rounds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
