#!/usr/bin/env python3
"""A small CPU causality simulator for chunked free-run and draft+verify+rollback.

The important index fact is the one lx.session documents: prediction ``i`` is for token
``i + 1``.  Therefore, after a chunk ending at position ``e``, only one token of the next
chunk is known causally (the prediction at ``e``).  The rest must be drafted.  If verification
rejects a draft token, every model state byte produced by that token or a later one is
causally poisoned and must be rolled back before the accepted prefix is replayed/committed.

This file is deliberately stdlib-only.  It does not emulate a transformer; it makes the
causality invariant executable and exact in a toy recurrent system, and can also measure how
well a past-only copy drafter fits the real ray0 lx.ref token stream.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

M32 = 0xFFFFFFFF


class ToyLM:
    """A deterministic input-sensitive recurrent token model.

    ``step(x)`` consumes input token ``x`` at the current position and returns the argmax-like
    prediction for the next input.  The nonlinear state update makes wrong draft tokens change
    all later predictions, so failure to roll back is observable immediately.
    """

    def __init__(self, seed: int = 0xC0FFEE, vocab: int = 16, state: int | None = None):
        self.vocab = vocab
        self.state = seed if state is None else state

    def initial_next(self) -> int:
        return ((self.state >> 13) ^ (self.state >> 3)) % self.vocab

    def step(self, token: int) -> int:
        token = int(token) % self.vocab
        self.state = (self.state * 1664525 + 1013904223 + token * 2654435761) & M32
        self.state ^= (self.state >> 11) & M32
        return ((self.state >> 17) ^ (self.state >> 5) ^ token) % self.vocab

    def clone(self) -> "ToyLM":
        return ToyLM(vocab=self.vocab, state=self.state)

    def restore(self, other: "ToyLM") -> None:
        self.state = other.state


@dataclass
class SpecStats:
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rollback_rounds: int = 0
    rollback_replay_tokens: int = 0
    full_accept_rounds: int = 0
    acceptance_histogram: dict[int, int] = field(default_factory=dict)

    def note(self, draft_len: int, accepted: int, rollback: int) -> None:
        self.rounds += 1
        self.draft_tokens += draft_len
        self.accepted_tokens += accepted
        self.acceptance_histogram[accepted] = self.acceptance_histogram.get(accepted, 0) + 1
        if rollback:
            self.rollback_rounds += 1
            self.rollback_replay_tokens += rollback
        if accepted == draft_len:
            self.full_accept_rounds += 1

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rollback_rounds": self.rollback_rounds,
            "rollback_replay_tokens": self.rollback_replay_tokens,
            "full_accept_rounds": self.full_accept_rounds,
            "acceptance_histogram": {str(k): v for k, v in sorted(self.acceptance_histogram.items())},
        }


def eval_block(model, tokens: list[int]) -> list[int]:
    """Stand-in for one chunked prefill: consume a block, return next-token predictions.

    A GPU transformer evaluates the block in parallel, but the mathematical contract is the
    same: prediction j is conditioned on inputs through token j and predicts input j+1.
    """
    return [model.step(t) for t in tokens]


def sequential_free(model, n_tokens: int) -> tuple[list[int], int]:
    """Exact one-token-at-a-time free-run reference."""
    nxt = model.initial_next()
    out = []
    while len(out) < n_tokens:
        x = nxt
        out.append(x)
        nxt = model.step(x)
    return out, nxt


class OracleDrafter:
    """Drafter with access to the already-computed reference stream, for algorithm tests.

    This is not a deployable drafter.  It exists to make the verify/rollback state machine exact
    and to inject controlled wrong guesses.  ``corrupt_positions`` are absolute stream positions.
    """

    def __init__(self, oracle: list[int], corrupt_positions: set[int] | None = None):
        self.oracle = oracle
        self.corrupt_positions = corrupt_positions or set()

    def draft(self, next_token: int, generated: list[int], pos: int, size: int) -> list[int]:
        out = []
        for p in range(pos, min(pos + size, len(self.oracle))):
            t = self.oracle[p]
            if p in self.corrupt_positions:
                t = (t + 1) % 16
            out.append(t)
        while len(out) < size:
            out.append(out[-1] if out else next_token)
        # A deterministic causal drafter always knows the pending first token.  Even the oracle
        # test drafter is constrained by this host-visible fact.
        out[0] = next_token
        return out


class CopyDrafter:
    """Past-only continuation drafter, in the spirit of linuxformer lx/tools/draft_sim.py."""

    def __init__(self, ngram: int = 6):
        self.ngram = max(1, ngram)

    def _next_from_history(self, context: list[int]) -> int:
        # Find the longest suffix that occurs earlier, then copy the token that followed that
        # earlier occurrence.  Only indices < len(context) are visible, so there is no future leak.
        max_n = min(self.ngram, len(context) - 1)
        for n in range(max_n, 0, -1):
            suffix = context[-n:]
            for end in range(len(context) - n, 0, -1):
                if context[end - n:end] == suffix and end < len(context):
                    return context[end]
        return context[-1]

    def draft(self, next_token: int, generated: list[int], pos: int, size: int) -> list[int]:
        out = [next_token]
        context = list(generated)
        while len(out) < size:
            t = self._next_from_history(context)
            out.append(t)
            context.append(t)
        return out[:size]


def speculative_free(model, n_tokens: int, drafter, draft_size: int = 16
                     ) -> tuple[list[int], SpecStats, int]:
    """Exact deterministic draft+verify+rollback free-run.

    Invariant at a round boundary: ``pos`` is the index of the pending next token; model state
    has consumed exactly positions ``0..pos-1``; ``generated`` contains positions ``0..pos``
    (one token of causal lookahead).  A draft covers positions ``pos..pos+D-1``.  Verification
    accepts draft[0] iff it is the pending token and accepts draft[j] iff it equals prediction
    j-1.  On a short accept, the polluted block state is discarded, the round-start state is
    restored, and only the accepted prefix is replayed.  The correction token is lookahead; it
    is not consumed until a later round.
    """
    nxt = model.initial_next()
    generated = [nxt]
    pos = 0
    stats = SpecStats()
    while len(generated) < n_tokens:
        draft = drafter.draft(nxt, generated, pos, draft_size)
        if not draft:
            raise ValueError("empty draft")
        # This is a hard causality requirement in deterministic argmax decoding: the host already
        # knows the first token of the block, so proposing anything else is a bug in the drafter.
        draft[0] = nxt
        start = model.clone()
        preds = eval_block(model, draft)

        accepted = 1
        while accepted < len(draft) and draft[accepted] == preds[accepted - 1]:
            accepted += 1
        correction = preds[accepted - 1]

        replay = 0
        if accepted < len(draft):
            # The current model state has consumed rejected suffix tokens.  It is not merely
            # useless; it belongs to a different counterfactual timeline.  Restore, then replay
            # only causal tokens.  The correction remains lookahead and is deliberately not fed.
            model.restore(start)
            for t in draft[:accepted]:
                model.step(t)
                replay += 1
        # If the full draft was accepted, the block state is already exactly the replayed state.

        generated.extend(draft[1:accepted] + [correction])
        pos += accepted
        nxt = correction
        stats.note(len(draft), accepted, replay if accepted < len(draft) else 0)
    return generated[:n_tokens], stats, nxt


def naive_chunked_free(model, n_tokens: int, chunk: int) -> list[int]:
    """The incorrect stale-prediction loop used here as a negative control.

    It teacher-forces one burn-in chunk, then uses the previous chunk's prediction vector as the
    next input block.  Those predictions belong to positions inside the previous block; only the
    final prediction belongs to the first position of the next block.  This reproduces the causal
    indexing bug without needing a GPU.
    """
    oracle_model = model.clone()
    burn, nxt = sequential_free(oracle_model, chunk)
    inp = burn[:chunk]
    out = []
    preds = []
    while len(out) < n_tokens:
        preds = eval_block(model, inp)
        out.extend(inp)
        # Wrong: preds[j] predicts the successor of inp[j], not the token at the same index in
        # the next block.  In particular preds[-1] alone is the causal first token of inp-next.
        inp = list(preds)
    return out[:n_tokens]


def toy_demo(n_tokens: int = 96, chunk: int = 16) -> dict:
    # Reference trajectory.
    ref_model = ToyLM()
    reference, _ = sequential_free(ref_model, n_tokens + 4 * chunk)

    # Negative control.
    naive_model = ToyLM()
    naive = naive_chunked_free(naive_model, n_tokens, chunk)
    first_diff = next((i for i, (a, b) in enumerate(zip(reference, naive)) if a != b), None)

    # Exact speculative path.  Corruptions force rollback; oracle drafting elsewhere isolates
    # the state-machine correctness from drafter quality.
    corrupt = {chunk + 3, 2 * chunk + 9, 3 * chunk + 1}
    spec_model = ToyLM()
    drafter = OracleDrafter(reference, corrupt)
    spec, stats, _ = speculative_free(spec_model, n_tokens, drafter, chunk)

    return {
        "reference_first24": reference[:24],
        "naive_first_difference": first_diff,
        "naive_matches_reference": naive == reference[:n_tokens],
        "speculative_matches_reference": spec == reference[:n_tokens],
        "injected_corrupt_positions": sorted(corrupt),
        "speculative_stats": stats.as_dict(),
    }


def load_lx_stream(elf_path: Path, n_tokens: int, lxsrc: Path) -> list[int]:
    """Produce the first n_tokens of the ray0 exact lx.ref stream (host CPU, no torch)."""
    sys.path.insert(0, str(lxsrc / "lx"))
    from lx import spec as S  # type: ignore
    from lx.ref import Machine  # type: ignore
    from asmelf import Elf32  # type: ignore

    elf = Elf32(elf_path)
    cfg = elf.machine_config()
    mem = bytearray(elf.load_memory(cfg["ram_size"]))
    m = Machine(mem, text_end=cfg["text_end"])
    m.x[2] = S.RAM_BASE + len(mem) - 16
    m.anchor = (elf.entry - S.RAM_BASE) // 4
    m.n = 0
    out = []
    while len(out) < n_tokens and not m.halted:
        out.extend(m.step())
    return out[:n_tokens]


def stream_draft_stats(stream: list[int], draft_size: int = 64, ngram: int = 6) -> dict:
    """Round count of causal copy-drafting on a fixed exact stream.

    This measures drafter quality only.  Model-state rollback is what the ToyLM half of this
    file proves; a fixed stream has no model state to roll back.
    """
    pos = 0
    rounds = 0
    rollback_rounds = 0
    accepted_total = 0
    drafter = CopyDrafter(ngram)
    while pos < len(stream) - 1:
        generated = stream[:pos + 1]
        draft = drafter.draft(stream[pos], generated, pos,
                              min(draft_size, len(stream) - pos))
        draft[0] = stream[pos]
        accepted = 1
        while accepted < len(draft) and pos + accepted < len(stream) and \
                draft[accepted] == stream[pos + accepted]:
            accepted += 1
        rounds += 1
        accepted_total += accepted
        rollback_rounds += int(accepted < len(draft))
        pos += accepted
    return {
        "stream_tokens": len(stream),
        "draft_size": draft_size,
        "rounds": rounds,
        "positions_advanced_per_round": round((len(stream) - 1) / max(rounds, 1), 3),
        "rollback_rounds": rollback_rounds,
        "full_or_longest_accept_rate": round(accepted_total / max(len(stream) - 1, 1), 6),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elf", type=Path, help="optional ray0 ELF: measure copy-drafting on lx.ref stream")
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--draft", type=int, default=64)
    ap.add_argument("--ngram", type=int, default=6)
    ap.add_argument("--lxsrc", type=Path, default=Path(__file__).resolve().parents[3] /
                    "Doom-On-Qwen" / "lxsrc")
    a = ap.parse_args(argv)

    result = {"toy_causality": toy_demo()}
    if a.elf:
        stream = load_lx_stream(a.elf, a.tokens, a.lxsrc)
        result["ray0_lx_ref_stream"] = stream_draft_stats(stream, a.draft, a.ngram)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
