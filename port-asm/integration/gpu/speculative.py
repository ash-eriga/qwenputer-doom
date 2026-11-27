#!/usr/bin/env python3
"""Causally exact draft+verify+rollback for LX Sessions.

This module is deliberately torch-free.  It uses only the small Session surface that matters
(``verify``, ``_state``, ``_set_state`` and the metadata mutated by ``verify``), so the state
machine can be tested on CPU with a toy recurrent model.  The indexing contract is the one in
lx.session: prediction ``i`` is for input token ``i + 1``.

Boundary invariant after ``prime()``: ``generated`` contains absolute stream positions
``0..pos``; the model state has consumed positions ``0..pos-1``; and ``generated[pos]`` is the
already-known pending token (the model's prediction for position ``pos``).  A draft covers
``pos..pos+D-1``.  On rejection, the polluted state is discarded, the round-start state is
restored, and only the accepted prefix is replayed.  The correction token remains lookahead and
is not consumed in the same round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class VerifySession(Protocol):
    n: int
    next: int | None
    last: int | None
    waiting: int

    def verify(self, toks: list[int]) -> dict: ...
    def _state(self): ...
    def _set_state(self, state) -> None: ...


@dataclass
class SessionSnapshot:
    state: Any
    n: int
    next: int | None
    last: int | None
    waiting: int


def snapshot_session(session: VerifySession) -> SessionSnapshot:
    """Capture every scalar mutated by Session.verify plus a detached cache-state copy.

    ``lx.session.Session._state()`` already clones its tensors; implementations used with this
    module must provide the same detached-copy contract.
    """
    state = session._state()
    return SessionSnapshot(state, session.n, session.next, session.last,
                           getattr(session, "waiting", 0))


def restore_session(session: VerifySession, snap: SessionSnapshot) -> None:
    session._set_state(snap.state)
    session.n = snap.n
    session.next = snap.next
    session.last = snap.last
    session.waiting = snap.waiting


def _as_list(x) -> list[int]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [int(v) for v in x]


@dataclass
class RoundResult:
    position: int
    draft_len: int
    accepted: int
    correction: int
    rollback: bool
    replay_tokens: int = 0
    replay_shape_mismatch: bool = False


@dataclass
class SpeculativeStats:
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rollback_rounds: int = 0
    rollback_replay_tokens: int = 0
    full_accept_rounds: int = 0
    replay_shape_mismatches: int = 0
    acceptance_histogram: dict[int, int] = field(default_factory=dict)

    def note(self, r: RoundResult) -> None:
        self.rounds += 1
        self.draft_tokens += r.draft_len
        self.accepted_tokens += r.accepted
        self.acceptance_histogram[r.accepted] = self.acceptance_histogram.get(r.accepted, 0) + 1
        if r.rollback:
            self.rollback_rounds += 1
            self.rollback_replay_tokens += r.replay_tokens
        else:
            self.full_accept_rounds += 1
        self.replay_shape_mismatches += int(r.replay_shape_mismatch)

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rollback_rounds": self.rollback_rounds,
            "rollback_replay_tokens": self.rollback_replay_tokens,
            "full_accept_rounds": self.full_accept_rounds,
            "replay_shape_mismatches": self.replay_shape_mismatches,
            "mean_accepted_per_round": round(self.accepted_tokens / max(self.rounds, 1), 3),
            "acceptance_histogram": {str(k): v for k, v in sorted(self.acceptance_histogram.items())},
        }


class CopyDrafter:
    """Past-only n-gram continuation drafter.

    The drafter sees only ``generated`` (positions already committed plus the pending token) and
    therefore has no future leak.  It searches for the longest prior occurrence of the current
    suffix, excluding the current suffix itself, and copies the token that followed that
    occurrence.  This is a correctness-safe but deliberately simple drafter; better learned or
    machine-structure-aware drafters can use the same interface.
    """

    def __init__(self, ngram: int = 6):
        self.ngram = max(1, int(ngram))

    def _next_from_history(self, context: list[int]) -> int:
        max_n = min(self.ngram, len(context) - 1)
        for n in range(max_n, 0, -1):
            suffix = context[-n:]
            # end is exclusive.  end == len(context) is the live suffix itself, so stop before it.
            for end in range(len(context) - 1, n - 1, -1):
                if context[end - n:end] == suffix:
                    return context[end]
        return context[-1]

    def draft(self, next_token: int, generated: list[int], pos: int, size: int) -> list[int]:
        del pos
        out = [int(next_token)]
        context = list(generated)
        while len(out) < size:
            t = self._next_from_history(context)
            out.append(t)
            context.append(t)
        return out[:size]


class OracleDrafter:
    """Future-stream drafter for tests and upper-bound measurements only.

    ``stream[pos]`` is the exact token at stream position ``pos``.  This is not deployable as a
    free-running drafter, but verification/rollback still enforces the causal contract, so it is
    useful for validating the mechanism and measuring the cost of rejections independently from
    drafter quality.
    """

    def __init__(self, stream: list[int], corrupt_positions: set[int] | None = None,
                 vocab: int | None = None):
        self.stream = stream
        self.corrupt_positions = corrupt_positions or set()
        self.vocab = vocab

    def draft(self, next_token: int, generated: list[int], pos: int, size: int) -> list[int]:
        del generated
        out = []
        for p in range(pos, pos + size):
            if p < len(self.stream):
                t = self.stream[p]
                if p in self.corrupt_positions:
                    if self.vocab is None:
                        t = t + 1
                    else:
                        t = (t + 1) % self.vocab
                out.append(int(t))
            else:
                out.append(out[-1] if out else int(next_token))
        out[0] = int(next_token)  # even an oracle may not contradict known causal lookahead
        return out


class CausalSpeculator:
    """Drive a verify-capable LX Session with exact rollback on draft rejection."""

    def __init__(self, session: VerifySession, drafter, draft_size: int = 64,
                 go_token: int | None = None, history: list[int] | None = None):
        if draft_size <= 0:
            raise ValueError("draft_size must be positive")
        self.session = session
        self.drafter = drafter
        self.draft_size = int(draft_size)
        self.go_token = go_token
        self.generated = list(history or [])
        self.pos = len(self.generated) - 1 if self.generated else -1
        self.stats = SpeculativeStats()

    @property
    def pending(self) -> int:
        if self.pos < 0 or self.pos >= len(self.generated):
            raise RuntimeError("prime() before stepping")
        return int(self.generated[self.pos])

    def prime(self) -> None:
        """Establish the boundary invariant.

        With ``go_token`` (the LX machine integration), position zero is host control input GO.
        Feeding it produces the model's prediction for the first machine token.  Without a GO
        token (toy tests), the session must already expose ``next`` for position zero.
        """
        if self.pos >= 0:
            return
        if self.go_token is not None and self.session.n == 0:
            self.session.verify([int(self.go_token)])
            if self.session.next is None:
                raise RuntimeError("session produced no prediction after GO")
            self.generated = [int(self.go_token), int(self.session.next)]
            self.pos = 1
            return
        if self.session.next is None:
            raise RuntimeError("session.next is empty; provide go_token or resume history")
        self.generated = [int(self.session.next)]
        self.pos = 0

    def _check_invariant(self) -> None:
        if self.pos < 0 or self.pos >= len(self.generated):
            raise RuntimeError(f"bad speculative boundary pos={self.pos} len={len(self.generated)}")
        if self.session.next is not None and int(self.session.next) != self.pending:
            raise RuntimeError(
                f"boundary mismatch: session.next={self.session.next} pending={self.pending}")

    def step(self, draft_size: int | None = None) -> RoundResult:
        """Run one draft/verify/rollback round and commit only its causal prefix."""
        self.prime()
        self._check_invariant()
        size = self.draft_size if draft_size is None else max(1, int(draft_size))
        pos = self.pos
        pending = self.pending
        draft = list(self.drafter.draft(pending, self.generated, pos, size))
        if not draft:
            raise ValueError("empty draft")
        if len(draft) > size:
            draft = draft[:size]
        # The first token is not a guess: it is already known from the previous prediction.
        draft[0] = pending

        start = snapshot_session(self.session)
        result = self.session.verify(draft)
        preds = _as_list(result.get("predicted", []))
        if len(preds) != len(draft):
            raise RuntimeError(f"verify returned {len(preds)} predictions for {len(draft)} inputs")

        accepted = 1
        while accepted < len(draft) and draft[accepted] == preds[accepted - 1]:
            accepted += 1
        correction = int(preds[accepted - 1])

        rollback = accepted < len(draft)
        replay_tokens = 0
        replay_shape_mismatch = False
        if rollback:
            # The verifier state now belongs to a counterfactual suffix.  Restore the causal
            # state, replay exactly the accepted prefix, and keep the correction as lookahead.
            restore_session(self.session, start)
            replay = self.session.verify(draft[:accepted])
            replay_preds = _as_list(replay.get("predicted", []))
            replay_tokens = accepted
            if len(replay_preds) != accepted:
                raise RuntimeError("replay returned the wrong number of predictions")
            prefix_bad = any(draft[j] != replay_preds[j - 1] for j in range(1, accepted))
            if prefix_bad or self.session.next is None:
                raise RuntimeError("accepted-prefix replay failed verification")
            if int(self.session.next) != correction:
                # In exact causal arithmetic this cannot happen: prediction accepted-1 cannot
                # depend on the rejected suffix.  If kernels make results shape-dependent, the
                # replayed causal state is the authoritative one, but the event is counted.
                replay_shape_mismatch = True
                correction = int(self.session.next)
        else:
            # A full acceptance consumed exactly the causal block, so no rollback is needed.
            if self.session.next is not None and int(self.session.next) != correction:
                raise RuntimeError("session.next does not equal the final block prediction")

        self.generated.extend(draft[1:accepted] + [correction])
        self.pos += accepted
        r = RoundResult(position=pos, draft_len=len(draft), accepted=accepted,
                        correction=correction, rollback=rollback,
                        replay_tokens=replay_tokens,
                        replay_shape_mismatch=replay_shape_mismatch)
        self.stats.note(r)
        self._check_invariant()
        return r

    def run(self, n_tokens: int) -> list[int]:
        """Generate until at least ``n_tokens`` stream positions are known."""
        self.prime()
        while len(self.generated) < n_tokens:
            self.step(min(self.draft_size, n_tokens - len(self.generated)))
        return self.generated[:n_tokens]
