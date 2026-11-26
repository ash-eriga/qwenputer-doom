"""Minimal GDN-state RAM readout used by the ray0 GPU runner.

This is the same decoding contract as doom-lx/ops/work/free/readout.py, kept here so the ray0
runner can be copied with the integration directory without depending on the doom-lx operations
checkout.  Words below TEXT_END are ROM and intentionally cannot be read from RAM heads.
"""
from __future__ import annotations

import struct

import torch

A_BIT = 2**7
THRESH = A_BIT / 2
BIT_COLS = slice(2, 34)
RAM_BLOCK_WORDS = 2048
POW = (1 << torch.arange(32, dtype=torch.int64))


def _node_index(c, name: str) -> int:
    node = c.tables[name]
    for i, n in enumerate(c.nodes):
        if n is node:
            return i
    raise KeyError(name)


class GDNReadout:
    """Expose ``read_bytes(offset, n)`` from the LX Session cache."""

    def __init__(self, c, cache) -> None:
        from lx import hf

        self.c, self.cache = c, cache
        self.r = c.target.value_heads // c.target.key_heads
        self.blk = {}
        for b in c.ram_blocks:
            name = f"ram{b}"
            if name not in c.tables:
                continue
            ni = _node_index(c, name)
            L, j = hf.head_slots(c)[ni]
            self.blk[b] = (L, j, len(c.nodes[ni].vheads()))
        if not self.blk:
            raise KeyError("no RAM blocks found in this circuit")

    def _words(self, w: torch.Tensor) -> torch.Tensor:
        """Guest word indices ``w`` (from RAM_BASE) as uint32 values."""
        first = next(iter(self.blk))
        dev = self.cache.layers[self.blk[first][0]].recurrent_states[0].device
        w = w.to(dev)
        out = torch.zeros(len(w), dtype=torch.int64, device=dev)
        done = torch.zeros(len(w), dtype=torch.bool, device=dev)
        for b, (L, j, _) in sorted(self.blk.items()):
            sel = ((w >> 11) == b) & ~done
            if not sel.any():
                continue
            st = self.cache.layers[L].recurrent_states[0]
            wi = w[sel] & (RAM_BLOCK_WORDS - 1)
            bits = st[0, j * self.r + (wi & 7), wi >> 3, BIT_COLS].float()
            out[sel] = ((bits > THRESH).double() @ POW.double().to(dev)).long()
            done |= sel
        if not done.all():
            raise KeyError(f"words {w[~done].tolist()[:4]} are in no RAM block (below TEXT_END?)")
        return out

    def read_bytes(self, base: int, n: int) -> bytes:
        """Read ``n`` guest-RAM bytes at byte offset ``base`` from RAM_BASE."""
        if n < 0 or base < 0:
            raise ValueError("negative readout offset/length")
        w0, off = base >> 2, base & 3
        nw = (off + n + 3) >> 2
        words = self._words(torch.arange(w0, w0 + nw, dtype=torch.long))
        raw = b"".join(struct.pack("<I", int(v) & 0xFFFFFFFF) for v in words.tolist())
        return raw[off:off + n]

    def read_u32(self, byte_offset: int) -> int:
        return struct.unpack("<I", self.read_bytes(byte_offset, 4))[0]
