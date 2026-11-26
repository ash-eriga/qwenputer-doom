"""The ray0 machine played on the megakernel, the model alone.

The host never touches the machine's memory: its only input is a typed byte (a KEY0 token at a
record boundary, with the receive FIFO empty), which ray0 reads over the UART as its key mask
(bit 0 up, 1 down, 2 left, 3 right; bit 7 hands control back to the autopilot). Every other
token is the model's own prediction, fed back on the GPU. Frames are read out of the model's GDN
state. lx.ref can run beside it (Shadow): it is given the same typed bytes and checks every token
the model ran, and it is never fed to the model.

    game = MegaGame(runtime, session, mega)
    game.request(0b0001)        # forward, typed at the next record boundary
    if game.step(1024):         # about 1024 tokens; True when a new frame was committed
        png = game.png()
"""
from __future__ import annotations

import copy
import struct
import time
import zlib

from asm_readout import AsmColumnReadout
from gdn_readout import GDNReadout


def png_bytes(indices: bytes, width: int, height: int, palette: bytes | None,
              scale: int = 1) -> bytes:
    """Row-major palette indexes as an RGB PNG (stdlib only), each pixel scale x scale."""
    if palette is None:
        palette = bytes(v for i in range(256) for v in (i, i, i))
    rows = []
    for y in range(height):
        line = b"".join(palette[3 * v:3 * v + 3] * scale
                        for v in indices[y * width:(y + 1) * width])
        rows.extend([b"\0" + line] * scale)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    w, h = width * scale, height * scale
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


class Shadow:
    """lx.ref beside the model: each token the model ran, checked against the reference machine
    (typed bytes go into both, at the same place). It never feeds the model."""

    def __init__(self, machine0):
        from lx import spec as S

        self.S, self.m = S, copy.deepcopy(machine0)
        self.stream = [S.tok("GO")]
        self.pos = 0
        self.divergence = None
        self.key0, _ = S.FAMILIES["KEY0"]

    def feed(self, tok: int, typed: bool) -> None:
        if self.divergence is not None:
            return
        p = self.pos
        if typed:
            if len(self.stream) != p:
                self.divergence = {"position": p, "reason": "typed off a reference record boundary"}
                return
            self.stream.extend(self.m.key(tok - self.key0))
        while len(self.stream) <= p and not self.m.halted:
            self.stream.extend(self.m.step())
        if p >= len(self.stream) or self.stream[p] != tok:
            want = self.stream[p] if p < len(self.stream) else None
            self.divergence = {"position": p, "want": want, "got": tok,
                               "context": self.stream[max(0, p - 8):p]}
            return
        self.pos += 1

    @property
    def ok(self) -> bool:
        return self.divergence is None


class MegaGame:
    def __init__(self, runtime, session, mega, check: bool = True,
                 palette: bytes | None = None):
        from lx import spec as S
        from lx.session import _final_ids

        self.S, self.rt, self.session, self.mega = S, runtime, session, mega
        self.final = _final_ids().tolist()
        self.key0, _ = S.FAMILIES["KEY0"]
        self.rx0, self.nrx = S.FAMILIES["RX"]
        self.next_input = S.tok("GO")
        self.last_input = None
        self.tokens = 0
        self.waiting = 0  # typed bytes the guest has not read yet
        self.shadow = Shadow(runtime.machine0) if check else None
        self.view = AsmColumnReadout(GDNReadout(runtime.c, session.cache), runtime.cfg,
                                     palette=palette)
        self.fc_off = int(runtime.cfg["symbols"]["qwen_frame_count"]["offset"])
        self.frame = 0
        self.want_byte = None  # the byte the host wants the guest to have
        self.typed_byte = None
        self.typed = 0
        self.run_seconds = 0.0  # inside Mega.run (the kernel and its launch)
        self.check_seconds = 0.0
        self.inputs_checked = 0
        self._inp = None

    def request(self, b: int) -> None:
        """Have the guest read byte b (typed at the next record boundary with the FIFO empty;
        a newer request replaces one not typed yet)."""
        assert 0 <= b < 256
        self.want_byte = b

    def at_boundary(self) -> bool:
        return self.last_input is not None and bool(self.final[self.last_input])

    def _run(self, first: int, n: int, typed: bool) -> None:
        import torch

        if self._inp is None or self._inp.numel() < n:
            self._inp = torch.zeros(max(n, 4096), dtype=torch.int32, device=self.mega.dev)
        t0 = time.time()
        preds = self.mega.run([first], n, inputs=self._inp)
        t1 = time.time()
        inputs = [first] + preds[:-1]
        # what the kernel embedded: the host's token first, then the model's own predictions
        got = self._inp[:n].tolist()
        if got != inputs:
            i = next(i for i in range(n) if got[i] != inputs[i])
            raise RuntimeError(f"kernel input {i} of the launch is {got[i]}, not {inputs[i]}")
        self.inputs_checked += n
        for i, t in enumerate(inputs):
            is_typed = typed and i == 0
            if is_typed:
                self.waiting += 1
            elif self.rx0 <= t < self.rx0 + self.nrx:
                self.waiting -= 1
            if self.shadow is not None:
                self.shadow.feed(t, is_typed)
        self.check_seconds += time.time() - t1
        self.run_seconds += t1 - t0
        self.last_input, self.next_input = inputs[-1], preds[-1]
        self.tokens += n

    def step(self, n: int) -> bool:
        """About n tokens. Returns whether the guest committed a new frame."""
        if self.want_byte is not None and self.want_byte != self.typed_byte and self.waiting == 0:
            for _ in range(100_000):  # single tokens up to a record boundary (usually 0 or 1)
                if self.at_boundary():
                    break
                self._run(self.next_input, 1, False)
            else:
                raise RuntimeError("no record boundary in 100k tokens")
            b = self.want_byte
            self._run(self.key0 + b, n, True)
            self.typed_byte = b
            self.typed += 1
        else:
            self._run(self.next_input, n, False)
        fc = self.view.readout.read_u32(self.fc_off)
        new = fc != self.frame
        self.frame = fc
        return new

    def sync(self) -> None:
        """Single tokens up to a record boundary, where the model's state and lx.ref's machine
        (if checking) stand at the same point."""
        for _ in range(100_000):
            if self.at_boundary() and (self.shadow is None
                                       or len(self.shadow.stream) == self.shadow.pos):
                return
            self._run(self.next_input, 1, False)
        raise RuntimeError("no record boundary in 100k tokens")

    def frame_indices(self) -> bytes:
        return self.view.frame_indices()

    def png(self, scale: int = 1) -> bytes:
        return png_bytes(self.frame_indices(), self.view.width, self.view.height,
                         self.view.palette, scale)

    def status(self) -> dict:
        return {"frame": self.frame, "tokens": self.tokens, "typed": self.typed,
                "waiting": self.waiting, "inputs_checked": self.inputs_checked,
                "verified": None if self.shadow is None else self.shadow.ok,
                "divergence": None if self.shadow is None else self.shadow.divergence,
                "run_seconds": round(self.run_seconds, 3),
                "check_seconds": round(self.check_seconds, 3)}


class CpuMega:
    """Mega.run's interface on gpu/cpu_decode.py's decoder: the host side (typing, Shadow,
    readout, PNGs, the live server) without a GPU, at about 10 tok/s."""

    def __init__(self, runtime, cache=None):
        from cpu_decode import CpuDecoder

        self.dec, self.cache, self.dev = CpuDecoder(runtime), cache, "cpu"
        self.stats = {"device": "cpu (cpu_decode.CpuDecoder)"}

    def run(self, forced, n, inputs=None):
        out, t = [], forced[0]
        for i in range(n):
            if i < len(forced):
                t = forced[i]
            if inputs is not None:
                inputs[i] = t
            t = self.dec.step(self.cache, t)
            out.append(t)
        return out
