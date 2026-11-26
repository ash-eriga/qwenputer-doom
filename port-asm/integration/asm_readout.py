#!/usr/bin/env python3
"""Column-major framebuffer readout for the ray0 assembly probe.

ray0 cost builds store pixels as fb_cols[x * stride + y].  The guest intentionally does not
transpose them: that would cost guest instructions/tokens inside the circuit.  This module is
the host side of that contract.  It uses an existing doom-lx ``Readout`` object for all GDN
state access and only rearranges the returned host bytes into row-major display order.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

RAM_BASE = 0x80000000


def column_major_to_rows(data: bytes | bytearray, width: int, height: int, stride: int) -> bytes:
    """Return row-major pixel indexes from ray0's column-major bytes.

    ``stride`` may be larger than ``height`` because the column kernels write whole aligned
    words; padding bytes are read but never displayed.
    """
    if width <= 0 or height <= 0 or stride < height:
        raise ValueError(f"bad geometry width={width} height={height} stride={stride}")
    need = width * stride
    if len(data) < need:
        raise ValueError(f"short column framebuffer: got {len(data)}, need {need}")
    return bytes(data[x * stride + y] for y in range(height) for x in range(width))


def rows_ppm(indices: bytes | bytearray, width: int, height: int, palette: bytes | None) -> bytes:
    """Encode row-major palette indexes as a binary PPM."""
    if len(indices) != width * height:
        raise ValueError(f"frame has {len(indices)} pixels, expected {width * height}")
    out = bytearray(f"P6\n{width} {height}\n255\n".encode())
    if palette is None:
        # Deterministic fallback: enough for smoke tests and geometry checks.  Real runs should
        # use a PLAYPAL palette (the assembly guest itself never reads a WAD).
        for v in indices:
            out += bytes((v, v, v))
    else:
        if len(palette) < 768:
            raise ValueError("PLAYPAL palette must have at least 768 bytes")
        for v in indices:
            out += palette[3 * v:3 * v + 3]
    return bytes(out)


def column_major_ppm(column_data: bytes | bytearray, width: int, height: int, stride: int,
                     palette: bytes | None = None) -> bytes:
    return rows_ppm(column_major_to_rows(column_data, width, height, stride),
                    width, height, palette)


class AsmColumnReadout:
    """Adapter around doom-lx's ``readout.Readout``.

    The wrapped object provides ``read_bytes(offset, n)`` and optionally ``_words``; this class
    deliberately does not duplicate the GDN bit decoding.  By default it reads the ``fb_cols``
    symbol directly.  ``follow_pointer=True`` instead dereferences DG_ScreenBuffer through the
    wrapped Readout, which is useful for visual/debug ELFs.
    """

    def __init__(self, readout, config: dict, palette: bytes | None = None,
                 follow_pointer: bool = False):
        self.readout = readout
        self.config = config
        fb = config["framebuffer"]
        self.width, self.height, self.stride = fb["width"], fb["height"], fb["stride"]
        self.symbols = config["symbols"]
        self.palette = palette
        self.follow_pointer = follow_pointer

    @property
    def fb_offset(self) -> int:
        return int(self.symbols["fb_cols"]["offset"])

    def _read_word(self, byte_offset: int) -> int:
        raw = self.readout.read_bytes(byte_offset, 4)
        return struct.unpack("<I", raw)[0]

    def framebuffer_bytes(self) -> bytes:
        """Raw column-major bytes, including row padding, straight from the model state."""
        if self.follow_pointer:
            ptr_off = int(self.symbols["DG_ScreenBuffer"]["offset"])
            fb = self._read_word(ptr_off) - RAM_BASE
        else:
            fb = self.fb_offset
        return self.readout.read_bytes(fb, self.width * self.stride)

    def frame_indices(self) -> bytes:
        """Row-major palette indexes; the transpose happens here, on the host."""
        return column_major_to_rows(self.framebuffer_bytes(), self.width, self.height, self.stride)

    def frame(self) -> bytes:
        """Row-major PPM bytes, API-compatible with doom-lx ``Readout.frame()``."""
        return rows_ppm(self.frame_indices(), self.width, self.height, self.palette)


def read_playpal(wad_path: str | Path) -> bytes:
    """Read the first PLAYPAL palette from a WAD, or raise FileNotFoundError/ValueError."""
    b = Path(wad_path).read_bytes()
    if len(b) < 12:
        raise ValueError("short WAD")
    n, directory = struct.unpack_from("<II", b, 4)
    for i in range(n):
        off, size, name = struct.unpack_from("<II8s", b, directory + 16 * i)
        if name.rstrip(b"\0") == b"PLAYPAL" and size >= 768:
            return b[off:off + 768]
    raise ValueError("PLAYPAL not found")


def write_palette_wad(path: str | Path, palette: bytes | None = None) -> Path:
    """Write a tiny valid WAD containing only PLAYPAL.

    The assembly probe never reads a WAD in-guest.  This artifact exists solely so existing GPU
    plumbing that expects a file named doom1.wad can obtain a deterministic host display palette.
    """
    path = Path(path)
    if palette is None:
        # A simple ANSI-ish ramp.  It is not Doom's art assets and is only a host visualization
        # palette; replace with a real DOOM1.WAD PLAYPAL for faithful colors.
        palette = bytes((v // 2 if c == 0 else min(255, v) if c == 1 else max(0, v - 32))
                        for v in range(256) for c in range(3))
    if len(palette) != 768:
        raise ValueError("palette must be exactly 768 bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"PWAD" + struct.pack("<II", 1, 12 + len(palette))
    directory = struct.pack("<II8s", 12, len(palette), b"PLAYPAL\0")
    path.write_bytes(header + palette + directory)
    return path


def load_config(path_or_elf) -> dict:
    p = Path(path_or_elf)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    from asmelf import Elf32, infer_fb_geometry
    elf = Elf32(p)
    cfg = elf.machine_config()
    cfg["framebuffer"] = infer_fb_geometry(elf)
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Host-side column-major framebuffer utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    wt = sub.add_parser("write-palette-wad")
    wt.add_argument("out")
    cv = sub.add_parser("convert", help="convert a raw column-major dump to PPM")
    cv.add_argument("config")
    cv.add_argument("raw")
    cv.add_argument("out")
    cv.add_argument("--wad")
    a = ap.parse_args(argv)
    if a.cmd == "write-palette-wad":
        print(write_palette_wad(a.out))
        return 0
    cfg = load_config(a.config)
    fb = cfg["framebuffer"]
    pal = read_playpal(a.wad) if a.wad else None
    ppm = column_major_ppm(Path(a.raw).read_bytes(), fb["width"], fb["height"], fb["stride"], pal)
    Path(a.out).write_bytes(ppm)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
