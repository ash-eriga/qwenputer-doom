#!/usr/bin/env python3
"""ELF32/RISC-V inspection for the assembly raycaster, with no binutils dependency.

The GPU runners need byte offsets in the LX RAM image, not virtual addresses.  This module
reads the ELF symbol table directly and derives those offsets by subtracting lx.spec's
RAM_BASE (0x80000000).  It also reads the linker-script absolute symbols TEXT_END and
RAM_SIZE, so a rebuilt assembly probe does not require manually editing runner constants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

RAM_BASE = 0x80000000
SHN_ABS = 0xFFF1
SHT_SYMTAB = 2
SHT_NOBITS = 8
SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    info: int
    shndx: int

    @property
    def absolute(self) -> bool:
        return self.shndx == SHN_ABS

    @property
    def offset(self) -> int:
        return self.value - RAM_BASE


@dataclass
class Section:
    name: str
    typ: int
    flags: int
    addr: int
    off: int
    size: int
    link: int
    entsize: int

    @property
    def alloc(self) -> bool:
        return bool(self.flags & SHF_ALLOC)

    @property
    def executable(self) -> bool:
        return bool(self.flags & SHF_EXECINSTR)


class Elf32:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.bytes = self.path.read_bytes()
        b = self.bytes
        if b[:4] != b"\x7fELF" or b[4] != 1 or b[5] != 1:
            raise ValueError(f"{self.path}: not a little-endian ELF32 file")
        self.entry = struct.unpack_from("<I", b, 0x18)[0]
        phoff = struct.unpack_from("<I", b, 0x1C)[0]
        shoff = struct.unpack_from("<I", b, 0x20)[0]
        self.phentsize, self.phnum = struct.unpack_from("<HH", b, 0x2A)
        self.shentsize, self.shnum, self.shstrndx = struct.unpack_from("<HHH", b, 0x2E)

        self.segments: list[tuple[int, int, int, int, int]] = []
        for i in range(self.phnum):
            typ, off, va, _pa, filesz, memsz, flags, _align = struct.unpack_from(
                "<IIIIIIII", b, phoff + i * self.phentsize)
            if typ == 1 and memsz:
                self.segments.append((off, va, filesz, memsz, flags))

        raw_sections = []
        for i in range(self.shnum):
            sh = struct.unpack_from("<IIIIIIIIII", b, shoff + i * self.shentsize)
            raw_sections.append(sh)
        shstr_off = raw_sections[self.shstrndx][4] if self.shstrndx < len(raw_sections) else 0
        self.sections: list[Section] = []
        for name_off, typ, flags, addr, off, size, link, _info, _align, entsize in raw_sections:
            name = self._cstring(shstr_off + name_off) if name_off else ""
            self.sections.append(Section(name, typ, flags, addr, off, size, link, entsize))

        self.symbols: dict[str, Symbol] = {}
        for sec in self.sections:
            if sec.typ != SHT_SYMTAB or not sec.entsize:
                continue
            strtab = self.sections[sec.link]
            for pos in range(sec.off, sec.off + sec.size, sec.entsize):
                nm, value, size, info, _other, shndx = struct.unpack_from("<IIIBBH", b, pos)
                if not nm:
                    continue
                end = b.index(b"\0", strtab.off + nm)
                name = b[strtab.off + nm:end].decode("utf-8", "replace")
                self.symbols[name] = Symbol(name, value, size, info, shndx)

    def _cstring(self, pos: int) -> str:
        end = self.bytes.index(b"\0", pos)
        return self.bytes[pos:end].decode("utf-8", "replace")

    def require(self, name: str) -> Symbol:
        try:
            return self.symbols[name]
        except KeyError as exc:
            raise KeyError(f"{self.path}: required symbol {name!r} is missing") from exc

    def offset(self, name: str) -> int:
        return self.require(name).offset

    def md5(self) -> str:
        return hashlib.md5(self.bytes).hexdigest()

    def load_memory(self, ram_size: int | None = None) -> bytes:
        ram_size = ram_size or self.machine_config()["ram_size"]
        mem = bytearray(ram_size)
        for off, va, filesz, memsz, _flags in self.segments:
            if va < RAM_BASE or va + memsz > RAM_BASE + ram_size:
                raise ValueError(f"segment outside LX image: {va:#x}+{memsz:#x}")
            mem[va - RAM_BASE:va - RAM_BASE + filesz] = self.bytes[off:off + filesz]
        return bytes(mem)

    def machine_config(self) -> dict:
        """LX machine dimensions and runner-relevant symbols, all derived from this ELF."""
        text_end = int(self.symbols.get("TEXT_END", Symbol("TEXT_END", 0x40000, 0, 0, SHN_ABS)).value)
        ram_size = int(self.symbols.get("RAM_SIZE", Symbol("RAM_SIZE", 0x600000, 0, 0, SHN_ABS)).value)
        names = ("qwen_frame_count", "qwen_key_state", "qwen_demo_enabled",
                 "DG_ScreenBuffer", "I_VideoBuffer", "fb_cols", "fb_linear")
        syms = {}
        for name in names:
            if name in self.symbols:
                s = self.symbols[name]
                syms[name] = {
                    "addr": f"0x{s.value:x}",
                    "offset": s.offset,
                    "offset_hex": f"0x{s.offset:x}",
                    "size": s.size,
                }
        missing = [n for n in ("qwen_frame_count", "qwen_key_state", "qwen_demo_enabled",
                               "DG_ScreenBuffer") if n not in syms]
        if missing:
            raise KeyError(f"{self.path}: missing required symbols: {', '.join(missing)}")
        return {
            "elf": str(self.path),
            "elf_md5": self.md5(),
            "entry": f"0x{self.entry:x}",
            "ram_base": f"0x{RAM_BASE:x}",
            "text_end": text_end,
            "text_end_hex": f"0x{text_end:x}",
            "ram_size": ram_size,
            "ram_size_hex": f"0x{ram_size:x}",
            "symbols": syms,
        }


def infer_fb_geometry(elf: Elf32) -> dict:
    """Infer (width, height, stride) from the adjacent fb_cols/fb_linear BSS symbols.

    ray0.S lays out fb_cols (width*stride bytes), then fb_linear (width*height bytes), then
    qwen_frame_count.  The cost build never uses fb_linear, but its presence makes the two
    region sizes visible in the symbol table.  Among integer solutions, prefer the Doom-like
    8:5 aspect and then minimal row padding.  For DoomGeneric ELFs without fb_cols, fall back
    to a linear row-major framebuffer (width/height overridable by the runner).
    """
    if "fb_cols" not in elf.symbols or "fb_linear" not in elf.symbols:
        return {"layout": "linear", "width": 160, "height": 100, "stride": 160,
                "pitch_bytes": 16000, "linear_bytes": 16000,
                "aspect_error": 0.0, "row_padding": 0}
    fc = elf.require("fb_cols").value
    fl = elf.require("fb_linear").value
    qf = elf.require("qwen_frame_count").value
    pitch = fl - fc
    linear = qf - fl
    if pitch <= 0 or linear <= 0:
        raise ValueError(f"bad framebuffer symbol ordering: pitch={pitch}, linear={linear}")
    candidates = []
    for w in range(8, 641):
        if pitch % w or linear % w:
            continue
        stride, h = pitch // w, linear // w
        if not (8 <= h <= 400 and h <= stride <= 4 * max(h, 1) and stride % 4 == 0):
            continue
        aspect_score = abs((w / h) - 1.6)
        pad = stride - h
        pow2_bonus = 0 if stride & (stride - 1) == 0 else 1
        candidates.append((aspect_score, pad, pow2_bonus, stride, w, h))
    if not candidates:
        raise ValueError(f"no plausible framebuffer geometry for pitch={pitch}, linear={linear}")
    aspect_score, pad, _pow2, stride, w, h = sorted(candidates)[0]
    return {"width": w, "height": h, "stride": stride,
            "pitch_bytes": pitch, "linear_bytes": linear,
            "aspect_error": round(aspect_score, 6), "row_padding": pad}


def runner_flags(cfg: dict, fb: dict) -> str:
    s = cfg["symbols"]
    return (f"--fc-off {s['qwen_frame_count']['offset_hex']} "
            f"--key-off {s['qwen_key_state']['offset_hex']} "
            f"--demo-off {s['qwen_demo_enabled']['offset_hex']} "
            f"--fb-off {s['DG_ScreenBuffer']['offset_hex']} "
            f"--fb-w {fb['width']} --fb-h {fb['height']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("elf")
    ap.add_argument("--json", action="store_true", help="print the full JSON config")
    ap.add_argument("--shell", action="store_true", help="print runner flags for this ELF")
    ap.add_argument("--write", type=Path, help="write the full JSON config here")
    a = ap.parse_args(argv)

    elf = Elf32(a.elf)
    cfg = elf.machine_config()
    cfg["framebuffer"] = infer_fb_geometry(elf)
    if a.write:
        a.write.parent.mkdir(parents=True, exist_ok=True)
        a.write.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    if a.shell:
        print(runner_flags(cfg, cfg["framebuffer"]))
    elif a.json or not a.write:
        print(json.dumps(cfg, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
