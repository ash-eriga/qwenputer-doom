#!/usr/bin/env python3
"""Run two Doom ELFs on lx.ref.Machine and compare their framebuffers frame by frame.

Used to prove the fast build (pruned WAD, assembly renderer, -O2 hot files) draws exactly the
frames a stock -Os C build with the full WAD draws. Both guests are started from reset and
stepped until the requested frame's DG_DrawFrame has completed, so the comparison covers the
whole boot + level load + render path, not just the last function.

Usage: compare_frames.py A.elf B.elf [--frames 6,8,10] [--png-dir DIR] [--scale 2] [--wad W]
"""
import argparse
import hashlib
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LX = ROOT.parent / 'Doom-On-Qwen' / 'lxsrc' / 'lx'
sys.path.insert(0, str(LX))
sys.path.insert(0, str(ROOT / 'port-asm'))

from lx import spec as S                      # noqa: E402
from lx.ref import Machine                    # noqa: E402
from measure_ray import Elf, load_memory, machine_config   # noqa: E402
from ppm_to_png import write_png              # noqa: E402

RAM_BASE = S.RAM_BASE


def read_pal(wad_path):
    raw = Path(wad_path).read_bytes()
    n_lumps, dir_off = struct.unpack_from('<II', raw, 4)
    for i in range(n_lumps):
        off, size, name = struct.unpack_from('<II8s', raw, dir_off + 16 * i)
        if name.rstrip(b'\0') == b'PLAYPAL':
            return raw[off:off + 768]
    raise SystemExit('PLAYPAL not in WAD')


class Guest:
    def __init__(self, path, wad_path):
        self.elf = Elf(path)
        ram_size, text_end = machine_config(self.elf)
        self.ram_size, self.text_end = ram_size, text_end
        self.m = Machine(load_memory(self.elf, ram_size), text_end=text_end)
        self.m.x[2] = RAM_BASE + ram_size - 16
        self.m.anchor = (self.elf.entry - RAM_BASE) // 4
        self.m.n = 0
        self.fc = self.elf.symbols['qwen_frame_count'] - RAM_BASE
        self.fbp = self.elf.symbols['I_VideoBuffer'] - RAM_BASE
        self.w, self.h = self.elf.symbols['__lf_width'], self.elf.symbols['__lf_height']
        self.pal = read_pal(wad_path)

    def frame(self):
        return struct.unpack_from('<I', self.m.ram, self.fc)[0]

    def run_to(self, frame, limit=200_000_000):
        steps = 0
        while self.frame() < frame and not self.m.halted and steps < limit:
            self.m.step()
            steps += 1
        if self.m.halted:
            out = bytes(self.m.out).decode('latin1')
            raise SystemExit(f'{self.elf.symbols and "guest"} halted before frame {frame}:\n{out[-1200:]}')

    def pixels(self):
        addr = struct.unpack_from('<I', self.m.ram, self.fbp)[0] - RAM_BASE
        return bytes(self.m.ram[addr:addr + self.w * self.h])

    def png(self, path, scale):
        px = self.pixels()
        rgb = bytearray(self.w * self.h * 3)
        for i, b in enumerate(px):
            rgb[3 * i:3 * i + 3] = self.pal[3 * b:3 * b + 3]
        write_png(path, self.w, self.h, bytes(rgb), scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--frames', default='6,8,10')
    ap.add_argument('--png-dir')
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--wad', default=str(ROOT / 'build' / 'doom1.wad'))
    a = ap.parse_args()

    frames = [int(x) for x in a.frames.split(',')]
    ga = Guest(a.a, a.wad)
    gb = Guest(a.b, a.wad)
    if (ga.w, ga.h) != (gb.w, gb.h):
        raise SystemExit(f'different screen sizes: {ga.w}x{ga.h} vs {gb.w}x{gb.h}')
    print(f'{a.a}: {ga.w}x{ga.h}, ram_size={ga.ram_size:#x}, text_end={ga.text_end:#x}')
    print(f'{a.b}: {gb.w}x{gb.h}, ram_size={gb.ram_size:#x}, text_end={gb.text_end:#x}')

    ok = True
    for f in sorted(frames):
        ga.run_to(f)
        gb.run_to(f)
        pa, pb = ga.pixels(), gb.pixels()
        same = pa == pb
        ok &= same
        diff = sum(1 for x, y in zip(pa, pb) if x != y)
        print(f'frame {f}: {"" if same else "DIFF"} bytes={len(pa)} differing={diff} '
              f'sha256={hashlib.sha256(pa).hexdigest()[:16]} / {hashlib.sha256(pb).hexdigest()[:16]}')
        if a.png_dir:
            out = Path(a.png_dir)
            out.mkdir(parents=True, exist_ok=True)
            ga.png(out / f'frame-{f:04d}.png', a.scale)
            print(f'  wrote {out / ("frame-%04d.png" % f)}')
    print('IDENTICAL' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
