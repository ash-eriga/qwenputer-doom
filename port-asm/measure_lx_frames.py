#!/usr/bin/env python3
"""Measure exact LX tokens and instructions per frame with lx.ref.Machine.

This is the slow but authoritative CPU-only oracle. It works for doom-lf.elf and for the
assembly probes. Symbols are read from the ELF when available; raw offsets can be supplied
for stripped/other guests.
"""
import argparse
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for cand in (ROOT / 'Doom-On-Qwen' / 'lxsrc' / 'lx', ROOT / 'linuxformer' / 'lx'):
    if (cand / 'lx' / 'ref.py').exists():
        sys.path.insert(0, str(cand))
        break

from lx import spec as S  # noqa: E402
from lx.ref import Machine  # noqa: E402
from measure_ray import Elf, load_memory, read_script  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--script', default='')
    ap.add_argument('--fc-off', type=lambda x: int(x, 0))
    ap.add_argument('--key-off', type=lambda x: int(x, 0))
    ap.add_argument('--demo-off', type=lambda x: int(x, 0))
    ap.add_argument('--out', default='/tmp/lx-frame-counts.csv')
    a = ap.parse_args()

    elf = Elf(a.elf)

    def off(name, raw):
        if raw is not None:
            return raw
        return elf.symbols[name] - S.RAM_BASE

    fc_off = off('qwen_frame_count', a.fc_off)
    key_off = off('qwen_key_state', a.key_off)
    demo_off = off('qwen_demo_enabled', a.demo_off)

    ram = load_memory(elf)
    m = Machine(ram, 0x40000)
    m.x[2] = S.RAM_BASE + len(ram) - 16
    m.anchor = (elf.entry - S.RAM_BASE) >> 2
    m.n = 0
    sched = read_script(a.script) if a.script else []
    last = -1
    tokens = 0
    rows = []
    t0 = time.time()

    while not m.halted:
        fc = struct.unpack_from('<I', m.ram, fc_off)[0]
        if fc != last:
            if sched:
                mask = 0
                for f, k in sched:
                    if f <= fc:
                        mask = k
                struct.pack_into('<I', m.ram, demo_off, 0)
                struct.pack_into('<I', m.ram, key_off, mask)
            if fc > 0:
                rows.append((fc, m.instret, tokens))
                print(f'{fc},{m.instret},{tokens}', flush=True)
                if fc >= a.frames:
                    break
            last = fc
        toks = m.step()
        tokens += len(toks)

    dt = time.time() - t0
    with open(a.out, 'w') as f:
        f.write('frame,cum_instructions,cum_tokens\n')
        for r in rows:
            f.write('%d,%d,%d\n' % r)
    print(f'DONE frames={len(rows)} instructions={m.instret} tokens={tokens} '
          f'tok_per_instr={tokens / max(m.instret, 1):.6f} elapsed={dt:.2f}s '
          f'mips={m.instret / max(dt, 1e-9) / 1e6:.2f}', flush=True)


if __name__ == '__main__':
    main()
