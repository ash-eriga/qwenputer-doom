#!/usr/bin/env python3
"""Exact LX token count for the assembly probe, using lx.ref.Machine as the executable spec."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Doom-On-Qwen' / 'lxsrc' / 'lx'))

from lx.ref import Machine  # noqa: E402
from lx import spec as S  # noqa: E402
from measure_ray import Elf, load_memory, read_script  # noqa: E402

RAM_BASE = S.RAM_BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf')
    ap.add_argument('frames', type=int)
    ap.add_argument('script', nargs='?')
    a = ap.parse_args()

    elf = Elf(a.elf)
    mem = load_memory(elf)
    m = Machine(mem, text_end=0x40000)
    m.x[2] = RAM_BASE + len(mem) - 16
    m.anchor = (elf.entry - RAM_BASE) // 4
    m.n = 0

    fc_addr = elf.symbols['qwen_frame_count'] - RAM_BASE
    key_addr = elf.symbols['qwen_key_state'] - RAM_BASE
    schedule = read_script(a.script)
    if schedule and 'qwen_demo_enabled' in elf.symbols:
        demo = elf.symbols['qwen_demo_enabled'] - RAM_BASE
        m.ram[demo:demo + 4] = (0).to_bytes(4, 'little')
    sp = 0
    frames = 0
    tokens = 0
    records = 0
    start_instret = m.instret

    while frames < a.frames and records < 100_000_000:
        toks = m.step()
        tokens += len(toks)
        records += 1
        nf = int.from_bytes(m.ram[fc_addr:fc_addr + 4], 'little')
        if nf != frames:
            frames = nf
            while sp < len(schedule) and schedule[sp][0] <= frames:
                m.ram[key_addr:key_addr + 4] = schedule[sp][1].to_bytes(4, 'little')
                sp += 1

    inst = m.instret - start_instret
    print(f'frames={frames} records={records} instructions={inst} exact_lx_tokens={tokens}')
    if frames:
        print(f'instr/frame={inst / frames:.1f} tokens/frame={tokens / frames:.1f} '
                  f'tokens/instr={tokens / max(inst, 1):.3f}')
    print(f'20s_at_10fps_tokens={tokens / max(frames, 1) * 200:.0f}')


if __name__ == '__main__':
    main()
