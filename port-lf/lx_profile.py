#!/usr/bin/env python3
"""Attribute exact LX instructions and tokens to guest functions, using lx.ref.Machine.

This is the profiling half of port-asm/measure_exact_lx.py: same machine, same token rules, but
each record's tokens are charged to the function whose code the PC is in when it commits. Frames
are delimited by the guest's qwen_frame_count, so a run measures exactly what a display frame
costs.

Usage:
    lx_profile.py ELF [--from 6] [--to 10] [--top 10] [--csv /tmp/x.csv] [--png-dir DIR]
"""
import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LX = ROOT.parent / 'Doom-On-Qwen' / 'lxsrc' / 'lx'
sys.path.insert(0, str(LX))
sys.path.insert(0, str(ROOT / 'port-asm'))

from lx import spec as S              # noqa: E402
from lx.ref import Machine            # noqa: E402
from measure_ray import Elf, load_memory, machine_config  # noqa: E402

RAM_BASE = S.RAM_BASE


def frame_count_addr(elf):
    return elf.symbols['qwen_frame_count'] - RAM_BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf')
    ap.add_argument('--from', dest='first', type=int, default=6)
    ap.add_argument('--to', dest='last', type=int, default=10)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--tok-per-s', type=float, default=13000.0,
                    help='assumed GPU sequential tok/s for per-component time estimate (default 13k for H200 @ 20L)')
    ap.add_argument('--csv')
    ap.add_argument('--png-dir')
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--width', type=int, default=192)
    ap.add_argument('--height', type=int, default=120)
    ap.add_argument('--wad', default=str(ROOT / 'build' / 'doom1.wad'))
    a = ap.parse_args()

    elf = Elf(a.elf)
    ram_size, text_end = machine_config(elf)
    mem = load_memory(elf, ram_size)
    m = Machine(mem, text_end=text_end)
    m.x[2] = RAM_BASE + ram_size - 16
    m.anchor = (elf.entry - RAM_BASE) // 4
    m.n = 0
    fc_addr = frame_count_addr(elf)

    # function ranges from the ELF symbols
    funcs = sorted(((addr, name) for name, addr in elf.symbols.items()
                    if addr >= elf.symbols.get('_start', RAM_BASE)),
                   key=lambda t: t[0])
    addrs = [f[0] for f in funcs]

    import bisect
    def fn_of(pc):
        i = bisect.bisect_right(addrs, pc) - 1
        if i < 0:
            return '?'
        return funcs[i][1]

    per_fn = {}
    per_frame = {}
    frames = 0
    cur_instr = 0
    cur_tokens = 0
    frame_base_instr = 0
    frame_base_tokens = 0
    last_fn = None
    last_pc = 0

    fb_addr = elf.symbols.get('I_VideoBuffer')
    wad = Path(a.wad).read_bytes()
    n_lumps, dir_off = struct.unpack_from('<II', wad, 4)
    pal = None
    for i in range(n_lumps):
        off, size, name = struct.unpack_from('<II8s', wad, dir_off + 16 * i)
        if name.rstrip(b'\0') == b'PLAYPAL':
            pal = wad[off:off + 768]
            break
    sys.path.insert(0, str(ROOT / 'port-asm'))
    from ppm_to_png import write_png  # noqa: E402

    def dump(tag):
        """Write the guest's live framebuffer (192x120 indices) as a PNG in the real palette."""
        if not a.png_dir or fb_addr is None:
            return
        addr = struct.unpack_from('<I', m.ram, fb_addr - RAM_BASE)[0] - RAM_BASE
        w, h = a.width, a.height
        rgb = bytearray(w * h * 3)
        for i in range(w * h):
            b = m.ram[addr + i]
            rgb[3 * i:3 * i + 3] = pal[3 * b:3 * b + 3]
        out = Path(a.png_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_png(out / f'frame-{tag:04d}.png', w, h, bytes(rgb), a.scale)

    stepped = 0
    while frames <= a.last and not m.halted and stepped < 400_000_000:
        f0 = int.from_bytes(m.ram[fc_addr:fc_addr + 4], 'little')
        pc = m.pc
        fn = fn_of(pc)
        toks = m.step()
        if fn != last_fn:
            last_fn = fn
        n = len(toks)
        cur_tokens += n
        cur_instr += 1
        rec = per_fn.get(fn)
        if rec is None:
            rec = per_fn[fn] = [0, 0]
        if a.first <= f0 < a.last:
            rec[0] += 1
            rec[1] += n
        stepped += 1
        f1 = int.from_bytes(m.ram[fc_addr:fc_addr + 4], 'little')
        if f1 != f0:
            frames = f1
            per_frame[frames] = (cur_instr - frame_base_instr, cur_tokens - frame_base_tokens)
            if a.first <= frames <= a.last:
                dump(frames)
            frame_base_instr, frame_base_tokens = cur_instr, cur_tokens

    sel = {k: v for k, v in per_frame.items() if a.first < k <= a.last}
    nfr = max(len(sel), 1)
    tot_i = sum(v[0] for v in sel.values())
    tot_t = sum(v[1] for v in sel.values())
    print(f'frames {a.first}->{a.last}: instr/frame={tot_i / nfr:.1f} '
          f'tokens/frame={tot_t / nfr:.1f} tokens/instr={tot_t / max(tot_i, 1):.3f}')
    for f in sorted(sel):
        print(f'  frame {f}: instr={sel[f][0]} tokens={sel[f][1]}')
    print(f'halted={m.halted} guest_bytes={len(m.out)}')
    if m.halted or b'rror' in bytes(m.out):
        print('--- guest output (tail) ---')
        print(bytes(m.out).decode('latin1')[-2000:])

    rows = sorted(per_fn.items(), key=lambda kv: -kv[1][1])
    total_tok = sum(v[1] for v in per_fn.values()) or 1
    total_inst = sum(v[0] for v in per_fn.values())
    print(f'\nwhole run: instr={total_inst} tokens={total_tok}')
    print(f'{"function":34s} {"instr":>10s} {"tokens":>10s} {"tok%":>6s} {"tok/frame":>10s} '
          f'{"tok/instr":>9s}')
    for name, (i, t) in rows[:a.top]:
        print(f'{name:34s} {i:10d} {t:10d} {100.0 * t / total_tok:6.2f} {t / nfr:10.1f} '
              f'{t / max(i, 1):9.3f}')

    def component_of(fn):
        if 'Span' in fn or 'Plane' in fn:
            return 'Floor/Ceiling (Spans & Planes)'
        if 'Column' in fn and 'Masked' not in fn:
            return 'Walls (Columns)'
        if fn in ('R_RenderSegLoop', 'R_StoreWallRange', 'R_GetColumn', 'R_ScaleFromGlobalAngle',
                  'R_ClipSolidWallSegment', 'R_ClipPassWallSegment'):
            return 'Walls (Segs & Projection)'
        if fn in ('R_RenderBSPNode', 'R_Subsector', 'R_CheckBBox', 'R_AddLine', 'R_PointOnSide',
                  'R_PointToAngle', 'R_PointToAngle2', 'SlopeDiv'):
            return 'BSP & Visibility'
        if 'Sprite' in fn or 'Masked' in fn or 'PSprite' in fn or 'Things' in fn:
            return 'Sprites & Weapon'
        if fn.startswith(('P_', 'G_', 'A_', 'T_')):
            return 'Gameplay / Physics / AI'
        if fn in ('__divdi3', '__udivdi3', 'FixedDiv', 'FixedDiv2', 'FixedMul'):
            return 'Fixed-Point Math / Div'
        return 'Other (WAD/HUD/Misc)'

    comps = {}
    for name, (i, t) in rows:
        c = component_of(name)
        ci, ct = comps.get(c, (0, 0))
        comps[c] = (ci + i, ct + t)
    print(f'\nby component (at {a.tok_per_s:,.0f} tok/s):')
    print(f'{"component":34s} {"tok/frame":>10s} {"tok%":>6s} {"sec/frame":>10s}')
    for c, (_i, t) in sorted(comps.items(), key=lambda kv: -kv[1][1]):
        tf = t / nfr
        print(f'{c:34s} {tf:10.1f} {100.0 * t / total_tok:6.2f} {tf / a.tok_per_s:9.2f}s')
    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('function,instr,tokens,tokens_per_frame\n')
            for name, (i, t) in rows:
                f.write(f'{name},{i},{t},{t / nfr:.1f}\n')


if __name__ == '__main__':
    main()
