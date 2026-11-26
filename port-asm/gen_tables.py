#!/usr/bin/env python3
"""Generate fixed-point tables, WAD-derived geometry and column kernels for ray0.S.

The runtime does not parse the WAD.  When build/doom1.wad is available this script uses
wadprobe/extract_wad.py to rasterize the real E1M1 solid-linedef window, quantize the dominant
real texture to five palette entries, and use the dominant real floor/ceiling flat colors.  A
small synthetic fallback keeps the probe buildable on machines without the shareware WAD.
"""
import argparse
import collections
import math
import sys
from pathlib import Path

W, H = 96, 60
NTAB = 256
GRID = 16
SUB = 4                 # quarter-tile units; keeps the direct position table at 4 KiB
XSUBS = GRID * SUB      # linear packed row stride: xsub + ysub*64
POSTAB_SIZE = XSUBS * XSUBS
KSTRIDE = 64            # fifteen aligned SW plus one jump, exactly 16 instructions
CASES = 64

ap = argparse.ArgumentParser()
ap.add_argument('--step-tiles', type=float, default=1.0, help='ray step length in tiles')
ap.add_argument('--ray-shift', type=int, default=6, help='assembly trig shift for this build')
ap.add_argument('--suffix', default='full', help='output suffix')
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def synthetic_world():
    rows = [
        '1111111111111111',
        '1000000000000001',
        '1000000000000001',
        '1000002000000001',
        '1000002000000001',
        '1000000000000001',
        '1000000000000001',
        '1111000000001111',
        '1000000000000001',
        '1000000000000001',
        '1000000000000001',
        '1000000000000001',
        '1000000000000001',
        '1000000000000001',
        '1000000000000001',
        '1111111111111111',
    ]
    grid = [[int(c) for c in row] for row in rows]
    # Fallback colors are the hand-picked Doom palette indexes used by the original probe.
    return {
        'source': 'synthetic fallback',
        'grid': grid,
        'ceil': 64,
        'floor': 96,
        'reps': [144, 128, 96, 64, 32],
        'class_at': lambda _tx, _ty: 0,
        'texture_name': 'synthetic',
    }


def wad_world():
    wad_path = ROOT / 'build' / 'doom1.wad'
    if not wad_path.exists():
        return synthetic_world()
    sys.path.insert(0, str(HERE / 'wadprobe'))
    try:
        import extract_wad

        wad = extract_wad.Wad(wad_path)
        verts, lines, sidedefs, sectors, player, _ml = extract_wad.parse_map(wad)
        grid32, meta = extract_wad.rasterize(verts, lines, sidedefs, player)

        # The extractor produces a 32x32 tile / 2048-Doom-unit window centered on the spawn.
        # Keep the central 16x16 tiles.  This covers the same number of tiles as the original
        # probe but with real E1M1 geometry, while keeping the direct position lookup at 4 KiB.
        lo = (32 - GRID) // 2
        hi = lo + GRID
        grid = [row[lo:hi] for row in grid32[lo:hi]]

        texname = meta['texture_counts'].most_common(1)[0][0] if meta['texture_counts'] else 'STARTAN3'
        if texname == '-':
            texname = 'STARTAN3'
        texture = extract_wad.texture_decoders(wad)
        tw, th, img = texture(texname)
        pal = extract_wad.palette(wad)
        reps, nearest = extract_wad.choose_color_clusters([v for row in img for v in row], pal, 5)

        floor_counts = collections.Counter(s['floor'] for s in sectors)
        ceil_counts = collections.Counter(s['ceiling'] for s in sectors)
        floor_idx, _floor_size = extract_wad.flat_common_index(wad, floor_counts.most_common(1)[0][0])
        ceil_idx, _ceil_size = extract_wad.flat_common_index(wad, ceil_counts.most_common(1)[0][0])

        # Pick the most visually active source-texture column.  Only one lateral phase is baked
        # so the column kernels stay inside the 8 KiB small-image ROM; vertical variation still
        # comes from the actual composited TEXTURE1/PNAMES data.
        def score(tx):
            cls = [nearest(img[y][tx]) for y in range(th)]
            return (len(set(cls)), sum(a != b for a, b in zip(cls, cls[1:])), -tx)
        tx = max(range(tw), key=score)

        return {
            'source': f'{wad_path.name} E1M1/{texname} tx{tx}',
            'grid': grid,
            'ceil': ceil_idx,
            'floor': floor_idx,
            'reps': reps,
            'class_at': lambda xtx, ty: nearest(img[min(th - 1, max(0, ty))][xtx]),
            'texture_tx': tx,
            'texture_name': texname,
        }
    except Exception as exc:
        print(f'gen_tables: warning: WAD extraction failed ({exc}); using synthetic fallback',
              file=sys.stderr)
        return synthetic_world()


world = wad_world()
grid = [list(row) for row in world['grid']]
assert len(grid) == GRID and all(len(row) == GRID for row in grid)
# The ray position is a single linear postab address updated by one ADD.  Clamp the probe
# window with solid edge cells so a ray can never cross from one row into another or run past
# the table before reporting a hit.
for i in range(GRID):
    grid[0][i] = grid[GRID - 1][i] = 1
    grid[i][0] = grid[i][GRID - 1] = 1

lines = ['# generated by gen_tables.py; do not edit',
         f'# source: {world["source"]}',
         '    .section .rodata',
         '    .align 1',
         '    .globl sin_tab',
         'sin_tab:']
sin_vals = [round(math.sin(2 * math.pi * i / NTAB) * 256) for i in range(NTAB)]
for i in range(0, NTAB, 16):
    lines.append('    .half ' + ','.join(map(str, sin_vals[i:i + 16])))
lines += ['    .globl cos_tab', 'cos_tab:']
cos_vals = [round(math.cos(2 * math.pi * i / NTAB) * 256) for i in range(NTAB)]
for i in range(0, NTAB, 16):
    lines.append('    .half ' + ','.join(map(str, cos_vals[i:i + 16])))

# Camera angle offsets are pre-doubled byte offsets into raydelta_tab, biased by +64 so the
# ray loop can use angle+offset directly without a modulo/mask instruction.
lines += ['    .globl camoff_tab', 'camoff_tab:']
fov_units = 256 * 60 / 360
cam = [64 + 2 * round((x / (W - 1) - 0.5) * fov_units) for x in range(W)]
for i in range(0, W, 16):
    lines.append('    .byte ' + ','.join(map(str, cam[i:i + 16])))

lines += ['    .align 1', '    .globl raydelta_tab', 'raydelta_tab:']
base_delta = [(cos_vals[i] >> args.ray_shift) + ((sin_vals[i] >> args.ray_shift) << 6)
              for i in range(NTAB)]
raydelta = [base_delta[(i - 32) & 255] for i in range(320)]
for i in range(0, len(raydelta), 16):
    lines.append('    .half ' + ','.join(map(str, raydelta[i:i + 16])))

# Wall height by fixed-step count; 45 pixels at one tile.
h = [0] * (CASES + 1)
for steps in range(1, CASES + 1):
    h[steps] = min(H, max(1, round(45 / (steps * args.step_tiles))))
lines += ['    .globl height_tab', 'height_tab:']
for i in range(0, CASES + 1, 16):
    chunk = h[i:i + 16]
    if chunk:
        lines.append('    .byte ' + ','.join(map(str, chunk)))

# Compact row-major collision map (16 bytes per row).  Values are orientation bits from the
# extracted linedefs: 1 mostly horizontal, 2 mostly vertical, 3 ambiguous/corner.
lines += ['    .globl map', 'map:']
for row in grid:
    lines.append('    .byte ' + ','.join(map(str, row)))

# Direct fixed-point position lookup.  The hot loop keeps address=postab+xsub+ysub*64 in one
# register.  Because that representation is linear rather than lane-packed, negative ray steps
# update it correctly with a single ADD; the artificial border prevents x/y range wraparound.
lines += ['    .globl postab', 'postab:']
for ysub in range(XSUBS):
    ytile = ysub // SUB
    row = []
    for xsub in range(XSUBS):
        row.append(grid[ytile][xsub // SUB])
    for i in range(0, XSUBS, 16):
        lines.append('    .byte ' + ','.join(map(str, row[i:i + 16])))

# Runtime color words are real palette indexes replicated to four pixels per word.  Kernel
# register use is: a2 ceiling, a4 floor, then a3/t3/t4/t5/t6 for the five texture clusters.
def repword(v):
    return v | (v << 8) | (v << 16) | (v << 24)

color_words = [world['ceil'], world['floor']] + world['reps']
lines += ['    .align 2', '    .globl render_words', 'render_words:']
for v in color_words:
    lines.append(f'    .word 0x{repword(v):08x}')

out = HERE / f'tables-{args.suffix}.S'
out.write_text('\n'.join(lines) + '\n')

# Column kernels: exactly fifteen aligned SW instructions cover the visible 60 scanlines
# (four pixels per word).  The old sixteenth store only wrote hidden stride padding.  Group
# color is chosen at the center of each four-scanline group; wall groups sample the extracted
# texture vertically, so distant columns still use the true quantized STARTAN3/BROWN colors.
kernel = ['# generated column kernels',
          f'# source: {world["source"]}',
          '    .section .text',
          '    .align 4',
          '    .globl colkern_base',
          'colkern_base:',
          '    .space 64']
texregs = ['a3', 't3', 't4', 't5', 't6']
tx = world.get('texture_tx', 0)
for steps in range(1, CASES + 1):
    hh = h[steps]
    top = (H - hh) // 2
    colors = []
    for group in range(15):
        y = group * 4 + 1.5
        if y < top:
            colors.append('a2')
        elif y < top + hh:
            v = (y - top) / max(hh, 1)
            ty = min(127, int(v * 128))
            cls = world['class_at'](tx, ty)
            colors.append(texregs[cls])
        else:
            colors.append('a4')
    assert len(colors) == 15
    kernel.append(f'colkern_{steps}:')
    for i, reg in enumerate(colors):
        kernel.append(f'    sw {reg},{4 * i}(s4)')
    kernel.append('    j colkern_done')
    assert 16 * 4 == KSTRIDE

kout = HERE / f'kernels-{args.suffix}.S'
kout.write_text('\n'.join(kernel) + '\n')
print(out)
print(kout)
