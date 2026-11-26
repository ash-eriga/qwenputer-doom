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
FB_STRIDE = 64          # bytes per framebuffer column: 60 visible pixels + one hidden word
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
# ray loop can use angle+offset directly without a modulo/mask instruction.  They become load
# immediates in the generated per-group code.
fov_units = 256 * 60 / 360
cam = [64 + 2 * round((x / (W - 1) - 0.5) * fov_units) for x in range(W)]

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

# Every framebuffer word starts at zero.  At run time the hidden stride word (bytes 60..63) of
# the first column of each ray group holds the id of the column pattern that group last drew.
lines += ['    .section .data', '    .align 4', '    .globl fb_cols', 'fb_cols:',
          f'    .zero {W * FB_STRIDE}']

out = HERE / f'tables-{args.suffix}.S'
out.write_text('\n'.join(lines) + '\n')

# A column pattern is the fifteen colour registers of one fixed-distance column: fifteen
# aligned SW cover the visible 60 scanlines (four pixels per word).  Group colour is chosen at
# the centre of each four-scanline group; wall groups sample the extracted texture vertically,
# so distant columns still use the true quantized STARTAN3/BROWN colours.  Distances that give
# the same pixels share one pattern id, which also lets the delta check skip them.
texregs = ['a3', 't3', 't4', 't5', 't6']
tx = world.get('texture_tx', 0)


def column_colors(steps):
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
            colors.append(texregs[world['class_at'](tx, ty)])
        else:
            colors.append('a4')
    return tuple(colors)


pattern_ids = {}
pattern_of = [0] * (CASES + 1)
for steps in range(1, CASES + 1):
    pattern_of[steps] = pattern_ids.setdefault(column_colors(steps), len(pattern_ids) + 1)
patterns = {pid: colors for colors, pid in pattern_ids.items()}

# Columns whose camera offsets are equal cast the same ray, so each run of equal offsets is one
# ray group: one march, one delta check and one kernel that writes all of its columns.
groups = []  # [first column, width, biased camera byte offset]
for x, v in enumerate(cam):
    if groups and groups[-1][2] == v:
        groups[-1][1] += 1
    else:
        groups.append([x, 1, v])
widths = sorted({w for _x, w, _v in groups})
assert max(widths) * FB_STRIDE <= 2047

kernel = ['# generated renderer: ray groups, unrolled marches and column kernels',
          f'# source: {world["source"]}',
          f'# {len(groups)} ray groups for {W} columns, {len(patterns)} column patterns',
          '    .section .text',
          '    .align 2',
          '    .globl render_groups',
          'render_groups:']
# s4 is the first column of the group, s7 the player's biased raydelta_tab address.
for i, (_x, w, v) in enumerate(groups):
    kernel += [f'    lh s6,{v}(s7)', f'    jal ra,march_{w}']
    if i + 1 < len(groups):
        kernel.append(f'    addi s4,s4,{FB_STRIDE * w}')
kernel += ['    j render_done',
           '    .globl colkern_done',
           'colkern_done:',
           '    ret',
           '    .globl colkern_base',
           'colkern_base:']


def kernel_lines(w, pid):
    out = [f'    .globl colkern_{w}_{pid}', f'colkern_{w}_{pid}:',
           f'    lw t1,{H}(s4)',
           f'    li t2,{pid}',
           '    beq t1,t2,colkern_done',
           f'    sw t2,{H}(s4)']
    for c in range(w):
        for i, reg in enumerate(patterns[pid]):
            out.append(f'    sw {reg},{FB_STRIDE * c + 4 * i}(s4)')
    out.append('    ret')
    return out


# Every probe branches straight to its pattern's kernel.  The march sits between the two halves
# of its kernels so every target stays inside the +-4 KiB conditional-branch range; an
# out-of-range branch would be relaxed into a branch plus a jump, one more token per ray.
for w in widths:
    pids = sorted(patterns)
    half = (len(pids) + 1) // 2
    for pid in pids[:half]:
        kernel += kernel_lines(w, pid)
    kernel += [f'    .globl march_{w}', f'march_{w}:',
               '    add s5,s8,s6',
               '    lbu s10,0(s5)',
               f'    bne s10,zero,colkern_{w}_{pattern_of[1]}']
    for steps in range(2, CASES + 1):
        kernel += ['    add s5,s5,s6',
                   '    lbu s10,0(s5)',
                   f'    bne s10,zero,colkern_{w}_{pattern_of[steps]}']
    kernel.append(f'    j colkern_{w}_{pattern_of[CASES]}')
    for pid in pids[half:]:
        kernel += kernel_lines(w, pid)

kout = HERE / f'kernels-{args.suffix}.S'
kout.write_text('\n'.join(kernel) + '\n')
print(out)
print(kout)
