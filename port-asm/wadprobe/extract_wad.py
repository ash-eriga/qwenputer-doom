#!/usr/bin/env python3
"""Extract a compact Doom/E1M1 fidelity subset from DOOM1.WAD and generate LX assembly.

The extraction is deliberately lossy and explicit:
  * true E1M1 one-sided/impassable linedefs are rasterized into a 32x32 tile occupancy grid
    centered on the player start (64 Doom units per tile), with one bit of wall orientation;
  * the dominant middle texture in that window is composited from TEXTURE1+PNAMES patches;
  * the texture is reduced to five palette colors and baked into straight-line column kernels;
  * dominant E1M1 floor/ceiling flats contribute their most common palette indices.

The guest never parses the WAD. This is the point of the probe: WAD fidelity is paid at build
time, while the in-weights frame loop pays only for geometry and pixels it actually uses.
"""
import argparse
import collections
import json
import math
import struct
from pathlib import Path

W, H = 96, 60
NTAB = 256
GRID = 32
TILE_DOOM_UNITS = 64
WINDOW = GRID * TILE_DOOM_UNITS
KERNEL_STRIDE = 80
PHASES = 8
CASES = 64


def cname(raw):
    return raw.split(b'\0')[0].decode('ascii', 'replace')


def repbyte(v):
    v &= 0xff
    return v | (v << 8) | (v << 16) | (v << 24)


class Wad:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        if len(self.data) < 12:
            raise ValueError('not a WAD')
        self.magic, self.count, self.diroff = struct.unpack_from('<III', self.data, 0)
        if self.magic not in (0x44415749, 0x44415750):  # IWAD/PWAD little endian
            raise ValueError(f'bad WAD magic {self.magic:#x}')
        self.lumps = []
        for i in range(self.count):
            off, size, raw = struct.unpack_from('<II8s', self.data, self.diroff + i * 16)
            self.lumps.append({'name': cname(raw), 'off': off, 'size': size, 'index': i})
        self.byname = {}
        for lump in self.lumps:
            self.byname.setdefault(lump['name'], []).append(lump)

    def lump(self, name, occurrence=0):
        ent = self.byname[name][occurrence]
        return self.data[ent['off']:ent['off'] + ent['size']]

    def lump_entry(self, name, occurrence=0):
        return self.byname[name][occurrence]

    def map_lumps(self, marker='E1M1'):
        idx = next(i for i, x in enumerate(self.lumps) if x['name'] == marker)
        out = {}
        for ent in self.lumps[idx + 1:idx + 11]:
            out[ent['name']] = ent
        return out


def parse_map(wad):
    ml = wad.map_lumps('E1M1')
    def part(name):
        ent = ml[name]
        return wad.data[ent['off']:ent['off'] + ent['size']]

    vd = part('VERTEXES')
    verts = [struct.unpack_from('<hh', vd, i * 4) for i in range(len(vd) // 4)]

    ld = part('LINEDEFS')
    lines = []
    for i in range(len(ld) // 14):
        v1, v2, flags, typ, tag, rs, ls = struct.unpack_from('<HHHHHHH', ld, i * 14)
        lines.append({'v1': v1, 'v2': v2, 'flags': flags, 'type': typ, 'tag': tag,
                      'right': rs, 'left': ls})

    td = part('THINGS')
    things = [struct.unpack_from('<hhhhh', td, i * 10) for i in range(len(td) // 10)]
    player = next(x for x in things if x[3] == 1)

    sd = part('SIDEDEFS')
    sidedefs = []
    for i in range(len(sd) // 30):
        xoff, yoff = struct.unpack_from('<hh', sd, i * 30)
        upper = cname(sd[i * 30 + 4:i * 30 + 12])
        lower = cname(sd[i * 30 + 12:i * 30 + 20])
        middle = cname(sd[i * 30 + 20:i * 30 + 28])
        sector = struct.unpack_from('<H', sd, i * 30 + 28)[0]
        sidedefs.append({'upper': upper, 'lower': lower, 'middle': middle,
                         'sector': sector, 'xoff': xoff, 'yoff': yoff})

    secd = part('SECTORS')
    sectors = []
    for i in range(len(secd) // 26):
        fh, ch = struct.unpack_from('<hh', secd, i * 26)
        floor = cname(secd[i * 26 + 4:i * 26 + 12])
        ceil = cname(secd[i * 26 + 12:i * 26 + 20])
        light, special, tag = struct.unpack_from('<hhh', secd, i * 26 + 20)
        sectors.append({'floor': floor, 'ceiling': ceil, 'floor_h': fh, 'ceiling_h': ch,
                        'light': light, 'special': special, 'tag': tag})
    return verts, lines, sidedefs, sectors, player, ml


def decode_patch(data):
    w, h, left, top = struct.unpack_from('<hhhh', data, 0)
    cols = struct.unpack_from('<' + 'I' * w, data, 8)
    pix = [[0 for _ in range(w)] for _ in range(h)]
    for x in range(w):
        p = cols[x]
        guard = 0
        while p < len(data) and data[p] != 255 and guard < 256:
            topdelta = data[p]
            length = data[p + 1]
            for i in range(length):
                y = topdelta + i
                if 0 <= y < h:
                    pix[y][x] = data[p + 3 + i]
            p += length + 4
            guard += 1
    return w, h, pix


def texture_decoders(wad):
    pnames_raw = wad.lump('PNAMES')
    npatch = struct.unpack_from('<I', pnames_raw, 0)[0]
    pnames = [cname(pnames_raw[4 + i * 8:12 + i * 8]) for i in range(npatch)]
    t1 = wad.lump('TEXTURE1')
    ntex = struct.unpack_from('<I', t1, 0)[0]
    offsets = struct.unpack_from('<' + 'I' * ntex, t1, 4)
    patch_cache = {}

    def patch(name):
        if name not in patch_cache:
            patch_cache[name] = decode_patch(wad.lump(name))
        return patch_cache[name]

    def texture(name):
        for off in offsets:
            if cname(t1[off:off + 8]) != name:
                continue
            _masked, tw, th, _cd, pcount = struct.unpack_from('<IhhIH', t1, off + 8)
            img = [[0 for _ in range(tw)] for _ in range(th)]
            for k in range(pcount):
                ox, oy, pi, _stepdir, _cmap = struct.unpack_from('<hhhhh', t1, off + 22 + 10 * k)
                pw, ph, pp = patch(pnames[pi])
                for y in range(ph):
                    yy = y + oy
                    if not 0 <= yy < th:
                        continue
                    for x in range(pw):
                        xx = x + ox
                        if 0 <= xx < tw and pp[y][x]:
                            img[yy][xx] = pp[y][x]
            return tw, th, img
        raise KeyError(f'texture {name} not found')
    return texture


def palette(wad):
    raw = wad.lump('PLAYPAL')[:768]
    return [tuple(raw[i * 3:i * 3 + 3]) for i in range(256)]


def choose_color_clusters(indices, pal, k=5):
    """Small deterministic weighted k-means in RGB, returning representative palette indices."""
    weights = collections.Counter(indices)
    if not weights:
        return [1] * k, lambda _v: 0
    entries = sorted(weights.items(), key=lambda kv: sum(pal[kv[0]]), reverse=False)
    total = sum(weights.values())
    reps = []
    for q in ((1 / (2 * k)) * (2 * i + 1) for i in range(k)):
        acc = 0
        target = q * total
        for idx, cnt in entries:
            acc += cnt
            if acc >= target:
                reps.append(idx)
                break
        else:
            reps.append(entries[-1][0])

    for _ in range(12):
        centroids = [pal[r] for r in reps]
        buckets = [collections.Counter() for _ in range(k)]
        for idx, cnt in weights.items():
            pr, pg, pb = pal[idx]
            best = min(range(k), key=lambda j: ((pr - centroids[j][0]) ** 2 +
                                                 (pg - centroids[j][1]) ** 2 +
                                                 (pb - centroids[j][2]) ** 2))
            buckets[best][idx] += cnt
        newreps = []
        for j, bucket in enumerate(buckets):
            if not bucket:
                newreps.append(reps[j])
                continue
            cr = sum(pal[i][0] * n for i, n in bucket.items()) / sum(bucket.values())
            cg = sum(pal[i][1] * n for i, n in bucket.items()) / sum(bucket.values())
            cb = sum(pal[i][2] * n for i, n in bucket.items()) / sum(bucket.values())
            newreps.append(min(bucket, key=lambda i: ((pal[i][0] - cr) ** 2 +
                                                       (pal[i][1] - cg) ** 2 +
                                                       (pal[i][2] - cb) ** 2)))
        if newreps == reps:
            break
        reps = newreps

    centroids = [pal[r] for r in reps]
    def nearest(v):
        pr, pg, pb = pal[v]
        return min(range(k), key=lambda j: ((pr - centroids[j][0]) ** 2 +
                                             (pg - centroids[j][1]) ** 2 +
                                             (pb - centroids[j][2]) ** 2))
    return reps, nearest


def rasterize(verts, lines, sidedefs, player):
    px, py, angle, _typ, _flags = player
    cx, cy = px, py
    grid = [[0 for _ in range(GRID)] for _ in range(GRID)]
    texture_counts = collections.Counter()
    used_lines = 0

    def grid_pos(x, y):
        return (x - cx) / TILE_DOOM_UNITS + GRID / 2, (y - cy) / TILE_DOOM_UNITS + GRID / 2

    def mark(x, y, val):
        gx = math.floor((x - cx) / TILE_DOOM_UNITS + GRID / 2)
        gy = math.floor((y - cy) / TILE_DOOM_UNITS + GRID / 2)
        if 0 <= gx < GRID and 0 <= gy < GRID:
            grid[gy][gx] |= val

    half = WINDOW / 2
    for line in lines:
        solid = line['left'] == 0xffff or (line['flags'] & 1)
        if not solid:
            continue
        x1, y1 = verts[line['v1']]
        x2, y2 = verts[line['v2']]
        if (max(x1, x2) < cx - half or min(x1, x2) > cx + half or
                max(y1, y2) < cy - half or min(y1, y2) > cy + half):
            continue
        used_lines += 1
        if line['right'] != 0xffff:
            texture_counts[sidedefs[line['right']]['middle'] or '-'] += 1
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        oval = 1 if horizontal else 2
        # Sixteen samples per tile are intentionally conservative for a rasterized thin-wall
        # approximation. This is build-time work; the guest sees only the resulting byte grid.
        length = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(1, math.ceil(length / (TILE_DOOM_UNITS / 16)))
        for k in range(steps + 1):
            x = x1 + (x2 - x1) * k / steps
            y = y1 + (y2 - y1) * k / steps
            mark(x, y, oval)

    # Border is artificial but necessary: the compact fixed-point position format has only
    # eight bits per axis, and the border also keeps rays/movement inside the extracted window.
    for i in range(GRID):
        grid[0][i] |= 2
        grid[GRID - 1][i] |= 2
        grid[i][0] |= 1
        grid[i][GRID - 1] |= 1

    solid_cells = sum(1 for row in grid for v in row if v)
    spawn_gx = GRID // 2
    spawn_gy = GRID // 2
    ascii_rows = []
    chars = '.hv+'
    for row in reversed(grid):
        ascii_rows.append(''.join(chars[v & 3] for v in row))
    return grid, {
        'spawn_world': [px, py],
        'spawn_tile': [spawn_gx, spawn_gy],
        'doom_angle_degrees': angle,
        'lines_in_window': used_lines,
        'solid_cells': solid_cells,
        'solid_fraction': solid_cells / (GRID * GRID),
        'texture_counts': texture_counts,
        'ascii': ascii_rows,
    }


def flat_common_index(wad, name):
    try:
        data = wad.lump(name)
    except (KeyError, IndexError):
        return 96, 0
    if len(data) < 4096:
        return 96, len(data)
    cnt = collections.Counter(data[:4096])
    return cnt.most_common(1)[0][0], len(data)


def height_table(step_tiles):
    h = [0] * (CASES + 1)
    for steps in range(1, CASES + 1):
        h[steps] = min(H, max(1, round(45 / (steps * step_tiles))))
    return h


def asm_bytes(name, vals, per=16):
    lines = [f'    .globl {name}', f'{name}:']
    for i in range(0, len(vals), per):
        lines.append('    .byte ' + ','.join(str(v) for v in vals[i:i + per]))
    return lines


def generate_common(out, grid, texinfo, ceil_idx, floor_idx):
    texname, tw, th, img, reps, nearest = texinfo
    lines = ['# generated by extract_wad.py; do not edit', '    .section .rodata', '    .align 1']
    vals = [round(math.sin(2 * math.pi * i / NTAB) * 256) for i in range(NTAB)]
    lines += asm_bytes('sin_tab', [], 1)[:2]  # replaced below; keeps symbol style compact
    lines = lines[:-2]
    lines += ['    .globl sin_tab', 'sin_tab:']
    for i in range(0, NTAB, 16):
        lines.append('    .half ' + ','.join(map(str, vals[i:i + 16])))
    vals = [round(math.cos(2 * math.pi * i / NTAB) * 256) for i in range(NTAB)]
    lines += ['    .globl cos_tab', 'cos_tab:']
    for i in range(0, NTAB, 16):
        lines.append('    .half ' + ','.join(map(str, vals[i:i + 16])))

    fov_units = 256 * 60 / 360
    cam = [round((x / (W - 1) - 0.5) * fov_units) for x in range(W)]
    lines += ['    .globl camoff_tab', 'camoff_tab:']
    for i in range(0, W, 16):
        lines.append('    .byte ' + ','.join(map(str, cam[i:i + 16])))

    lines += ['    .globl map32', 'map32:']
    for row in grid:
        lines.append('    .byte ' + ','.join(map(str, row)))

    # Padded rows preserve ray0's packed-index trick: after shifting the packed 5.3 position,
    # xtile occupies bits 0..4 and ytile bits 8..12, so the lookup address is base+x+y*256.
    lines += ['    .globl map256', 'map256:']
    for row in grid:
        lines.append('    .byte ' + ','.join(map(str, row)))
        lines.append('    .zero 224')

    (out / 'wad_data.S').write_text('\n'.join(lines) + '\n')

    cfg = [
        '# generated extraction constants',
        f'#define CEIL_WORD 0x{repbyte(ceil_idx):08x}',
        f'#define FLOOR_WORD 0x{repbyte(floor_idx):08x}',
    ]
    for i, idx in enumerate(reps):
        cfg.append(f'#define TEX{i}_WORD 0x{repbyte(idx):08x}')
    (out / 'wad_config.h').write_text('\n'.join(cfg) + '\n')
    return reps


def generate_tables_and_kernels(out, suffix, step_tiles, texinfo):
    texname, tw, th, img, reps, nearest = texinfo
    htab = height_table(step_tiles)
    lines = ['# generated height table', '    .section .rodata', '    .align 1',
             '    .globl height_tab', 'height_tab:']
    for i in range(0, CASES + 1, 16):
        lines.append('    .byte ' + ','.join(map(str, htab[i:i + 16])))
    (out / f'wad_tables_{suffix}.S').write_text('\n'.join(lines) + '\n')

    # Five texture colors live in a3,t3,t4,t5,t6. The kernels still execute exactly sixteen
    # aligned SW instructions plus one jump: texture fidelity is selected at codegen time.
    regs = ['a3', 't3', 't4', 't5', 't6']
    kernel = ['# generated WAD-texture column kernels', '    .section .text', '    .align 4',
              '    .globl colkern_base', 'colkern_base:']
    for phase in range(PHASES):
        # Each fixed-point fraction bit selects a sixteenth-of-texture strip. Sampling at the
        # strip center avoids making every phase a hard edge of the 128px source texture.
        tx = min(tw - 1, int((phase + 0.5) * tw / PHASES))
        kernel.append('    .space 80')  # case 0 invalid, stride remains exact
        for steps in range(1, CASES + 1):
            hh = htab[steps]
            top = (H - hh) // 2
            floorpix = H - top - hh
            ceiln = (top + 3) // 4
            floorn = (floorpix + 3) // 4
            walln = 16 - ceiln - floorn
            colors = ['a2'] * ceiln
            for gi in range(walln):
                # Wall vertical coordinate in source-texture space. walln is in four-pixel
                # groups, so this is necessarily a coarse fidelity probe rather than patch-perfect
                # patch composition at runtime.
                v = (gi + 0.5) / max(walln, 1)
                ty = min(th - 1, int(v * th))
                src = img[ty][tx]
                colors.append(regs[nearest(src)])
            colors += ['a4'] * floorn
            assert len(colors) == 16
            kernel.append(f'colkern_p{phase}_{steps}:')
            for i, reg in enumerate(colors):
                kernel.append(f'    sw {reg},{4 * i}(s4)')
            kernel.append('    j colkern_done')
            kernel.append(f'    .space {KERNEL_STRIDE - 17 * 4}')
    (out / f'wad_kernels_{suffix}.S').write_text('\n'.join(kernel) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wad', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wad = Wad(args.wad)
    verts, lines, sidedefs, sectors, player, ml = parse_map(wad)
    grid, meta = rasterize(verts, lines, sidedefs, player)
    texname = meta['texture_counts'].most_common(1)[0][0] if meta['texture_counts'] else 'STARTAN3'
    if texname == '-':
        texname = 'STARTAN3'
    texture = texture_decoders(wad)
    tw, th, img = texture(texname)
    pal = palette(wad)
    flat_indices = [v for row in img for v in row]
    reps, nearest = choose_color_clusters(flat_indices, pal, 5)

    floor_counts = collections.Counter(s['floor'] for s in sectors)
    ceil_counts = collections.Counter(s['ceiling'] for s in sectors)
    floor_flat = floor_counts.most_common(1)[0][0]
    ceil_flat = ceil_counts.most_common(1)[0][0]
    floor_idx, floor_size = flat_common_index(wad, floor_flat)
    ceil_idx, ceil_size = flat_common_index(wad, ceil_flat)

    texinfo = (texname, tw, th, img, reps, nearest)
    generate_common(out, grid, texinfo, ceil_idx, floor_idx)
    generate_tables_and_kernels(out, 'full', 1.0, texinfo)
    generate_tables_and_kernels(out, 'half', 0.5, texinfo)

    solid_line_count = sum(1 for x in lines if x['left'] == 0xffff or x['flags'] & 1)
    geom_bytes = sum(ml[n]['size'] for n in ('LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SECTORS'))
    patch_entries = []
    for name in ('SW19_1', 'SW19_2'):
        try:
            ent = wad.lump_entry(name)
            patch_entries.append({'name': name, 'bytes': ent['size']})
        except KeyError:
            pass
    report = {
        'wad': str(wad.path),
        'wad_bytes': len(wad.data),
        'map': 'E1M1',
        'grid': {'tiles': GRID, 'tile_doom_units': TILE_DOOM_UNITS,
                 'coverage_doom_units': WINDOW, 'occupancy_bytes_compact': GRID * GRID,
                 'lookup_bytes_padded': GRID * 256},
        'geometry_source': {
            'vertices': len(verts), 'linedefs': len(lines), 'solid_linedefs': solid_line_count,
            'sidedefs': len(sidedefs), 'sectors': len(sectors),
            'raw_linedef_sidedef_vertex_sector_bytes': geom_bytes,
        },
        'extraction': {k: v for k, v in meta.items() if k not in ('texture_counts', 'ascii')},
        'dominant_window_textures': meta['texture_counts'].most_common(8),
        'texture': {'name': texname, 'width': tw, 'height': th, 'representative_palette_indices': reps,
                    'runtime_replicated_words': len(reps), 'phases': PHASES,
                    'patch_lumps': patch_entries},
        'flats': {'floor_name': floor_flat, 'floor_palette_index': floor_idx,
                  'ceiling_name': ceil_flat, 'ceiling_palette_index': ceil_idx,
                  'floor_lump_bytes': floor_size, 'ceiling_lump_bytes': ceil_size},
        'palette': {'entries': 256, 'first_palette_bytes': 768},
        'assembly_codegen': {'cases_per_phase': CASES, 'phases': PHASES,
                             'kernel_stride_bytes': KERNEL_STRIDE,
                             'kernel_text_bytes': PHASES * (CASES + 1) * KERNEL_STRIDE},
        'map_ascii_top_to_bottom': meta['ascii'],
    }
    (out / 'wad_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('map', 'grid', 'extraction', 'texture', 'flats', 'assembly_codegen')}, indent=2))


if __name__ == '__main__':
    main()
