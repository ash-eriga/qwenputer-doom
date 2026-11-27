#!/usr/bin/env python3
"""Prune DOOM.WAD (shareware) down to exactly what E1M1 needs, for the LX fast build.

The LX machine loads the whole guest image as RAM, and the WAD lives in the image, so every
byte of lump data is a byte of ``RAM_SIZE`` -- the lever the circuit size scales with.  This
script keeps only the lumps a 192x120 DoomGeneric that boots straight into E1M1 and plays it
touches, and rebuilds TEXTURE1/PNAMES so no unused patch has to exist:

  * E1M1 and its ten map lumps (contiguous, P_SetupLevel addresses them by relative index);
  * every wall texture E1M1's sidedefs name, plus SKY1, plus the switch textures E1M1 uses
    (boot needs all episode-1 switch names to resolve, so the rest become 1-patch stubs that
    keep the name, and no pixels, because no sidedef references them);
  * every patch those textures reference;
  * every flat E1M1's sectors name, plus the rest of each flat animation cycle it starts
    (NUKAGE1..3), so P_UpdateSpecials animates exactly as in the real WAD;
  * every sprite reachable from E1M1's THINGS through the state table plus the player, the
    obtainable weapons' psprite states and the effects E1M1 can spawn (puffs, blood, imp
    fireball, teleport fog, barrel explosion) -- minus the frames a single player's first-person
    camera can never see (the player's own body is behind the view plane, tz < MINZ);
  * the Doom font ST_Init/HU_Init load unconditionally, PLAYPAL and COLORMAP.

The status bar cannot be blitted into a 192-pixel-wide frame buffer and is never drawn at
screenblocks 11, so the LX build compiles st_stuff.c with -DLF_NO_STBAR and this script leaves
its 70 KiB of ST* graphics (and ENDOOM, shown only on a clean exit) out of the WAD.  Pass
--with-stbar to keep them, which is what the -Os reference build of the same guest uses.

Everything else -- all sound/music lumps, E1M2..E1M9, intermission/menu/title art, unused
textures, patches, flats and sprites -- is dropped.  Output is a regular IWAD that
DoomGeneric's own W_Init/R_InitData/ST_Init/P_SetupLevel accept unchanged.

Usage:
    prune_wad.py [--src build/doom1.wad] [--out build/e1m1-pruned.wad]
                 [--dg doomgeneric/doomgeneric] [--stats]
"""
import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------- WAD reading

class Wad:
    def __init__(self, path):
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        magic, count, diroff = struct.unpack_from('<4sii', self.raw, 0)
        if magic not in (b'IWAD', b'PWAD'):
            raise SystemExit(f'{path}: not a WAD')
        self.lumps = []          # [name, filepos, size]
        for i in range(count):
            pos, size, name = struct.unpack_from('<ii8s', self.raw, diroff + 16 * i)
            self.lumps.append([name.split(b'\0')[0].decode('latin1'), pos, size])
        self.byname = {}
        for i, lump in enumerate(self.lumps):
            self.byname.setdefault(lump[0].upper(), i)

    def data(self, i):
        name, pos, size = self.lumps[i]
        return self.raw[pos:pos + size]

    def find(self, name):
        return self.byname.get(name.upper())

    def need(self, name):
        i = self.find(name)
        if i is None:
            raise SystemExit(f'{self.path}: missing lump {name}')
        return i

    def range_between(self, start, end):
        """Lump indices strictly between the markers `start` and `end`."""
        a = self.need(start) + 1
        b = self.need(end)
        return list(range(a, b))


# ---------------------------------------------------------------------------- TEXTURE1/PNAMES

def read_pnames(wad):
    i = wad.need('PNAMES')
    raw = wad.data(i)
    count = struct.unpack_from('<i', raw, 0)[0]
    return [raw[4 + 8 * k: 12 + 8 * k].split(b'\0')[0].decode('latin1').upper()
            for k in range(count)]


def read_texture1(wad):
    """[{name, header(bytes up to patches), patches:[(ox,oy,patchidx,stepdir,colormap)]}]"""
    i = wad.need('TEXTURE1')
    off0 = wad.lumps[i][1]
    raw = wad.raw
    size = wad.lumps[i][2]
    count = struct.unpack_from('<i', raw, off0)[0]
    textures = []
    for k in range(count):
        off = struct.unpack_from('<i', raw, off0 + 4 + 4 * k)[0]
        if off >= size:
            raise SystemExit('bad TEXTURE1 directory')
        b = off0 + off
        name = raw[b:b + 8].split(b'\0')[0].decode('latin1').upper()
        w, h = struct.unpack_from('<hh', raw, b + 12)
        pc = struct.unpack_from('<h', raw, b + 20)[0]
        patches = []
        for j in range(pc):
            ox, oy, pidx, sd, cm = struct.unpack_from('<hhhhh', raw, b + 22 + 10 * j)
            patches.append((ox, oy, pidx, sd, cm))
        textures.append(dict(name=name, header=raw[b:b + 22], width=w, height=h,
                             patches=patches))
    return textures


# ---------------------------------------------------------------------------- info.c tables

def parse_doom_info(dg):
    """Sprite names, the state table and mobjinfo, straight from DoomGeneric's info.c."""
    import re
    text = (dg / 'info.c').read_text()
    sprnames = re.findall(r'"([A-Z0-9]+)"',
                          re.search(r'char \*sprnames\[\] = \{(.*?)\};', text, re.S).group(1))

    states = []
    pat = (r'\{SPR_(\w+),\s*(\d+),\s*(-?\d+),\s*\{([^}]*)\},\s*(S_\w+),'
           r'\s*(\d+),\s*(\d+)\},\s*//\s*(S_\w+)')
    prev = -(1 << 30)
    for m in re.finditer(pat, text):
        # state order in the file is the state enum order; skip repeats from the #if 0 blocks
        if m.start() < prev:
            continue
        prev = m.start()
        states.append(dict(spr=m.group(1), frame=int(m.group(2)) & 0x7fff,
                           nxt=m.group(5), name=m.group(8)))
    sidx = {}
    for i, st in enumerate(states):
        sidx.setdefault(st['name'], i)

    mobj = {}
    cur = None
    for line in text.splitlines():
        m = re.search(r'\{\s*//\s*(MT_\w+)', line)
        if m:
            cur = m.group(1)
            mobj[cur] = {}
            continue
        if cur is not None:
            m = re.match(r'\s*(.+?),\s*//\s*(\w+)\s*$', line)
            if m:
                mobj[cur][m.group(2)] = m.group(1).strip()
            if line.strip().startswith('}'):
                cur = None
    return sprnames, states, sidx, mobj


def parse_switch_list(dg):
    """(name1, name2) pairs P_InitSwitchList resolves for the shareware episode."""
    import re
    text = (dg / 'p_switch.c').read_text()
    body = re.search(r'alphSwitchList\[\]\s*=\s*\{(.*?)\};', text, re.S).group(1)
    pairs = []
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*\}', body):
        if int(m.group(3)) <= 1:
            pairs.append((m.group(1).upper(), m.group(2).upper()))
    return pairs


def parse_anims(dg):
    """(istexture, startname, endname): the animation cycles P_InitPicAnims sets up."""
    import re
    text = (dg / 'p_spec.c').read_text()
    body = re.search(r'animdef_t\s+animdefs\[\]\s*=\s*\{(.*?)\};', text, re.S).group(1)
    anims = []
    for m in re.finditer(r'\{\s*(true|false|0|1)\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*(\d+)\s*\}',
                         body):
        istexture = m.group(1) in ('true', '1')
        # animdef_t is {istexture, endname, startname, speed}: the cycle starts at the second
        # name and runs forward through WAD order to the first.
        anims.append((istexture, m.group(3).upper(), m.group(2).upper()))
    return anims


def parse_weapon_psprite_states(dg, weapons=None):
    """Psprite states of the weapons the level can hand the player.

    weaponinfo[] is in weapon_t order (fist, pistol, shotgun, chaingun, ...); the player always
    owns the first two, and a level can hand over the rest only if it contains the pickup. Every
    other weapon's psprite art can never reach the screen, so it is not worth its bytes.
    """
    import re
    text = (dg / 'd_items.c').read_text()
    m = re.search(r'weaponinfo_t\s+weaponinfo\[NUMWEAPONS\]\s*=\s*\{(.*?)\n\};', text, re.S)
    if not m:
        raise SystemExit('d_items.c: weaponinfo not found')
    entries = re.split(r'\n\s*\{\s*//', m.group(1))
    states = set()
    for i, entry in enumerate(entries):
        if weapons is not None and i not in weapons:
            continue
        states.update(re.findall(r'(S_[A-Z0-9_]+)', entry))
    return states


# Pickup mobj type -> weapon_t index (d_items.c order), for the weapons a level can provide.
WEAPON_PICKUPS = {'MT_SHOTGUN': 2, 'MT_CHAINGUN': 3, 'MT_LAUNCHER': 4, 'MT_PLASMA': 5,
                  'MT_BFG': 6, 'MT_CHAINSAW': 7, 'MT_SUPERSHOTGUN': 8}


def parse_mobj_flags(dg):
    """{MT_*: set(flag names)} so things that cannot exist in E1M1 can be skipped."""
    import re
    text = (dg / 'info.c').read_text()
    flags = {}
    cur = None
    for line in text.splitlines():
        m = re.search(r'\{\s*//\s*(MT_\w+)', line)
        if m:
            cur = m.group(1)
            continue
        if cur is not None:
            m = re.match(r'\s*(.+?),\s*//\s*flags\s*$', line)
            if m:
                flags[cur] = set(re.findall(r'MF_(\w+)', m.group(1)))
                cur = None
    return flags


# ---------------------------------------------------------------------------- pruning

MAP_LUMPS = ['THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS', 'SSECTORS', 'NODES',
             'SECTORS', 'REJECT', 'BLOCKMAP']

STATUSBAR_LUMPS = (
    ['STTNUM%d' % i for i in range(10)] + ['STYSNUM%d' % i for i in range(10)] +
    ['STTPRCNT', 'STKEYS0', 'STKEYS1', 'STKEYS2', 'STKEYS3', 'STKEYS4', 'STKEYS5', 'STARMS'] +
    ['STGNUM%d' % i for i in range(2, 8)] +
    ['STFB0', 'STBAR', 'STTMINUS', 'STFGOD0', 'STFDEAD0'] +
    [x for i in range(5) for x in
     ['STFST%d%d' % (i, j) for j in range(3)] +
     ['STFTR%d0' % i, 'STFTL%d0' % i, 'STFOUCH%d' % i, 'STFEVL%d' % i, 'STFKILL%d' % i]]
)

# STCFN0nn is the Doom font HU_Init loads for player messages ("You got the shotgun!"), which
# are drawn on the fullscreen view too.  Everything else in STATUSBAR_LUMPS is status-bar art.
HUD_LUMPS = ['STCFN%.3d' % i for i in range(33, 96)] + ['PLAYPAL', 'COLORMAP']

# The player's own body: its frames belong to MT_PLAYER, whose mobj sits exactly at the camera
# (viewx/viewy are its x/y), so R_ProjectSprite rejects it at tz < MINZ and it is never drawn.
# A level does show bodies of this sprite: E1M1 has thing type 15 (MT_MISC62, "dead player",
# spawnstate S_PLAY_DIE7) and type 10 (MT_MISC68, "dead player, XDEAD", spawnstate
# S_PLAY_XDIE9), so the frames those spawn in must stay.  Everything else in PLAY (74 KiB, the
# standing/running/dying marine) can never reach the screen.
PLAYER_BODY_SPRITES = {'PLAY'}

# Mobj types DoomGeneric can only create at runtime (doomednum -1) whose sprites belong to
# monsters E1M1 does not contain: their projectiles/effects can never be spawned there.
# Leaving them out removes 35 KiB of sprite data and changes no pixel E1M1 can show.
MONSTER_ONLY_SPRITES = {'BAL2', 'BAL7', 'MANF', 'FATB', 'APLS', 'APBX', 'BOSF', 'SKUL',
                        'BFS1', 'BFE1', 'BFE2'}


def build_keep_set(wad, dg, stubs=True, verbose=False, with_stbar=False,
                   keep_player_sprite=False):
    # ---- E1M1 map
    e1m1 = wad.need('E1M1')
    map_idx = [e1m1] + [e1m1 + 1 + k for k in range(len(MAP_LUMPS))]
    for k, name in enumerate(MAP_LUMPS):
        if wad.lumps[e1m1 + 1 + k][0].upper() != name:
            raise SystemExit(f'E1M1: unexpected map lump {wad.lumps[e1m1 + 1 + k][0]}')

    # ---- sidedef textures and sector flats
    sd = wad.lumps[e1m1 + 3][1]
    raw = wad.raw
    map_tex = set()
    for o in range(0, wad.lumps[e1m1 + 3][2], 30):
        for k in range(3):
            nm = raw[sd + o + 4 + 8 * k: sd + o + 12 + 8 * k].split(b'\0')[0].decode('latin1')
            if nm and nm != '-':
                map_tex.add(nm.upper())
    se = wad.lumps[e1m1 + 8][1]
    map_flats = set()
    for o in range(0, wad.lumps[e1m1 + 8][2], 26):
        for k in range(2):
            map_flats.add(raw[se + o + 4 + 8 * k: se + o + 12 + 8 * k]
                          .split(b'\0')[0].decode('latin1').upper())

    # ---- textures: sidedefs + sky + switches + animation frames
    pnames = read_pnames(wad)
    textures = read_texture1(wad)
    tex_by_name = {t['name']: t for t in textures}

    switch_pairs = parse_switch_list(dg)
    switch_names = {n for pair in switch_pairs for n in pair}
    switch_used = {n for pair in switch_pairs for n in pair if n in map_tex}

    anims = parse_anims(dg)
    tex_order = [t['name'] for t in textures]

    keep_tex = set(map_tex) | {'SKY1'} | switch_used
    missing = keep_tex - set(tex_by_name)
    if missing:
        raise SystemExit(f'textures missing from WAD: {sorted(missing)}')
    # A cycle animates every texture between its start and end in WAD order, so if any member
    # is drawn every member is drawn: keep the whole run.
    keep_tex = _close_cycles(keep_tex, tex_order, anims, istexture=True)
    # A flipped switch shows its partner, so a switch E1M1 uses needs both names real.
    for a, b in switch_pairs:
        if a in keep_tex or b in keep_tex:
            keep_tex.add(a)
            keep_tex.add(b)
    # Boot resolves every episode-1 switch name (P_InitSwitchList), but only the switches E1M1's
    # sidedefs actually use are ever drawn: give the rest a named 1-patch stub with no pixels.
    keep_tex |= switch_names
    stub_tex = {n for n in switch_names if n not in switch_used} if stubs else set()
    real_tex = keep_tex - stub_tex

    keep_patch = set()
    for nm in sorted(real_tex):
        for ox, oy, pidx, sd_, cm in tex_by_name[nm]['patches']:
            keep_patch.add(pnames[pidx].upper())

    # ---- flats: map flats + every frame of a cycle any kept flat belongs to
    flat_order = [wad.lumps[i][0].upper() for i in wad.range_between('F_START', 'F_END')]
    keep_flat = set(map_flats) | {'F_SKY1'}
    keep_flat.discard('F1_START')
    keep_flat.discard('F1_END')
    keep_flat = _close_cycles(keep_flat, flat_order, anims, istexture=False)
    missing = keep_flat - set(flat_order)
    if missing:
        raise SystemExit(f'flats missing from WAD: {sorted(missing)}')

    # ---- sprites: state-graph closure from E1M1's things + player + weapons + effects
    sprnames, states, sidx, mobj = parse_doom_info(dg)
    bydoom = {}
    for mt, fields in mobj.items():
        dn = fields.get('doomednum', '-1').strip()
        try:
            bydoom.setdefault(int(dn), []).append(mt)
        except ValueError:
            pass
    thing_types = set()
    th = wad.lumps[e1m1 + 1][1]
    for o in range(0, wad.lumps[e1m1 + 1][2], 10):
        thing_types.add(struct.unpack_from('<h', raw, th + o + 6)[0])

    seed_mt = {'MT_PLAYER'}
    for t in thing_types:
        seed_mt.update(bydoom.get(t, []))
    for mt, fields in mobj.items():
        if fields.get('doomednum', '').strip() == '-1':
            seed_mt.add(mt)

    # Weapons the player can actually hold in this level: the starting fist and pistol plus any
    # pickup the THINGS list contains.
    weapons = {0, 1}
    for mt in seed_mt:
        if mt in WEAPON_PICKUPS:
            weapons.add(WEAPON_PICKUPS[mt])

    stack = []
    for mt in sorted(seed_mt):
        f = mobj[mt]
        for key in ('spawnstate', 'seestate', 'painstate', 'meleestate', 'missilestate',
                    'deathstate', 'xdeathstate', 'raisestate'):
            if key in f and f[key] in sidx:
                stack.append(f[key])
    for st in parse_weapon_psprite_states(dg, weapons):
        if st in sidx:
            stack.append(st)

    seen = set()
    while stack:
        name = stack.pop()
        if name in seen or name not in sidx:
            continue
        seen.add(name)
        nxt = states[sidx[name]]['nxt']
        if nxt in sidx:
            stack.append(nxt)
    keep_spr = {states[sidx[s]]['spr'] for s in seen}
    keep_spr -= MONSTER_ONLY_SPRITES

    # Frames of the player's body that this level can actually show.  The player's own mobj is
    # never drawn: it sits exactly at the camera (viewx/viewy are its x/y), so R_ProjectSprite
    # rejects it at tz < MINZ.  The level does show dead marines, though: E1M1 places thing type
    # 15 (MT_MISC62, S_PLAY_DIE7) and type 10 (MT_MISC68, S_PLAY_XDIE9), whose spawn states are
    # the corpse frames.  So keep the frames every non-player thing of that sprite can reach.
    # A frame is a lump name whose fifth character is 'A' + its index.
    body_frames = set()
    frame_stack = []
    for mt in seed_mt:
        if mt == 'MT_PLAYER':
            continue
        f = mobj[mt]
        for key in ('spawnstate', 'seestate', 'painstate', 'meleestate', 'missilestate',
                    'deathstate', 'xdeathstate', 'raisestate'):
            if key in f and f[key] in sidx:
                frame_stack.append(f[key])
    fseen = set()
    while frame_stack:
        sname = frame_stack.pop()
        if sname in fseen or sname not in sidx:
            continue
        fseen.add(sname)
        st = states[sidx[sname]]
        if st['spr'] in PLAYER_BODY_SPRITES:
            body_frames.add(st['frame'])
        if st['nxt'] in sidx:
            frame_stack.append(st['nxt'])
    if not keep_player_sprite:
        keep_spr -= PLAYER_BODY_SPRITES
    # R_InitSpriteDefs requires every frame letter from A up to the highest one present to have
    # at least one lump, so a sprite that only keeps its corpse frame needs the earlier frames
    # filled in.  They can never be drawn (see PLAYER_BODY_SPRITES), so a 1x1 transparent stub
    # each is enough -- 13 bytes, and pixel-identical to not drawing them.
    body_stub_frames = []
    if body_frames and not keep_player_sprite:
        body_stub_frames = [f for f in range(max(body_frames)) if f not in body_frames]

    # ---- assemble the keep list in WAD order, then add the rebuilt lumps
    hud_names = HUD_LUMPS + (STATUSBAR_LUMPS if with_stbar else [])
    hud_idx = []
    for nm in hud_names:
        if nm in ('PLAYPAL', 'COLORMAP', 'ENDOOM'):
            continue
        i = wad.find(nm)
        if i is None:
            raise SystemExit(f'HUD lump missing from WAD: {nm}')
        hud_idx.append(i)

    patch_idx = []
    for i in wad.range_between('P_START', 'P_END'):
        if wad.lumps[i][0].upper() in keep_patch:
            patch_idx.append(i)
    got = {wad.lumps[i][0].upper() for i in patch_idx}
    lost = keep_patch - got
    if lost:
        raise SystemExit(f'patches missing from WAD: {sorted(lost)}')

    spr_idx = [i for i in wad.range_between('S_START', 'S_END')
               if wad.lumps[i][0][:4].upper() in keep_spr
               or (wad.lumps[i][0][:4].upper() in PLAYER_BODY_SPRITES
                   and ord(wad.lumps[i][0][4].upper()) - ord('A') in body_frames)]
    flat_idx = [i for i in wad.range_between('F_START', 'F_END')
                if wad.lumps[i][0].upper() in keep_flat]

    if verbose:
        print(f'  map        {len(map_idx):5d} lumps {sum(wad.lumps[i][2] for i in map_idx):9d} B')
        print(f'  patches    {len(patch_idx):5d} lumps {sum(wad.lumps[i][2] for i in patch_idx):9d} B'
              f' ({len(real_tex)} real textures, {len(stub_tex)} stubbed)')
        print(f'  sprites    {len(spr_idx):5d} lumps {sum(wad.lumps[i][2] for i in spr_idx):9d} B')
        print(f'  flats      {len(flat_idx):5d} lumps {sum(wad.lumps[i][2] for i in flat_idx):9d} B')
        print(f'  hud        {len(hud_idx):5d} lumps {sum(wad.lumps[i][2] for i in hud_idx):9d} B')

    return dict(map_idx=map_idx, hud_idx=hud_idx, patch_idx=patch_idx, spr_idx=spr_idx,
                flat_idx=flat_idx, real_tex=real_tex, stub_tex=stub_tex,
                keep_tex=keep_tex, textures=textures, pnames=pnames, keep_patch=keep_patch,
                keep_flat=keep_flat, keep_spr=keep_spr, body_stub_frames=body_stub_frames)


def _close_cycles(keep, order, anims, istexture):
    """Add every lump of an animation cycle that has at least one member in `keep`.

    P_InitPicAnims sets up a cycle when its start name exists and then walks forward through
    WAD order to the end name, so a partial cycle would animate the wrong pictures.
    """
    keep = set(keep)
    for istex, start, end in anims:
        if istex != istexture:
            continue
        if start not in order or end not in order:
            continue
        a, b = order.index(start), order.index(end)
        lo, hi = min(a, b), max(a, b)
        run = order[lo:hi + 1]
        if any(nm in keep for nm in run):
            keep.update(run)
    return keep


# ---------------------------------------------------------------------------- WAD writing

# The shared stub patch: 1x1, one column, and that column is empty (a single end marker). 13
# bytes, fully transparent wherever it lands, and -- unlike picking a real patch -- identical on
# every run, so the pruned WAD and the ELF built from it are reproducible byte for byte.
STUB_PATCH_SIZE = (1, 1)
STUB_PATCH_DATA = struct.pack('<hhhh', 1, 1, 0, 0) + struct.pack('<i', 12) + b'\xff'


def make_stub_texture(name, patch_width, patch_height):
    """A named texture that resolves at boot, costs one shared patch, and shows no map pixels."""
    hdr = bytearray(22)
    hdr[0:len(name)] = name.encode('ascii')
    struct.pack_into('<i', hdr, 8, 0)                     # masked
    struct.pack_into('<hh', hdr, 12, patch_width, patch_height)
    struct.pack_into('<i', hdr, 16, 0)                    # obsolete
    struct.pack_into('<h', hdr, 20, 1)                    # patchcount
    patch = struct.pack('<hhhhh', 0, 0, 0, 0, 0)          # origin 0,0 -> patch 0
    return bytes(hdr) + patch


def write_wad(path, entries):
    """entries: [(name, bytes)] in directory order.

    Every lump starts on a 4-byte boundary, like the id Software WADs, so the guest can read
    a lump's contents with word/long accesses where the code does and point at the directory
    in place (the RISC-V machine traps misaligned data accesses).
    """
    out = bytearray()
    out += b'IWAD' + struct.pack('<ii', len(entries), 0)
    dirents = []
    for name, data in entries:
        out += b'\0' * (-len(out) % 4)
        pos = len(out)
        out += data
        dirents.append((pos, len(data), name))
    out += b'\0' * (-len(out) % 4)
    diroff = len(out)
    for pos, size, name in dirents:
        out += struct.pack('<ii8s', pos, size, name.encode('ascii')[:8].ljust(8, b'\0'))
    struct.pack_into('<ii', out, 4, len(dirents), diroff)
    Path(path).write_bytes(out)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(ROOT / 'build' / 'doom1.wad'))
    ap.add_argument('--out', default=str(ROOT / 'build' / 'e1m1-pruned.wad'))
    ap.add_argument('--dg', default=str(ROOT / 'doomgeneric' / 'doomgeneric'))
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--with-stbar', action='store_true',
                    help='keep the status-bar graphics too (the pre-fullscreen build)')
    ap.add_argument('--keep-player-sprite', action='store_true',
                    help='keep the PLAY* body sprites (A/B check that they are never drawn)')
    a = ap.parse_args()

    dg = Path(a.dg)
    wad = Wad(a.src)
    keep = build_keep_set(wad, dg, verbose=True, with_stbar=a.with_stbar,
                          keep_player_sprite=a.keep_player_sprite)

    real_tex = keep['real_tex']
    stub_tex = keep['stub_tex']
    textures = {t['name']: t for t in keep['textures']}
    pnames = keep['pnames']

    # ---- new PNAMES: the real patches, plus one shared stub patch for stubbed textures
    new_pnames = sorted(keep['keep_patch'])
    stub_patch_name = 'PSTUB'
    while stub_patch_name in new_pnames:
        stub_patch_name += '_'
    patch_index = {n: i for i, n in enumerate(new_pnames)}
    if stub_tex:
        new_pnames.append(stub_patch_name)
        stub_patch_index = len(new_pnames) - 1

    # The stub's pixels: an empty 1x1 patch, so a stubbed texture that somehow reached the screen
    # would be blank rather than some other wall's art (and so repeated runs agree).
    stub_w, stub_h = STUB_PATCH_SIZE
    pnames_blob = struct.pack('<i', len(new_pnames))
    for n in new_pnames:
        pnames_blob += n.encode('ascii')[:8].ljust(8, b'\0')

    # ---- new TEXTURE1: every kept name, in the original order (animation indices stay valid)
    order = [t['name'] for t in keep['textures'] if t['name'] in keep['keep_tex']]
    blobs = []
    for name in order:
        t = textures[name]
        if name in stub_tex:
            blobs.append(make_stub_texture(name, stub_w, stub_h))
        else:
            body = bytearray(t['header']) + bytearray(10 * len(t['patches']))
            for j, (ox, oy, pidx, sd, cm) in enumerate(t['patches']):
                struct.pack_into('<hhhhh', body, 22 + 10 * j, ox, oy,
                                 patch_index[pnames[pidx].upper()], sd, cm)
            blobs.append(bytes(body))
    head = struct.pack('<i', len(blobs))
    off = 4 + 4 * len(blobs)
    dirs = []
    for b in blobs:
        dirs.append(struct.pack('<i', off))
        off += len(b)
    texture1 = head + b''.join(dirs) + b''.join(blobs)

    # ---- final directory order
    entries = []
    def add_idx(i):
        entries.append((wad.lumps[i][0], wad.data(i)))

    add_idx(wad.need('PLAYPAL'))
    add_idx(wad.need('COLORMAP'))
    if a.with_stbar:
        add_idx(wad.need('ENDOOM'))
    entries.append(('PNAMES', pnames_blob))
    entries.append(('TEXTURE1', texture1))
    # map: E1M1 followed by its ten lumps, addressed by relative index
    for i in keep['map_idx']:
        add_idx(i)
    # HUD graphics, outside the patch/sprite/flat marker ranges
    for i in keep['hud_idx']:
        if wad.lumps[i][0].upper() in ('PLAYPAL', 'COLORMAP', 'ENDOOM'):
            continue
        add_idx(i)
    entries.append(('P_START', b''))
    for i in keep['patch_idx']:
        add_idx(i)
    if stub_tex:
        entries.append((stub_patch_name, STUB_PATCH_DATA))
    entries.append(('P_END', b''))
    entries.append(('S_START', b''))
    for i in keep['spr_idx']:
        add_idx(i)
    for f in keep['body_stub_frames']:
        entries.append((sorted(PLAYER_BODY_SPRITES)[0] + chr(ord('A') + f) + '0',
                        STUB_PATCH_DATA))
    entries.append(('S_END', b''))
    entries.append(('F_START', b''))
    for i in keep['flat_idx']:
        add_idx(i)
    entries.append(('F_END', b''))

    size = write_wad(a.out, entries)
    print(f'wrote {a.out}: {size} bytes ({len(entries)} lumps, was {len(wad.raw)})')
    if a.stats:
        print(f'  real textures {len(real_tex)}, stub textures {len(stub_tex)}, '
              f'patches {len(keep["patch_idx"])}, sprites {len(keep["spr_idx"])}, '
              f'flats {len(keep["flat_idx"])}')


if __name__ == '__main__':
    main()
