#!/bin/bash
# Build the small-RAM, low-token DoomGeneric for the linuxformer LX machine.
#
# Same guest as port-lf/build.sh -- the real engine, real E1M1, real BSP, textures, lighting,
# COLORMAP, monsters and weapons at 192x120 with DETAIL=0 -- but three things differ, and each
# one is aimed at a circuit cost:
#
#   1. build/e1m1-pruned.wad: port-lf/prune_wad.py drops everything E1M1 cannot touch (all
#      sound/music, E1M2..E1M9, intermission/menu art, unused textures/patches/flats/sprites).
#      The WAD sits in the guest image, so those bytes were RAM_SIZE bytes.
#   2. port-lf/r_draw_asm.S replaces R_DrawSpan and R_DrawColumn: the same pixels written with
#      aligned word stores instead of one byte store per pixel (an SB record costs the circuit
#      two data tokens plus the store; a whole 4-pixel SW costs one).
#   3. the renderer's hot files are compiled -O3 while cold files stay -Os so .text still fits
#      the circuit's 256 KB test_end.
#
# Cost levers that are on for every DETAIL=1 build, all of them pixel-exact (compare_frames.py
# reports differing=0 on frames 6, 10, 60 and 150 against a REF=1 build):
#
#   * gp-relative small data.  doom-lf-fast.ld groups .srodata/.sdata/.sbss and defines
#     __global_pointer$, crt0.S loads gp, and the link uses --relax, so every global scalar
#     load/store is one lw/sw rd,off(gp) instead of auipc+addi+load (-16.5k tok/frame at
#     160x100 DETAIL=1).
#   * a second optimization tier (MID below): w_wad/p_maputl/p_mobj/p_tick/v_video/z_zone at
#     -O2, and r_main/tables/r_data promoted to the renderer's -O3 tier.
#   * r_segs.c caches the wall texture's column base per segment instead of calling R_GetColumn
#     per pixel, and tables.h inlines SlopeDiv into R_PointToAngle.
#   * r_plane.c keeps a 16-entry (64 KiB) cache of pre-shaded 64x64 flat tiles keyed on
#     (ds_source, ds_colormap) -- exact for static and animated flats because ds_source is the
#     memory-mapped WAD address -- and R_DrawSpanLowShadedFast draws full-width (viewwidth)
#     spans from it with one LBU per view pixel instead of two.  Table-only spans skip the
#     cache, so there is no 4 KiB fill for a short span; in steady state the cache takes zero
#     fills and R_DrawSpanLow* drops from 119,474 to 104,680 tok/frame (-12.1k tok/frame net,
#     frames 6-10 and 101-110), measured by lx_profile.py.
#
# Output: build/doom-lf-fast.elf
#
# Usage: build_fast.sh
# Knobs (env): RAM_SIZE, STACK, HEAP, ZONE, TEXT_END, OPT_COLD, OPT_MID, OPT_HOT, HOT_FILES,
#              MID_FILES, RESX, RESY, DETAIL, BLOCKS, SHADED,
#              REF=1 (reference build: full WAD, no assembly, -Os everywhere, 6 MB RAM)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
DG=$ROOT/doomgeneric/doomgeneric
PLAT=$ROOT/port

PICO=${PICO:-/usr/riscv64-unknown-elf}
GCC=${GCC:-riscv64-unknown-elf-gcc}
MARCH=rv32im

# Shipped configuration: 160x100, horizontal low detail (DETAIL=1 doubles every view pixel, so
# the renderer touches half as many texels per frame).  At frame 6 that is 6,915 span pixels and
# 1,245 column pixels; see the report for the token breakdown.
RESX=${RESX:-160}
RESY=${RESY:-100}
DETAIL=${DETAIL:-1}
# 11 = Doom's own fullscreen view size: at 192x120 the 320-pixel status bar cannot be blitted
# into the frame buffer (V_CopyRect copies 320-wide rows), so the authentic choice is the view
# size Doom itself uses when the status bar is hidden.
BLOCKS=${BLOCKS:-11}
MAP=${MAP:-1}
SKILL=${SKILL:-3}

REF=${REF:-0}
if [ "$REF" = 1 ]; then
  OUT=$ROOT/build/lf-ref
  BIN=$ROOT/build/doom-lf-ref.elf
  WAD=$ROOT/build/doom1.wad
  RAM_SIZE=${RAM_SIZE:-0x600000}
  ZONE=${ZONE:-0xC0000}
else
  OUT=$ROOT/build/lf-fast
  BIN=$ROOT/build/doom-lf-fast.elf
  WAD=$ROOT/build/e1m1-pruned.wad
  # Sized from the measured zone peak and from what the image plus heap plus stack actually
  # needs; every byte here is a circuit head.  The measured zone peak through frame 151 is
  # 205,428 (0x32274, -DLF_ZONE_STATS), so 0x34000 leaves ~7.5 KB of headroom; the 69 KB this
  # frees over the old 0x45000 pays for the 64 KiB pre-shaded flat cache in r_plane.c.  RAM
  # stays at 0x1B0000: that packs to 20 circuit layers (0x1B8000 is the first size needing 21).
  RAM_SIZE=${RAM_SIZE:-0x1B0000}
  ZONE=${ZONE:-0x34000}
fi
STACK=${STACK:-0x3000}
HEAP=${HEAP:-0x1000}
TEXT_END=${TEXT_END:-0x38000}
OPT_COLD=${OPT_COLD:-s}
OPT_HOT=${OPT_HOT:-3}
mkdir -p "$OUT"

LIBC=$(find "$PICO/lib" -path '*rv32im/ilp32/libc.a' | head -1)
if [ -z "$LIBC" ]; then
  echo "picolibc for rv32im/ilp32 not found under $PICO" >&2
  exit 1
fi
LIBGCC=$($GCC -march=$MARCH -mabi=ilp32 -print-libgcc-file-name)

if [ ! -f "$WAD" ] || [ "${PRUNE:-0}" = 1 ]; then
  echo "=== pruning $ROOT/build/doom1.wad ==="
  /tmp/lxenv/bin/python "$HERE/prune_wad.py" --out "$WAD" --stats ${PRUNE_ARGS:-}
fi

CFLAGS="-march=$MARCH -mabi=ilp32 -DNORMALUNIX -DCMAP256 \
-DDOOMGENERIC_RESX=$RESX -DDOOMGENERIC_RESY=$RESY -DDG_SCREENWIDTH=$RESX -DDG_SCREENHEIGHT=$RESY \
-isystem $PICO/include -I$HERE -I$PLAT -I$DG -I$PLAT/env \
-Wno-unused-parameter -Wno-unused-variable -DDG_SCREEN_IS_VIDEOBUFFER -DDG_NO_SCREEN_MALLOC \
-DDG_DETAILLEVEL=$DETAIL -DDG_SCREENBLOCKS=$BLOCKS -DQWEN_ZONE_BYTES=$ZONE \
-DLF_WARP -DLF_MAP=\"$MAP\" -DLF_SKILL=\"$SKILL\" -DLF_NO_WIPE -DLF_NO_WADSUM -DLF_EVERY=1 \
-ffunction-sections -fdata-sections -fno-common ${EXTRA_CFLAGS:-}"
if [ "$REF" != 1 ]; then
  CFLAGS="$CFLAGS -DR_DRAW_ASM -DLF_FIXED_ASM -DLF_SPAN_PACK=${SPAN_PACK:-1} -DLF_NO_MUSIC"
  CFLAGS="$CFLAGS -DLF_SHADED_FLAT=${SHADED:-1}"
  # DETAIL=1 draws through R_DrawSpanLow/R_DrawColumnLow; their assembly versions are exact
  # (compare_frames.py: differing=0 on frames 6, 10, 60 and 150) and cheaper than the C loops.
  if [ "$DETAIL" = 1 ]; then
    CFLAGS="$CFLAGS -DR_DRAW_ASM_SPANLOW=1 -DR_DRAW_ASM_COLLOW=1"
  fi
fi
if [ "${ASM:-1}" = 0 ]; then
  # Bisect knob: the pure-C fast build (pruned WAD, -O2 hot files) with the C renderer loops.
  CFLAGS=$(echo "$CFLAGS" | sed -e 's/-DR_DRAW_ASM_SPANLOW=1//' -e 's/-DR_DRAW_ASM_COLLOW=1//' \
    -e 's/-DR_DRAW_ASM//' -e 's/-DLF_SPAN_PACK=[01]//')
fi

# At screenblocks 11 the status bar is never drawn (the 320-pixel bar cannot be blitted into a
# narrower frame buffer); st_stuff.c stubs its entry points so --gc-sections drops the widget
# code, and prune_wad.py leaves its ST* graphics out of the WAD.  Both are the same lever: the
# guest image, which the circuit's layer count scales with.
if [ "$BLOCKS" = 11 ] && [ "$REF" != 1 ]; then
  CFLAGS="$CFLAGS -DLF_NO_STBAR"
  PRUNE_ARGS=${PRUNE_ARGS:-}
fi

# Renderer files that dominate a frame's tokens; everything else stays -Os so .text fits the
# ROM region the circuit packs (256 KB).
HOT=${HOT_FILES:-r_draw r_segs r_plane r_bsp r_things m_fixed r_main tables r_data}
# Second tier: files that are not renderer-hot but still pay per-frame tokens (-O2 keeps them
# well under the -O3 size of the renderer files while fitting the 224 KB ROM region).
MID=${MID_FILES:-w_wad p_maputl p_mobj p_tick v_video z_zone}
if [ "$REF" = 1 ]; then HOT=""; MID=""; fi

SRCS="dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop \
d_main d_mode d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus \
i_endoom i_joystick i_scale i_sound i_timer memio m_argv m_bbox m_cheat \
m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy \
p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg \
p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw \
r_main r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff \
s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone \
i_input i_video doomgeneric"

OBJS=""
for s in $SRCS; do
  O=$OPT_COLD
  for h in $HOT; do
    if [ "$h" = "$s" ]; then O=$OPT_HOT; fi
  done
  for m in $MID; do
    if [ "$m" = "$s" ]; then O=${OPT_MID:-2}; fi
  done
  o=$OUT/$s.o
  $GCC $CFLAGS -O$O -c "$DG/$s.c" -o "$o"
  OBJS="$OBJS $o"
done
$GCC $CFLAGS -O$OPT_COLD -c "$PLAT/src/i_system.c" -o "$OUT/i_system.o"
$GCC $CFLAGS -O$OPT_COLD -c "$PLAT/doomgeneric_qwenputer.c" -o "$OUT/doomgeneric_platform.o"
$GCC $CFLAGS -O$OPT_COLD -c "$PLAT/w_file_qwenputer.c" -o "$OUT/w_file_platform.o"
$GCC $CFLAGS -O$OPT_COLD -c "$HERE/stubs.c" -o "$OUT/stubs.o"
$GCC $CFLAGS -O$OPT_COLD -c "$HERE/sys.c" -o "$OUT/sys.o"
$GCC $CFLAGS -c "$HERE/crt0.S" -o "$OUT/crt0.o"

if [ "$REF" != 1 ]; then
  if [ "${ASM:-1}" = 1 ]; then
    $GCC $CFLAGS -O$OPT_HOT -c "$HERE/r_draw_asm.S" -o "$OUT/r_draw_asm.o"
    OBJS="$OBJS $OUT/r_draw_asm.o"
  fi
fi

echo "$WAD" > "$OUT/wad_path.txt"
sed "s|placeholder.wad|$WAD|" "$PLAT/wad_data.S" > "$OUT/wad_data_gen.S"
(cd "$PLAT" && $GCC $CFLAGS -c "$OUT/wad_data_gen.S" -o "$OUT/wad_data.o")
OBJS="$OBJS $OUT/i_system.o $OUT/doomgeneric_platform.o $OUT/w_file_platform.o $OUT/stubs.o \
$OUT/sys.o $OUT/crt0.o $OUT/wad_data.o"

$GCC -march=$MARCH -mabi=ilp32 -nostdlib -nostartfiles -static -T"$HERE/doom-lf-fast.ld" \
  -Wl,--gc-sections -Wl,--relax \
  -Wl,--defsym,RAM_SIZE=$RAM_SIZE -Wl,--defsym,TEXT_END=$TEXT_END \
  -Wl,--defsym,STACK_BYTES=$STACK -Wl,--defsym,HEAP_BYTES=$HEAP \
  -Wl,--defsym,__lf_width=$RESX -Wl,--defsym,__lf_height=$RESY \
  -o "$BIN" $OBJS -Wl,--start-group "$LIBC" "$LIBGCC" -Wl,--end-group

echo "=== $(basename "$BIN") at ${RESX}x${RESY}, RAM_SIZE=$RAM_SIZE, TEXT_END=$TEXT_END ==="
riscv64-unknown-elf-size -A "$BIN" | grep -E '^\.(text|rodata|data|tdata|tbss|bss)' || true
riscv64-unknown-elf-size "$BIN"
NM=$(riscv64-unknown-elf-nm "$BIN")
for s in __heap_start __heap_end __stack_top; do
  printf '%-14s %s\n' "$s" "$(echo "$NM" | awk -v s="$s" '$3==s {print $1}')"
done
IMAGE_END=$(echo "$NM" | awk '$3=="__stack_top" {print $1}')
echo "image: $(printf '%d' $((0x$IMAGE_END - 0x80000000 + 16))) bytes of the ${RAM_SIZE} RAM_SIZE"
