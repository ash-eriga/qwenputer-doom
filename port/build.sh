#!/bin/bash
# Build Doom (doomgeneric) for the Qwenputer RV32 machine.
# Usage: build.sh [stock|big]   (linker map; default big)
# Output: build/doom-<map>.elf + a size report.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
DG=$ROOT/doomgeneric/doomgeneric
RVENV=${RVENV:-$HERE/env}
MAP=${1:-big}
OUT=$ROOT/build/$MAP
mkdir -p "$OUT"

PICO=${PICO:-/usr/lib/picolibc/riscv64-unknown-elf}
GCC=${GCC:-riscv64-unknown-elf-gcc}
MARCH=rv32im_zba
LIBC=$(find "$PICO/lib" -path '*rv32im/ilp32/libc.a' | head -1)
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
echo "libc: $LIBC"

if [ "$MAP" = stock ]; then LD=$RVENV/rom.ld; else LD=$HERE/qwenputer-big.ld; fi

CFLAGS="-march=$MARCH -mabi=ilp32 -Os -DNORMALUNIX -DCMAP256 \
-DDOOMGENERIC_RESX=${RESX:-320} -DDOOMGENERIC_RESY=${RESY:-200} -DDG_SCREENWIDTH=${RESX:-320} -DDG_SCREENHEIGHT=${RESY:-200} -DDG_DETAILLEVEL=${DETAIL:-0} -DDG_SCREENBLOCKS=${BLOCKS:-10} -DQWEN_ZONE_BYTES=${ZONE:-524288} \
-isystem $PICO/include -I$HERE -I$DG -I$RVENV -Wno-unused-parameter -Wno-unused-variable -DDG_SCREEN_IS_VIDEOBUFFER"

# doomgeneric sources (from its Makefile.soso list), minus the soso platform,
# minus w_file_stdc (replaced by the DROM WAD class), minus i_system
# (replaced by port/src/i_system.c with a static zone).
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
  o=$OUT/$s.o
  $GCC $CFLAGS -c "$DG/$s.c" -o "$o"
  OBJS="$OBJS $o"
done
$GCC $CFLAGS -c "$HERE/src/i_system.c" -o "$OUT/i_system.o"
$GCC $CFLAGS -c "$HERE/doomgeneric_qwenputer.c" -o "$OUT/doomgeneric_qwenputer.o"
$GCC $CFLAGS -c "$HERE/w_file_qwenputer.c" -o "$OUT/w_file_qwenputer.o"
$GCC $CFLAGS -c "$HERE/stubs.c" -o "$OUT/stubs.o"
$GCC $CFLAGS -c "$RVENV/sys.c" -o "$OUT/sys.o"
$GCC $CFLAGS -c "$RVENV/crt0.S" -o "$OUT/crt0.o"
# WAD payload: the real shareware doom1.wad when present, else a generated
# 12-byte placeholder header (link testing only).
if [ -f "$ROOT/build/doom1.wad" ]; then
  WAD=../build/doom1.wad
else
  printf 'IWAD\000\000\000\000\000\000\000\000' > "$OUT/placeholder.wad"
  WAD="$OUT/placeholder.wad"
fi
sed "s|placeholder.wad|$WAD|" "$HERE/wad_data.S" > "$OUT/wad_data_gen.S"
(cd "$HERE" && $GCC $CFLAGS -c "$OUT/wad_data_gen.S" -o "$OUT/wad_data.o")
OBJS="$OBJS $OUT/i_system.o $OUT/doomgeneric_qwenputer.o $OUT/w_file_qwenputer.o $OUT/stubs.o $OUT/sys.o $OUT/crt0.o $OUT/wad_data.o"

$GCC -march=$MARCH -mabi=ilp32 -nostdlib -nostartfiles -static -T"$LD" \
  -o "$ROOT/build/doom-$MAP.elf" $OBJS \
  -Wl,--start-group "$LIBC" "$LIBGCC" -Wl,--end-group
echo "=== doom-$MAP.elf section sizes ==="
riscv64-unknown-elf-size -A "$ROOT/build/doom-$MAP.elf" | grep -E '^\.(text|rodata|data|bss|tbss|tdata)' || true
riscv64-unknown-elf-size "$ROOT/build/doom-$MAP.elf"
