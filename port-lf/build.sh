#!/bin/bash
# Build Doom (doomgeneric) for the linuxformer LX machine: RV32IMA without Zba, one contiguous
# image at 0x80000000, code and read-only data below text_end, everything writable above it.
#
# Usage: build.sh [resx resy]     default 160 100
# Output: build/doom-lf.elf
#
# The vendored doomgeneric/ tree is gitignored (get-doomgeneric.sh fetches it), so the five
# instruction-count fixes this machine needs are kept in doomgeneric-lf.patch next to this
# script. Apply it once after fetching the tree:
#
#     (cd ../doomgeneric && git apply ../port-lf/doomgeneric-lf.patch)
#
# Without it the game still builds and runs; it just costs ~19.6M instructions of startup
# instead of ~4.6M. See work/STATE.md sections 2 and 3.
#
# Knobs: OPT (default s; 2 is cheaper per frame for the same image), DETAIL, BLOCKS, ZONE, MAP,
# SKILL. TEXT_END in doom-lf.ld must stay 0x40000 to match the packed circuit.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
DG=$ROOT/doomgeneric/doomgeneric
# The platform layer is shared with the Qwenputer port: only the runtime differs.
PLAT=$ROOT/port
OUT=$ROOT/build/lf
mkdir -p "$OUT"

PICO=${PICO:-/usr/lib/picolibc/riscv64-unknown-elf}
GCC=${GCC:-riscv64-unknown-elf-gcc}
MARCH=rv32im
RESX=${1:-320}
RESY=${2:-200}
# The two levers that cut instructions per frame: the detail level (1 = no column textures) and
# the screen blocks (a smaller viewport draws fewer columns). Measured at 320x200 on the
# reference machine: detail 0 -> 713k instr/frame, and blocks 3 -> 135k.
DETAIL=${DETAIL:-0}
BLOCKS=${BLOCKS:-10}
ZONE=${ZONE:-0xC0000}
MAP=${MAP:-1}
SKILL=${SKILL:-3}
# Render every FRAMESKIP tics (default 1 = every tic). 3 makes a 10 fps clip show
# real-time play: the game still simulates all 35 tics/s, and a rendered frame
# covers 3 of them. See the LF_EVERY gate in D_Display.
FRAMESKIP=${FRAMESKIP:-1}
LIBC=$(find "$PICO/lib" -path '*rv32im/ilp32/libc.a' | head -1)
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
echo "libc: $LIBC"

CFLAGS="-march=$MARCH -mabi=ilp32 -O${OPT:-s} -DNORMALUNIX -DCMAP256 \
-DDOOMGENERIC_RESX=$RESX -DDOOMGENERIC_RESY=$RESY -DDG_SCREENWIDTH=$RESX -DDG_SCREENHEIGHT=$RESY \
-isystem $PICO/include -I$HERE -I$PLAT -I$DG -I$PLAT/env \
-Wno-unused-parameter -Wno-unused-variable -DDG_SCREEN_IS_VIDEOBUFFER \
-DDG_DETAILLEVEL=$DETAIL -DDG_SCREENBLOCKS=$BLOCKS -DQWEN_ZONE_BYTES=$ZONE \
-DLF_WARP -DLF_MAP=\"$MAP\" -DLF_SKILL=\"$SKILL\" -DLF_NO_WIPE -DLF_NO_WADSUM -DLF_EVERY=$FRAMESKIP \
${EXTRA_CFLAGS:-} \
-ffunction-sections -fdata-sections"

# doomgeneric sources (from its Makefile.soso list), minus the soso platform, minus w_file_stdc
# (replaced by the DROM WAD class), minus i_system (replaced by the static-zone overlay).
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
$GCC $CFLAGS -c "$PLAT/src/i_system.c" -o "$OUT/i_system.o"
$GCC $CFLAGS -c "$PLAT/doomgeneric_qwenputer.c" -o "$OUT/doomgeneric_platform.o"
$GCC $CFLAGS -c "$PLAT/w_file_qwenputer.c" -o "$OUT/w_file_platform.o"
$GCC $CFLAGS -c "$HERE/stubs.c" -o "$OUT/stubs.o"
$GCC $CFLAGS -c "$HERE/sys.c" -o "$OUT/sys.o"
$GCC $CFLAGS -c "$HERE/crt0.S" -o "$OUT/crt0.o"

if [ -f "$ROOT/build/doom1.wad" ]; then
  WAD=../build/doom1.wad
else
  printf 'IWAD\000\000\000\000\000\000\000\000' > "$OUT/placeholder.wad"
  WAD="$OUT/placeholder.wad"
fi
sed "s|placeholder.wad|$WAD|" "$PLAT/wad_data.S" > "$OUT/wad_data_gen.S"
(cd "$PLAT" && $GCC $CFLAGS -c "$OUT/wad_data_gen.S" -o "$OUT/wad_data.o")
OBJS="$OBJS $OUT/i_system.o $OUT/doomgeneric_platform.o $OUT/w_file_platform.o $OUT/stubs.o \
$OUT/sys.o $OUT/crt0.o $OUT/wad_data.o"

$GCC -march=$MARCH -mabi=ilp32 -nostdlib -nostartfiles -static -T"$HERE/doom-lf.ld" \
  -Wl,--gc-sections -o "$ROOT/build/doom-lf.elf" $OBJS \
  -Wl,--start-group "$LIBC" "$LIBGCC" -Wl,--end-group
echo "=== doom-lf.elf at ${RESX}x${RESY} ==="
riscv64-unknown-elf-size -A "$ROOT/build/doom-lf.elf" | grep -E '^\.(text|rodata|data|bss|tbss|tdata)' || true
riscv64-unknown-elf-size "$ROOT/build/doom-lf.elf"
