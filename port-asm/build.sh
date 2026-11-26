#!/bin/sh
# Build the assembly raycaster cost probes:
#   build/asm-ray0-full.elf      1-tile ray steps: fastest, occasional corner artifacts
#   build/asm-ray0-half.elf      1/2-tile ray steps: cleaner walls, about 1.4x the cost
#   build/asm-ray0-vis.elf       full-step visual/debug build with a linear framebuffer
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
python3 "$HERE/gen_tables.py" --step-tiles 1.0 --ray-shift 6 --suffix full >/dev/null
python3 "$HERE/gen_tables.py" --step-tiles 0.5 --ray-shift 7 --suffix half >/dev/null
GCC=${GCC:-riscv64-unknown-elf-gcc}
CFLAGS="-march=rv32im -mabi=ilp32 -Os -nostdlib -nostartfiles -static -T$HERE/../port-lf/doom-lf.ld"
mkdir -p "$ROOT/build"
$GCC $CFLAGS -DVISUAL=0 -DRAY_SHIFT=6 -o "$ROOT/build/asm-ray0-full.elf" \
    "$HERE/ray0.S" "$HERE/tables-full.S" "$HERE/kernels-full.S"
$GCC $CFLAGS -DVISUAL=0 -DRAY_SHIFT=7 -o "$ROOT/build/asm-ray0-half.elf" \
    "$HERE/ray0.S" "$HERE/tables-half.S" "$HERE/kernels-half.S"
$GCC $CFLAGS -DVISUAL=1 -DRAY_SHIFT=6 -o "$ROOT/build/asm-ray0-vis.elf" \
    "$HERE/ray0.S" "$HERE/tables-full.S" "$HERE/kernels-full.S"
$GCC -march=rv32im -mabi=ilp32 -Os -nostdlib -nostartfiles -static \
    -T"$HERE/ray-small.ld" -DVISUAL=0 -DRAY_SHIFT=6 -o "$ROOT/build/asm-ray0-small.elf" \
    "$HERE/ray0.S" "$HERE/tables-full.S" "$HERE/kernels-full.S"
cp "$ROOT/build/asm-ray0-full.elf" "$ROOT/build/asm-ray0.elf"
riscv64-unknown-elf-size -A "$ROOT/build/asm-ray0-full.elf" | grep -E '^\.(text|rodata|data|bss)' || true
echo "built asm-ray0-full.elf, asm-ray0-half.elf, asm-ray0-vis.elf and asm-ray0-small.elf"
