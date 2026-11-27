#!/bin/sh
# Build and measure the assembly probe. The exact mode uses lx.ref.Machine from a sibling
# Doom-On-Qwen checkout (../Doom-On-Qwen/lxsrc); the approximate mode is self-contained.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
bash "$HERE/build.sh"
cd "$ROOT"
echo '== approximate, no autopilot script, 40 frames =='
python3 "$HERE/measure_ray.py" build/asm-ray0-full.elf 40 | sed -n '1,2p'
python3 "$HERE/measure_ray.py" build/asm-ray0-half.elf 40 | sed -n '1,2p'
if [ -d "$ROOT/../Doom-On-Qwen/lxsrc/lx" ]; then
  echo '== exact lx.ref, baked autopilot, 20 frames =='
  python3 "$HERE/measure_exact_lx.py" build/asm-ray0-full.elf 20 | sed -n '1,3p'
else
  echo 'exact lx.ref mode skipped: sibling Doom-On-Qwen/lxsrc not present'
fi
