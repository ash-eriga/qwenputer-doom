# Benchmarks and before/after evidence

All token counts below are exact CPU-only counts from `lx.ref.Machine`, not extrapolated from a
constant. The scripts are versioned in this directory.

## Current before/after snapshot

Interval: frames 5→20 inclusive boundary deltas (15 rendered-frame intervals).

| guest | resolution/fidelity | instr/frame | exact LX tokens/frame | tokens/instr |
|---|---|---:|---:|---:|
| `build/doom-lf.elf` local DoomGeneric build | 192x120, real Doom renderer/game | 614,807 | 968,034 | 1.575 |
| `build/asm-ray0-full.elf` | 96x60, assembly raycaster cost probe | 10,128 | 11,837 | 1.169 |

Reduction in this interval:

- **60.7× fewer instructions/frame**
- **81.8× fewer exact LX tokens/frame**

Twenty output seconds at 10 fps (200 frames) at the measured steady rate:

- DoomGeneric local build: **~193.6M tokens**
- assembly cost probe: **~2.37M tokens**

## Important comparability caveat

This is **not yet an apples-to-apples Doom replacement**. The baseline is real DoomGeneric at
192x120; the probe is a much smaller hardcoded raycaster at 96x60 with baked autopilot. The
comparison is still useful because both counts use the same executable LX cost semantics, but it
proves a cost floor and an implementation direction — not final feature parity.

## Reproduction

Baseline DoomGeneric exact measurement (slow; about 70 seconds for 20 frames on the current
workspace CPU):

```sh
python3 port-asm/measure_lx_frames.py \
  build/doom-lf.elf \
  --frames 20 \
  --script ../Doom-On-Qwen/doom-lx/oracle/sched.txt \
  --out /tmp/doom-lx-baseline-20.csv
```

Assembly probe exact measurement:

```sh
python3 port-asm/measure_lx_frames.py \
  build/asm-ray0-full.elf \
  --frames 20 \
  --out /tmp/asm-lx20.csv
```

Fast approximate measurement used in CI:

```sh
python3 port-asm/measure_ray.py build/asm-ray0-full.elf 20
python3 port-asm/measure_ray.py build/asm-ray0-half.elf 20
```

## Raw local measurements

Baseline cumulative CSV frame 5→20:

```text
frame 5:  instructions=7,569,921  tokens=11,106,119
frame 20: instructions=16,792,029 tokens=25,626,626
delta/15: instructions=614,807/frame tokens=968,034/frame
```

Assembly cumulative CSV frame 5→20:

```text
frame 5:  instructions=48,335    tokens=56,394
frame 20: instructions=200,256   tokens=233,950
delta/15: instructions=10,128/frame tokens=11,837/frame
```
