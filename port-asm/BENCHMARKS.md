# Benchmarks and before/after evidence

All token counts below are exact CPU-only counts from `lx.ref.Machine`, not extrapolated from a
constant. The scripts are versioned in this directory.

## Current before/after snapshot

Interval: frames 5→20 inclusive boundary deltas (15 rendered-frame intervals).

| guest | resolution/fidelity | instr/frame | exact LX tokens/frame | tokens/instr |
|---|---|---:|---:|---:|
| `build/doom-lf.elf` local DoomGeneric build | 192x120, real Doom renderer/game | 614,807 | 968,034 | 1.575 |
| `build/asm-ray0-full.elf` | 96x60, assembly raycaster with extracted E1M1 geometry/colors | 4,202 | 5,131 | 1.221 |

Reduction in this interval:

- **146.3× fewer instructions/frame**
- **188.7× fewer exact LX tokens/frame**

Twenty output seconds at 10 fps (200 frames) at the measured steady rate:

- DoomGeneric local build: **~193.6M tokens**
- assembly cost probe: **~1.03M tokens**

## Circuit size lever: small memory

The LX circuit allocates one RAM key head per 2048 words. The normal Doom image uses
`RAM_SIZE=0x600000`, producing **55 layers**. The assembly probe also builds as
`asm-ray0-small.elf` with `TEXT_END=0x2000` and `RAM_SIZE=0x8000`; building the circuit IR
for that configuration produces **14 layers** and 32 RAM-related heads instead of 55 layers and
768 heads. Token cost per frame is unchanged because guest semantics are identical, but each
token is much cheaper to execute.

Combined first-order effect before executor-specific optimizations:

- ~188.7× fewer tokens/frame;
- ~3.93× fewer layers/work per token;
- **~742× less transformer work per output frame**.

This is an architectural reduction in circuit size, not a throughput measurement. GPU tok/s
must still be measured after emission.


## Local small-circuit emission smoke test

Command:

```sh
python3 -m venv /tmp/lxenv
/tmp/lxenv/bin/pip install numpy torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
/tmp/lxenv/bin/pip install transformers==5.22 fla-core==0.5.2 flash-linear-attention==0.5.2
/tmp/lxenv/bin/pip install --no-deps torchcodec
/tmp/lxenv/bin/python port-asm/emit_small.py build/asm-ray0-small.elf \
  --data /tmp/asm-emit-data --work /tmp/asm-emit-work
```

Observed locally:

```text
emit config ram_size=0x8000 text_end=0x2000
circuit {'layers': 14, 'dims': 3572, 'heads': 32, 'ffn_units': 36720,
         'max_units': 8102, 'max_heads': 10} hidden 3840 layers 14
streamed 219s, packed nonzeros 0.11 GiB
wrote /tmp/asm-emit-work/packed.pt 3.89 GiB
```

The large file size is dominated by dense model tensors, especially the token embedding/head,
not by the small circuit's sparse layer payload. `port-asm/packedzip.py` compresses the emitted artifact for transport; the local 3.89 GiB `packed.pt` becomes a 33 MiB gzip stream (`sha256=3100ef0748f62f62310932d0ec58dd60b04484d8c1c0461807cba6d5db58b030`).

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
frame 5:  instructions=24,160    tokens=29,781
frame 20: instructions=87,185    tokens=106,744
delta/15: instructions=4,202/frame tokens=5,131/frame
```
