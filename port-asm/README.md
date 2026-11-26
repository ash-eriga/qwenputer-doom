# port-asm: assembly-native cost probe

`ray0.S` is not DoomGeneric. It is a small Doom-like renderer written directly for the LX
machine's cost model, intended to answer: how cheap can a frame become when the guest program
is designed for the transformer CPU instead of for a normal CPU?

Tricks used:

- RV32IM assembly only; no libc, no boot, no Doom startup cost.
- 96x60 framebuffer in column-major order, so vertical spans are contiguous memory.
- 15 aligned `sw` writes per column; generated straight-line column kernels, one per distinct
  pixel pattern, replace span loops.
- columns that share a camera angle share one ray: 43 marches for 96 columns.
- each ray group remembers the pattern it last drew in its hidden stride word and skips the
  kernel when the pattern is unchanged; a frame whose position and angle are unchanged is
  skipped whole. Every displayed pixel stays identical to the per-column renderer
  (`verify/reference_frames.json`).
- no division in the frame loop; projection comes from generated tables/code.
- packed 4.4 fixed-point X/Y ray state: one `add` advances both axes.
- expanded `map256`, indexed directly by packed tile bits.
- baked autopilot by default, because a free-running in-weights clip cannot depend on host
  feedback; `qwen_demo_enabled=0` switches back to `qwen_key_state` for native scripts/live input.
- live input over the UART: each frame the guest checks the line status register (4
  instructions, 5 tokens when nothing is waiting) and takes a waiting byte as its key mask (bit 0
  up, 1 down, 2 left, 3 right); a byte with bit 7 set hands control back to the autopilot. The
  host only ever types bytes (KEY0 tokens); it never pokes guest memory.

Build and measure:

```sh
bash port-asm/build.sh
bash port-asm/bench.sh
```

Outputs:

- `build/asm-ray0-full.elf` — one-tile ray steps, fastest, occasional corner artifacts
- `build/asm-ray0-half.elf` — half-tile ray steps, cleaner walls, higher cost
- `build/asm-ray0-vis.elf` — debug/visual build with a linear copy for `fastemu-lf` dumps
- `build/asm-ray0-small.elf` — same cost build linked into a 32 KiB LX image (`TEXT_END=8 KiB`), reducing the circuit shape from 55 layers to 14 layers

Current local measurement on the baked autopilot (`measure_exact_lx.py`, exact `lx.ref.Machine`
token stream, full-step build, 20 frames, first frame included):

```text
instr/frame=1475.0
tokens/frame=1785.8
tokens/instr=1.211
20s_at_10fps_tokens=357170
```

The fastemu visual build is only for debugging: its linear transpose is intentionally excluded
from the cost build. A real in-weights integration should teach the host readout to unpack the
column-major framebuffer instead of paying for a guest-side transpose.

Caveat: this proves the cost floor and the assembly techniques. It is not yet E1M1/Doom physics,
BSP, sprites, weapons, or WAD-backed geometry.
