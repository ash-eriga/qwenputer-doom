# Doom on Qwenputer

Doom (doomgeneric) ported to the **Qwenputer** — the RV32IM+Zba CPU compiled
into the weights of a Qwen3.5-4B model
([divagr1925/Qwen3.5-4B-Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer)).

The target is the CPU *inside the model*: every instruction executed by the
transformer circuit, no host CPU emulating the guest, no bootstrapped
hardware. This repo holds the game port, the machine sizing, and the
verification harness.

## Status

**In-weights run started (2026-11-05, overnight):** the Doom circuit emitted locally
(packed.pt, 55 layers, head_cols 227 — EXACT vs the repo emitter), and the LX machine's
trace verified on GPU: ~4.4M tokens predicted across the night's runs with **zero wrong
predictions**, including 2,064,385 in the final budget-capped run (checkpoint saved).
`kaggle/` holds the runner (doomrun.py, fast16.py — fp32-precast GEMMs, two-GPU split,
resume), `oracle/` the verification gate. T4-measured ceiling: ~1,030 tok/s (fla-bound);
split across 2x T4 measured slower (866) — single GPU is the config. The boot (7.35M
tokens) is 28% paid via `session.pt`; frames come at ~6.8 min each on a T4 after it
finishes. Full ledger: the qwenputer repo's `doom-lx/ops/work/` + the overnight
`DECISIONS.md`/`REPORT.md`.

**Proven (2026-10-31):** the port boots and plays on the project's reference
machine (`ref.py`) — the executable spec the in-weights circuit is verified
against instruction-for-instruction. 60M instructions, zero faults: title
screen, menu navigation via scripted input, E1M1 loaded and rendering.

![Title screen rendered by the guest CPU](docs/title.svg)
![E1M1 rendered by the guest CPU](docs/e1m1.svg)

**Also proven:** a native fast runner (`native/fastemu.c`) that executes the
same reference semantics without building the circuit's token trace —
852M instructions/s on an M-series Mac (~650 fps in gameplay, vs 0.37 fps
for the Python reference). Verified state-exact against `ref.py`: at frames
10 and 30 the full 2 MB RAM, all registers and the PC hash identically, and
rendered frames match byte-for-byte. It accelerates testing only; the
in-weights circuit remains the real target.

**Next:** the in-weights circuit run on Executor2 (Triton). The game fits
the circuit's constraints with room to spare on code; the memory map has to
grow (see below). Route decided: **recompile the chip layers with the grown
map**. The alternative — host-side paging against the stock circuit — is
mechanically impossible: a store commits as a single token with no
address/value visible to the host, and loads resolve entirely inside the
circuit, so the host can neither snoop stores nor serve loads. Full
analysis in `PAGING-DESIGN-NOTES.md`.

## The machine Doom needs (measured, not guessed)

| Region | Stock circuit | This port | Notes |
|---|---|---|---|
| ROM (code) | 256 KB @ 0x40000 | **228 KB** @ 0x800000 | fits the circuit's hard 256 KB cap — the trace format's anchors are 16-bit |
| RAM | 32 KB | **2 MB** @ 0x0 | 320×200 framebuffer (64 KB) + 512 KB zone cache + engine statics |
| DROM (read-only data) | 128 KB | **6 MB** @ 0x200000 | engine tables (105 KB) + the WAD |

On the stock map the link overflows RAM by 860 KB — the framebuffer alone is
twice the stock RAM. The growth is in the *map*, not the CPU: RV32IM+Zba,
`ecall` putchar/exit, and the CSR stubs are all stock.

## Layout

- `port/` — the port sources:
  - `doomgeneric_qwenputer.c` — platform layer: static framebuffer, a frame
    clock (one rendered frame = one 35 Hz tic), scripted input keyed by
    frame number
  - `w_file_qwenputer.c` — WAD file class reading lumps zero-copy from DROM
  - `stubs.c` — filesystem stubs (opens fake-succeed: the WAD lives in DROM,
    config/savegames are skipped by design)
  - `src/i_system.c` — overlay with a static 512 KB zone (no heap growth)
  - `qwenputer-big.ld` — the grown memory map above
  - `env/` — crt0/syscalls/linker scripts vendored from the Qwenputer fork's
    rv env, so this repo builds standalone
  - `build.sh` — `build.sh [big|stock]` (stock fails the link, on purpose:
    it's the size proof)
  - `run_ref.py` — reference-machine shakeout: runs `doom-big.elf`, dumps
    rendered frames as PPM from guest RAM
- `native/` — `fastemu.c`, the native fast runner (see Status): same
  semantics as the reference machine, no trace; frame-exact stop/dump on
  the store that commits each frame, optional `--pace` for watchable
  live play
- `live/` — the watchable, playable live view: `index.html` (frame view
  + on-screen/keyboard controls), `control_server.py` (serves the page
  and writes the runner's control file), `run_live.py` (reference
  machine + PNG frames). The demo — intro script, then a looping
  patrol — drives until a viewer takes control; the page's toggle
  flips the guest's demo flag and pokes the key-state cell through
  fastemu's `--control` channel, so a human can play, then hand it
  back to the patrol
- `PAGING-DESIGN-NOTES.md` — why host paging can't work, and the concrete
  change list for recompiling the chip layers with the grown map
- `docs/` — frames captured from the reference-machine run
- `doomgeneric/` — upstream doomgeneric (fetched by `get-doomgeneric.sh`,
  GPL-2.0, © id Software / Chocolate Doom contributors)

## Build & shakeout

```sh
sudo apt install gcc-riscv64-unknown-elf picolibc-riscv64-unknown-elf
./get-doomgeneric.sh
# Supply your own shareware DOOM1.WAD v1.9 (4,196,020 bytes,
# MD5 f0cefca49926d00903cf57551d901abe) as build/doom1.wad —
# e.g. Debian's doom-wad-shareware package ships exactly that file.
./port/build.sh big
python3 port/run_ref.py 130 60000000   # needs the qwenputer fork's rv package on the path
```

Fast runner (same semantics, no trace):

```sh
cc -O3 -o fastemu native/fastemu.c -lz
./fastemu build/doom-big.elf --frames 120 --wad build/doom1.wad \
    --dump 10,100,120 --dump-dir frames   # frame dumps verify against run_ref.py
./fastemu build/doom-big.elf --live latest.png --wad build/doom1.wad --pace 60
```

Playable live view (demo until you take the controls):

```sh
# symbol addresses below are this build's (nm build/doom-big.elf)
cd live && python3 control_server.py &   # page + controls on :8950
./fastemu build/doom-big.elf --live live/latest.png --wad build/doom1.wad \
    --pace 60 --control live/control.bin \
    --fb 0xCA3AC --fc 0xD9DD0 --keys 0xD9DCC --demo 0xEB84
```

## Credits

- Qwenputer: divagr1925 — the CPU-in-weights design this builds on
- doomgeneric: ozkl; Doom © id Software
- Port: Duckets + Juno 🦆
