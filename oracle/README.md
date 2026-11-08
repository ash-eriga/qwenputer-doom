# Oracle of record

The ground truth the in-weights run is verified against: `fastemu-lf` (native runner, same
semantics as `lx.ref.Machine`) playing this build with the E1M5 key schedule.

- `play.script` — the per-frame key masks (decimal; bit0 up, 1 down, 2 left, 3 right, 4 fire,
  5 use, 6 enter, 7 escape), one `frame mask` pair per line (`;` is a comment in the loader).
- `frames.md5` — md5 of every rendered frame 1..160, produced from `doom-lf-s3.elf`
  (FRAMESKIP=3, OPT=2 MAP=5 DETAIL=1 BLOCKS=3, .text 247,380 <= TEXT_END 0x40000).
- `frame-*.png` — a human-viewable sample (every 10th frame; the PPMs are the bytes that count).

Regenerate (the ELFs were built with Debian gcc-riscv64
14.2 + picolibc 1.8.9):

    cc -O3 -o build/fastemu-lf native/fastemu-lf.c -lm
    ./build/fastemu-lf build/doom-lf-s3.elf --wad build/doom1.wad \
      --fbp-sym DG_ScreenBuffer --fc-sym qwen_frame_count \
      --keys-sym qwen_key_state --demo-sym qwen_demo_enabled \
      --frames 161 --script oracle/play.script \
      --dump-every 1 --dump-from 1 --dump-to 160 \
      --dump-dir build/oracle-frames --guest-out none
    md5sum build/oracle-frames/frame-*.ppm   # must match frames.md5

The repo's older `frames/frame-0060.png` came from a build tree that no longer exists, so it is
superseded: an in-weights frame is genuine iff it matches THIS oracle byte for byte.
