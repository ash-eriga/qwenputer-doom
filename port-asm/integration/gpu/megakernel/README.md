# The ray0 megakernel (L4, H100/H200)

ray0's 14-layer circuit decoded one token at a time as one persistent CUDA kernel: every token's
chip layers and the output head run in one launch, and each argmax is fed back as the next input
on the GPU. It is linuxformer's megakernel (`lx/csrc/mega.cu`, `lx/lx/mega.py`, MIT), ported to
this circuit (`csrc/`, `mega.py` and `mk_build.py` keep linuxformer's license,
`LICENSE-linuxformer`):

- the hidden size is a build flag (`-DMK_HIDDEN=3840` here, 2560 in linuxformer) and the
  RMSNorm sum of squares is templated on it, in the same order as torch's reduction;
- the output head has 460,032 rows: when they do not fit one pass per CTA, each CTA takes
  several partitions (`head_parts`), keeping the first-max tie rule;
- a CTA's data that does not fit shared memory stays in global memory (`res_cut`), so the
  kernel runs on the L4's 100 KB of shared memory, not only on the H100's 227 KB;
- the build targets the device (`sm_89` on L4/L40S, `sm_90a` on H100/H200) with CUDA 12 or 13;
- the decode plan comes from `packed.pt` alone (`packed_model.build_decode_plan`): no dense
  weights are needed;
- the GDN layers that seldom change the residual stream (6 and 8, from
  `gpu/cpu_decode.py`'s activity counts) run alongside their MLP and are redone when they did.

Live input is a byte over ray0's UART (a KEY0 token at a record boundary): the key mask, or 0x80
to hand control back to the autopilot. The host never pokes memory, and every other token is the
model's own prediction.

## Files

| file | what |
|---|---|
| `csrc/mega.cu`, `csrc/mk_math.cuh` | the kernel and its exact arithmetic |
| `csrc/mk_test.cu` | the arithmetic alone, for the bit-for-bit checks |
| `mk_build.py` | builds them with nvcc for the device (cached in `build/mk`) |
| `mega.py` | packing and launch (`Mega(fast, model, cache, 0).run(forced, n)`) |
| `plan.py` | sizes a GPU's layout on the CPU (no GPU needed) |
| `check_pack.py` | CPU check: the packed head decodes back to `w_head` exactly |
| `../cpu_decode.py` | the same decode plan in torch on the CPU, checked against lx.ref |
| `../mega_play.py` | `MegaGame`: typing, frame readout, PNGs, lx.ref beside the model |
| `../mega_check.py` | the GPU validation, step by step |
| `../mega_live.py`, `../mega_live.html` | live play in a browser |

## Layout per GPU (from `plan.py`)

The packed data is about 4.7-5.0 MB (the head's 13.8M sparse entries are read from global memory
on every GPU; this is each CTA's resident data). Shared memory per CTA includes the 55,168-byte
base.

| GPU | CTAs | shared memory / CTA (cap) | head passes a CTA | CTA data left in global memory |
|---|---|---|---|---|
| L4 | 58 | 101,344 (101,376) | 3 (2,644 rows each) | 2.15 MB of 4.73 MB, all 58 CTAs |
| L40S | 142 | 101,376 (101,376) | 1 | 54 KB, 3 CTAs |
| A100 | 108 | 129,568 (166,912) | 2 | none |
| H100 / H200 | 132 | 121,776 (232,448) | 1 | none |

`check_pack.py` decodes the packed head back for each of these and finds it equal to
`w_head` element for element; every CTA's shared-memory cut falls on a blob boundary.

## What was checked without a GPU

- `cpu_decode.py` (the megakernel's decode plan, in torch on the CPU): 9,000 tokens free-running
  from reset with three bytes typed over the UART (0x09 after frame 1, 0x01 after frame 2, 0x00
  after frame 3), 0 tokens different from lx.ref, 36 frames.
- `check_pack.py`: OK for L4, L40S, A100, H100.
- `nvcc` builds of `mega.cu` and `mk_test.cu` for `sm_89` and `sm_90a` with CUDA 13 (128
  registers, no spills on `sm_89`). CUDA 12 was not compiled here: this machine's glibc 2.42
  headers do not build with CUDA 12.8/12.9 (a glibc/CUDA mismatch that standard L4 images with
  Ubuntu 22.04/24.04 do not have).
- `mega_check.py --device cpu`: the host side on the CPU decoder. 7,001 tokens with two bytes
  typed, and 6,200 tokens with one, every token and every kernel input checked, PASSED. Frames 1
  and 2, read out of the model's state at a record boundary, are byte for byte lx.ref's
  framebuffer.
- `mega_live.py`'s server and page with a stand-in game in a browser: the frame, the status,
  the token check, and a key tapped within one launch still typed (0, 1, 0, then 0x80 for the
  autopilot).
- Nothing has run on a GPU yet: the kernel's speed and its exactness on the device are what the
  L4 run measures.

## On the L4

The stack is the one the Doom-On-Qwen runs pin: torch 2.11, transformers 5.22,
flash-linear-attention 0.5.2, Python 3.11 or newer. nvcc must have torch's CUDA major version.

```sh
nvidia-smi --query-gpu=name,driver_version --format=csv   # driver >= 580: CUDA 13 works
# CUDA 13 (driver >= 580): nvcc comes from a pip wheel, nothing else to install
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install nvidia-cuda-nvcc                       # nvidia/cu13/bin/nvcc, found automatically
# CUDA 12 (older driver): torch for CUDA 12 and a system CUDA 12 toolkit
#   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
#   export LX_CUDA_HOME=/usr/local/cuda            # CUDA 12's pip wheel has no nvcc
pip install transformers==5.22 fla-core==0.5.2 flash-linear-attention==0.5.2 accelerate safetensors
pip install --no-deps torchcodec                   # transformers 5.22 reads its metadata
```

`packed.pt` is the 14-layer build (4.2 GB; `run_ray0.py` can decompress `packed.pt.gz`). The
ELF is `build/asm-ray0-small.elf` (`--elf` to point elsewhere): copy it over, or build it there
with `bash port-asm/build.sh` (riscv64-unknown-elf-gcc; with `build/doom1.wad` present the
tables come from E1M1, without it from a synthetic map, which changes the picture and the
`verify/` baselines but not the checks below: lx.ref runs the same ELF). `packed.pt` does not
depend on the ELF's contents, only on its 32 KiB layout. Then,
in order (each step prints one JSON line and stops the run at the first failure):

```sh
cd qwenputer-doom
P=/path/packed.pt
# 1. env, math (bit for bit against torch and fla), build, megakernel vs lx.fast's decode vs
#    lx.ref (2,000 tokens), then 60,000 tokens of play with typed keys, every token checked
python port-asm/integration/gpu/mega_check.py --packed $P --png-dir /tmp/mega-frames \
    --out /tmp/mega-check.json
# 2. a long free run at full speed (launches of 64k tokens)
python port-asm/integration/gpu/mega_check.py --packed $P --skip-math --stock 0 \
    --tokens 2000000 --every 65536 --keys 3:128 --out /tmp/mega-long.json
# 3. the pairing off, to compare speed (same tokens)
python port-asm/integration/gpu/mega_check.py --packed $P --skip-math --stock 0 --no-spec \
    --tokens 200000 --every 65536 --out /tmp/mega-nospec.json
```

What to look at: `math` all zero, `build.smem` and `build.res_global` (how much stays in global
memory), `stock.first_difference_vs_ref` null, `play.verified` true with
`inputs_checked == tokens`, and `play.tok_per_s_kernel`. A frame costs 491 tokens on average
under the autopilot and 1,167 under random keys (the slowest about 2,900, the first 3,285), so
10 fps needs about 5k-12k tok/s on average and about 30k tok/s for the slowest frames.

If something fails: `cpu_decode.py` runs the same plan on the CPU (no GPU), `check_pack.py`
checks the packing, and `mega_check.py --ncta N` changes the CTA count.

## Playing live

```sh
python port-asm/integration/gpu/mega_live.py --packed $P --port 8950 --token "$(openssl rand -hex 8)"
cloudflared tunnel --url http://localhost:8950     # open the printed URL with ?token=...
```

The page shows the latest frame read out of the model's state, the frame count, tok/s, fps and
whether every token so far matches lx.ref. Arrows or WASD take control (the guest reads one typed
byte a frame); "Autopilot" types 0x80 and hands control back. `--chunk` sets the tokens a launch:
smaller answers keys sooner, larger runs faster. The server binds 127.0.0.1; for a public URL put
Cloudflare Access in front of the tunnel instead of relying on the token alone.

`--device cpu` runs the same server with the CPU decoder (about 10 tok/s), to try the page
without a GPU.
