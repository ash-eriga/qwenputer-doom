#!/usr/bin/env python3
"""Size the megakernel's layout for a GPU without one: pack the ray0 circuit (packed.pt) for a
GPU profile's SM count and shared memory, on the CPU, and report what fits in shared memory and
what stays in global memory.

    python port-asm/integration/gpu/megakernel/plan.py --packed /tmp/asm-emit-work-v2/packed.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU = HERE.parent
for p in (HERE, GPU):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mega  # noqa: E402
from packed_model import build_decode_plan, load_packed_runtime  # noqa: E402

# name: (SMs, cudaDevAttrMaxSharedMemoryPerBlockOptin)
PROFILES = {"L4": (58, mega.L4_SMEM), "L40S": (142, mega.L4_SMEM), "A100": (108, 166912),
            "H100": (132, mega.H100_SMEM), "H200": (132, mega.H100_SMEM)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elf", default=str(HERE.parents[3] / "build" / "asm-ray0-small.elf"))
    ap.add_argument("--packed", required=True)
    ap.add_argument("--lxsrc", default=str(HERE.parents[4] / "Doom-On-Qwen" / "lxsrc"))
    ap.add_argument("--gpus", default="L4,H100")
    ap.add_argument("--pair-layers", default=",".join(map(str, mega.PAIR_LAYERS)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch

    torch.set_grad_enabled(False)
    rt = load_packed_runtime(a.elf, a.packed, a.lxsrc, device="cpu", compile_fast=False,
                             quiet=True)
    rt.fast.dec = build_decode_plan(rt.fast, rt.model)
    pairs = tuple(int(x) for x in a.pair_layers.split(",") if x)
    out = {}
    for name in a.gpus.split(","):
        ncta, cap = PROFILES[name]
        t0 = time.time()
        m = mega.Mega(rt.fast, rt.model, None, 0, ncta=ncta, smem_cap=cap, plan_only=True,
                      pair_layers=pairs)
        st = dict(m.stats, plan_seconds=round(time.time() - t0, 1))
        out[name] = st
        print(name, json.dumps(st), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
