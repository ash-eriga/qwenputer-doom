#!/usr/bin/env python3
"""CPU checks of the megakernel's packing for the ray0 circuit, without a GPU: for each GPU
profile (plan.PROFILES) the packed output head, decoded back from what the kernel reads (each
CTA's dense codes in its data, the sparse entries by partition), must be lx.fast's w_head
exactly; and every CTA's data must have its shared-memory part end on a blob boundary.

    python port-asm/integration/gpu/megakernel/check_pack.py --packed /tmp/asm-emit-work-v2/packed.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GPU = HERE.parent
for p in (HERE, GPU):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mega  # noqa: E402
from packed_model import build_decode_plan, load_packed_runtime  # noqa: E402
from plan import PROFILES  # noqa: E402


def check_layout(m) -> dict:
    """Every CTA's blobs: 16-byte aligned, the first at 0, and the shared-memory cut at a blob
    start (or at the end), so no blob is partly in shared memory."""
    ncta = m.ncta
    sched = m.sched.cpu().numpy().reshape(m.n_phases, ncta)
    res_len, res_cut = m.res_len.cpu().numpy(), m.res_cut.cpu().numpy()
    bad = []
    for c in range(ncta):
        offs = sorted({int(x) for x in sched[:, c] if x >= 0})
        cut, end = int(res_cut[c]), int(res_len[c])
        ok = (offs[0] == 0 and all(o % 16 == 0 for o in offs) and cut % 16 == 0
              and (cut == end or cut in offs) and all(o < end for o in offs))
        if not ok:
            bad.append(c)
    return {"ctas": ncta, "bad_ctas": bad}


def decode_head(m, w_head) -> dict:
    """The head as the kernel sees it, against w_head (bf16, rows x head columns)."""
    ncta, parts, hr, V = m.ncta, m.head_parts, m.head_rows, m.n_vocab
    nd, ns, W = m.n_dense, m.n_sparse, -(-m.head_rows // 16)
    npart = ncta * parts
    blob = m.blob.cpu().numpy()
    sched = m.sched.cpu().numpy().reshape(m.n_phases, ncta)
    res_off = m.res_off.cpu().numpy()
    hk = m.n_phases - 1
    assert m.phases[hk].kind == 2
    dense = m.hdense.cpu().numpy()[:nd].astype(np.int64)
    sparse = m.hsparse.cpu().numpy()[:ns].astype(np.int64)
    htab = m.htab.cpu().numpy()
    n_cols = w_head.shape[1]
    got = np.zeros((npart * hr, n_cols), np.float32)
    for c in range(ncta):
        base = int(res_off[c]) + int(sched[hk, c])
        hd = blob[base:base + 16].view(np.int32)
        n, body = int(hd[0]), int(hd[1])
        assert n == n_cols, (n, n_cols)
        cols = blob[base + 16:base + 16 + 2 * n].view(np.int16)
        assert np.array_equal(cols, m.fast.head_cols.cpu().numpy())
        if nd:
            words = blob[base + body:base + body + 4 * parts * nd * W].view(np.uint32)
            words = words.reshape(parts, nd, W).astype(np.uint64)
            shifts = (2 * np.arange(16)).astype(np.uint64)
            codes = (words[..., None] >> shifts) & np.uint64(3)  # (parts, nd, W, 16)
            codes = codes.reshape(parts, nd, W * 16)[:, :, :hr]
            for j in range(parts):
                p = c * parts + j
                got[p * hr:(p + 1) * hr, dense] = htab[codes[j].T.astype(np.int64)]
    hptr = m.hptr.cpu().numpy().reshape(npart, ns + 1).astype(np.int64)
    hnz = m.hnz.cpu().numpy().view(np.uint32)
    for p in range(npart):
        for j in range(ns):
            e = hnz[hptr[p, j]:hptr[p, j + 1]]
            rows = p * hr + (e & 0xffff).astype(np.int64)
            vals = ((e >> 16).astype(np.uint32) << 16).view(np.float32)
            got[rows, sparse[j]] = vals
    want = w_head.float().cpu().numpy()
    diff = int((got[:V] != want).sum())
    return {"head_rows": hr, "head_parts": parts, "partitions": npart, "dense": nd,
            "sparse": ns, "elements_differ": diff, "padding_rows": npart * hr - V,
            "entries": int(hnz.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elf", default=str(HERE.parents[3] / "build" / "asm-ray0-small.elf"))
    ap.add_argument("--packed", required=True)
    ap.add_argument("--lxsrc", default=str(HERE.parents[4] / "Doom-On-Qwen" / "lxsrc"))
    ap.add_argument("--gpus", default="L4,H100,A100")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch

    torch.set_grad_enabled(False)
    rt = load_packed_runtime(a.elf, a.packed, a.lxsrc, device="cpu", compile_fast=False,
                             quiet=True)
    rt.fast.dec = build_decode_plan(rt.fast, rt.model)
    res, ok = {}, True
    for name in a.gpus.split(","):
        ncta, cap = PROFILES[name]
        m = mega.Mega(rt.fast, rt.model, None, 0, ncta=ncta, smem_cap=cap, plan_only=True)
        r = {"layout": check_layout(m), "head": decode_head(m, rt.fast.w_head),
             "smem": m.smem, "res_global": m.stats["res_global"]}
        # rows past the vocabulary (the last partition's padding, dense code 0) are never read:
        # head_part takes rows below n_vocab only
        ok &= not r["layout"]["bad_ctas"] and r["head"]["elements_differ"] == 0
        res[name] = r
        print(name, json.dumps(r), flush=True)
        del m
    print("OK" if ok else "FAILED", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
