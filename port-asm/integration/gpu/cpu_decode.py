#!/usr/bin/env python3
"""The packed ray0 circuit's single-token decode on the CPU, checked against lx.ref.

This is the plan the megakernel runs (packed_model.build_decode_plan, lx.fast's Fast.decode
steps), run one token at a time in torch on the CPU, with fla's recurrent step written out in
torch (fp32, the delta rule with in-kernel l2 norms). It needs no GPU, so a decode-plan or
circuit problem shows up here before any GPU time is spent:

* ``--mode free``: the model alone. Each prediction is the next input, and it is compared with
  lx.ref's stream (the first divergence stops the run).
* ``--mode forced``: lx.ref's stream is the input and every prediction is checked.
* ``--keys 1:9,4:0``: after frame 1's commit type key mask 9 (a KEY0 token, at that record
  boundary, into the model and lx.ref alike), after frame 4's mask 0: the live host's input path.

It also counts, per phase, the steps whose GDN or MLP changed the residual stream. That is the
activity the megakernel's optimistic pairing (mega.PAIR_LAYERS) is chosen from.

    python port-asm/integration/gpu/cpu_decode.py --packed /tmp/asm-emit-work-v2/packed.pt \\
        --tokens 4000 --out /root/cpu-decode.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from packed_model import (add_lxsrc, build_decode_plan, load_packed_runtime,  # noqa: E402
                          make_session)


def recurrent_step(self, d, q, k, v, beta, g, rs):
    """fla's fused recurrent gated delta rule for one token (use_qk_l2norm_in_kernel), in torch,
    on the live value heads' part of rs (updated in place). Value head i reads key head i // R."""
    import torch

    K = q.shape[3]
    nhv = v.shape[2]
    rep = nhv // q.shape[2]
    qf, kf = q[0, 0].float(), k[0, 0].float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * K ** -0.5
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    qf, kf = qf.repeat_interleave(rep, 0), kf.repeat_interleave(rep, 0)
    h = rs[0, d.hv] if isinstance(d.hv, slice) else rs[0, d.hv].clone()  # (nhv, K, V)
    if g is not None:
        h.mul_(g[0, 0].float().exp()[:, None, None])
    s = torch.einsum("hkv,hk->hv", h, kf)
    vn = (v[0, 0].float() - s) * beta[0, 0].float()[:, None]
    h.add_(kf[:, :, None] * vn[:, None, :])
    o = torch.einsum("hkv,hk->hv", h, qf)
    if not isinstance(d.hv, slice):
        rs[0, d.hv] = h
    return o.to(v.dtype).view(1, 1, nhv, v.shape[3])


class CpuDecoder:
    def __init__(self, runtime):
        import torch

        self.torch, self.rt = torch, runtime
        f, model = runtime.fast, runtime.model
        f.dec = build_decode_plan(f, model)
        f.compile = False
        f._gdn_rec = types.MethodType(recurrent_step, f)
        self.f, self.model = f, model
        # the head in fp32: bf16 products are exact in fp32, so only the sum order differs from
        # a bf16 matmul's fp32 accumulation, and the circuit's head sums are exact
        self.w_head32 = f.w_head.float()
        self.layers_with = [(L, d.gdn, d.mlp) for L, d in enumerate(f.dec)]
        self.activity = {f"{L}.{k}": 0 for L, d in enumerate(f.dec)
                         for k, on in (("gdn", d.gdn), ("mlp", d.mlp)) if on}

    def step(self, cache, tok: int) -> int:
        torch, f = self.torch, self.f
        from lx.fast import _Norm

        x = self.model.model.embed_tokens.weight[tok][None].clone()
        pre, post, mlp = f._gdn_pre, f._gdn_post, f._mlp_step
        for L, d in enumerate(f.dec):
            if d.gdn:
                cl = cache.layers[L]
                before = x.clone()
                qkv = pre(d, x, cl.conv_states[0])
                if qkv is not None:
                    o = f._gdn_rec(d, *qkv[:5], cl.recurrent_states[0])
                    post(self.model.model.layers[L].linear_attn.norm, d, x, o, qkv[5])
                if not torch.equal(before, x):
                    self.activity[f"{L}.gdn"] += 1
            if d.mlp:
                before = x.clone()
                mlp(d, x)
                if not torch.equal(before, x):
                    self.activity[f"{L}.mlp"] += 1
        hn = _Norm(self.model.model.norm, x)(f.head_cols)
        return int((hn.float() @ self.w_head32.T).argmax(-1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elf", default=str(HERE.parents[2] / "build" / "asm-ray0-small.elf"))
    ap.add_argument("--packed", required=True)
    ap.add_argument("--lxsrc", default=str(HERE.parents[3] / "Doom-On-Qwen" / "lxsrc"))
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--mode", choices=["free", "forced"], default="free")
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keys", default="", help="frame:mask,... typed after that frame's commit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import copy

    import torch

    add_lxsrc(a.lxsrc)
    from lx import spec as S

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    rt = load_packed_runtime(a.elf, a.packed, a.lxsrc, device="cpu", compile_fast=False)
    dec = CpuDecoder(rt)
    session = make_session(rt, chunk=1)
    ref_m = copy.deepcopy(rt.machine0)
    keys = dict((int(f), int(k, 0)) for f, k in (x.split(":") for x in a.keys.split(",") if x))
    fc_off = int(rt.cfg["symbols"]["qwen_frame_count"]["offset"])
    frame_seen, typed_at = 0, set()
    stream = [S.tok("GO")]
    tok, wrong, first_wrong = stream[0], 0, None
    t0 = time.time()
    n = 0
    for n in range(1, a.tokens + 1):
        pred = dec.step(session.cache, tok)
        if len(stream) <= n and not ref_m.halted:
            # a record boundary: everything lx.ref emitted has been run
            fc = int.from_bytes(ref_m.ram[fc_off:fc_off + 4], "little")
            if fc != frame_seen:
                frame_seen = fc
                if fc in keys:
                    toks = ref_m.key(keys[fc])
                    typed_at.update(range(len(stream), len(stream) + len(toks)))
                    stream.extend(toks)
                    print(f"  frame {fc}: typed {keys[fc]:#x} at token {n}", flush=True)
            while len(stream) <= n and not ref_m.halted:
                stream.extend(ref_m.step())
        if len(stream) <= n:
            break
        want = stream[n]
        if n in typed_at:  # the host's input: the model's prediction here is not used
            tok = want
            continue
        if pred != want:
            wrong += 1
            if first_wrong is None:
                first_wrong = {"position": n, "want": want, "got": pred,
                               "context": stream[max(0, n - 8):n]}
                print(f"  mismatch at {n}: want {want} got {pred}", flush=True)
            if a.mode == "free":
                break
        tok = pred if a.mode == "free" else want
        if a.progress and n % a.progress == 0:
            el = time.time() - t0
            print(f"  {n:,}/{a.tokens:,} tokens, {n / el:.1f} tok/s, wrong={wrong}", flush=True)
    el = time.time() - t0
    res = {"mode": a.mode, "tokens": n, "wrong": wrong, "first_wrong": first_wrong,
           "frames_completed": frame_seen, "typed_positions": sorted(typed_at),
           "seconds": round(el, 1), "tok_per_s": round(n / max(el, 1e-9), 2),
           "activity": dec.activity, "packed": a.packed}
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
