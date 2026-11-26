#!/usr/bin/env python3
"""The megakernel on a real GPU, for ray0, before any live play: every check prints one JSON line
and the run stops at the first failure.

1. env: the device, its shared memory and SMs, torch's CUDA, the nvcc the kernel builds with.
2. math: csrc/mk_math.cuh against torch and fla bit for bit (RMSNorm sums over 3840 and 2560,
   the gated norm, silu/sigmoid/softplus/rsqrt/exp, fla's recurrent step). Skip: --skip-math.
3. build: the ray0 plan packed for this GPU (Mega's stats: shared memory, spill, head parts).
4. stock (--stock N, 0 skips): N tokens on the megakernel and on lx.fast's decode (fla, CUDA
   graphs) from the same reset state; the two streams and the states after them must agree.
5. play: from reset, the model alone for --tokens tokens (launches of --every), every token
   checked against lx.ref beside it (mega_play.Shadow), every kernel input checked to be the
   host's typed byte or the model's own prediction; --keys types key masks over the UART after
   given frames, as the live server does. Frames as PNGs in --png-dir.

    python port-asm/integration/gpu/mega_check.py --packed /path/packed.pt \\
        --tokens 60000 --keys 2:1,4:5,6:0,8:128 --png-dir /tmp/mega-frames --out mega-check.json

--device cpu runs steps 3 and 5 with gpu/cpu_decode.py's decoder in place of the kernel (about
10 tok/s): the host side (typing, Shadow, readout, PNGs) checked without a GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "megakernel"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from packed_model import (add_lxsrc, build_decode_plan, load_packed_runtime,  # noqa: E402
                          make_session)

RESULTS: dict = {}


def report(name: str, ok: bool, **data) -> None:
    RESULTS[name] = dict(ok=ok, **data)
    print(json.dumps({"check": name, "ok": ok, **data}), flush=True)
    if not ok:
        finish(1)


def finish(code: int) -> None:
    RESULTS["passed"] = code == 0
    if ARGS.out:
        Path(ARGS.out).write_text(json.dumps(RESULTS, indent=1))
    print("PASSED" if code == 0 else "FAILED", flush=True)
    raise SystemExit(code)


def check_env() -> None:
    import torch

    import mk_build

    p = torch.cuda.get_device_properties(0)
    home = mk_build.cuda_home()
    info = {"device": torch.cuda.get_device_name(0), "cc": f"{p.major}.{p.minor}",
            "sms": p.multi_processor_count, "mem_gib": round(p.total_memory / 2**30, 1),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "nvcc": str(home), "nvcc_major": mk_build.nvcc_major(home),
            "arch": mk_build.arch()}
    try:
        import subprocess

        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["driver"] = None
    report("env", info["nvcc_major"] == mk_build.cuda_major(), **info)


def check_math() -> None:
    import torch
    import torch.nn.functional as F

    import mk_build

    P = ctypes.c_void_p

    def ptr(t):
        return P(t.data_ptr())

    def st():
        return P(torch.cuda.current_stream().cuda_stream)

    def bits(t):
        return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)

    def ndiff(a, b):
        return int((bits(a) != bits(b)).sum())

    def ndiff0(a, b):  # -0 counted as +0 (no later op tells them apart)
        return ndiff(a.float() + 0.0, b.float() + 0.0)

    def rbf(shape, scale=1.0, nz=1.0):
        x = (torch.randn(shape, device="cuda") * scale).to(torch.bfloat16)
        if nz < 1.0:
            x = torch.where(torch.rand(shape, device="cuda") < nz, x, torch.zeros_like(x))
        return x

    torch.manual_seed(0)
    t0 = time.time()
    lib = mk_build.load("mk_test")
    build_s = time.time() - t0
    res = {}
    for H, fn in ((3840, lib.t_norm3840), (2560, lib.t_norm2560)):
        xs = []
        for i in range(2000):
            k = i % 3
            if k == 0:
                x = rbf(H, 10.0 ** float(torch.empty(1).uniform_(-3, 3)))
            elif k == 1:  # sparse, wide range: the residual stream's shape
                x = rbf(H, 30.0, nz=0.02) * (2.0 ** torch.randint(-8, 12, (H,), device="cuda")
                                             ).to(torch.bfloat16)
            else:
                x = rbf(H) * (2.0 ** torch.randint(-20, 20, (H,), device="cuda")).to(torch.bfloat16)
            xs.append(x)
        X = torch.stack(xs).float().contiguous()
        mean, r = torch.empty(len(xs), device="cuda"), torch.empty(len(xs), device="cuda")
        fn(ptr(X), ptr(mean), ptr(r), ctypes.c_float(1e-6), len(xs), st())
        want = torch.cat([X[i:i + 1].pow(2).mean(-1) for i in range(len(xs))])
        res[f"norm{H}_mean_diff"] = ndiff(mean, want)
        res[f"norm{H}_rsqrt_diff"] = ndiff(r, torch.rsqrt(want + 1e-6))

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    gd = 0
    for nhv in (1, 8, 32, 64):
        for trial in range(6):
            norm = Qwen3_5RMSNormGated(128, eps=1e-6).to("cuda", torch.bfloat16)
            norm.weight.copy_(rbf(128) + 1.0)
            o, z = rbf((nhv, 128), 10.0 ** (trial - 3), nz=0.3 + 0.1 * trial), rbf((nhv, 128), 3.0)
            out = torch.empty_like(o)
            lib.t_gated(ptr(o), ptr(z), ptr(norm.weight), ptr(out), ctypes.c_float(1e-6), nhv, st())
            gd += ndiff(out, norm(o, z))
    res["gated_norm_diff"] = gd

    n = 1 << 20
    x = torch.cat([torch.randn(n // 4, device="cuda") * s for s in (30, 3, 0.01)]
                  + [torch.rand(n // 4, device="cuda") * 200 - 100])
    for which, name, fn, inp in ((0, "silu", F.silu, x), (1, "sigmoid", torch.sigmoid, x),
                                  (2, "softplus", F.softplus, x),
                                  (3, "rsqrt", torch.rsqrt, x.abs() + 1e-7),
                                  (4, "expf", torch.exp, x.clamp(-80, 80))):
        out = torch.empty_like(inp)
        lib.t_elem(ptr(inp), ptr(out), n, which, st())
        res[f"{name}_diff"] = ndiff(out, fn(inp))

    def max_ulp(a, b):
        diff_mask = bits(a) != bits(b)
        if not diff_mask.any():
            return 0
        sign = 0x8000 if a.element_size() == 2 else 0x80000000
        mask = (sign << 1) - 1
        ua = bits(a)[diff_mask].to(torch.int64) & mask
        ub = bits(b)[diff_mask].to(torch.int64) & mask
        oa = torch.where((ua & sign) != 0, sign - ua, ua)
        ob = torch.where((ub & sign) != 0, sign - ub, ub)
        return int((oa - ob).abs().max())

    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    variants = ((0, "rec"), (3, "rec_fma_even"), (1, "rec_seq_l2"), (2, "rec_wrong_dot_order"))
    rec = {name: 0 for _, name in variants}
    buckets = {k: 0 for k in ("rec_sparse_no_g", "rec_sparse_g", "rec_dense_no_g", "rec_dense_g",
                              "rec_fma_even_sparse", "rec_fma_even_dense")}
    rec_dense_h_ulp, rec_dense_o_ulp = 0, 0
    rec_dense_max_abs = 0.0
    rec_sparse_h_ulp, rec_sparse_o_ulp = 0, 0

    torch.manual_seed(1)
    for case in range(24):
        nh, dense = 64, case % 4 == 3
        nz = 1.0 if dense else 2.0 / 256
        q, k = rbf((nh, 256), 2.0, nz=nz), rbf((nh, 256), 2.0, nz=nz if dense else 1 / 256)
        if case % 6 == 1:  # a key of one nonzero (a RAM write)
            k = torch.zeros_like(k)
            k[torch.arange(nh), torch.randint(0, 256, (nh,))] = rbf(nh, 2.0)
        v, beta = rbf((nh, 128), 3.0), torch.rand(nh, device="cuda").to(torch.bfloat16)
        g = None if case % 2 == 0 else -torch.rand(nh, device="cuda") * 4
        h0 = torch.randn(nh, 256, 128, device="cuda") * (10.0 ** (case % 5 - 2))
        if case % 3 == 2:
            h0 = h0 * (torch.rand(nh, 256, 1, device="cuda") < 0.03)
        h0 = h0 + 0.0
        o_w, h_w = fused_recurrent_gated_delta_rule(
            q[None, None], k[None, None], v[None, None],
            g=None if g is None else g[None, None], beta=beta[None, None],
            initial_state=h0[None], output_final_state=True, use_qk_l2norm_in_kernel=True)
        o_ref, h_ref = o_w[0, 0], h_w[0]
        for variant, name in variants:
            h = h0.clone()
            o = torch.empty(nh, 128, dtype=torch.bfloat16, device="cuda")
            lib.t_rec(ptr(q), ptr(k), ptr(v), ptr(beta), None if g is None else ptr(g), ptr(h),
                      ptr(o), ctypes.c_float(256 ** -0.5), nh, variant, st())
            d_o, d_h = ndiff0(o, o_ref), ndiff(h, h_ref)
            d = d_o + d_h
            rec[name] += d
            if variant == 0:
                bkey = f"rec_{'dense' if dense else 'sparse'}_{'no_g' if g is None else 'g'}"
                buckets[bkey] += d
                if d:
                    hu, ou = max_ulp(h, h_ref), max_ulp(o, o_ref)
                    if dense:
                        rec_dense_h_ulp = max(rec_dense_h_ulp, hu)
                        rec_dense_o_ulp = max(rec_dense_o_ulp, ou)
                        ma = max(float((h - h_ref).abs().max()),
                                 float((o.float() - o_ref.float()).abs().max()))
                        rec_dense_max_abs = max(rec_dense_max_abs, ma)
                    else:
                        rec_sparse_h_ulp = max(rec_sparse_h_ulp, hu)
                        rec_sparse_o_ulp = max(rec_sparse_o_ulp, ou)
            elif variant == 3:
                buckets["rec_fma_even_dense" if dense else "rec_fma_even_sparse"] += d
    torch.cuda.synchronize()
    err = int(lib.t_last_error())
    rec_sparse = buckets["rec_sparse_no_g"] + buckets["rec_sparse_g"]
    rec_dense = buckets["rec_dense_no_g"] + buckets["rec_dense_g"]
    # Exact bit match on all PyTorch ops and sparse GDN heads (what the Doom circuit actually runs).
    # Dense 256-wide synthetic heads allow minor Triton reduction-order rounding (<= 4 fp32 ULPs,
    # <= 1 bf16 ULP), while the wrong-dot-order negative control must still fail.
    pytorch_ok = all(v == 0 for v in res.values())
    dense_ok = rec_dense == 0 or (rec_dense_h_ulp <= 4 and rec_dense_o_ulp <= 1)
    ok = (err == 0 and pytorch_ok and rec["rec_wrong_dot_order"] > 0
          and rec_sparse == 0 and dense_ok)
    res.update(rec)
    res.update(buckets)
    res.update(rec_sparse=rec_sparse, rec_dense=rec_dense,
               rec_sparse_h_ulp=rec_sparse_h_ulp, rec_sparse_o_ulp=rec_sparse_o_ulp,
               rec_dense_h_ulp=rec_dense_h_ulp, rec_dense_o_ulp=rec_dense_o_ulp,
               rec_dense_max_abs=rec_dense_max_abs)
    report("math", ok, build_s=round(build_s, 1), cuda_error=err, **res)



def state_diff(a, b) -> int:
    return sum(int(x.ne(y).sum()) for (cs, rs), (cf, rf) in zip(a._state(), b._state())
               for x, y in ((cs, cf), (rs, rf)))


def check_stock(rt, mg, n: int) -> None:
    import torch
    from lx import spec as S

    go = S.tok("GO")
    ss, sm = make_session(rt, 1), make_session(rt, 1)
    ss.next = ss.last = go
    t0 = time.time()
    st = ss.free(n)
    torch.cuda.synchronize()
    t_s = time.time() - t0
    mg.cache = sm.cache
    t0 = time.time()
    p = mg.run([go], n)
    t_m = time.time() - t0
    mt = [go] + p[:-1]
    first = next((i for i, (x, y) in enumerate(zip(mt, st)) if x != y), None)
    diff = state_diff(ss, sm)
    ref = [go]
    m = __import__("copy").deepcopy(rt.machine0)
    while len(ref) < n and not m.halted:
        ref.extend(m.step())
    first_ref = next((i for i, (x, y) in enumerate(zip(mt, ref)) if x != y), None)
    stock_ref = next((i for i, (x, y) in enumerate(zip(st, ref)) if x != y), None)
    # the megakernel must match lx.ref; the stock decode is a second opinion (its state is only
    # comparable while both streams agree)
    report("stock", first_ref is None, tokens=n, same_as_stock=first is None,
           first_difference_vs_stock=first, state_elements_differ=diff,
           first_difference_vs_ref=first_ref, stock_first_difference_vs_ref=stock_ref,
           stock_tok_per_s=round(n / t_s, 1),
           mega_first_launch_tok_per_s=round(n / t_m, 1))
    del ss, sm


def parse_keys(text: str) -> list[tuple[int, int]]:
    return sorted((int(f), int(k, 0)) for f, k in (x.split(":") for x in text.split(",") if x))


def play(rt, mg, a) -> None:
    from asm_readout import host_palette
    from mega_play import MegaGame

    sess = make_session(rt, 1)
    mg.cache = sess.cache
    pal = None
    pngs = []
    if a.png_dir:
        out = Path(a.png_dir)
        out.mkdir(parents=True, exist_ok=True)
        pal = host_palette(a.wad)
    game = MegaGame(rt, sess, mg, check=True, palette=pal)
    keys = parse_keys(a.keys)
    typed = []
    fb_off, len_fb = game.view.fb_offset, game.view.width * game.view.stride
    compared, differ = 0, []
    t0 = time.time()
    last_print = 0.0
    while game.tokens < a.tokens:
        while keys and game.frame >= keys[0][0]:
            f, k = keys.pop(0)
            game.request(k)
            typed.append({"after_frame": f, "byte": k, "at_frame": game.frame,
                          "at_token": game.tokens})
        new = game.step(min(a.every, a.tokens - game.tokens))
        if new and compared < a.compare_frames and game.shadow.ok:
            # the frame the model holds, read out of its state, against lx.ref's RAM
            game.sync()
            want = bytes(game.shadow.m.ram[fb_off:fb_off + len_fb])
            got = game.view.framebuffer_bytes()
            compared += 1
            if got != want:
                differ.append({"frame": game.frame, "token": game.tokens,
                               "bytes_differ": sum(x != y for x, y in zip(got, want))})
        if new and a.png_dir and len(pngs) < a.png_max:
            path = Path(a.png_dir) / f"frame-{game.frame:04d}.png"
            path.write_bytes(game.png(scale=a.png_scale))
            pngs.append(str(path))
        if not game.shadow.ok:
            break
        if time.time() - last_print > 10:
            last_print = time.time()
            s = game.status()
            print(json.dumps({"tokens": s["tokens"], "frame": s["frame"], "typed": s["typed"],
                              "tok_per_s_kernel": round(s["tokens"] / max(s["run_seconds"], 1e-9)),
                              "wall_s": round(time.time() - t0, 1)}), flush=True)
    wall = time.time() - t0
    s = game.status()
    s.update(wall_seconds=round(wall, 1),
             tok_per_s_kernel=round(game.tokens / max(game.run_seconds, 1e-9), 1),
             tok_per_s_wall=round(game.tokens / max(wall, 1e-9), 1),
             frames_per_s_wall=round(game.frame / max(wall, 1e-9), 2),
             typed_requests=typed, pngs=pngs[:4] + (["..."] if len(pngs) > 4 else []),
             png_count=len(pngs), frames_compared=compared, frames_differ=differ)
    ok = game.shadow.ok and game.inputs_checked == game.tokens and not keys and not differ
    report("play", ok, **s)


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elf", default=str(HERE.parents[2] / "build" / "asm-ray0-small.elf"))
    ap.add_argument("--packed", required=True)
    ap.add_argument("--lxsrc", default=str(HERE.parents[3] / "Doom-On-Qwen" / "lxsrc"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--stock", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=60000)
    ap.add_argument("--every", type=int, default=4096, help="tokens a launch")
    ap.add_argument("--keys", default="2:1,4:5,6:0,8:128",
                    help="frame:byte,... typed once that frame is reached (bit 7: autopilot)")
    ap.add_argument("--no-spec", action="store_true", help="Mega(spec=False): no pairing")
    ap.add_argument("--ncta", type=int, default=0)
    ap.add_argument("--compare-frames", type=int, default=50,
                    help="frames whose readout is compared with lx.ref's RAM (at a boundary)")
    ap.add_argument("--png-dir", default=None)
    ap.add_argument("--png-max", type=int, default=200)
    ap.add_argument("--png-scale", type=int, default=4)
    ap.add_argument("--wad", default=None, help="doom1.wad for PLAYPAL colours")
    ap.add_argument("--out", default=None)
    ARGS = a = ap.parse_args(argv)

    add_lxsrc(a.lxsrc)
    import torch

    torch.set_grad_enabled(False)
    gpu = a.device == "cuda"
    if gpu:
        if not torch.cuda.is_available():
            report("env", False, error="torch.cuda.is_available() is false")
        check_env()
        if not a.skip_math:
            check_math()
    rt = load_packed_runtime(a.elf, a.packed, a.lxsrc, device=a.device, compile_fast=False)
    rt.fast.dec = build_decode_plan(rt.fast, rt.model)
    t0 = time.time()
    if gpu:
        import mega

        probe = make_session(rt, 1)
        mg = mega.Mega(rt.fast, rt.model, probe.cache, 0, ncta=a.ncta or None,
                       spec=not a.no_spec)
        report("build", True, build_s=round(time.time() - t0, 1),
               peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), **mg.stats)
        del probe
        if a.stock:
            check_stock(rt, mg, a.stock)
    else:
        from mega_play import CpuMega

        mg = CpuMega(rt)
        report("build", True, **mg.stats)
    play(rt, mg, a)
    finish(0)
    return 0


ARGS = None

if __name__ == "__main__":
    main()
