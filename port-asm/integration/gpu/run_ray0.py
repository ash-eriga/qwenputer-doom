#!/usr/bin/env python3
"""GPU runner for packed.pt emitted from build/asm-ray0-small.elf.

Modes:

* ``speculative``: causal draft+verify+rollback.  ``copy`` is the deployable past-only drafter;
  ``oracle`` uses lx.ref's future stream only as a correctness/performance-test instrument.
* ``verify``: teacher-forced lx.ref control.  This checks whether the packed circuit predicts
  the exact machine stream and is the correctness gate before free-running.

The small assembly guest uses a column-major framebuffer.  Frames are decoded out of the GDN
state with GDNReadout and transposed on the host by AsmColumnReadout; the guest transpose is not
executed and is not part of the cost build.
"""
from __future__ import annotations

import argparse
import copy as copylib
import gzip
import hashlib
import json
import shutil
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parent
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from asm_readout import AsmColumnReadout, read_playpal, write_palette_wad  # noqa: E402
from asmelf import Elf32, infer_fb_geometry  # noqa: E402
from speculative import CausalSpeculator, CopyDrafter  # noqa: E402


class ReferenceStreamDrafter:
    """lx.ref-backed oracle drafter for mechanism tests, not a deployable free-run drafter."""

    def __init__(self, machine, go_offset: int = 1, corrupt_positions: set[int] | None = None,
                 vocab: int | None = None):
        self.machine = machine
        self.stream: list[int] = []
        self.go_offset = int(go_offset)
        self.corrupt_positions = corrupt_positions or set()
        self.vocab = vocab
        self.exhausted = False

    def _ensure(self, n: int) -> None:
        while len(self.stream) < n and not self.machine.halted:
            self.stream.extend(self.machine.step())
        self.exhausted = len(self.stream) < n and self.machine.halted

    def draft(self, next_token: int, generated: list[int], pos: int, size: int) -> list[int]:
        del generated
        start = int(pos) - self.go_offset
        self._ensure(start + size)
        out = []
        for i in range(size):
            p = start + i
            if 0 <= p < len(self.stream):
                t = int(self.stream[p])
                if pos + i in self.corrupt_positions:
                    t = (t + 1) % self.vocab if self.vocab else t + 1
                out.append(t)
            else:
                out.append(out[-1] if out else int(next_token))
        out[0] = int(next_token)
        return out


def elf_config(elf_path: str | Path) -> dict:
    elf = Elf32(elf_path)
    cfg = elf.machine_config()
    cfg["framebuffer"] = infer_fb_geometry(elf)
    return cfg


def parse_ranges(text: str) -> set[int]:
    out = set()
    for part in (text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def parse_positions(text: str) -> set[int]:
    return {int(x) for x in (text or "").replace(" ", "").split(",") if x}


def palette_for_run(args, work: Path) -> bytes | None:
    if args.wad:
        return read_playpal(args.wad)
    # The assembly guest never reads a WAD.  This deterministic ramp only makes host-side PPM
    # visualization non-grayscale; pass --wad doom1.wad for Doom's real PLAYPAL colors.
    synthetic = work / "host-palette.wad"
    write_palette_wad(synthetic)
    return read_playpal(synthetic)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_packed(src: str, work: Path) -> tuple[Path, str]:
    """Return a local packed.pt and its SHA-256, transparently decompressing packed.pt.gz."""
    src_path = Path(src)
    if src_path.name.endswith(".gz"):
        packed = work / "packed.pt"
        with gzip.open(src_path, "rb") as fi, open(packed, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        print(f"packed input {src_path} sha256={sha256_file(src_path)}", flush=True)
        print(f"decompressed packed {packed} bytes={packed.stat().st_size}", flush=True)
    else:
        packed = src_path
    digest = sha256_file(packed)
    print(f"packed.pt sha256={digest}", flush=True)
    return packed, digest


def make_frame_capture(session, runtime, palette: bytes | None, outdir: Path, wanted: set[int]):
    from gdn_readout import GDNReadout

    base = GDNReadout(runtime.c, session.cache)
    ro = AsmColumnReadout(base, runtime.cfg, palette=palette, follow_pointer=False)
    fc_off = int(runtime.cfg["symbols"]["qwen_frame_count"]["offset"])
    saved = set()

    def capture(force: bool = False):
        fc = base.read_u32(fc_off)
        if fc not in wanted or (fc in saved and not force):
            return None
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"frame-{fc:04d}.ppm"
        path.write_bytes(ro.frame())
        saved.add(fc)
        print(f"  frame {fc} from GDN state -> {path}", flush=True)
        return path

    return capture, saved


def extend_reference(machine, pre: list[int], need: int) -> None:
    while len(pre) < need and not machine.halted:
        pre.extend(machine.step())


def run_verify(session, runtime, capture, args) -> dict:
    """Teacher-forced exact-stream control using Session.verify's chunked prefill."""
    from lx import spec as S

    proposer = session.ref
    pre: list[int] = []
    wrong = 0
    first_wrong = []
    chunks = 0
    t0 = time.time()
    go = S.tok("GO")
    while session.n < args.tokens and not proposer.halted:
        first = session.n == 0
        need = args.chunk - 1 if first else args.chunk
        extend_reference(proposer, pre, need)
        if len(pre) < need:
            break
        seg = ([go] if first else []) + pre[:need]
        del pre[:need]
        r = session.verify(seg)
        wrong += r["wrong"]
        first_wrong.extend(r["first_wrong"][:max(0, 8 - len(first_wrong))])
        chunks += 1
        capture()
        if args.progress_chunks and chunks % args.progress_chunks == 0:
            el = time.time() - t0
            print(f"  verify: {session.n:,}/{args.tokens:,} tokens, "
                  f"{session.n / max(el, 1e-9):.0f} tok/s, wrong={wrong}", flush=True)
    el = time.time() - t0
    return {"mode": "verify", "tokens": session.n, "wrong": wrong,
            "first_wrong": first_wrong[:8], "chunks": chunks,
            "seconds": round(el, 3), "tokens_per_second": round(session.n / max(el, 1e-9), 2)}


def make_speculator(session, args):
    from lx import spec as S

    if args.drafter == "copy":
        drafter = CopyDrafter(args.copy_ngram)
    elif args.drafter == "oracle":
        drafter = ReferenceStreamDrafter(
            session.ref, go_offset=1, corrupt_positions=parse_positions(args.oracle_corrupt),
            vocab=S.VOCAB)
    else:
        raise ValueError(f"unknown drafter {args.drafter}")
    spec = CausalSpeculator(session, drafter, draft_size=args.draft,
                            go_token=S.tok("GO"), history=args._resume_history)
    if args.drafter == "oracle" and args._resume_oracle_stream is not None:
        drafter.stream = list(args._resume_oracle_stream)
    if args._resume_pos is not None:
        spec.pos = args._resume_pos
    if args._resume_stats is not None:
        spec.stats = args._resume_stats
    return spec


def run_speculative(session, runtime, capture, args) -> dict:
    from lx import spec as S

    spec = make_speculator(session, args)
    spec.prime()
    t0 = time.time()
    rounds0 = spec.stats.rounds
    n0 = session.n
    last_ckpt = session.n

    # Optional semantics diagnostic: independently step lx.ref and compare the committed free-run
    # stream.  These tokens are never fed to the model, so this cannot teacher-force or leak into
    # the trajectory; it only reports where the in-weights machine leaves the reference trajectory.
    ref_machine = copylib.deepcopy(runtime.machine0) if args.check_reference else None
    ref_stream: list[int] = []
    ref_checked = 0
    first_divergence = None

    def check_reference_prefix():
        nonlocal ref_checked, first_divergence
        if ref_machine is None or first_divergence is not None:
            return
        while ref_checked < len(spec.generated):
            pos = ref_checked
            if pos == 0:
                want = S.tok("GO")
            else:
                while len(ref_stream) < pos and not ref_machine.halted:
                    ref_stream.extend(ref_machine.step())
                if len(ref_stream) < pos:
                    return
                want = ref_stream[pos - 1]
            got = spec.generated[pos]
            if got != want:
                first_divergence = {"position": pos, "want": want, "got": got}
                print(f"  reference divergence at stream position {pos}: want {want}, got {got}",
                      flush=True)
                return
            ref_checked += 1

    while len(spec.generated) < args.tokens:
        remaining = args.tokens - len(spec.generated)
        r = spec.step(min(args.draft, remaining))
        capture()
        check_reference_prefix()
        if first_divergence is not None and args.strict_reference:
            print("  stopping at the first reference divergence (--strict-reference)", flush=True)
            break
        if args.progress_rounds and spec.stats.rounds % args.progress_rounds == 0:
            el = time.time() - t0
            fed = session.n - n0
            print(f"  speculative: known={len(spec.generated):,}/{args.tokens:,} "
                  f"fed={session.n:,} (+{fed:,}) {fed / max(el, 1e-9):.0f} tok/s "
                  f"accepted={spec.stats.accepted_tokens}/{spec.stats.draft_tokens} "
                  f"rollbacks={spec.stats.rollback_rounds}", flush=True)
        if args.checkpoint and args.checkpoint_every and \
                session.n - last_ckpt >= args.checkpoint_every:
            from packed_model import save_checkpoint

            save_checkpoint(session, args.checkpoint, generated=spec.generated, pos=spec.pos,
                            stats=spec.stats, drafter=args.drafter,
                            oracle_stream=getattr(spec.drafter, "stream", None))
            print(f"  checkpoint -> {args.checkpoint}", flush=True)
            last_ckpt = session.n

    if args.checkpoint:
        from packed_model import save_checkpoint

        save_checkpoint(session, args.checkpoint, generated=spec.generated, pos=spec.pos,
                        stats=spec.stats, drafter=args.drafter,
                        oracle_stream=getattr(spec.drafter, "stream", None))
    el = time.time() - t0
    fed = session.n - n0
    return {"mode": "speculative", "drafter": args.drafter,
            "known_tokens": len(spec.generated), "fed_tokens_total": session.n,
            "fed_tokens_this_run": fed, "rounds_this_run": spec.stats.rounds - rounds0,
            "seconds": round(el, 3),
            "fed_tokens_per_second": round(fed / max(el, 1e-9), 2),
            "stats": spec.stats.as_dict(),
            "reference_checked_positions": ref_checked,
            "first_reference_divergence": first_divergence}


def static_config_json(elf_path: str | Path) -> str:
    return json.dumps(elf_config(elf_path), indent=2, sort_keys=True)


def run_cpu_self_test() -> int:
    import test_speculative

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_speculative)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elf", default=str(HERE.parents[2] / "build" / "asm-ray0-small.elf"))
    ap.add_argument("--packed", help="packed.pt emitted by port-asm/emit_small.py")
    ap.add_argument("--lxsrc", default=str(HERE.parents[3] / "Doom-On-Qwen" / "lxsrc"))
    ap.add_argument("--work", default="/tmp/asm-ray0-gpu")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--mode", choices=["speculative", "verify"], default="speculative")
    ap.add_argument("--drafter", choices=["copy", "oracle"], default="copy")
    ap.add_argument("--draft", type=int, default=64)
    ap.add_argument("--copy-ngram", type=int, default=6)
    ap.add_argument("--oracle-corrupt", default="",
                    help="comma-separated absolute positions to corrupt in oracle test drafts")
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--dump", default="1-20", help="frames/ranges to write; empty disables")
    ap.add_argument("--wad", default=None, help="optional WAD for host-side PLAYPAL colors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-block", type=int, default=32768)
    ap.add_argument("--head-vblock", type=int, default=8192)
    ap.add_argument("--fp32-head", action="store_true")
    ap.add_argument("--check-reference", action="store_true",
                    help="compare committed stream with lx.ref without feeding it to the model")
    ap.add_argument("--strict-reference", action="store_true",
                    help="stop at the first --check-reference divergence")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=2_000_000)
    ap.add_argument("--progress-rounds", type=int, default=100)
    ap.add_argument("--progress-chunks", type=int, default=8)
    ap.add_argument("--expect-packed-sha256", default=None,
                    help="required SHA-256 of the materialized packed.pt")
    ap.add_argument("--load-only", action="store_true",
                    help="load the packed circuit and exit before token execution")
    ap.add_argument("--print-config", action="store_true", help="print ELF-derived config and exit")
    ap.add_argument("--self-test", action="store_true", help="run CPU-only causality tests and exit")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        return run_cpu_self_test()
    if args.print_config:
        print(static_config_json(args.elf))
        return 0
    if not args.packed:
        raise SystemExit("--packed is required for GPU execution (use --print-config or --self-test locally)")
    if args.draft > args.chunk:
        raise SystemExit("--draft must be <= --chunk so verify and rollback replay use one causal block")
    if args.mode == "verify" and args.drafter != "copy":
        print("note: --drafter is ignored in verify mode", flush=True)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    outdir = Path(args.outdir or work / "frames")
    wanted = parse_ranges(args.dump)

    from packed_model import load_checkpoint, load_packed_runtime, make_session

    packed, packed_sha256 = materialize_packed(args.packed, work)
    if args.expect_packed_sha256 and packed_sha256 != args.expect_packed_sha256.lower():
        raise SystemExit(f"packed.pt sha256 mismatch: got {packed_sha256}, "
                         f"expected {args.expect_packed_sha256.lower()}")
    runtime = load_packed_runtime(args.elf, packed, args.lxsrc, chunk=args.chunk,
                                  device=args.device, compile_fast=not args.no_compile,
                                  head_block=args.head_block, head_vblock=args.head_vblock,
                                  fp32_head=args.fp32_head)
    if args.load_only:
        print(json.dumps({"mode": "load-only", "device": args.device,
                          "packed_sha256": packed_sha256,
                          "load_seconds": round(runtime.load_seconds, 3),
                          "circuit_layers": runtime.c.n_layers(),
                          "packed_layers": runtime.fast.n_chip,
                          "head_columns": int(runtime.fast.head_cols.numel())},
                         indent=2, sort_keys=True), flush=True)
        return 0
    args._resume_history = None
    args._resume_pos = None
    args._resume_stats = None
    args._resume_oracle_stream = None
    if args.resume:
        if not args.checkpoint:
            raise SystemExit("--resume requires --checkpoint")
        session, extra = load_checkpoint(runtime, args.checkpoint, args.chunk)
        args._resume_history = extra.get("generated")
        args._resume_pos = extra.get("pos")
        args._resume_stats = extra.get("stats")
        args._resume_oracle_stream = extra.get("oracle_stream")
        print(f"resumed checkpoint at token {session.n:,}", flush=True)
    else:
        session = make_session(runtime, args.chunk)
        print("fresh session", flush=True)

    pal = palette_for_run(args, work)
    capture, saved = make_frame_capture(session, runtime, pal, outdir, wanted)
    if args.mode == "verify":
        result = run_verify(session, runtime, capture, args)
    else:
        result = run_speculative(session, runtime, capture, args)
    result["frames_written"] = sorted(saved)
    result["outdir"] = str(outdir)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
