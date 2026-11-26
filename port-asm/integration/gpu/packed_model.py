"""packed.pt loading and runtime construction for the small ray0 LX circuit.

The code follows Doom-On-Qwen/lxsrc/run_kaggle.py and doom-lx's free runner, but all machine
dimensions come from the assembly ELF through port-asm/integration/asmelf.py.  No constant here
may be hardcoded to the 6 MiB Doom layout.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parents[1]
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))

from asmelf import Elf32, RAM_BASE, infer_fb_geometry  # noqa: E402


@dataclass
class PackedRuntime:
    torch: object
    c: object
    model: object
    fast: object
    machine0: object
    cfg: dict
    elf: Elf32
    blob: dict
    device: str
    load_seconds: float


def load_elf_config(elf_path: str | Path) -> tuple[Elf32, dict]:
    elf = Elf32(elf_path)
    cfg = elf.machine_config()
    cfg["framebuffer"] = infer_fb_geometry(elf)
    return elf, cfg


def add_lxsrc(lxsrc: str | Path) -> Path:
    lxsrc = Path(lxsrc).resolve()
    for p in (lxsrc / "lx", lxsrc):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return lxsrc


def build_fast(blob: dict):
    """Reconstruct lx.fast.Fast's packed tensors from run_kaggle's packed.pt blob."""
    from lx.fast import Fast, _GDN, _MLP

    f = Fast.__new__(Fast)
    f.n_chip = blob["meta"]["n_chip"]
    f.nk, f.nv = blob["meta"]["nk"], blob["meta"]["nv"]
    f.kd, f.vd = blob["meta"]["kd"], blob["meta"]["vd"]
    f.r = f.nv // f.nk
    f.head_cols, f.w_head = blob["head_cols"], blob["w_head"]
    f.layers = [(_GDN(**g), _MLP(**m)) for g, m in zip(blob["gdn"], blob["mlp"])]
    f.dec, f.stats, f._conv_c, f._steps = None, None, None, None
    return f


def make_initial_machine(elf: Elf32, cfg: dict):
    from lx.ref import Machine

    mem = bytearray(elf.load_memory(cfg["ram_size"]))
    m = Machine(mem, text_end=cfg["text_end"])
    m.x[2] = RAM_BASE + len(mem) - 16
    m.anchor = (elf.entry - RAM_BASE) // 4
    m.n = 0
    return m


def _materialize_streamed_model(torch, model, blob, proj_prefixes):
    """Load packed state_dict entries and resolve intentionally meta tensors.

    Projections held packed by Fast are dropped, as in run_kaggle.py; every other meta tensor is
    zero (the one stock attention layer after the chip).  Confusing those cases would either
    allocate the dense model or corrupt the circuit.
    """
    model.load_state_dict(blob["sd"], strict=False, assign=True)
    n_zero = n_drop = 0
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p is None or p.device.type != "meta":
            continue
        parts = name.split(".")
        obj = model
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        if any(name.split(".", 3)[-1].startswith(x) for x in proj_prefixes):
            empty = torch.empty(0)
            setattr(obj, parts[-1], torch.nn.Parameter(empty, requires_grad=False)
                    if isinstance(p, torch.nn.Parameter) else empty)
            n_drop += 1
            continue
        z = torch.zeros(p.shape, dtype=p.dtype)
        setattr(obj, parts[-1], torch.nn.Parameter(z, requires_grad=False)
                if isinstance(p, torch.nn.Parameter) else z)
        n_zero += 1
    return n_zero, n_drop


def load_packed_runtime(elf_path: str | Path, packed_pt: str | Path, lxsrc: str | Path,
                        chunk: int = 32768, device: str = "cuda", compile_fast: bool = True,
                        head_block: int = 32768, head_vblock: int = 8192,
                        fp32_head: bool = False, quiet: bool = False) -> PackedRuntime:
    """Build the circuit object, load packed.pt, move it to the GPU, and return the runtime."""
    del chunk  # the session owns chunking; keeping it in the signature documents the pairing
    t_start = time.time()
    elf, cfg = load_elf_config(elf_path)
    lxsrc = add_lxsrc(lxsrc)

    import torch
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        maj, mi = torch.cuda.get_device_capability()
        os.environ["LX_ARCH"] = f"sm_{maj}{mi}a" if maj == 9 else f"sm_{maj}{mi}"

    import colab_bench as cb
    from lx import core, hf
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling

    c = core.build(core.Config(ram_size=cfg["ram_size"], text_end=cfg["text_end"]))
    lay = hf.pick_layout(len(c.dims))
    machine0 = make_initial_machine(elf, cfg)

    if not quiet:
        print(f"ELF ram_size={cfg['ram_size_hex']} text_end={cfg['text_end_hex']} "
              f"md5={cfg['elf_md5']}", flush=True)
        print(f"circuit layers={c.n_layers()} dims={len(c.dims)}", flush=True)

    t0 = time.time()
    blob = torch.load(Path(packed_pt), map_location="cpu", weights_only=False)
    got_layers = int(blob["meta"]["n_chip"])
    if got_layers != c.n_layers():
        raise ValueError(f"packed.pt/circuit mismatch: packed n_chip={got_layers}, "
                         f"ELF-derived circuit layers={c.n_layers()}")
    if int(blob["tc"].get("hidden_size", -1)) != lay.hidden:
        raise ValueError(f"packed.pt/layout mismatch: hidden={blob['tc'].get('hidden_size')}, "
                         f"expected {lay.hidden}")

    cfg35 = Qwen3_5TextConfig(**blob["tc"])
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg35)
    n_zero, n_drop = _materialize_streamed_model(torch, model, blob, tuple(cb.PROJ_PREFIXES))
    model.model.rotary_emb = qwen35_modeling.Qwen3_5TextRotaryEmbedding(config=cfg35)
    model.layout = lay
    model.tie_weights()
    if not quiet:
        print(f"meta tensors: {n_zero} zeroed, {n_drop} dropped (Fast holds them packed)",
              flush=True)

    f = build_fast(blob)
    f.model = model
    f.head_block, f.head_vblock = int(head_block), int(head_vblock)
    f.lean, f.compile = True, bool(compile_fast)
    model.to(device)
    cb.to_gpu(f, device)

    if fp32_head:
        from lx.fast import _Norm

        w32 = f.w_head.float()
        norm, cols = model.model.norm, f.head_cols

        def head_argmax32(h, block=None):
            out, block = [], block or f.head_block
            for i in range(0, h.shape[0], block):
                hn = _Norm(norm, h[i:i + block])(cols)
                out.append((hn.float() @ w32.T).argmax(-1))
            return torch.cat(out)

        f.head_argmax = head_argmax32
        if not quiet:
            print("using fp32 output head", flush=True)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        mem_gib = torch.cuda.memory_allocated() / 2**30
        if not quiet:
            print(f"GPU {torch.cuda.get_device_name(0)} sm_{props.major}{props.minor}; "
                  f"loaded packed build in {time.time() - t0:.0f}s, allocated {mem_gib:.2f} GiB",
                  flush=True)
    elif not quiet:
        print(f"loaded packed build on CPU in {time.time() - t0:.0f}s", flush=True)

    return PackedRuntime(torch=torch, c=c, model=model, fast=f, machine0=machine0, cfg=cfg,
                         elf=elf, blob=blob, device=device,
                         load_seconds=time.time() - t_start)


def make_session(runtime: PackedRuntime, chunk: int):
    """Create a fresh LX Session with the initial ray0 machine injected."""
    from lx import session as SS

    return SS.Session(runtime.c, runtime.model, runtime.machine0, chunk, 0, runtime.fast)


def save_checkpoint(session, path: str | Path, **extra) -> None:
    """Save the same state as Session.save plus runner-specific fields such as history."""
    import torch as th

    d = {"state": [(cs.cpu(), rs.cpu()) for cs, rs in session._state()],
         "next": session.next, "last": session.last, "n": session.n,
         "waiting": session.waiting, "ref": session.ref}
    d.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    th.save(d, path)


def load_checkpoint(runtime: PackedRuntime, path: str | Path, chunk: int):
    """Restore a checkpoint written by save_checkpoint; returns (session, extra_dict)."""
    from lx import session as SS

    d = runtime.torch.load(Path(path), map_location="cpu", weights_only=False)
    s = SS.Session(runtime.c, runtime.model, None, chunk, 0, runtime.fast)
    s._set_state([(cs.to(s.dev), rs.to(s.dev)) for cs, rs in d["state"]])
    s.next, s.last, s.n, s.waiting, s.ref = d["next"], d["last"], d["n"], d["waiting"], d["ref"]
    extra = {k: v for k, v in d.items()
             if k not in {"state", "next", "last", "n", "waiting", "ref"}}
    return s, extra
