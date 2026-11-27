"""Build the megakernel's CUDA sources (csrc/*.cu) with nvcc into shared libraries, called through
ctypes on torch's tensors and stream, linked to torch's own CUDA runtime so they share its
context. Ported from linuxformer's lx/mk_build.py: the target is the device's own architecture
(sm_89 on an L4, sm_90a on an H100/H200) and torch builds for CUDA 12 or 13 both work.

    lib = mk_build.load("mega", ["-DMK_HIDDEN=3840"])   # build/mk/mega-<hash>.so, rebuilt on change

Environment: LX_CUDA_HOME / CUDA_HOME (a toolkit with bin/nvcc), LX_ARCH (e.g. sm_89),
LX_NVCC_FLAGS (extra nvcc flags, e.g. -allow-unsupported-compiler for a host gcc newer than the
toolkit knows), QWENPUTER_MK_BUILD (output directory).
"""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parent / "csrc"
OUT = Path(os.environ.get("QWENPUTER_MK_BUILD",
                          Path(__file__).resolve().parents[4] / "build" / "mk"))
# nvcc's own contractions are kept (as PyTorch's build has them): the exact paths write every
# rounding step as an intrinsic (__fmul_rn, __fadd_rn, __fmaf_rn), which nvcc never fuses
FLAGS = ["-O3", "-std=c++17", "-lineinfo", "-shared", "-Xcompiler", "-fPIC"]


def _torch_nvidia() -> Path:
    import torch

    return Path(torch.__file__).resolve().parents[1] / "nvidia"


def cuda_major() -> int:
    import torch

    if not torch.version.cuda:
        raise RuntimeError("this torch has no CUDA")
    return int(torch.version.cuda.split(".")[0])


def arch() -> str:
    """LX_ARCH, else the current device's: sm_90a for compute capability 9.0 (as linuxformer
    builds for the H100), sm_XY otherwise."""
    a = os.environ.get("LX_ARCH")
    if a:
        return a
    import torch

    maj, mi = torch.cuda.get_device_capability()
    return f"sm_{maj}{mi}a" if (maj, mi) == (9, 0) else f"sm_{maj}{mi}"


def cuda_home() -> Path:
    nv = _torch_nvidia()
    pip = [nv / "cu13", nv / "cuda_nvcc"]  # nvidia-cuda-nvcc wheels (CUDA 13, CUDA 12 layouts)
    for c in (os.environ.get("LX_CUDA_HOME"), os.environ.get("CUDA_HOME"), *pip,
              "/usr/local/cuda"):
        if c and (Path(c) / "bin" / "nvcc").exists():
            return Path(c)
    raise RuntimeError("nvcc not found: set LX_CUDA_HOME to a CUDA toolkit matching torch's "
                       f"CUDA {cuda_major()}")


def nvcc_major(home: Path) -> int:
    r = subprocess.run([str(home / "bin" / "nvcc"), "--version"], capture_output=True, text=True)
    for tok in r.stdout.replace(",", " ").split():
        if tok.startswith("V") and tok[1:2].isdigit():
            return int(tok[1:].split(".")[0])
    raise RuntimeError(f"cannot read nvcc's version:\n{r.stdout}{r.stderr}")


def cudart() -> Path:
    """torch's libcudart.so.<major> (the pip wheels' copy, else the toolkit's)."""
    m = cuda_major()
    nv = _torch_nvidia()
    cands = [nv / "cu13" / "lib", nv / "cuda_runtime" / "lib"]
    try:
        h = cuda_home()
        cands += [h / "lib64", h / "lib"]
    except RuntimeError:
        pass
    for d in cands:
        if (d / f"libcudart.so.{m}").exists():
            return d / f"libcudart.so.{m}"
    raise RuntimeError(f"libcudart.so.{m} not found in {[str(c) for c in cands]}")


def includes(home: Path) -> list[str]:
    """CUDA 12's pip wheels split the toolkit: nvcc (nvidia/cuda_nvcc) does not carry
    cuda_runtime.h or cuda_bf16.h, which are in nvidia/cuda_runtime/include."""
    out = []
    for d in (home / "include", home.parent / "cuda_runtime" / "include"):
        if (d / "cuda_runtime.h").exists() and str(d) not in out:
            out += ["-I", str(d)]
    return out


def load(name: str, extra: list[str] = ()) -> ctypes.CDLL:
    src = SRC / f"{name}.cu"
    flags = FLAGS + [f"-arch={arch()}"] + os.environ.get("LX_NVCC_FLAGS", "").split() + list(extra)
    h = hashlib.sha256()
    for p in sorted(SRC.glob("*.cu*")):
        h.update(p.name.encode() + p.read_bytes())
    h.update(" ".join(flags).encode())
    h.update(str(cuda_major()).encode())
    so = OUT / f"{name}-{h.hexdigest()[:12]}.so"
    if not so.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        home, rt = cuda_home(), cudart()
        if nvcc_major(home) != cuda_major():
            raise RuntimeError(f"nvcc in {home} is CUDA {nvcc_major(home)}, torch is CUDA "
                               f"{cuda_major()}: set LX_CUDA_HOME to a matching toolkit")
        tmp = so.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [str(home / "bin" / "nvcc"), *flags, "-I", str(SRC), *includes(home),
               "-o", str(tmp), str(src),
               "-L", str(rt.parent), f"-l:{rt.name}", "-Xlinker", f"-rpath={rt.parent}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"nvcc failed:\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        tmp.rename(so)
    return ctypes.CDLL(str(so))
