#!/usr/bin/env bash
# Everything a fresh GPU VM (L4, H100/H200) needs to run ray0 on the megakernel, then the
# validation: run from a clone of qwenputer-doom, nothing to copy by hand.
#
#   bash port-asm/integration/gpu/setup_gpu.sh              # set up, then mega_check.py
#   bash port-asm/integration/gpu/setup_gpu.sh --no-check   # set up only
#   STEPS="assets lxsrc" bash port-asm/integration/gpu/setup_gpu.sh --no-check   # some steps
#
# Steps (each skips what is already there):
#   python  a virtualenv in $WORK/venv with the pinned stack: torch 2.11 for CUDA 13.0 (driver
#           >= 580) or 12.8, transformers 5.22, flash-linear-attention 0.5.2, and nvcc matching
#           torch's CUDA (CUDA 13: pip wheels; CUDA 12: the system toolkit, else NVIDIA's
#           redistributable archives in $WORK)
#   lxsrc   ../Doom-On-Qwen/lxsrc: Doom-On-Qwen cloned beside this checkout, then its
#           make_lxsrc.sh (linuxformer at the commit Doom-On-Qwen pins)
#   assets  the release ray0-gpu-assets-v1: packed.pt (decompressed into $WORK),
#           build/asm-ray0-small.elf and $WORK/playpal.wad (Doom's palette, for the frames'
#           colours), all checked against SHA256SUMS
#
# Environment: WORK (default <repo>/build/gpu), PYTHON (default python3), CUDA_FLAVOR (cu130 or
# cu128; default from the driver), STEPS (default "python lxsrc assets"), RES (96 or 192; default 96).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK=${WORK:-$REPO/build/gpu}
PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-python lxsrc assets}
RES=${RES:-96}
RELEASE=https://github.com/ash-eriga/qwenputer-doom/releases/download/ray0-gpu-assets-v1
DOOM_ON_QWEN=https://github.com/ash-eriga/Doom-On-Qwen
LXSRC=$(dirname "$REPO")/Doom-On-Qwen/lxsrc
VENV=$WORK/venv
CHECK=1
while [ $# -gt 0 ]; do
    case $1 in
        --no-check) CHECK=0 ;;
        --res) shift; RES=${1:?missing value for --res} ;;
        --res=*) RES=${1#--res=} ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
case $RES in
    96)  ELF_NAME=asm-ray0-small.elf; PACKED_NAME=packed.pt ;;
    192) ELF_NAME=asm-ray0-192.elf;   PACKED_NAME=packed-192.pt ;;
    *) echo "unsupported RES=$RES (use 96 or 192)" >&2; exit 2 ;;
esac
mkdir -p "$WORK"
say() { printf '\n== %s\n' "$*"; }
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }

step_python() {
    say "python stack in $VENV"
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver" >&2; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    local drv flavor
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    flavor=${CUDA_FLAVOR:-$([ "$drv" -ge 580 ] && echo cu130 || echo cu128)}
    echo "driver $drv -> torch for $flavor"
    if [ ! -x "$VENV/bin/python" ]; then
        if ! "$PYTHON" -m venv "$VENV" 2>/dev/null; then
            # Debian/Ubuntu images ship python3 without ensurepip
            local v
            v=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
            if command -v apt-get >/dev/null; then
                local sudo=""
                [ "$(id -u)" = 0 ] || sudo="sudo -n"
                # update first: a stale package index 404s on the .deb
                $sudo apt-get update -q >/dev/null 2>&1 || true
                $sudo apt-get install -y -q "python$v-venv" >/dev/null
            fi
            rm -rf "$VENV"
            "$PYTHON" -m venv "$VENV" || { echo "cannot create a virtualenv: install python$v-venv" >&2; exit 1; }
        fi
    fi
    local py=$VENV/bin/python
    "$py" -m pip install -q --retries 10 -U pip
    # multi-GB wheels: keep pip's scratch off a small /tmp and resume dropped downloads
    mkdir -p "$WORK/tmp"
    pipi() { TMPDIR=$WORK/tmp "$py" -m pip install -q --retries 10 --resume-retries 20 "$@"; }
    if ! "$py" -c 'import torch, sys; sys.exit(torch.__version__.split("+")[0] != "2.11.0")' 2>/dev/null; then
        pipi "torch==2.11.0" --index-url "https://download.pytorch.org/whl/$flavor"
    fi
    pipi "transformers==5.22.*" "fla-core==0.5.2" "flash-linear-attention==0.5.2" \
        accelerate safetensors numpy
    # transformers 5.22 reads torchcodec's metadata on import; audio is never used
    pipi --no-deps torchcodec
    local tcuda
    tcuda=$("$py" -c 'import torch; print(torch.version.cuda)')
    echo "torch $("$py" -c 'import torch; print(torch.__version__)') (CUDA $tcuda)"
    case $tcuda in
        13.*)
            # nvcc and its headers at torch's own CUDA version, into torch's nvidia/cu13
            pipi "nvidia-cuda-nvcc==$tcuda.*" "nvidia-cuda-crt==$tcuda.*" \
                "nvidia-nvvm==$tcuda.*" "nvidia-cuda-cccl==$tcuda.*" "nvidia-cuda-runtime==$tcuda.*"
            ;;
        12.*)
            local home=""
            for c in "${LX_CUDA_HOME:-}" "${CUDA_HOME:-}" /usr/local/cuda "/usr/local/cuda-$tcuda"; do
                if [ -n "$c" ] && [ -x "$c/bin/nvcc" ] \
                    && "$c/bin/nvcc" --version | grep -q "release 12\."; then home=$c; break; fi
            done
            if [ -z "$home" ]; then
                # CUDA 12's pip wheels have no nvcc: NVIDIA's redistributable archives instead
                home=$WORK/cuda-$tcuda
                if [ ! -x "$home/bin/nvcc" ]; then
                    local base=https://developer.download.nvidia.com/compute/cuda/redist json=""
                    for rel in "$tcuda.1" "$tcuda.0" "$tcuda.2"; do
                        json=$(curl -fsSL "$base/redistrib_$rel.json" 2>/dev/null) && break || json=""
                    done
                    [ -n "$json" ] || { echo "no CUDA $tcuda redistributable found" >&2; exit 1; }
                    mkdir -p "$home"
                    for k in cuda_nvcc cuda_cudart cuda_cccl cuda_crt cuda_nvvm; do
                        local p
                        p=$(printf '%s' "$json" | "$py" -c "import json,sys; d=json.load(sys.stdin); print(d['$k']['linux-x86_64']['relative_path'] if '$k' in d else '')")
                        [ -n "$p" ] && curl -fsSL "$base/$p" | tar -xJ -C "$home" --strip-components=1
                    done
                fi
            fi
            echo "nvcc: $home"
            echo "export LX_CUDA_HOME=$home" > "$WORK/env.sh"
            ;;
    esac
    "$py" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"GPU {torch.cuda.get_device_name(0)} sm_{p.major}{p.minor}, {p.multi_processor_count} SMs")
import fla, transformers  # noqa: F401
print("transformers", transformers.__version__, "fla", fla.__version__)
PY
}

step_lxsrc() {
    say "lxsrc in $LXSRC"
    if [ -f "$LXSRC/lx/lx/ref.py" ] && [ -f "$LXSRC/colab_bench.py" ]; then
        echo "already there"
        return
    fi
    local dq
    dq=$(dirname "$LXSRC")
    [ -d "$dq/.git" ] || git clone -q "$DOOM_ON_QWEN" "$dq"
    sh "$dq/doom-lx/ops/work/make_lxsrc.sh" "$LXSRC"
}

step_assets() {
    say "assets in $WORK (RES=$RES: $ELF_NAME, $PACKED_NAME)"
    cd "$WORK"
    # the query string skips a stale cached copy after the release's assets change
    curl -fsSL -o SHA256SUMS "$RELEASE/SHA256SUMS?t=$(date +%s)"
    for f in "${PACKED_NAME}.gz" "$ELF_NAME" playpal.wad; do
        if [ ! -f "$f" ] || ! grep " $f\$" SHA256SUMS | sha256sum -c --status; then
            echo "downloading $f"
            curl -fsSL --retry 3 -o "$f" "$RELEASE/$f"
        fi
        grep " $f\$" SHA256SUMS | sha256sum -c
    done
    if [ ! -f "$PACKED_NAME" ] || ! grep " $PACKED_NAME\$" SHA256SUMS | sha256sum -c --status; then
        echo "decompressing $PACKED_NAME (~4 GB)"
        gzip -dc "${PACKED_NAME}.gz" > "${PACKED_NAME}.tmp" && mv "${PACKED_NAME}.tmp" "$PACKED_NAME"
    fi
    grep " $PACKED_NAME\$" SHA256SUMS | sha256sum -c
    mkdir -p "$REPO/build"
    local elf=$REPO/build/$ELF_NAME
    if [ -f "$elf" ] && ! cmp -s "$elf" "$ELF_NAME"; then
        echo "keeping your own build/$ELF_NAME (the release one is $WORK/$ELF_NAME)"
    else
        cp "$ELF_NAME" "$elf"
    fi
    cd "$REPO"
}

for s in python lxsrc assets; do
    if has "$s"; then "step_$s"; fi
done

if [ -f "$WORK/env.sh" ]; then . "$WORK/env.sh"; fi
CMD=("$VENV/bin/python" port-asm/integration/gpu/mega_check.py --elf "$REPO/build/$ELF_NAME"
     --packed "$WORK/$PACKED_NAME" --wad "$WORK/playpal.wad"
     --png-dir "$WORK/frames" --out "$WORK/mega-check.json")
if [ "$CHECK" = 1 ]; then
    say "validation: ${CMD[*]}"
    cd "$REPO"
    "${CMD[@]}" 2>&1 | tee "$WORK/mega-check.log"
    echo "result: $WORK/mega-check.json, log: $WORK/mega-check.log, frames: $WORK/frames"
else
    say "ready"
    if [ -f "$WORK/env.sh" ]; then echo ". $WORK/env.sh"; fi
    echo "${CMD[*]}"
    echo "$VENV/bin/python port-asm/integration/gpu/mega_live.py --elf $REPO/build/$ELF_NAME --packed $WORK/$PACKED_NAME --wad $WORK/playpal.wad --token \$(openssl rand -hex 8)"
fi
