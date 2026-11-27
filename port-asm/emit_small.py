#!/usr/bin/env python3
"""Emit packed.pt for a small-memory assembly ELF using the stock doom-lx emitter.

run_kaggle.py hardcodes the 6 MiB Doom layout as module constants. This wrapper overrides only
those constants after reading the linker symbols in the ELF, then calls the stock emit() path.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from measure_ray import Elf, machine_config  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf')
    ap.add_argument('--lxsrc', default=str(ROOT / 'Doom-On-Qwen' / 'lxsrc'))
    ap.add_argument('--work', required=True)
    ap.add_argument('--data', required=True)
    a = ap.parse_args()

    elf = Elf(a.elf)
    ram_size, text_end = machine_config(elf)
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    target = data / 'doom-lf.elf'
    if target.resolve() != Path(a.elf).resolve():
        target.write_bytes(Path(a.elf).read_bytes())

    lxsrc = Path(a.lxsrc)
    sys.path.insert(0, str(lxsrc / 'lx'))
    sys.path.insert(0, str(lxsrc))
    rk = load_module(lxsrc / 'run_kaggle.py', 'doom_run_kaggle')
    rk.RAM_SIZE = ram_size
    rk.TEXT_END = text_end
    print(f'emit config ram_size={ram_size:#x} text_end={text_end:#x}', flush=True)
    rk.emit(SimpleNamespace(data=str(data), lxsrc=str(lxsrc), work=str(a.work)))


if __name__ == '__main__':
    main()
