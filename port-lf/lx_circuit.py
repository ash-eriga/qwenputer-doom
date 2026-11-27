#!/usr/bin/env python3
"""Report the LX circuit size of a built Doom ELF.

The LX machine holds the guest's RAM in its KV cache: one 256-byte cache line per RAM word,
packed into circuit layers. So the only ELF facts that matter are RAM_SIZE and TEXT_END, and
both are recorded in the image by port-lf/doom-lf-fast.ld (__lx_ram_size / __lx_text_end).
This script reads them back and builds the circuit the linuxformer would have to pack.

Usage: lx_circuit.py ELF [ELF ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'Doom-On-Qwen' / 'lxsrc' / 'lx'))
sys.path.insert(0, str(ROOT / 'port-asm'))

from lx import core                        # noqa: E402
from measure_ray import Elf, machine_config  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        elf = Elf(path)
        ram_size, text_end = machine_config(elf)
        st = core.build(core.Config(ram_size=ram_size, text_end=text_end)).stats()
        print(f'{Path(path).name}: ram_size={ram_size:#x} text_end={text_end:#x}')
        print('  ' + ' '.join(f'{k}={v}' for k, v in st.items()))


if __name__ == '__main__':
    main()
