#!/usr/bin/env python3
"""Pure-Python ELF32/RISC-V inspection helpers for the assembly verification suite."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
PORT_ASM_DIR = VERIFY_DIR.parent
if str(PORT_ASM_DIR) not in sys.path:
    sys.path.insert(0, str(PORT_ASM_DIR))

import measure_ray  # noqa: E402
from lx_harness import decode_instruction  # noqa: E402

EM_RISCV = 243
ET_EXEC = 2
PT_LOAD = 1
SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4


def inspect_elf(path: str | Path) -> dict[str, object]:
    elf = measure_ray.Elf(str(path))
    b = elf.bytes
    if len(b) < 52 or b[:4] != b'\x7fELF':
        raise ValueError('not an ELF file')
    phoff = struct.unpack_from('<I', b, 0x1C)[0]
    shoff = struct.unpack_from('<I', b, 0x20)[0]
    phentsize, phnum = struct.unpack_from('<HH', b, 0x2A)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', b, 0x2E)

    program_headers = []
    for i in range(phnum):
        vals = struct.unpack_from('<IIIIIIII', b, phoff + i * phentsize)
        program_headers.append(dict(zip(
            ('type', 'offset', 'vaddr', 'paddr', 'filesz', 'memsz', 'flags', 'align'), vals)))

    raw_sections = []
    for i in range(shnum):
        vals = struct.unpack_from('<IIIIIIIIII', b, shoff + i * shentsize)
        raw_sections.append(dict(zip(
            ('name_off', 'type', 'flags', 'addr', 'offset', 'size', 'link', 'info', 'align', 'entsize'), vals)))

    shstr = b''
    if 0 <= shstrndx < len(raw_sections):
        sh = raw_sections[shstrndx]
        shstr = b[sh['offset']:sh['offset'] + sh['size']]

    def secname(sh):
        no = sh['name_off']
        if not shstr or no >= len(shstr):
            return ''
        end = shstr.index(b'\0', no)
        return shstr[no:end].decode()

    sections = []
    for sh in raw_sections:
        sh = dict(sh)
        sh['name'] = secname(sh)
        sections.append(sh)

    return {
        'path': str(path),
        'elf': elf,
        'bytes': b,
        'class': b[4],
        'data': b[5],
        'type': struct.unpack_from('<H', b, 0x10)[0],
        'machine': struct.unpack_from('<H', b, 0x12)[0],
        'entry': struct.unpack_from('<I', b, 0x18)[0],
        'program_headers': program_headers,
        'sections': sections,
        'symbols': elf.symbols,
    }


def scan_executable_instructions(path: str | Path) -> dict[str, object]:
    """Decode every nonzero 32-bit word in executable sections.

    Generated column kernels intentionally contain zero padding so each case has a fixed
    80-byte stride.  Zero words are counted separately; every nonzero word must decode as a
    supported RV32IM instruction.
    """
    info = inspect_elf(path)
    counts: dict[str, int] = {}
    division_sites = []
    invalid_sites = []
    zero_words = 0
    total_words = 0
    for sh in info['sections']:
        if not (sh['flags'] & SHF_EXECINSTR) or sh['type'] == 8:  # skip NOBITS defensively
            continue
        data = info['bytes'][sh['offset']:sh['offset'] + sh['size']]
        if len(data) % 4:
            invalid_sites.append({'section': sh['name'], 'reason': 'section size not multiple of 4'})
        for pos in range(0, len(data) - 3, 4):
            total_words += 1
            ir = struct.unpack_from('<I', data, pos)[0]
            if ir == 0:
                zero_words += 1
                counts['zero-padding'] = counts.get('zero-padding', 0) + 1
                continue
            mnem = decode_instruction(ir)
            counts[mnem] = counts.get(mnem, 0) + 1
            pc = sh['addr'] + pos
            if mnem in ('div', 'divu', 'rem', 'remu'):
                division_sites.append({'pc': pc, 'mnemonic': mnem, 'section': sh['name']})
            if mnem in ('illegal', 'compressed-or-illegal'):
                invalid_sites.append({'pc': pc, 'mnemonic': mnem, 'section': sh['name'], 'ir': ir})
    return {
        'counts': dict(sorted(counts.items())),
        'division_sites': division_sites,
        'invalid_sites': invalid_sites,
        'zero_words': zero_words,
        'total_words': total_words,
    }
