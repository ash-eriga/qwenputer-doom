#!/usr/bin/env python3
"""Exact CPU-only LX harness for the assembly renderer verification suite.

The machine semantics and token accounting come from the existing port-asm scripts:
``measure_exact_lx.py`` supplies ``lx.ref.Machine``/``lx.spec`` and ``measure_ray.py``
supplies the ELF loader and host-schedule parser.  This module adds per-frame capture,
framebuffer hashing and dynamic hot-loop instruction observation without modifying either
script.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
PORT_ASM_DIR = VERIFY_DIR.parent
ROOT = PORT_ASM_DIR.parent

if str(PORT_ASM_DIR) not in sys.path:
    sys.path.insert(0, str(PORT_ASM_DIR))

import measure_exact_lx  # noqa: E402  (the exact lx.ref entry point)
import measure_ray  # noqa: E402  (ELF/parser and independent approximate counter)

Machine = measure_exact_lx.Machine
S = measure_exact_lx.S

W = 96
H = 60
STRIDE = 64
FB_COLS_SIZE = W * STRIDE


def sha256_hex(data: bytes | bytearray) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def framebuffer_bytes(ram: bytes | bytearray, symbols: dict[str, int]) -> tuple[bytes, bytes]:
    """Return (raw column-major fb_cols, canonical row-major 96x60 bytes)."""
    off = symbols['fb_cols'] - S.RAM_BASE
    raw = bytes(ram[off:off + FB_COLS_SIZE])
    linear = bytearray(W * H)
    p = 0
    for y in range(H):
        for x in range(W):
            linear[p] = raw[x * STRIDE + y]
            p += 1
    return raw, bytes(linear)


def decode_instruction(ir: int) -> str:
    """Small RV32IM decoder sufficient for provenance and forbidden-op detection."""
    if ir == 0:
        return 'zero-padding'
    if ir & 3 != 3:
        return 'compressed-or-illegal'
    opc = ir & 0x7F
    f3 = (ir >> 12) & 7
    f7 = (ir >> 25) & 0x7F

    if opc == 0x37:
        return 'lui'
    if opc == 0x17:
        return 'auipc'
    if opc == 0x6F:
        return 'jal'
    if opc == 0x67:
        return 'jalr' if f3 == 0 else 'illegal'
    if opc == 0x63:
        return {0: 'beq', 1: 'bne', 4: 'blt', 5: 'bge', 6: 'bltu', 7: 'bgeu'}.get(f3, 'illegal')
    if opc == 0x03:
        return {0: 'lb', 1: 'lh', 2: 'lw', 4: 'lbu', 5: 'lhu'}.get(f3, 'illegal')
    if opc == 0x23:
        return {0: 'sb', 1: 'sh', 2: 'sw'}.get(f3, 'illegal')
    if opc == 0x13:
        if f3 == 0:
            return 'addi'
        if f3 == 1:
            return 'slli' if f7 == 0 else 'illegal'
        if f3 == 2:
            return 'slti'
        if f3 == 3:
            return 'sltiu'
        if f3 == 4:
            return 'xori'
        if f3 == 5:
            return 'srli' if f7 == 0 else 'srai' if f7 == 0x20 else 'illegal'
        if f3 == 6:
            return 'ori'
        if f3 == 7:
            return 'andi'
        return 'illegal'
    if opc == 0x33:
        if f7 == 1:
            return {0: 'mul', 1: 'mulh', 2: 'mulhsu', 3: 'mulhu',
                    4: 'div', 5: 'divu', 6: 'rem', 7: 'remu'}[f3]
        if f7 == 0:
            return {0: 'add', 1: 'sll', 2: 'slt', 3: 'sltu', 4: 'xor',
                    5: 'srl', 6: 'or', 7: 'and'}.get(f3, 'illegal')
        if f7 == 0x20:
            return {0: 'sub', 5: 'sra'}.get(f3, 'illegal')
        return 'illegal'
    if opc == 0x0F:
        return 'fence'
    if opc == 0x2F:
        return 'atomic'
    if opc == 0x73:
        return 'system-or-csr'
    if opc in (0x07, 0x27, 0x43, 0x47, 0x4B, 0x4F, 0x53):
        return 'floating-point'
    return 'illegal'


# Division/remainder are the central prohibition.  Byte/halfword stores are also prohibited
# inside the renderer hot loop because the cost design requires aligned SW-only framebuffer
# writes.  The other classes are outside the RV32IM subset this probe is allowed to execute.
FORBIDDEN_HOT_MNEMONICS = {
    'div', 'divu', 'rem', 'remu',
    'sb', 'sh',
    'atomic', 'system-or-csr', 'floating-point', 'compressed-or-illegal', 'illegal',
    'zero-padding',
}


class ObservedMachine(Machine):
    """lx.ref.Machine plus a non-invasive dynamic instruction observer."""

    def __init__(self, *args, hot_range: tuple[int, int] | None = None, **kwargs):
        self.hot_range = hot_range or (0, 0)
        self.executed: dict[int, dict[str, object]] = {}
        self.forbidden_events: list[dict[str, object]] = []
        super().__init__(*args, **kwargs)

    def instruction(self):  # noqa: D102 - behavior documented on the class
        pc = self.pc
        ir = self.word(self.pcw) if 0 <= self.pcw < self.size // 4 else 0
        mnem = decode_instruction(ir)
        hot = self.hot_range[0] <= pc < self.hot_range[1]
        ent = self.executed.setdefault(pc, {'count': 0, 'mnemonic': mnem, 'hot': hot})
        ent['count'] = int(ent['count']) + 1
        if mnem in FORBIDDEN_HOT_MNEMONICS:
            ev = {'pc': pc, 'mnemonic': mnem, 'hot': hot, 'ir': ir}
            self.forbidden_events.append(ev)
        return super().instruction()


@dataclass
class ExactRun:
    elf: str
    frames_requested: int
    frames_completed: int
    instructions: int
    exact_tokens: int
    records: int
    token_families: dict[str, int]
    per_frame: list[dict[str, object]]
    machine: ObservedMachine
    symbols: dict[str, int]

    def baseline_view(self) -> dict[str, object]:
        return {
            'frames': self.frames_completed,
            'totals': {
                'instructions': self.instructions,
                'exact_tokens': self.exact_tokens,
                'records': self.records,
                'instr_per_frame': self.instructions / max(self.frames_completed, 1),
                'tokens_per_frame': self.exact_tokens / max(self.frames_completed, 1),
            },
            'token_families': dict(sorted(self.token_families.items())),
            'per_frame': self.per_frame,
        }


def executable_text_bounds(elf: measure_ray.Elf) -> tuple[int, int]:
    """Return the hot-range upper bound from the LOAD segment containing the entry."""
    for _off, va, filesz, memsz in elf.segments:
        if va <= elf.entry < va + memsz:
            return elf.symbols.get('main_loop', elf.entry), va + memsz
    return elf.symbols.get('main_loop', elf.entry), elf.entry


def run_exact(elf_path: str | Path, frames: int, script_path: str | Path | None = None,
              max_records: int = 100_000_000) -> ExactRun:
    """Run ``frames`` rendered frames on lx.ref.Machine and capture exact per-frame data.

    Host-script application intentionally follows ``measure_exact_lx.py``: because the frame
    counter starts at zero, an event stamped frame N is installed immediately after the frame
    counter is observed at N.  Thus frame 0 input affects the next rendered frame, matching the
    existing measurement script byte-for-byte in timing.
    """
    elf_path = str(elf_path)
    elf = measure_ray.Elf(elf_path)
    mem = measure_ray.load_memory(elf)
    hot_range = executable_text_bounds(elf)
    m = ObservedMachine(mem, text_end=measure_ray.TEXT_END, hot_range=hot_range)
    m.x[2] = S.RAM_BASE + len(mem) - 16
    m.anchor = (elf.entry - S.RAM_BASE) // 4
    m.n = 0

    fc_addr = elf.symbols['qwen_frame_count'] - S.RAM_BASE
    key_addr = elf.symbols['qwen_key_state'] - S.RAM_BASE
    schedule = measure_ray.read_script(script_path) if script_path else []
    if schedule and 'qwen_demo_enabled' in elf.symbols:
        demo = elf.symbols['qwen_demo_enabled'] - S.RAM_BASE
        m.ram[demo:demo + 4] = (0).to_bytes(4, 'little')

    sp = 0
    frames_seen = 0
    tokens = 0
    records = 0
    start_instret = m.instret
    boundary_instret = m.instret
    boundary_tokens = 0
    boundary_records = 0
    token_families: dict[str, int] = {}
    per_frame: list[dict[str, object]] = []

    while frames_seen < frames and records < max_records:
        toks = m.step()
        tokens += len(toks)
        records += 1
        for token in toks:
            fam = S.family(token)[0]
            token_families[fam] = token_families.get(fam, 0) + 1

        nf = int.from_bytes(m.ram[fc_addr:fc_addr + 4], 'little')
        if nf != frames_seen:
            if nf != frames_seen + 1:
                raise RuntimeError(f'frame counter jumped {frames_seen} -> {nf}')
            frames_seen = nf
            while sp < len(schedule) and schedule[sp][0] <= frames_seen:
                m.ram[key_addr:key_addr + 4] = schedule[sp][1].to_bytes(4, 'little')
                sp += 1

            raw_fb, linear_fb = framebuffer_bytes(m.ram, elf.symbols)
            per_frame.append({
                'frame': frames_seen,
                'instructions': m.instret - boundary_instret,
                'exact_tokens': tokens - boundary_tokens,
                'records': records - boundary_records,
                'fb_cols_sha256': sha256_hex(raw_fb),
                'fb_rowmajor_sha256': sha256_hex(linear_fb),
            })
            boundary_instret = m.instret
            boundary_tokens = tokens
            boundary_records = records

    if frames_seen != frames:
        raise RuntimeError(f'only completed {frames_seen}/{frames} frames after {records} records')

    return ExactRun(
        elf=elf_path,
        frames_requested=frames,
        frames_completed=frames_seen,
        instructions=m.instret - start_instret,
        exact_tokens=tokens,
        records=records,
        token_families=token_families,
        per_frame=per_frame,
        machine=m,
        symbols=dict(elf.symbols),
    )


def scenario_definitions() -> dict[str, dict[str, object]]:
    """Canonical deterministic scenarios used by tests and baseline generation."""
    return {
        'autopilot_full_20': {
            'elf': str(ROOT / 'build' / 'asm-ray0-full.elf'),
            'frames': 20,
            'script': None,
            'description': 'full-step ray marcher, baked autopilot',
        },
        'autopilot_half_20': {
            'elf': str(ROOT / 'build' / 'asm-ray0-half.elf'),
            'frames': 20,
            'script': None,
            'description': 'half-step ray marcher, baked autopilot',
        },
        'host_script_full_24': {
            'elf': str(ROOT / 'build' / 'asm-ray0-full.elf'),
            'frames': 24,
            'script': str(PORT_ASM_DIR / 'demo-20s.sched'),
            'description': 'full-step renderer with the existing host key schedule',
        },
        'idle_full_8': {
            'elf': str(ROOT / 'build' / 'asm-ray0-full.elf'),
            'frames': 8,
            'script': str(VERIFY_DIR / 'scenarios' / 'idle-off.sched'),
            'description': 'full-step renderer, autopilot disabled and no input',
        },
    }


def run_scenario(name: str, definition: dict[str, object]) -> ExactRun:
    return run_exact(definition['elf'], int(definition['frames']), definition.get('script'))


def mnemonic_counts(machine: ObservedMachine, hot_only: bool = True) -> dict[str, int]:
    out: dict[str, int] = {}
    for ent in machine.executed.values():
        if hot_only and not ent['hot']:
            continue
        mnem = str(ent['mnemonic'])
        out[mnem] = out.get(mnem, 0) + int(ent['count'])
    return out
