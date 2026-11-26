#!/usr/bin/env python3
"""Instruction and LX-token cost meter for asm-ray0 ELF files.

This is not a second machine semantics implementation for correctness; fastemu-lf remains the
reference runner. It is a small dynamic counter that uses the token rules in lx/spec.py:
rd-writing records cost one token when the value fits signed 16 bits and two otherwise, SW is
one control token, SB/SH carry the old word in two D tokens plus the store control, jumps and
branches are one token in this small image, and timer/anchor records add a small overhead.
"""
import struct
import sys
from pathlib import Path

M32 = 0xFFFFFFFF
RAM_BASE = 0x80000000
RAM_SIZE = 0x600000
TEXT_END = 0x40000


def sx(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def fits16(v):
    v &= M32
    return v < 0x8000 or v >= 0xFFFF8000


class Elf:
    def __init__(self, path):
        b = Path(path).read_bytes()
        if b[:4] != b'\x7fELF' or b[4] != 1:
            raise ValueError('not ELF32')
        self.bytes = b
        self.entry = struct.unpack_from('<I', b, 0x18)[0]
        phoff = struct.unpack_from('<I', b, 0x1c)[0]
        shoff = struct.unpack_from('<I', b, 0x20)[0]
        phentsize, phnum = struct.unpack_from('<HH', b, 0x2a)
        shentsize, shnum = struct.unpack_from('<HH', b, 0x2e)
        self.segments = []
        for i in range(phnum):
            typ, off, va, pa, filesz, memsz, flags, align = struct.unpack_from('<IIIIIIII', b, phoff + i * phentsize)
            if typ == 1 and memsz:
                self.segments.append((off, va, filesz, memsz))
        self.sections = []
        for i in range(shnum):
            self.sections.append(struct.unpack_from('<IIIIIIIIII', b, shoff + i * shentsize))
        self.symbols = {}
        for sh in self.sections:
            name_off, typ, flags, addr, off, size, link, info, align, entsize = sh
            if typ != 2:
                continue
            str_off = self.sections[link][4]
            for pos in range(off, off + size, 16):
                nm, value, sz, info, other, shndx = struct.unpack_from('<IIIBBH', b, pos)
                if not nm:
                    continue
                end = b.index(b'\0', str_off + nm)
                self.symbols[b[str_off + nm:end].decode()] = value


def load_memory(elf):
    mem = bytearray(RAM_SIZE)
    for off, va, filesz, memsz in elf.segments:
        if va < RAM_BASE or va + memsz > RAM_BASE + RAM_SIZE:
            raise ValueError(f'segment outside LX image: {va:#x}+{memsz:#x}')
        mem[va - RAM_BASE:va - RAM_BASE + filesz] = elf.bytes[off:off + filesz]
    return mem


def read_script(path):
    rows = []
    if not path:
        return rows
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or line.startswith(';'):
            continue
        parts = line.replace(';', ' ').split()
        if len(parts) >= 2:
            rows.append((int(parts[0], 0), int(parts[1], 0)))
    rows.sort()
    return rows


def run(path, frames=40, script_path=None):
    elf = Elf(path)
    mem = load_memory(elf)
    x = [0] * 32
    pc = elf.entry
    x[2] = RAM_BASE + RAM_SIZE - 16
    fc_addr = elf.symbols['qwen_frame_count']
    key_addr = elf.symbols.get('qwen_key_state', 0)
    schedule = read_script(script_path)
    schedule_pos = 0

    inst = 0
    tokens = 0
    frames_seen = 0
    per_pc = {}
    per_mnemonic = {}
    # Anchor approximation: jumps in this tiny image almost never change JH. The loop still
    # models the 63-commit refresh and the 64-commit timer tick as small additive records.
    commits_since_jump = 0
    cycle_mod = 0

    def add_cost(pc0, mnem, ninst, ntok):
        nonlocal inst, tokens
        inst += ninst
        tokens += ntok
        a = per_pc.setdefault(pc0, [0, 0, mnem])
        a[0] += ninst
        a[1] += ntok
        b = per_mnemonic.setdefault(mnem, [0, 0])
        b[0] += ninst
        b[1] += ntok

    def valcost(v, rd):
        return 1 if rd == 0 or fits16(v) else 2

    def lb(addr, signed=False):
        o = addr - RAM_BASE
        if not 0 <= o < len(mem):
            raise RuntimeError(f'load outside RAM {addr:#x}')
        v = mem[o]
        return sx(v, 8) & M32 if signed else v

    def lh(addr, signed=False):
        o = addr - RAM_BASE
        if addr & 1 or not 0 <= o + 1 < len(mem):
            raise RuntimeError(f'bad lh {addr:#x}')
        v = mem[o] | mem[o + 1] << 8
        return sx(v, 16) & M32 if signed else v

    def lw(addr):
        o = addr - RAM_BASE
        if addr & 3 or not 0 <= o + 3 < len(mem):
            raise RuntimeError(f'bad lw {addr:#x}')
        return int.from_bytes(mem[o:o + 4], 'little')

    def sb(addr, v):
        o = addr - RAM_BASE
        if not 0 <= o < len(mem):
            raise RuntimeError(f'store outside RAM {addr:#x}')
        mem[o] = v & 0xff

    def sh(addr, v):
        o = addr - RAM_BASE
        if addr & 1 or not 0 <= o + 1 < len(mem):
            raise RuntimeError(f'bad sh {addr:#x}')
        mem[o:o + 2] = (v & 0xffff).to_bytes(2, 'little')

    def sw(addr, v):
        nonlocal frames_seen, schedule_pos
        o = addr - RAM_BASE
        if addr & 3 or not 0 <= o + 3 < len(mem):
            raise RuntimeError(f'bad sw {addr:#x}')
        mem[o:o + 4] = (v & M32).to_bytes(4, 'little')
        if addr == fc_addr:
            frames_seen = v
            while schedule_pos < len(schedule) and schedule[schedule_pos][0] <= v:
                kv = schedule[schedule_pos][1]
                ko = key_addr - RAM_BASE
                mem[ko:ko + 4] = kv.to_bytes(4, 'little')
                schedule_pos += 1

    while frames_seen < frames:
        if pc < RAM_BASE or pc >= RAM_BASE + TEXT_END or pc & 3:
            raise RuntimeError(f'bad pc {pc:#x}')
        pc0 = pc
        w = lw(pc)
        pc = (pc + 4) & M32
        opc = w & 0x7f
        rd = (w >> 7) & 31
        f3 = (w >> 12) & 7
        rs1 = (w >> 15) & 31
        rs2 = (w >> 20) & 31
        f7 = w >> 25
        imm_i = sx(w >> 20, 12)
        cost = 1
        mnem = '?'

        if opc == 0x37:  # LUI
            v = w & 0xfffff000
            if rd: x[rd] = v
            cost = valcost(v, rd); mnem = 'lui'
        elif opc == 0x17:  # AUIPC
            v = (pc0 + (w & 0xfffff000)) & M32
            if rd: x[rd] = v
            cost = valcost(v, rd); mnem = 'auipc'
        elif opc == 0x6f:  # JAL
            imm = sx(((w >> 31) << 20) | ((w >> 21) & 0x3ff) << 1 | ((w >> 20) & 1) << 11 | ((w >> 12) & 0xff) << 12, 21)
            if rd: x[rd] = pc
            pc = (pc0 + imm) & M32
            commits_since_jump = 0
            mnem = 'jal'
        elif opc == 0x67:  # JALR
            if rd: x[rd] = pc
            pc = (x[rs1] + imm_i) & ~1 & M32
            commits_since_jump = 0
            mnem = 'jalr'
        elif opc == 0x63:  # branches
            imm = sx(((w >> 31) << 12) | ((w >> 7) & 1) << 11 | ((w >> 25) & 0x3f) << 5 | ((w >> 8) & 0xf) << 1, 13)
            a, b = x[rs1], x[rs2]
            sa, sbv = sx(a, 32), sx(b, 32)
            cond = (a == b if f3 == 0 else a != b if f3 == 1 else sa < sbv if f3 == 4 else
                    sa >= sbv if f3 == 5 else a < b if f3 == 6 else a >= b)
            if cond:
                pc = (pc0 + imm) & M32
                commits_since_jump = 0
                mnem = 'br.taken'
            else:
                commits_since_jump += 1
                mnem = 'br.nt'
        elif opc == 0x03:  # loads
            addr = (x[rs1] + imm_i) & M32
            if f3 == 0: v = lb(addr, True); mnem = 'lb'
            elif f3 == 1: v = lh(addr, True); mnem = 'lh'
            elif f3 == 2: v = lw(addr); mnem = 'lw'
            elif f3 == 4: v = lb(addr); mnem = 'lbu'
            elif f3 == 5: v = lh(addr); mnem = 'lhu'
            else: raise RuntimeError(f'bad load f3 {f3} at {pc0:#x}')
            if rd: x[rd] = v & M32
            cost = valcost(v, rd)
            commits_since_jump += 1
        elif opc == 0x23:  # stores
            imm = sx(((w >> 25) << 5) | ((w >> 7) & 31), 12)
            addr = (x[rs1] + imm) & M32
            if f3 == 0: sb(addr, x[rs2]); cost = 3; mnem = 'sb'
            elif f3 == 1: sh(addr, x[rs2]); cost = 3; mnem = 'sh'
            elif f3 == 2: sw(addr, x[rs2]); cost = 1; mnem = 'sw'
            else: raise RuntimeError(f'bad store f3 {f3} at {pc0:#x}')
            commits_since_jump += 1
        elif opc == 0x13:  # OP-IMM
            a = x[rs1]
            if f3 == 0: v = (a + imm_i) & M32; mnem = 'addi'
            elif f3 == 1: v = (a << (rs2 & 31)) & M32; mnem = 'slli'
            elif f3 == 2: v = int(sx(a, 32) < imm_i); mnem = 'slti'
            elif f3 == 3: v = int(a < (imm_i & M32)); mnem = 'sltiu'
            elif f3 == 4: v = a ^ (imm_i & M32); mnem = 'xori'
            elif f3 == 5:
                v = (sx(a, 32) >> (rs2 & 31)) & M32 if f7 == 0x20 else a >> (rs2 & 31)
                mnem = 'srai' if f7 == 0x20 else 'srli'
            elif f3 == 6: v = a | (imm_i & M32); mnem = 'ori'
            elif f3 == 7: v = a & (imm_i & M32); mnem = 'andi'
            else: raise RuntimeError(f'bad opimm f3 {f3} at {pc0:#x}')
            if rd: x[rd] = v & M32
            cost = valcost(v, rd)
            commits_since_jump += 1
        elif opc == 0x33:  # OP
            a, b = x[rs1], x[rs2]
            if f7 == 1:
                if f3 == 0: v = (a * b) & M32; mnem = 'mul'
                elif f3 == 1: v = ((sx(a, 32) * sx(b, 32)) >> 32) & M32; mnem = 'mulh'
                elif f3 == 2: v = ((sx(a, 32) * a) >> 32) & M32; mnem = 'mulhsu'
                elif f3 == 3: v = ((a * b) >> 32) & M32; mnem = 'mulhu'
                else: raise RuntimeError('division reached the hot loop')
            elif f7 == 0x10 and f3 in (2, 4, 6):
                shn = {2: 1, 4: 2, 6: 3}[f3]
                v = ((a << shn) + b) & M32; mnem = f'sh{shn}add'
            elif f7 in (0, 0x20):
                if f3 == 0: v = (a - b) & M32 if f7 == 0x20 else (a + b) & M32; mnem = 'sub' if f7 == 0x20 else 'add'
                elif f3 == 1: v = (a << (b & 31)) & M32; mnem = 'sll'
                elif f3 == 2: v = int(sx(a, 32) < sx(b, 32)); mnem = 'slt'
                elif f3 == 3: v = int(a < b); mnem = 'sltu'
                elif f3 == 4: v = a ^ b; mnem = 'xor'
                elif f3 == 5: v = (sx(a, 32) >> (b & 31)) & M32 if f7 == 0x20 else a >> (b & 31); mnem = 'sra' if f7 == 0x20 else 'srl'
                elif f3 == 6: v = a | b; mnem = 'or'
                elif f3 == 7: v = a & b; mnem = 'and'
            else:
                raise RuntimeError(f'bad op f7/f3 {f7:x}/{f3} at {pc0:#x}')
            if rd: x[rd] = v & M32
            cost = valcost(v, rd)
            commits_since_jump += 1
        elif opc == 0x0f:
            mnem = 'fence'
            commits_since_jump += 1
        else:
            raise RuntimeError(f'unimplemented opcode {opc:#x} at {pc0:#x}')

        # Timer tick and anchor refresh overhead. They are tiny here but kept explicit so the
        # token/instruction ratio does not look magically equal to the hot-loop cost.
        cycle_mod += 1
        overhead = 0
        if cycle_mod == 64:
            cycle_mod = 0
            overhead += 1
        if commits_since_jump >= 63:
            commits_since_jump = 0
            overhead += 1
        add_cost(pc0, mnem, 1, cost + overhead)

    return {
        'instructions': inst,
        'tokens': tokens,
        'frames': frames_seen,
        'per_pc': per_pc,
        'per_mnemonic': per_mnemonic,
        'symbols': elf.symbols,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'build/asm-ray0.elf'
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    script = sys.argv[3] if len(sys.argv) > 3 else None
    r = run(path, frames, script)
    print(f"frames={r['frames']} instructions={r['instructions']} tokens~={r['tokens']}")
    print(f"instr/frame={r['instructions'] / frames:.1f} tokens/frame~={r['tokens'] / frames:.1f} "
          f"tokens/instr~={r['tokens'] / r['instructions']:.3f}")
    print('\nby mnemonic:')
    for mnem, (ni, nt) in sorted(r['per_mnemonic'].items(), key=lambda kv: -kv[1][1]):
        print(f'  {mnem:10s} {ni:9d} inst  {nt:9d} tok  {nt / ni:.2f} tok/inst')
    print('\nhot PCs:')
    for pc, (ni, nt, mnem) in sorted(r['per_pc'].items(), key=lambda kv: -kv[1][0])[:16]:
        print(f'  {pc:#x} {mnem:10s} {ni:9d} inst  {nt:9d} tok')


if __name__ == '__main__':
    main()
