#!/usr/bin/env python3
"""Strong deterministic verification for the assembly-native LX renderer."""
from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import unittest
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
PORT_ASM_DIR = VERIFY_DIR.parent
ROOT = PORT_ASM_DIR.parent
if str(PORT_ASM_DIR) not in sys.path:
    sys.path.insert(0, str(PORT_ASM_DIR))
if str(VERIFY_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIR))

import measure_ray  # noqa: E402
from elfcheck import EM_RISCV, ET_EXEC, PT_LOAD, inspect_elf, scan_executable_instructions  # noqa: E402
from lx_harness import (  # noqa: E402
    FB_COLS_SIZE,
    S,
    framebuffer_bytes,
    mnemonic_counts,
    run_exact,
    run_scenario,
    scenario_definitions,
)

BASELINE_PATH = VERIFY_DIR / 'baselines.json'
COST_BUILDS = ('full', 'half')


def load_baselines() -> dict[str, object]:
    return json.loads(BASELINE_PATH.read_text())


class AsmProbeVerification(unittest.TestCase):
    _scenario_cache: dict[str, object] = {}

    @classmethod
    def scenario(cls, name: str):
        if name not in cls._scenario_cache:
            cls._scenario_cache[name] = run_scenario(name, scenario_definitions()[name])
        return cls._scenario_cache[name]

    def assert_baseline_scenario_equal(self, name: str):
        baseline = load_baselines()['scenarios'][name]
        run = self.scenario(name)
        got = run.baseline_view()
        self.assertEqual(baseline['totals']['instructions'], got['totals']['instructions'], name)
        self.assertEqual(baseline['totals']['exact_tokens'], got['totals']['exact_tokens'], name)
        self.assertEqual(baseline['totals']['records'], got['totals']['records'], name)
        self.assertEqual(baseline['token_families'], got['token_families'], name)
        for expected, actual in zip(baseline['per_frame'], got['per_frame']):
            for field in ('frame', 'instructions', 'exact_tokens', 'records',
                          'fb_cols_sha256', 'fb_rowmajor_sha256'):
                self.assertEqual(expected[field], actual[field], f'{name} frame {actual["frame"]} {field}')
        self.assertEqual(len(baseline['per_frame']), len(got['per_frame']), name)

    def test_elf_is_a_valid_lx_memory_image(self):
        required = {
            'fb_cols', 'fb_linear', 'qwen_frame_count', 'qwen_key_state', 'qwen_demo_enabled',
            'DG_ScreenBuffer', 'main_loop', 'colkern_base', 'colkern_done', '_start',
        }
        for kind in COST_BUILDS:
            path = ROOT / 'build' / f'asm-ray0-{kind}.elf'
            with self.subTest(elf=path.name):
                info = inspect_elf(path)
                self.assertEqual(info['class'], 1, 'ELF32 required')
                self.assertEqual(info['data'], 1, 'little-endian required')
                self.assertEqual(info['type'], ET_EXEC)
                self.assertEqual(info['machine'], EM_RISCV)
                self.assertEqual(info['entry'], S.RAM_BASE)
                self.assertEqual(info['entry'] & 3, 0)

                loads = [ph for ph in info['program_headers'] if ph['type'] == PT_LOAD]
                self.assertGreaterEqual(len(loads), 2)
                for ph in loads:
                    self.assertLessEqual(ph['filesz'], ph['memsz'])
                    self.assertGreaterEqual(ph['vaddr'], S.RAM_BASE)
                    self.assertLessEqual(ph['vaddr'] + ph['memsz'], S.RAM_BASE + measure_ray.RAM_SIZE)
                    self.assertEqual(ph['vaddr'] & 3, 0)
                self.assertTrue(any(ph['flags'] & 1 and ph['vaddr'] <= info['entry'] < ph['vaddr'] + ph['memsz']
                                    for ph in loads), 'entry must be in an executable LOAD')

                ordered = sorted((ph['vaddr'], ph['vaddr'] + ph['memsz']) for ph in loads)
                for (a0, a1), (b0, _b1) in zip(ordered, ordered[1:]):
                    self.assertLessEqual(a1, b0, 'LOAD segments must not overlap')

                syms = info['symbols']
                self.assertTrue(required <= set(syms), f'missing symbols: {required - set(syms)}')
                for name in ('fb_cols', 'qwen_frame_count', 'qwen_key_state', 'qwen_demo_enabled'):
                    self.assertEqual(syms[name] & 3, 0, name)
                self.assertEqual(syms['fb_cols'] & 15, 0, 'framebuffer columns should stay 16-byte aligned')

                elf = info['elf']
                mem = measure_ray.load_memory(elf)
                self.assertEqual(len(mem), measure_ray.RAM_SIZE)
                for _off, va, filesz, memsz in elf.segments:
                    rel = va - S.RAM_BASE
                    file_bytes = elf.bytes[_off:_off + filesz]
                    self.assertEqual(bytes(mem[rel:rel + filesz]), file_bytes, 'PT_LOAD file image mismatch')
                    self.assertEqual(bytes(mem[rel + filesz:rel + memsz]), b'\0' * (memsz - filesz),
                                     'PT_LOAD bss/zero-fill mismatch')

                fc_off = syms['qwen_frame_count'] - S.RAM_BASE
                key_off = syms['qwen_key_state'] - S.RAM_BASE
                demo_off = syms['qwen_demo_enabled'] - S.RAM_BASE
                screen_off = syms['DG_ScreenBuffer'] - S.RAM_BASE
                self.assertEqual(struct.unpack_from('<I', mem, fc_off)[0], 0)
                self.assertEqual(struct.unpack_from('<I', mem, key_off)[0], 0)
                self.assertEqual(struct.unpack_from('<I', mem, demo_off)[0], 1)
                self.assertEqual(struct.unpack_from('<I', mem, screen_off)[0], syms['fb_cols'])

    def test_static_instruction_scan_rejects_division_and_invalid_opcodes(self):
        for kind in COST_BUILDS:
            path = ROOT / 'build' / f'asm-ray0-{kind}.elf'
            with self.subTest(elf=path.name):
                scan = scan_executable_instructions(path)
                self.assertEqual(scan['division_sites'], [])
                self.assertEqual(scan['invalid_sites'], [])
                self.assertGreater(scan['zero_words'], 0, 'expected generated fixed-stride kernel padding')
                counts = scan['counts']
                self.assertNotIn('atomic', counts)
                self.assertNotIn('system-or-csr', counts)
                self.assertNotIn('floating-point', counts)
                self.assertGreater(counts.get('sw', 0), 1000, 'generated column kernels should be SW-heavy')

    def test_exact_lx_tokens_and_instructions_per_frame_match_reference(self):
        for name in scenario_definitions():
            with self.subTest(scenario=name):
                self.assert_baseline_scenario_equal(name)

    def test_framebuffer_hashes_are_stable_and_reproducible(self):
        # Golden per-frame hashes are checked for every canonical scenario.
        for name in scenario_definitions():
            with self.subTest(scenario=name):
                baseline = load_baselines()['scenarios'][name]
                run = self.scenario(name)
                self.assertEqual([f['fb_cols_sha256'] for f in baseline['per_frame']],
                                 [f['fb_cols_sha256'] for f in run.per_frame])
                self.assertEqual([f['fb_rowmajor_sha256'] for f in baseline['per_frame']],
                                 [f['fb_rowmajor_sha256'] for f in run.per_frame])

        # Independent repetition proves the harness itself is deterministic, not merely equal
        # to a checked-in value because of a one-off capture artifact.
        elf = ROOT / 'build' / 'asm-ray0-full.elf'
        a = run_exact(elf, 4)
        b = run_exact(elf, 4)
        self.assertEqual([f['fb_cols_sha256'] for f in a.per_frame],
                         [f['fb_cols_sha256'] for f in b.per_frame])
        self.assertEqual([f['exact_tokens'] for f in a.per_frame],
                         [f['exact_tokens'] for f in b.per_frame])

    def test_cost_regressions_exact_and_bounded(self):
        baselines = load_baselines()['scenarios']
        for name in scenario_definitions():
            with self.subTest(scenario=name):
                expected = baselines[name]['totals']
                run = self.scenario(name)
                self.assertLessEqual(run.instructions, expected['instructions'])
                self.assertLessEqual(run.exact_tokens, expected['exact_tokens'])
                max_expected = max(f['exact_tokens'] for f in baselines[name]['per_frame'])
                max_got = max(f['exact_tokens'] for f in run.per_frame)
                self.assertLessEqual(max_got, max_expected)

        full = self.scenario('autopilot_full_20')
        half = self.scenario('autopilot_half_20')
        self.assertLess(full.exact_tokens / full.frames_completed,
                        half.exact_tokens / half.frames_completed,
                        'the full-step build should retain its cost advantage over half-step')

    def test_dynamic_hot_loop_detector_and_positive_control(self):
        for kind in COST_BUILDS:
            with self.subTest(elf=f'asm-ray0-{kind}.elf'):
                run = run_exact(ROOT / 'build' / f'asm-ray0-{kind}.elf', 4)
                hot_events = [ev for ev in run.machine.forbidden_events if ev['hot']]
                self.assertEqual(hot_events, [])
                counts = mnemonic_counts(run.machine, hot_only=True)
                self.assertNotIn('div', counts)
                self.assertNotIn('divu', counts)
                self.assertNotIn('rem', counts)
                self.assertNotIn('remu', counts)
                # 16 aligned SW per column * 96 columns plus the frame-count commit per frame.
                self.assertEqual(counts.get('sw', 0), 4 * (16 * 96 + 1))
                self.assertNotIn('sb', counts)
                self.assertNotIn('sh', counts)

        # Positive control: the visual/debug build intentionally transposes with SB.  If the
        # detector does not fire here, the negative checks above would be vacuous.
        visual = run_exact(ROOT / 'build' / 'asm-ray0-vis.elf', 1)
        visual_hot = [ev for ev in visual.machine.forbidden_events if ev['hot']]
        self.assertTrue(any(ev['mnemonic'] == 'sb' for ev in visual_hot),
                        'detector positive control must catch the visual build SB transpose')

    def test_existing_measure_scripts_cross_check_exact_autopilot_cost(self):
        elf = ROOT / 'build' / 'asm-ray0-full.elf'
        exact = subprocess.run(
            [sys.executable, str(PORT_ASM_DIR / 'measure_exact_lx.py'), str(elf), '20'],
            text=True, capture_output=True, check=True).stdout
        approx = subprocess.run(
            [sys.executable, str(PORT_ASM_DIR / 'measure_ray.py'), str(elf), '20'],
            text=True, capture_output=True, check=True).stdout

        ex = re.search(r'instructions=(\d+) exact_lx_tokens=(\d+)', exact)
        ap = re.search(r'instructions=(\d+) tokens~=(\d+)', approx)
        self.assertIsNotNone(ex, exact)
        self.assertIsNotNone(ap, approx)
        baseline = load_baselines()['scenarios']['autopilot_full_20']['totals']
        self.assertEqual(int(ex.group(1)), baseline['instructions'],
                         'measure_exact_lx.py must independently reproduce the harness baseline')
        self.assertEqual(int(ex.group(2)), baseline['exact_tokens'],
                         'measure_exact_lx.py must independently reproduce the harness baseline')
        self.assertEqual(int(ex.group(1)), int(ap.group(1)),
                         'measure_ray.py instruction count must match lx.ref for this RV32IM subset')
        self.assertEqual(int(ex.group(2)), int(ap.group(2)),
                         'measure_ray.py token approximation must be exact for the autopilot probe')

    def test_framebuffer_memory_layout_after_exact_run(self):
        run = self.scenario('autopilot_full_20')
        raw, linear = framebuffer_bytes(run.machine.ram, run.symbols)
        self.assertEqual(len(raw), FB_COLS_SIZE)
        self.assertEqual(len(linear), 96 * 60)
        # The generated kernels write all 16 aligned words of every column, including the four
        # stride-padding rows.  A completely black/zero framebuffer or a missing render would
        # make these hashes trivially unstable across frames; require nonzero structure instead.
        self.assertGreaterEqual(len(set(raw)), 3, 'framebuffer should contain multiple palette indices')
        self.assertNotEqual(run.per_frame[0]['fb_cols_sha256'], run.per_frame[-1]['fb_cols_sha256'],
                            'autopilot frames should not collapse to a static image')

    def test_gpu_runner_cpu_self_test_and_static_config(self):
        runner = PORT_ASM_DIR / 'integration' / 'gpu' / 'run_ray0.py'
        self.assertTrue(runner.exists(), 'GPU runner is missing')
        subprocess.run([sys.executable, str(runner), '--self-test'],
                       cwd=ROOT, text=True, capture_output=True, check=True)
        out = subprocess.run([sys.executable, str(runner), '--print-config'],
                             cwd=ROOT, text=True, capture_output=True, check=True).stdout
        cfg = json.loads(out)
        self.assertEqual(cfg['ram_size'], 0x8000)
        self.assertEqual(cfg['text_end'], 0x2000)
        self.assertEqual(cfg['framebuffer']['width'], 96)
        self.assertEqual(cfg['framebuffer']['height'], 60)
        self.assertEqual(cfg['framebuffer']['stride'], 64)
        self.assertEqual(cfg['symbols']['qwen_frame_count']['offset'] & 3, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
