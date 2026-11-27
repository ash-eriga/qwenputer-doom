#!/usr/bin/env python3
"""One-command CPU-only verification for port-asm.

Default behavior builds the three probe ELFs and runs the deterministic unittest suite.  Use
``--no-build`` to verify the existing artifacts exactly as they are, or ``--update-baseline``
after an intentional renderer/cost change to regenerate the exact golden vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
PORT_ASM_DIR = VERIFY_DIR.parent
ROOT = PORT_ASM_DIR.parent
if str(VERIFY_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIR))

from lx_harness import run_scenario, scenario_definitions  # noqa: E402

BASELINE_PATH = VERIFY_DIR / 'baselines.json'


def build_probes() -> None:
    subprocess.run(['bash', str(PORT_ASM_DIR / 'build.sh')], cwd=ROOT, check=True)


def write_baseline() -> None:
    scenarios = {}
    for name, definition in scenario_definitions().items():
        run = run_scenario(name, definition)
        elf_path = Path(definition['elf'])
        scenarios[name] = {
            'description': definition['description'],
            'elf': str(elf_path.relative_to(ROOT)),
            'elf_sha256_when_generated': hashlib.sha256(elf_path.read_bytes()).hexdigest(),
            'script': str(Path(definition['script']).relative_to(ROOT)) if definition.get('script') else None,
            **run.baseline_view(),
        }
    payload = {
        'schema': 1,
        'semantics': 'lx.ref.Machine exact CPU-only token stream; frame boundary is the committed qwen_frame_count store',
        'scenarios': scenarios,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'wrote {BASELINE_PATH}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-build', action='store_true', help='do not rebuild; verify current build artifacts')
    ap.add_argument('--update-baseline', action='store_true', help='regenerate baselines.json from current artifacts')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if not args.no_build:
        build_probes()
    if args.update_baseline:
        write_baseline()
        return 0

    suite = unittest.defaultTestLoader.loadTestsFromName('test_asm_probe')
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
