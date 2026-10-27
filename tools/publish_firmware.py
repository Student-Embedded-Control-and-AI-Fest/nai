#!/usr/bin/env python3
"""Publish a built ESP32-S3 merged factory .bin in the NoodleAI GitHub Pages site.

Usage:
    python3 tools/publish_firmware.py /path/to/noodleai-esp32s3-int8-factory.bin

Run this from any directory. The destination is the web/firmware directory of
this script's project; commit the binary and manifest together.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

DEST_NAME = 'noodleai-esp32s3-int8-factory.bin'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, help='merged ESP32-S3 factory binary')
    parser.add_argument('--version', help='optional human-readable firmware version')
    args = parser.parse_args()
    src = args.binary.expanduser().resolve()
    if not src.is_file():
        parser.error(f'firmware not found: {src}')
    with src.open('rb') as inp:
        first = inp.read(4)
    size = src.stat().st_size
    if first[:1] != b'\xe9' or size < 65536:
        parser.error('not a plausible merged ESP32 boot image (0xE9 header and at least 64 KiB required)')
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    firmware_dir = Path(__file__).resolve().parents[1] / 'firmware'
    firmware_dir.mkdir(parents=True, exist_ok=True)
    target = firmware_dir / DEST_NAME
    if src != target.resolve():
        shutil.copy2(src, target)
    manifest = {
        'name': 'NoodleAI Mode 1 ESP32-S3 Matrix INT8',
        'version': args.version or f'int8-{sha[:10]}',
        'new_install_prompt_erase': True,
        'new_install_improv_wait_time': 0,
        'builds': [
            {
                'chipFamily': 'ESP32-S3',
                'improv': False,
                'parts': [{'path': DEST_NAME, 'offset': 0}],
            }
        ],
    }
    (firmware_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Copied: {target} ({size:,} bytes)')
    print(f'SHA256: {sha}')
    print(f'Updated: {firmware_dir / "manifest.json"}')
    print('Next: commit the binary + manifest and push to the GitHub Pages branch.')
    print('Note: Verify this is the correct board/flash-layout and INT8 build before publishing.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
