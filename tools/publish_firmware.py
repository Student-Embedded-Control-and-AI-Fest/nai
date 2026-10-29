#!/usr/bin/env python3
"""Publish a NoodleAI INT8 merged factory image without losing other boards.

Examples (from nai/web):
  python3 tools/publish_firmware.py --chip esp32 ../firmware/.pio/build/esp32-mpu6050-int8/noodleai-esp32-int8-factory.bin
  python3 tools/publish_firmware.py --chip esp32s3 ../firmware/.pio/build/esp32-s3-matrix-int8/noodleai-esp32s3-int8-factory.bin

Accept only merged factory images at *flash offset 0*, with chip-specific
bootloader placement. The manifest is shared by ESP32 and ESP32-S3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

TARGETS = {
    'esp32': {'chipFamily': 'ESP32', 'filename': 'noodleai-esp32-int8-factory.bin', 'bootOffset': 0x1000},
    'esp32s3': {'chipFamily': 'ESP32-S3', 'filename': 'noodleai-esp32s3-int8-factory.bin', 'bootOffset': 0x0},
}
MANIFEST_NAME = 'NoodleAI INT8 — ESP32 / ESP32-S3'


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_image(path: Path, chip: str) -> None:
    size = path.stat().st_size
    if not 0x10001 < size <= 4 * 1024 * 1024:
        raise ValueError(f'factory image size {size:,} bytes is out of range (need 4 MB target)')
    with path.open('rb') as inp:
        checks = [(TARGETS[chip]['bootOffset'], b'\xe9', 'bootloader'),
                  (0x8000, b'\xaa\x50', 'partition table'),
                  (0x10000, b'\xe9', 'application')]
        for offset, expected, label in checks:
            inp.seek(offset)
            if inp.read(len(expected)) != expected:
                raise ValueError(f'{label} header mismatch at 0x{offset:x}: check chip and merge offsets')
    # This validates the image *layout*, not its board-specific sensors or build flags.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, help='merged ESP32 / ESP32-S3 factory binary')
    parser.add_argument('--chip', required=True, choices=list(TARGETS), help='target chip, not the sensor board')
    parser.add_argument('--version', help='optional explicit manifest version')
    args = parser.parse_args()

    src = args.binary.expanduser().resolve()
    if not src.is_file():
        parser.error(f'file does not exist: {src}')
    try:
        validate_image(src, args.chip)
    except ValueError as err:
        parser.error(str(err))

    out_dir = Path(__file__).resolve().parents[1] / 'firmware'
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / TARGETS[args.chip]['filename']
    if src != target.resolve():
        shutil.copy2(src, target)

    manifest_file = out_dir / 'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            parser.error('existing manifest must be a JSON object')
    else:
        manifest = {}
    builds = manifest.get('builds', [])
    if not isinstance(builds, list):
        parser.error('existing manifest builds must be an array')

    entry = {
        'chipFamily': TARGETS[args.chip]['chipFamily'],
        'improv': False,
        'parts': [{'path': target.name, 'offset': 0}],
    }
    # Replace only the selected chip entry, while retaining others.
    builds = [item for item in builds if not (isinstance(item, dict) and item.get('chipFamily') == entry['chipFamily'])]
    builds.append(entry)
    order = {'ESP32-S3': 0, 'ESP32': 1}
    builds.sort(key=lambda item: order.get(item.get('chipFamily'), 9))

    # Fingerprint the collection of available files for a stable shared version.
    combined = hashlib.sha256()
    for chip in ('esp32s3', 'esp32'):
        path = out_dir / TARGETS[chip]['filename']
        if path.is_file():
            combined.update(chip.encode('ascii'))
            combined.update(bytes.fromhex(sha_file(path)))
    version = args.version or 'int8-' + combined.hexdigest()[:10]

    manifest['name'] = MANIFEST_NAME
    manifest['version'] = version
    manifest['new_install_prompt_erase'] = True
    manifest['new_install_improv_wait_time'] = 0
    manifest['builds'] = builds
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'Published {entry["chipFamily"]}: {target} ({target.stat().st_size:,} bytes)')
    print(f'SHA256: {sha_file(target)}')
    print(f'Manifest: {manifest_file} ({version})')
    for chip in TARGETS.values():
        image_path = out_dir / chip['filename']
        print(f'  {chip["chipFamily"]}: {"ready" if image_path.is_file() else "MISSING (copy/publish binary before web install)"}')
    print('Verify both images and board wiring before publishing to GitHub Pages.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
