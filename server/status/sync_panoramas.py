#!/usr/bin/env python3
"""Mirror station equirect panoramas into the public status page cache."""

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CAMERAS_JSON = SCRIPT_DIR.parent / 'data' / 'cameras.json'
CACHE_DIR = SCRIPT_DIR / 'panoramas'
REMOTE_FILES = {
    'equirect': '/meteor/equirect.jpg',
    'equirect_hd': '/meteor/equirect_hd.jpg',
}
SAFE_NAME = re.compile(r'^[a-zA-Z0-9_-]+$')
SSH_COMMAND = (
    'ssh -o BatchMode=yes -o ConnectTimeout=8 '
    '-o ControlMaster=auto -o ControlPath=/tmp/nmn-panorama-%C '
    '-o ControlPersist=60'
)


def station_name(entry):
    name = (entry.get('station') or {}).get('name')
    return name if name and SAFE_NAME.fullmatch(name) else None


def rsync_image(station_id, image_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    remote = REMOTE_FILES[image_name]
    tmp = output_dir / f'.{image_name}.tmp'
    target = output_dir / f'{image_name}.jpg'
    cmd = [
        'rsync', '-rlpt', '--timeout=15',
        '-e', SSH_COMMAND,
        f'{station_id}:{remote}', str(tmp),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    if result.returncode != 0:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        detail = result.stderr.strip().splitlines()
        return False, detail[-1] if detail else f'rsync {result.returncode}'
    if not tmp.is_file() or tmp.stat().st_size <= 0:
        return False, 'empty file'
    os.replace(tmp, target)
    os.chmod(target, 0o644)
    return True, 'ok'


def sync_station(station_id, name, image_types):
    output_dir = CACHE_DIR / name
    results = {}
    for image_type in image_types:
        ok, detail = rsync_image(station_id, image_type, output_dir)
        results[image_type] = detail if ok else f'failed: {detail}'
    return station_id, name, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--types', nargs='+', choices=REMOTE_FILES,
                        default=list(REMOTE_FILES),
                        help='image types to synchronize')
    parser.add_argument('--stations', nargs='+',
                        help='station ids to synchronize (default: all)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='maximum concurrent station synchronizations')
    parser.add_argument('--cameras-json', type=Path, default=CAMERAS_JSON,
                        help='cameras.json path')
    args = parser.parse_args()

    cameras = json.loads(args.cameras_json.read_text())
    selected = set(args.stations or cameras.keys())
    tasks = []
    for station_id in sorted(selected):
        if station_id not in cameras or not SAFE_NAME.fullmatch(station_id):
            continue
        name = station_name(cameras[station_id])
        if name:
            tasks.append((station_id, name))

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(sync_station, station_id, name, args.types):
            (station_id, name) for station_id, name in tasks}
        for future in as_completed(futures):
            station_id, name, results = future.result()
            failed = {k: v for k, v in results.items()
                      if v.startswith('failed:')}
            if failed:
                failures.append((station_id, name, failed))
                print(f'{station_id}/{name}: {failed}')
            else:
                print(f'{station_id}/{name}: ok')

    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
    print(f'{stamp}: synced {len(tasks) - len(failures)}/{len(tasks)} stations')
    return 1 if len(failures) == len(tasks) and tasks else 0


if __name__ == '__main__':
    raise SystemExit(main())
