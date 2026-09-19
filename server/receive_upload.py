#!/usr/bin/env python3
"""Receives a station-uploaded event archive (the push side of report.py).

upload.php streams the POSTed tar.gz into QUARANTINE_DIR and invokes this
script.  Archive members are strictly validated (regular files and dirs
only, relative paths, no '..', the same extension denylist as the rsync
pull, file-count and size caps), staged, copied into the event tree, then
merged and processed exactly like an rsync pull so downstream behaviour is
identical.
"""
import datetime
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch  # noqa: E402  (Config, find_and_merge_event_directory, ...)
from fetch import Config  # noqa: E402

QUARANTINE_DIR = Path('/var/www/incoming')
MAX_TOTAL_BYTES = 400 * 1024 * 1024
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_FILES = 500
MAX_MEMBERS = 1000

# Mirrors the rsync --exclude list used by the pull path in fetch.py, plus a
# couple of extra server-executable extensions for defence in depth.
DENY_BASENAMES = {'.htaccess', '.htpasswd'}
DENY_GLOBS = ('frame-*',)
DENY_SUFFIXES = ('.pkl', '.php', '.phar', '.cgi', '.pl', '.shtml',
                 '.py', '.pyc', '.pyo', '.sh', '.phtml', '.htaccess')
SAFE_NAME = re.compile(r'^[A-Za-z0-9_./+@=-]+$')


def _member_ok(member: tarfile.TarInfo) -> bool:
    name = member.name
    if not name or name.startswith('/') or '..' in name.split('/'):
        return False
    if not SAFE_NAME.match(name):
        return False
    if member.islnk() or member.issym() or member.isdev():
        return False
    if not (member.isfile() or member.isdir()):
        return False
    if member.isfile():
        base = name.rsplit('/', 1)[-1]
        if base in DENY_BASENAMES or any(base.startswith(g[:-1]) for g in DENY_GLOBS):
            return False
        low = base.lower()
        if '.pht' in low or any(low.endswith(s) for s in DENY_SUFFIXES):
            return False
    return True


def extract_safely(archive: Path, dest: Path) -> int:
    """Extract a tar.gz into dest. Returns the number of files written."""
    count = 0
    total = 0
    with tarfile.open(archive, 'r:*') as tf:
        members = tf.getmembers()
        if len(members) > MAX_MEMBERS:
            raise ValueError(f'archive has {len(members)} members (> {MAX_MEMBERS})')
        for m in members:
            if not _member_ok(m):
                raise ValueError(f'unsafe member: {m.name!r}')
            if m.isfile():
                total += m.size
                count += 1
                if m.size > MAX_FILE_BYTES:
                    raise ValueError(f'member too large: {m.name!r} ({m.size} bytes)')
                if total > MAX_TOTAL_BYTES:
                    raise ValueError('archive too large')
                if count > MAX_FILES:
                    raise ValueError('too many files')
        tf.extractall(dest, filter='data')
    return count


def main():
    if len(sys.argv) != 4:
        sys.exit(f'Usage: {sys.argv[0]} <archive.tar.gz> <station> <remote_dir>')
    archive = Path(sys.argv[1])
    station = re.sub(r'\W', '', sys.argv[2])
    remote_cleaned = re.sub(r'[^a-zA-Z0-9_/\-]', '', sys.argv[3])

    try:
        parts = [p for p in remote_cleaned.split('/') if p]
        if parts[0] != 'meteor' or not re.fullmatch(r'cam\d+', parts[1]) \
                or parts[2] != 'amsevents' \
                or not re.fullmatch(r'\d{8}', parts[3]) \
                or not re.fullmatch(r'\d{6}(_\d+)?', parts[4]):
            raise IndexError
        cam_name, date_str, time_str = parts[1], parts[3], parts[4]
    except IndexError:
        sys.exit(f'Bad remote dir: {remote_cleaned}')

    log_path = QUARANTINE_DIR / 'receive_upload.log'
    fetch.setup_logging(log_path)
    logging.info(f'Received upload: station={station} dir={remote_cleaned} '
                 f'archive={archive.name}')

    staging = Path(tempfile.mkdtemp(prefix='up_', dir=QUARANTINE_DIR))
    try:
        nfiles = extract_safely(archive, staging)
        logging.info(f'Validated {nfiles} files from {archive.name}')

        local_dir = Config.METEOR_DATA_DIR / date_str / time_str / station / cam_name
        local_dir.mkdir(parents=True, exist_ok=True)
        for f in staging.rglob('*'):
            if f.is_file():
                rel = f.relative_to(staging)
                d = local_dir / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(d))  # quarantine may be another fs
        logging.info(f'Copied {nfiles} files into {local_dir}')

        try:
            fetch.set_permissions(Config.METEOR_DATA_DIR / date_str)
        except Exception as e:
            logging.warning(f'Failed to set permissions: {e}')

        date_obj = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        fetch.create_centroid_file(local_dir)
        final_event_dir = fetch.find_and_merge_event_directory(
            date_obj, station, local_dir)

        fetch.setup_logging(final_event_dir / 'process.log')
        logging.info(f'--- Event merged into {final_event_dir}. '
                     'Starting main processing (push path). ---')
        Config.log_paths()
        proc = datetime.datetime.strptime(
            final_event_dir.parent.name + final_event_dir.name, '%Y%m%d%H%M%S')
        fetch.process_event_with_lock(final_event_dir, proc)
        logging.info('--- Script finished. ---')
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
