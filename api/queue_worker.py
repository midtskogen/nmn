#!/usr/bin/env python3
"""
Queue worker daemon for CPU-intensive API tasks.

Keeps the number of concurrent CPU-intensive jobs at or below
max(1, floor(cpu_count / 2)) and avoids starting new jobs when the 1-minute
load average is already above that threshold.  Jobs are read from
nmn/api/task_queue.jsonl; status files are written to the shared locks dir.
"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent.parent
SECRETS_DIR = PROJECT_DIR / 'etc'
DATA_DIR = Path(os.environ.get('NMN_DATA_DIR', PROJECT_DIR / 'data')).resolve()
LOCK_DIR = Path(os.environ.get('NMN_LOCK_DIR', DATA_DIR / 'locks')).resolve()
QUEUE_FILE = API_DIR / 'task_queue.jsonl'
PID_FILE = API_DIR / 'queue_worker.pid'
CONFIG_FILE = SECRETS_DIR / 'api_config.json'
os.environ.setdefault('NMN_CONFIG_FILE', str(SECRETS_DIR / 'config.json'))
os.environ.setdefault('NMN_CREDENTIALS_FILE', str(SECRETS_DIR / 'credentials.json'))

# Let the existing Python logging module handle our own messages.
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('queue_worker')


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_config_cache = {'mtime': None, 'data': {}, 'warned': False}


def load_config():
    """Read api_config.json, cached by mtime.

    Called several times per main-loop iteration; without caching a missing
    or unreadable config produced a warning every call and filled the log.
    """
    try:
        mtime = os.path.getmtime(CONFIG_FILE)
    except OSError:
        mtime = None
    if mtime == _config_cache['mtime'] and _config_cache['mtime'] is not None:
        return _config_cache['data']
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
        _config_cache.update({'mtime': mtime, 'data': data, 'warned': False})
        return data
    except Exception as e:
        if not _config_cache['warned']:
            logger.warning('Could not read %s: %s', CONFIG_FILE, e)
            _config_cache['warned'] = True
        _config_cache['mtime'] = mtime
        return _config_cache['data']


def _cpu_count():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 2
    except Exception:
        return os.cpu_count() or 2


def max_workers():
    cpus = _cpu_count()
    cfg = load_config().get('queue', {})
    factor = float(cfg.get('max_workers_factor', 0.5))
    return max(1, int(cpus * factor))


def prediction_workers():
    """Number of worker processes each queued prediction may use internally."""
    return max_workers()


def max_load():
    cfg = load_config().get('queue', {})
    factor = float(cfg.get('max_load_factor', 0.5))
    try:
        import psutil
        cpus = psutil.cpu_count(logical=False)
    except Exception:
        cpus = os.cpu_count() or 2
    return cpus * factor


def job_timeout():
    return int(load_config().get('queue', {}).get('job_timeout_seconds', 1800))


def cleanup_age():
    return int(load_config().get('queue', {}).get('cleanup_age_hours', 24)) * 3600


# ---------------------------------------------------------------------------
# Queue file helpers (using POSIX advisory locks, compatible with PHP flock)
# ---------------------------------------------------------------------------
def _read_queue():
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            jobs = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return jobs
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _write_queue(jobs):
    text = ''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in jobs)
    tmp = QUEUE_FILE.with_suffix(QUEUE_FILE.suffix + '.tmp.' + _uniq())
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(QUEUE_FILE)


def _atomic_update(updater):
    """Read queue, apply updater(jobs)->jobs, write back under exclusive lock."""
    with open(QUEUE_FILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            jobs = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            jobs = updater(jobs)
            f.truncate(0)
            f.seek(0)
            f.write(''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in jobs))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _uniq():
    return '{:06x}{:06x}'.format(int(time.time() * 1000) & 0xffffff, os.getpid() & 0xffffff)


# ---------------------------------------------------------------------------
# Status file helper
# ---------------------------------------------------------------------------
def update_status(task_id, status, **data):
    status_file = LOCK_DIR / f'{task_id}.json'
    try:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {'status': status}
        payload.update(data)
        tmp = status_file.with_suffix(status_file.suffix + '.tmp.' + _uniq())
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        tmp.replace(status_file)
    except Exception as e:
        logger.error('Could not write status for %s: %s', task_id, e)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------
def claim_next_pending_job():
    claimed = None

    def updater(jobs):
        nonlocal claimed
        for j in jobs:
            if j.get('status') == 'pending' and not claimed:
                claimed = dict(j, status='running', started_ts=int(time.time()), pid=os.getpid())
                j.update(claimed)
                break
        return jobs

    _atomic_update(updater)
    return claimed


def set_job_done(task_id, exit_code, error=None):
    def updater(jobs):
        for j in jobs:
            if j.get('task_id') == task_id and j.get('status') == 'running':
                j['status'] = 'error' if (exit_code != 0 or error) else 'done'
                j['finished_ts'] = int(time.time())
                j['exit_code'] = exit_code
                if error:
                    j['error'] = error
        return jobs
    _atomic_update(updater)


def reset_stale_jobs(timeout):
    now = int(time.time())

    def updater(jobs):
        for j in jobs:
            if j.get('status') == 'running':
                started = j.get('started_ts') or j.get('created_ts') or now
                pid = j.get('pid')
                alive = False
                if pid:
                    try:
                        os.kill(int(pid), 0)
                        alive = True
                    except OSError:
                        pass
                if not alive and (now - started) > 60:
                    j['status'] = 'pending'
                    j['started_ts'] = None
                    j['pid'] = None
                    j['error'] = 'worker_died'
                    logger.warning('Reset stale job %s to pending', j.get('task_id'))
                elif alive and (now - started) > timeout:
                    # Leave status as running; the process is still alive but suspicious.
                    logger.warning('Job %s has been running for > %ss', j.get('task_id'), timeout)
        return jobs

    _atomic_update(updater)


def cleanup_old_jobs(age):
    now = int(time.time())

    def updater(jobs):
        keep = []
        for j in jobs:
            if j.get('status') in ('done', 'error'):
                finished = j.get('finished_ts') or j.get('started_ts') or j.get('created_ts') or now
                if (now - finished) <= age:
                    keep.append(j)
            else:
                keep.append(j)
        return keep

    before = len(_read_queue())
    _atomic_update(updater)
    after = len(_read_queue())
    if before != after:
        logger.info('Cleaned up %s finished jobs from queue', before - after)


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------
def build_command(action, args):
    py = '/usr/bin/python3'
    task_id = args.get('task_id', '')
    if action == 'find_passes':
        cmd = [py, str(DATA_DIR / 'predict_sat.py'), task_id]
        if args.get('station'):
            cmd += ['--station', args['station']]
        if args.get('days'):
            cmd += ['--days', str(args['days'])]
        if args.get('start'):
            cmd += ['--start', args['start']]
        if args.get('end'):
            cmd += ['--end', args['end']]
        return cmd
    if action == 'find_aircraft_crossings':
        cmd = [py, str(DATA_DIR / 'predict_flight.py'), task_id]
        if args.get('station'):
            cmd += ['--station', args['station']]
        if args.get('days'):
            cmd += ['--days', str(args['days'])]
        if args.get('start'):
            cmd += ['--start', args['start']]
        if args.get('end'):
            cmd += ['--end', args['end']]
        return cmd
    if action == 'enhance_filter':
        return [py, str(DATA_DIR / 'controller.py'), 'enhance_filter', str(args.get('image', '')), str(int(args.get('filter', 0)))]
    raise ValueError(f'Unknown queued action: {action}')


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------
_active = {}
_shutdown = threading.Event()


def run_job(job):
    task_id = job['task_id']
    action = job['action']
    args = job.get('args', {})
    if isinstance(args, list) and not isinstance(args, dict):
        # Normalise legacy list args to dict if needed.
        args = {}
    args.setdefault('task_id', task_id)

    try:
        cmd = build_command(action, args)
    except Exception as e:
        logger.exception('Bad command for %s', task_id)
        update_status(task_id, 'error', message=str(e))
        set_job_done(task_id, 1, str(e))
        return

    update_status(task_id, 'running', message='status_calculating')
    logger.info('Starting %s job %s: %s', action, task_id, ' '.join(cmd))

    # Constrain each child to roughly the same CPU budget so that running a few
    # queued CPU jobs in parallel does not push the 1-minute load above the
    # target threshold.
    child_env = os.environ.copy()
    child_env['NMN_MAX_WORKERS'] = str(prediction_workers())
    child_env.setdefault('OMP_NUM_THREADS', '2')
    child_env.setdefault('OPENBLAS_NUM_THREADS', '2')
    child_env.setdefault('MKL_NUM_THREADS', '2')
    child_env.setdefault('NUMBA_NUM_THREADS', '2')

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
        )
        _active[task_id] = proc
        try:
            stdout, stderr = proc.communicate(timeout=job_timeout())
        finally:
            _active.pop(task_id, None)

        if proc.returncode != 0:
            err = (stderr or 'unknown error').strip()[:500]
            update_status(task_id, 'error', message='error_internal', debug=err)
            set_job_done(task_id, proc.returncode, err)
            logger.error('Job %s failed with code %s: %s', task_id, proc.returncode, err)
        else:
            # For enhance_filter the Python script prints the JSON result.
            if action == 'enhance_filter':
                try:
                    data = json.loads(stdout)
                    update_status(task_id, 'complete', data=data)
                except json.JSONDecodeError:
                    update_status(task_id, 'complete', image=stdout.strip())
            set_job_done(task_id, 0)
            logger.info('Job %s completed', task_id)
    except subprocess.TimeoutExpired:
        proc.kill()
        update_status(task_id, 'error', message='error_timeout')
        set_job_done(task_id, -1, 'timeout')
        logger.error('Job %s timed out', task_id)
    except Exception as e:
        logger.exception('Job %s crashed', task_id)
        update_status(task_id, 'error', message='error_internal', debug=str(e))
        set_job_done(task_id, -1, str(e))


def _signal_handler(signum, frame):
    logger.info('Received signal %s, shutting down gracefully...', signum)
    _shutdown.set()


def main():
    if '--dry-run' in sys.argv:
        config = load_config()
        assert DATA_DIR.is_dir(), f'Data directory not found: {DATA_DIR}'
        assert LOCK_DIR.is_dir(), f'Lock directory not found: {LOCK_DIR}'
        assert CONFIG_FILE.is_file(), f'API config not found: {CONFIG_FILE}'
        assert isinstance(config, dict), 'API config is not an object'
        print(f'queue_worker dry-run OK: data={DATA_DIR} config={CONFIG_FILE}')
        return

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    PID_FILE.write_text(str(os.getpid()))
    logger.info('Queue worker started (pid=%s, max_workers=%s, max_load=%.1f)',
                os.getpid(), max_workers(), max_load())

    last_cleanup = 0
    while not _shutdown.is_set():
        try:
            reset_stale_jobs(job_timeout())

            # Periodic cleanup of finished jobs (every ~60s).
            now = time.time()
            if now - last_cleanup > 60:
                cleanup_old_jobs(cleanup_age())
                last_cleanup = now

            active_count = sum(1 for p in _active.values() if p.poll() is None)
            load_ok = True
            try:
                load = os.getloadavg()[0]
                load_ok = load < max_load()
                if not load_ok:
                    logger.debug('Load %.2f >= %.2f, waiting', load, max_load())
            except OSError:
                pass

            if active_count < max_workers() and load_ok:
                job = claim_next_pending_job()
                if job:
                    t = threading.Thread(target=run_job, args=(job,), daemon=True)
                    t.start()

            time.sleep(2)
        except Exception:
            logger.exception('Main loop error')
            time.sleep(5)

    logger.info('Waiting for active jobs to finish...')
    deadline = time.time() + 30
    while _active and time.time() < deadline:
        time.sleep(1)
    for p in _active.values():
        try:
            p.terminate()
        except Exception:
            pass
    logger.info('Queue worker stopped')


if __name__ == '__main__':
    main()
