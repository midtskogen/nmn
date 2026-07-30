#!/usr/bin/env python3

import os
import json
import time
import logging
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone

def uniqid(prefix=''):
    """
    Generates a reasonably unique ID string by using the current high-resolution timestamp.
    This is useful for creating unique task IDs and filenames.
    
    Args:
        prefix (str): A string to prepend to the generated ID.
    
    Returns:
        str: The unique ID string.
    """
    return prefix + f"{time.time():.8f}".replace('.', '')


def atomic_json_write(file_path, data, indent=None):
    """
    Atomically writes data as JSON to a file by writing to a temporary file
    first and then renaming it into place. This ensures concurrent readers
    always see either the old or the new complete file, never a truncated one.
    """
    tmp_path = f"{file_path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=indent)
        os.rename(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass


def pid_cmdline_matches(pid, patterns):
    """
    Verifies that a process ID still belongs to an expected process by matching
    its /proc cmdline against the given substrings. Used to avoid killing an
    unrelated process that has reused a stale PID recorded in a state file.
    """
    try:
        with open(f'/proc/{int(pid)}/cmdline', 'rb') as f:
            cmdline = f.read().replace(b'\0', b' ').decode('utf-8', 'replace')
        return any(p in cmdline for p in patterns)
    except (IOError, OSError, ValueError):
        return False


def read_json_file(file_path, default=None):
    """
    Tolerantly reads a JSON file. Returns `default` if the file is missing
    or contains invalid JSON, so a corrupt cache never crashes the caller.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        logging.warning(f"Could not read JSON file {os.path.basename(file_path)}: {e}")
        return default


def update_status(status_file, status, data={}):
    """
    Writes a status update to a JSON file for a given task.
    This allows the frontend to poll for the progress and result of a long-running background process.

    Args:
        status_file (str): The full path to the status JSON file.
        status (str): The current status (e.g., 'progress', 'complete', 'error').
        data (dict): A dictionary of additional data to include in the status file.
    """
    if status_file:
        try:
            atomic_json_write(status_file, {"status": status, **data})
        except IOError as e:
            logging.error(f"Could not write to status file {status_file}: {e}")


@contextmanager
def atomic_json_rw(file_path, task_id_for_log=""):
    """
    A context manager to safely (atomically) read and write to a JSON file using file locks.
    This prevents race conditions where multiple processes might try to access the same file
    simultaneously, which is common for quota and tracker files.
    
    Args:
        file_path (str): The path to the JSON file to be accessed.
        task_id_for_log (str): An optional task ID for more descriptive logging.
        
    Yields:
        dict: The data read from the JSON file. This data can be modified within the
              'with' block, and it will be automatically written back upon exit.
    """
    lock_path = file_path + '.lock'

    lock_file_handle = None
    try:
        # Acquire an exclusive lock on the associated .lock file. The lock file
        # is persistent: it must never be unlinked, because a process blocked on
        # flock() of an unlinked inode would hold a lock on a deleted file while
        # a new process locks a freshly created inode (split-brain writers).
        # flock() is released automatically by the OS if the holder dies, so no
        # stale-lock cleanup is needed.
        lock_file_handle = open(lock_path, 'a')
        fcntl.flock(lock_file_handle, fcntl.LOCK_EX)
        
        data = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If the JSON file is corrupt, start fresh with an empty dictionary.
                    logging.warning(f"Task {task_id_for_log} - {os.path.basename(file_path)} is corrupt, creating a new one.")
        
        # Yield the loaded data to the `with` block for modification.
        yield data

        # After the `with` block completes, write the (potentially modified) data back to the file.
        atomic_json_write(file_path, data, indent=2)
            
    except Exception as e:
        logging.error(f"Task {task_id_for_log} - FAILED to perform atomic update on {os.path.basename(file_path)}: {e}")
        raise
    finally:
        # Always release the lock on exit, even if errors occurred. The lock
        # file itself is left in place (see comment above).
        if lock_file_handle:
            fcntl.flock(lock_file_handle, fcntl.LOCK_UN)
            lock_file_handle.close()


def update_quota_tracker(updates, task_id, user_ip, quota_tracker_file):
    """
    Updates the daily download quota tracker with the amount of data downloaded.
    It logs usage for each station and tracks it against both a total daily limit
    for the station and a per-user-IP limit for that station.
    
    Args:
        updates (dict): A dictionary mapping station_id to bytes_downloaded.
        task_id (str): The ID of the master task for logging.
        user_ip (str): The IP address of the user who initiated the download.
        quota_tracker_file (str): Path to the JSON file that stores quota usage.
    """
    if not updates:
        return
    logging.info(f"Task {task_id} - Updating quota for IP {user_ip} with: {updates}")
    with atomic_json_rw(quota_tracker_file, task_id) as tracker:
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today_str not in tracker:
            tracker[today_str] = {}

        for station_id, bytes_downloaded in updates.items():
            station_usage = tracker[today_str].get(station_id, {})
            
            # This handles migration from a previous, simpler data format if encountered.
            if isinstance(station_usage, int):
                station_usage = {"total": station_usage, "sites": {}}
            
            # Increment the total bytes downloaded for the station for the day.
            station_usage["total"] = station_usage.get("total", 0) + bytes_downloaded
            if "sites" not in station_usage:
                station_usage["sites"] = {}
            # Increment the total bytes downloaded by this specific user IP for this station.
            station_usage["sites"][user_ip] = station_usage["sites"].get(user_ip, 0) + bytes_downloaded
            tracker[today_str][station_id] = station_usage


def trim_log_file(log_path, max_lines, task_id):
    """
    Trims a log file to a maximum number of lines, keeping the most recent lines at the end.
    This prevents log files from growing indefinitely.
    
    Args:
        log_path (str): The path to the log file to trim.
        max_lines (int): The maximum number of lines to keep.
        task_id (str): The task ID for logging the trim action.
    """
    try:
        if not os.path.exists(log_path): return
        with open(log_path, 'r') as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            logging.info(f"Task {task_id} - Trimming log file {os.path.basename(log_path)} from {len(lines)} to {max_lines} lines.")
            with open(log_path, 'w') as f:
                f.writelines(lines[-max_lines:])
    except Exception as e:
        logging.error(f"Task {task_id} - Could not trim log file {log_path}: {e}")


def cleanup_old_files(directory, age_in_days, task_id, files_to_skip=[]):
    """
    Deletes files in a specified directory that are older than a given number of days.
    
    Args:
        directory (str): The path to the directory to clean up.
        age_in_days (int): Files older than this will be deleted.
        task_id (str): The task ID for logging the cleanup action.
        files_to_skip (list): A list of filenames to ignore and not delete.
    """
    logging.info(f"Task {task_id} - Running cleanup in '{directory}' for files older than {age_in_days} days.")
    now = time.time()
    age_limit_seconds = age_in_days * 86400
    try:
        for filename in os.listdir(directory):
            if filename in files_to_skip:
                continue
            file_path = os.path.join(directory, filename)
            # Check if the file's modification time is older than the limit.
            if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > age_limit_seconds:
                try:
                    os.remove(file_path)
                    logging.info(f"Task {task_id} - Deleted old file: {file_path}")
                except OSError as e:
                    logging.error(f"Task {task_id} - Error deleting file {file_path}: {e}")
    except FileNotFoundError:
        logging.warning(f"Task {task_id} - Cleanup directory not found: {directory}")
