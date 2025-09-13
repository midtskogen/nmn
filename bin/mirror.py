#!/usr/bin/env python3
"""
Monitors directories for new video and data files from AMS (Allsky7 Meteor System).

This script performs three main tasks concurrently:
1.  Watches for new one-minute video files (.mp4) and processes them.
2.  Watches for new meteor detection data files (*-reduced.json) and passes them
    to an external script for event processing.
3.  Watches for changes to itself and restarts the service if updated.
"""

import inotify.adapters
import argparse
import logging
from logging.handlers import RotatingFileHandler
import re
import os
import sys
import asyncio
import datetime
import tempfile
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import stack  # For direct JPG generation

# --- Constants for Configuration ---
AMS2EVENT_PY_DEFAULT = "/home/meteor/bin/ams2event.py"
METEORS_DIR_NAME = "meteors"
REDUCED_JSON_SUFFIX = "-reduced.json"

# --- Self-Restarting Logic ---
# Store the script's modification time at startup
_SELF_MOD_TIME = os.path.getmtime(__file__)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Monitor AMS directories for new files.')
parser.add_argument('-l', '--logfile', dest='logfile', help='logfile')
parser.add_argument('-d', '--delay', dest='videodelay', help='video delay in ms (default=150)', type=int, default=150)
parser.add_argument('-L', '--link', dest='link', help='create link instead of copy', action="store_true")
parser.add_argument('-r', '--remotedir', dest='remotedir', help='destination directory', default='/meteor')
parser.add_argument('-e', '--execute', dest='exefile', help=f'program to pass event file to (default: "{AMS2EVENT_PY_DEFAULT}")', default=AMS2EVENT_PY_DEFAULT)
parser.add_argument(action='store', dest='recdir', help='directory to be monitored recursively (for detections)')
parser.add_argument(nargs='*', action='store', dest='dirs', help='additional directories to be monitored non-recursively (for video files)')
args = parser.parse_args()

# --- Logging Configuration ---
# Set up a rotating log file handler to limit the log size.
# An average log line is ~100 bytes. 100,000 lines * 100 bytes/line = 10 MB.
LOG_MAX_BYTES = 10000000
LOG_BACKUP_COUNT = 5

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

if args.logfile:
    # Create a rotating file handler if a logfile is specified
    handler = RotatingFileHandler(
        args.logfile,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
else:
    # If no logfile is specified, log to the console (stderr)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Global State ---
processed_files = deque(maxlen=200)
FILENAME_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\w+)_(\w+)\.mp4")

# Create a process pool executor to isolate the stack.py call.
# This prevents it from crashing the main script or leaking memory into it.
process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())


# --- Asynchronous Helper for Blocking Notifier ---
async def async_event_gen(inotify_iterator):
    """Wraps a blocking iterator in an executor to make it async."""
    loop = asyncio.get_running_loop()
    while True:
        event = await loop.run_in_executor(None, next, inotify_iterator)
        yield event

async def run_command_async(*command):
    """Asynchronously runs a command via exec and logs its output."""
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        cmd_str = ' '.join(command)
        logging.error(f"Command failed with exit code {proc.returncode}: {cmd_str}")
        if stdout:
            logging.error(f"[stdout]\n{stdout.decode()}")
        if stderr:
            logging.error(f"[stderr]\n{stderr.decode()}")
    return proc.returncode

def stack_wrapper(*args, **kwargs):
    """
    Wrapper for stack.stack_video_frames that redirects its stdout to /dev/null
    and suppresses its INFO-level log messages.
    """
    # This function runs in a child process and inherits the parent's logging config.
    # To prevent stack.py's INFO messages from cluttering mirror.log, we raise
    # the logging level for the duration of this call to only show warnings/errors.
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    # Also redirect stdout to the null device to silence any print() statements.
    with open(os.devnull, 'w') as f_null:
        original_stdout = sys.stdout
        sys.stdout = f_null
        try:
            # The 'stack' module is available here because the worker process
            # inherits loaded modules from the parent.
            return stack.stack_video_frames(*args, **kwargs)
        finally:
            # Restore stdout and logging level to their original state.
            sys.stdout = original_stdout
            root_logger.setLevel(original_level)

# --- Core Watcher Functions ---
async def watch_videos():
    """Monitors directories for new video files, processes, and moves them."""
    logging.info('Video watcher started.')
    i = inotify.adapters.Inotify()
    for d in args.dirs:
        i.add_watch(d, mask=inotify.constants.IN_CLOSE_WRITE | inotify.constants.IN_MOVED_TO)
    
    event_iterator = iter(i.event_gen(yield_nones=False))
    async for event in async_event_gen(event_iterator):
        try:
            (_, _, eventdir, eventfile) = event
            filepath = os.path.join(eventdir, eventfile)

            if not os.path.exists(filepath):
                continue

            if filepath in processed_files:
                continue

            if not eventfile.endswith('.mp4'):
                continue

            match = FILENAME_PATTERN.match(eventfile)
            if not match:
                logging.info(f"Skipping file with unexpected name format: {filepath}")
                continue
            
            id_str = match.groups()[-1]

            if os.path.getsize(filepath) == 0:
                logging.info(f"Skipping zero-byte file: {filepath}")
                continue

            # --- Timestamp Calculation ---
            (year, month, day, hour, minute, second) = match.groups()[:6]
            dt_from_name = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            try:
                file_stat = os.stat(filepath)
                file_time = datetime.datetime.fromtimestamp(file_stat.st_birthtime)
            except (AttributeError, FileNotFoundError):
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
            final_dt = dt_from_name.replace(microsecond=file_time.microsecond)
            final_dt = final_dt - datetime.timedelta(milliseconds=args.videodelay)
            offset = final_dt.timestamp()
            creation_time_iso = final_dt.isoformat(timespec='microseconds') + "Z"

            # --- FFmpeg Timestamp Injection ---
            modified_file = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_f:
                    modified_file = temp_f.name
                ffmpeg_cmd_list = [
                    'ffmpeg', '-y', '-i', filepath, '-c', 'copy', '-movflags',
                    '+faststart', '-metadata', f"creation_time={creation_time_iso}",
                    '-output_ts_offset', str(offset), modified_file
                ]
                proc = await asyncio.create_subprocess_exec(*ffmpeg_cmd_list, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    with open(filepath, 'wb') as dest_f, open(modified_file, 'rb') as src_f:
                        shutil.copyfileobj(src_f, dest_f)
                else:
                    logging.error(f"FFmpeg failed for {filepath}. Stderr: {stderr.decode()}")
                    continue
            finally:
                if modified_file and os.path.exists(modified_file):
                    os.remove(modified_file)
            
            processed_files.append(filepath)

            # --- File Organization ---
            cam = int(id_str) % 10
            dt_for_path = final_dt + datetime.timedelta(seconds=20)
            remotedir = os.path.join(args.remotedir, f'cam{cam}', dt_for_path.strftime('%Y%m%d'), dt_for_path.strftime('%H'))
            os.makedirs(remotedir, exist_ok=True)
            file_prefix = 'mini_' if eventdir.endswith('SD') else 'full_'
            remotefile = os.path.join(remotedir, f"{file_prefix}{dt_for_path.strftime('%M')}.mp4")
            
            if args.link:
                if os.path.lexists(remotefile): os.remove(remotefile)
                os.link(filepath, remotefile)
            else:
                shutil.copy2(filepath, remotefile)
            
            # --- JPG Stacking ---
            if eventdir.endswith('HD') or eventdir.endswith('SD'):
                jpgfile = os.path.join(remotedir, f"{file_prefix}{dt_for_path.strftime('%M')}.jpg")
                stack_duration = 0.2 if eventdir.endswith('HD') else None

                try:
                    loop = asyncio.get_running_loop()
                    # Run stacking in the process_pool via a wrapper to isolate it and suppress stdout.
                    await loop.run_in_executor(
                        process_pool,             # Use the dedicated process pool
                        stack_wrapper,            # Call our wrapper to suppress stdout
                        # --- stack.py arguments (passed to wrapper) ---
                        [filepath],           # video_paths
                        jpgfile,              # output_path
                        0.0,                  # start_seconds
                        stack_duration,       # duration_seconds
                        False,                # denoise
                        10.0,                 # denoise_strength
                        95,                   # quality
                        os.cpu_count() or 1,  # num_threads
                        False,                # individual
                        True                  # enhance
                    )
                    
                    if eventdir.endswith('HD') and args.link:
                        snapshot_link = os.path.join(args.remotedir, f'cam{cam}', 'snapshot.jpg')
                        if os.path.lexists(snapshot_link): os.remove(snapshot_link)
                        os.link(jpgfile, snapshot_link)
                except BrokenProcessPool:
                    # This error means the worker process terminated abruptly. This can happen
                    # if stack.py calls sys.exit(). We log this for diagnostics.
                    logging.error(f"Stacking process for {filepath} terminated unexpectedly.")
                except Exception as stack_err:
                    # This will now catch other errors within stack.py (e.g., a ValueError)
                    # and log them without crashing the main application.
                    logging.error(f"Failed to stack video {filepath}: {stack_err}")

        except Exception as e:
            logging.exception(f"FATAL error in video watcher loop: {e}")
            continue

async def watch_detections():
    """Monitors for meteor detection JSON files and triggers an external script."""
    logging.info('Detections watcher started.')
    i = inotify.adapters.InotifyTree(args.recdir, mask=inotify.constants.IN_CLOSE_WRITE)
    
    event_iterator = iter(i.event_gen(yield_nones=False))
    async for event in async_event_gen(event_iterator):
        try:
            (_, type_names, eventdir, eventfile) = event
            filepath = os.path.join(eventdir, eventfile)
            
            if 'IN_CLOSE_WRITE' in type_names and eventfile.endswith(REDUCED_JSON_SUFFIX):
                logging.info(f"Executing: {args.exefile} {filepath}")
                await run_command_async(args.exefile, filepath)

        except Exception as e:
            logging.exception(f"Error in detection watcher loop: {e}")
            continue

async def watch_self_for_changes():
    """Periodically checks if the script file has been updated and restarts."""
    logging.info('Self-update watcher started.')
    while True:
        await asyncio.sleep(60)
        try:
            current_mod_time = os.path.getmtime(__file__)
            if current_mod_time > _SELF_MOD_TIME:
                logging.warning("Script has been updated. Restarting service...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except FileNotFoundError:
            continue
        except Exception as e:
            logging.error(f"Error in self-update watcher: {e}")

async def main():
    """Main function to run all watcher tasks concurrently."""
    logging.info("Starting AMS Mirror service.")
    await asyncio.gather(
        watch_detections(),
        watch_videos(),
        watch_self_for_changes()
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("AMS Mirror service stopped by user.")
    except Exception as e:
        # Add a final fail-safe. If any unhandled exception
        # reaches this point, log it and restart the entire script.
        logging.exception(f"UNHANDLED EXCEPTION. Restarting service. Error: {e}")
        try:
            # Attempt a clean shutdown of the process pool
            process_pool.shutdown(wait=False, cancel_futures=True)
        except Exception as shutdown_err:
            logging.error(f"Error during process pool shutdown: {shutdown_err}")
        # Restart the script using the same mechanism as the self-updater
        os.execv(sys.executable, [sys.executable] + sys.argv)
