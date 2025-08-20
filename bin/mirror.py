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
import re
import os
import sys
import asyncio
import datetime
import tempfile
import shutil
from collections import deque
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
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, level=logging.INFO)

# --- Global State ---
processed_files = deque(maxlen=200)
FILENAME_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\w+)_(\w+)\.mp4")

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

            # FIX: Handle race condition where file disappears before processing
            if not os.path.exists(filepath):
                continue

            if filepath in processed_files:
                continue

            if not eventfile.endswith('.mp4'):
                continue

            match = FILENAME_PATTERN.match(eventfile)
            if not match:
                # FIX: Change log level from warning to info for skipped files
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
            
            if eventdir.endswith('HD'):
                jpgfile = os.path.join(remotedir, f"{file_prefix}{dt_for_path.strftime('%M')}.jpg")
                
                # FIX: Replace shell script with direct, non-blocking call to stack.py
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, stack.stack_video_frames, [filepath], jpgfile, 0.0, 0.2,
                        False, 10.0, 95, os.cpu_count() or 1, False, True
                    )
                    
                    if args.link:
                        snapshot_link = os.path.join(args.remotedir, f'cam{cam}', 'snapshot.jpg')
                        if os.path.lexists(snapshot_link): os.remove(snapshot_link)
                        os.link(jpgfile, snapshot_link)
                except Exception as stack_err:
                    logging.error(f"Failed to stack video {filepath}: {stack_err}")

        except Exception as e:
            logging.exception(f"Error in video watcher loop: {e}")
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
            
            # FIX: Corrected logic to find files in subdirectories
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
        await asyncio.sleep(60)  # Check every 60 seconds
        try:
            current_mod_time = os.path.getmtime(__file__)
            if current_mod_time > _SELF_MOD_TIME:
                logging.warning("Script has been updated. Restarting service...")
                # This call replaces the current process with a new one
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except FileNotFoundError:
            # This could happen during a file-save operation; ignore and retry
            continue
        except Exception as e:
            logging.error(f"Error in self-update watcher: {e}")

async def main():
    """Main function to run all watcher tasks concurrently."""
    logging.info("Starting AMS Mirror service.")
    await asyncio.gather(
        watch_detections(),
        watch_videos(),
        watch_self_for_changes()  # Add the self-watcher to the main tasks
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("AMS Mirror service stopped by user.")
