#!/usr/bin/env python3
"""
Monitors directories for new video and data files from AMS (Allsky7 Meteor System).

This script performs two main tasks concurrently:
1.  Watches for new one-minute video files (.mp4). When a new video appears, it:
    - Calculates an accurate creation timestamp using the file's system birth time.
    - Uses FFmpeg to embed the high-precision timestamp into the video's metadata.
    - Overwrites the original file's content to apply the change while preserving ownership.
    - Copies or links the processed video to a structured destination directory.
    - Generates a JPG preview for HD videos.
2.  Watches for new meteor detection data files (*-reduced.json) and passes them
    to an external script for event processing.
"""

import inotify.adapters
import argparse
import logging
import re
import shlex
import subprocess
import os
import asyncio
import datetime
import tempfile
import shutil
from collections import deque

# --- Constants for Configuration ---
AMS2EVENT_PY_DEFAULT = "/home/meteor/bin/ams2event.py"
VID2JPG_SH = "/home/meteor/bin/vid2jpg.sh"
METEORS_DIR_NAME = "meteors"
REDUCED_JSON_SUFFIX = "-reduced.json"

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

            if filepath in processed_files:
                continue

            if not eventfile.endswith('.mp4'):
                continue

            match = FILENAME_PATTERN.match(eventfile)
            if not match:
                logging.warning(f"Skipping file with unexpected name format: {filepath}")
                continue
            
            id_str = match.groups()[-1]

            if os.path.getsize(filepath) == 0:
                logging.info(f"Skipping zero-byte file: {filepath}")
                continue

            # --- Timestamp Calculation ---
            # Get the base timestamp (Y-M-D H:M:S) from the filename.
            (year, month, day, hour, minute, second) = match.groups()[:6]
            dt_from_name = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

            # Get the precise system time to "borrow" its sub-second value.
            file_stat = os.stat(filepath)
            try:
                file_time = datetime.datetime.fromtimestamp(file_stat.st_birthtime)
            except AttributeError:
                file_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)

            # Combine the accurate filename time with the system's sub-second precision.
            final_dt = dt_from_name.replace(microsecond=file_time.microsecond)

            # Apply the configured delay.
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
                    # Overwrite original file content to preserve ownership and permissions
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
                await run_command_async(VID2JPG_SH, filepath, jpgfile)
                if args.link:
                    snapshot_link = os.path.join(args.remotedir, f'cam{cam}', 'snapshot.jpg')
                    if os.path.lexists(snapshot_link): os.remove(snapshot_link)
                    os.symlink(jpgfile, snapshot_link)

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
            dir_parts = eventdir.split('/')
            
            if 'IN_CLOSE_WRITE' in type_names and dir_parts[-1] == METEORS_DIR_NAME and eventfile.endswith(REDUCED_JSON_SUFFIX):
                logging.info(f"Executing: {args.exefile} {filepath}")
                await run_command_async(args.exefile, filepath)

        except Exception as e:
            logging.exception(f"Error in detection watcher loop: {e}")
            continue

async def main():
    """Main function to run all watcher tasks concurrently."""
    logging.info("Starting AMS Mirror service.")
    await asyncio.gather(
        watch_detections(),
        watch_videos()
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("AMS Mirror service stopped by user.")
