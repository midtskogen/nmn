#!/usr/bin/env python3

import os
import subprocess
import re
from pathlib import Path
from datetime import datetime

# Config
CAMERA_BASE = Path("/meteor")
DISK_THRESHOLD = 95  # percent
MIN_DIRS = 3         # don't delete if a cam has fewer than this many directories
CAM_DIR_REGEX = re.compile(r"^cam\d+$")

# Regex to match YYYYMMDD-style folder names
DATE_DIR_REGEX = re.compile(r"^20\d{6}$")

def get_disk_usage_percent(path):
    """Return the used percentage of the disk containing `path`."""
    try:
        result = subprocess.run(['df', '-k', '--output=pcent', str(path)],
                                capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        return int(lines[-1].strip().strip('%'))
    except Exception as e:
        print(f"Error checking disk usage for {path}: {e}")
        return 0

def find_dated_dirs(cam_path):
    """Return a list of (path, mtime) tuples for dated directories under cam_path."""
    try:
        dated_dirs = []
        for entry in cam_path.iterdir():
            if entry.is_dir() and DATE_DIR_REGEX.match(entry.name):
                mtime = entry.stat().st_mtime
                dated_dirs.append((entry, mtime))
        return dated_dirs
    except Exception as e:
        print(f"Error reading directories in {cam_path}: {e}")
        return []

def main():
    cams = [p for p in CAMERA_BASE.iterdir() if p.is_dir() and CAM_DIR_REGEX.match(p.name)]
    
    while get_disk_usage_percent(CAMERA_BASE) >= DISK_THRESHOLD:
        oldest = None

        # Scan all cams for oldest dated directory
        for cam in cams:
            dated_dirs = find_dated_dirs(cam)
            if len(dated_dirs) < MIN_DIRS:
                continue
            dated_dirs.sort(key=lambda x: x[1])  # Sort by mtime
            candidate = dated_dirs[0]
            if not oldest or candidate[1] < oldest[1]:
                oldest = candidate

        # If no directory qualifies for deletion, exit
        if not oldest:
            print("No more directories to delete that meet criteria.")
            break

        try:
            print(f"Deleting oldest dir: {oldest[0]}")
            subprocess.run(['rm', '-rf', str(oldest[0])], check=True)
        except Exception as e:
            print(f"Failed to delete {oldest[0]}: {e}")
            break

if __name__ == "__main__":
    main()
