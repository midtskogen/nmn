#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes a single meteor event detection. (Multilingual Version)

This script takes an event.txt file as input and performs the following steps:
1.  Validates the event against various criteria.
2.  Acquires a processing slot to limit concurrent jobs.
3.  Calls external scripts for video processing and classification.
4.  Generates Metrack-compatible data files (metrack, centroid, light).
5.  Updates the event.txt file with a summary section.
6.  Generates a translated brightness plot for each supported language.

NOTE: This updated version uses the 'posix_ipc' library.
Install it with:
pip install posix_ipc
"""

import argparse
import configparser
import datetime
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import ephem
import matplotlib
import psutil
from dateutil.parser import parse as dt_parse

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The posix_ipc library is required for the new semaphore-based locking.
try:
    import posix_ipc
except ImportError:
    print("Error: The 'posix_ipc' library is required. Please install it using 'pip install posix_ipc'")
    sys.exit(1)


class Settings:
    """Configuration constants for the script."""
    # Path to the script that processes video files
    MAKEVIDEOS_SCRIPT = Path('/home/httpd/norskmeteornettverk.no/bin/makevideos.py')
    CROP_SCRIPT = Path('/home/httpd/norskmeteornettverk.no/bin/meteorcrop.py')
    METEOR_TEST_SCRIPT = Path('/home/httpd/norskmeteornettverk.no/bin/meteor_test.sh')

    # --- Semaphore-based concurrency limiting settings ---
    SEMAPHORE_NAME = "/nmn_process_py_semaphore"
    CORE_COUNT = os.cpu_count() or 1
    MAX_CONCURRENT_JOBS = max(1, CORE_COUNT // 4)

    # --- Original thresholds ---
    MIN_SUN_ALTITUDE_FOR_CHECK = 1  # degrees
    MIN_SUN_SEPARATION_ARC = 20  # degrees
    CLASSIFICATION_THRESHOLD_DEFAULT = 0.05
    CLASSIFICATION_THRESHOLD_AMS = 0.05

# --- Internationalization (i18n) Setup ---
SUPPORTED_LANGS = ['nb', 'en', 'de', 'cs']
DEFAULT_LANG = 'nb'
LOC_DIR = Path(__file__).parent / 'loc'

def load_translations(lang_code: str) -> dict:
    """Loads the translation dictionary for a given language, with fallback to default."""
    default_path = LOC_DIR / f"{DEFAULT_LANG}.json"
    lang_path = LOC_DIR / f"{lang_code}.json"

    translations = {}
    if default_path.exists():
        with default_path.open('r', encoding='utf-8') as f:
            translations = json.load(f)

    if lang_code != DEFAULT_LANG and lang_path.exists():
        with lang_path.open('r', encoding='utf-8') as f:
            translations.update(json.load(f))
            
    return translations


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Processes a single meteor event from an event.txt file."
    )
    parser.add_argument(
        "event_file",
        type=Path,
        help="Path to the event.txt file for the detection."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds for acquiring a processing slot. Defaults to 900."
    )
    return parser.parse_args()


def load_configs(event_file: Path) -> tuple:
    """Loads configuration from the event file and the station's meteor.cfg."""
    if not event_file.is_file():
        print(f"Error: Event file not found at '{event_file}'")
        sys.exit(1)

    event_dir = event_file.parent
    station_cfg_file = event_dir / 'meteor.cfg'

    if not station_cfg_file.is_file():
        print(f"Error: meteor.cfg not found in '{event_dir}'")
        sys.exit(1)

    event_config = configparser.ConfigParser()
    event_config.read(event_file)

    station_config = configparser.ConfigParser()
    station_config.read(station_cfg_file)

    return event_config, station_config, event_dir


def calculate_angular_distance(az1: float, alt1: float, az2: float, alt2: float) -> float:
    """Calculates the great-circle distance between two points in az/alt."""
    x1, x2 = math.radians(az1), math.radians(az2)
    y1, y2 = math.radians(alt1), math.radians(alt2)

    a = math.sin((y2 - y1) / 2)**2 + math.cos(y1) * math.cos(y2) * math.sin((x2 - x1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return math.degrees(c)


def get_sun_position(station_config: configparser.ConfigParser, timestamp: str) -> tuple:
    """Calculates the sun's altitude and azimuth for a given time and location."""
    cleaned_timestamp = timestamp.rsplit('(', 1)[0].strip()

    obs = ephem.Observer()
    obs.lat = station_config.get('astronomy', 'latitude')
    obs.lon = station_config.get('astronomy', 'longitude')
    obs.elevation = float(station_config.get('astronomy', 'elevation'))
    obs.date = dt_parse(cleaned_timestamp)

    sun = ephem.Sun()
    sun.compute(obs)

    return math.degrees(sun.alt), math.degrees(sun.az)


def should_discard_event(event_config, station_config) -> bool:
    """Performs pre-checks to determine if an event should be discarded."""
    # This logic is for validation and remains unchanged.
    # ... (function content is identical to original)
    return False # Placeholder for brevity


def cleanup_directory_and_exit(event_dir: Path):
    """Removes the event directory and its contents, then exits."""
    # This logic is for system cleanup and remains unchanged.
    # ... (function content is identical to original)
    sys.exit(0) # Placeholder for brevity


def acquire_processing_slot(timeout: int) -> posix_ipc.Semaphore:
    """Acquires a slot from a system-wide counting semaphore to limit concurrency."""
    # This logic is for process management and remains unchanged.
    # ... (function content is identical to original)
    # This is a simplified version for demonstration.
    try:
        semaphore = posix_ipc.Semaphore(Settings.SEMAPHORE_NAME, posix_ipc.O_CREX, initial_value=Settings.MAX_CONCURRENT_JOBS)
    except posix_ipc.ExistentialError:
        semaphore = posix_ipc.Semaphore(Settings.SEMAPHORE_NAME)
    semaphore.acquire(timeout=timeout)
    return semaphore


def run_command(command: list, cwd: Path) -> subprocess.CompletedProcess:
    """Helper function to run an external command."""
    # This logic is for system interaction and remains unchanged.
    # ... (function content is identical to original)
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def run_video_processing(event_config, station_config, event_dir: Path) -> tuple:
    """Runs the makevideos.py script and calculates corrected duration."""
    # This logic is for video processing and remains unchanged.
    # ... (function content is identical to original)
    # Placeholder return for demonstration
    return 10.0, 20.0, 15.0, 15.0, 1.23


def update_event_summary(event_config, station_config, proc_results: tuple):
    """Adds or updates the [summary] section in the event config."""
    # This logic updates a data file and remains unchanged.
    # ... (function content is identical to original)
    pass # Placeholder for brevity


def run_classification(event_config, event_dir: Path):
    """Crops meteor image and cleans up directory if classification is low."""
    # This logic is for classification and remains unchanged.
    # ... (function content is identical to original)
    pass # Placeholder for brevity


def write_data_files(event_config, station_config, event_dir: Path):
    """Writes the machine-readable data files (metrack, centroid, light)."""
    # This function now only handles the non-translated data files.
    videostart_str_full = event_config.get('video', 'start')
    cleaned_videostart_str = videostart_str_full.rsplit('(', 1)[0].strip()
    videostart_ts = dt_parse(cleaned_videostart_str)

    station_name = station_config.get('station', 'name')
    base_name = f"{station_name}-{videostart_ts.strftime('%Y%m%d%H%M%S')}"

    timestamps = [float(t) for t in event_config.get('trail', 'timestamps').split()]
    coordinates = event_config.get('trail', 'coordinates').split()
    brightness = [float(b) for b in event_config.get('trail', 'brightness').split()]
    size = [float(s) for s in event_config.get('trail', 'size').split()]
    frame_brightness = [float(f) for f in event_config.get('trail', 'frame_brightness').split()]
    code = station_config.get('station', 'code')

    # Write Metrack file
    metrack_file = event_dir / f"{base_name}.txt"
    with metrack_file.open('w', encoding='utf-8') as f:
        # ... (metrack file writing logic is unchanged)
        pass

    # Write Centroid file
    centroid_file = event_dir / 'centroid.txt'
    with centroid_file.open('w', encoding='utf-8') as f:
        for i, (ts, coord) in enumerate(zip(timestamps, coordinates)):
            # ... (centroid file writing logic is unchanged)
            pass

    # Write Light data file
    light_file = event_dir / 'light.txt'
    with light_file.open('w', encoding='utf-8') as f:
        for ts, b, s, fb in zip(timestamps, brightness, size, frame_brightness):
            dt = ts - timestamps[0]
            f.write(f"{dt:.2f} {b} {s} {fb}\n")

def generate_brightness_plots(event_dir: Path, timestamps: list, brightness: list):
    """Generates translated brightness plots for all supported languages."""
    print("Generating multilingual brightness plots...")
    if not timestamps or not brightness:
        print("Warning: No data available to generate brightness plot.")
        return

    time_axis = [t - timestamps[0] for t in timestamps]

    for lang in SUPPORTED_LANGS:
        translations = load_translations(lang)
        file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'
        
        svg_path = event_dir / f'{file_prefix}brightness.svg'
        jpg_path = event_dir / f'{file_prefix}brightness.jpg'

        try:
            plt.figure()
            plt.plot(time_axis, brightness)
            # Use translated labels, with fallbacks
            plt.xlabel(translations.get("plot_time_x_label", "Time [s]"))
            plt.ylabel(translations.get("brightness", "Brightness"))
            plt.title(translations.get("brightness_plot_title", "Brightness over Time"))
            
            plt.savefig(svg_path)
            plt.savefig(jpg_path)
            plt.close()
            print(f"  - Saved {svg_path.name} and {jpg_path.name}")
        except Exception as e:
            print(f"Error generating plot for language '{lang}': {e}")


def main():
    """Main execution flow of the script."""
    args = get_args()
    # 1. Initial load
    event_config, station_config, event_dir = load_configs(args.event_file)

    if should_discard_event(event_config, station_config):
        cleanup_directory_and_exit(event_dir)

    semaphore = acquire_processing_slot(args.timeout)

    try:
        # 2. Run heavy processing (video, etc.)
        proc_results = run_video_processing(event_config, station_config, event_dir)

        # 3. Reload config to get any changes made by external scripts
        event_config, station_config, event_dir = load_configs(args.event_file)

        # 4. Update summary, run classification, and save updated event file
        update_event_summary(event_config, station_config, proc_results)
        run_classification(event_config, event_dir)
        with args.event_file.open('w', encoding='utf-8') as f:
            event_config.write(f)

        # 5. Write non-translated data files
        write_data_files(event_config, station_config, event_dir)
        
        # 6. Generate all translated brightness plots
        timestamps = [float(t) for t in event_config.get('trail', 'timestamps').split()]
        brightness = [float(b) for b in event_config.get('trail', 'brightness').split()]
        generate_brightness_plots(event_dir, timestamps, brightness)

    finally:
        if semaphore:
            semaphore.release()
            semaphore.close()
            print(f"Process {os.getpid()} released its processing slot.")

    # Exiting with 1 is a convention to signal that a discard/cleanup did NOT happen.
    # The calling script `fetch.py` checks for exit code 0 to know if it was discarded.
    sys.exit(1)


if __name__ == "__main__":
    main()
