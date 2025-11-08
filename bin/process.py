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
    try:
        is_flash = event_config.getboolean('video', 'flash')
        coords = event_config.get('trail', 'coordinates').split()
        start_alt = float(coords[0].split(',')[1])
        end_alt = float(coords[-1].split(',')[1])

        if end_alt > start_alt and end_alt < 20 and not is_flash:
            print("Discarding: Event is moving upwards near the horizon.")
            return True
    except (configparser.NoOptionError, IndexError):
        pass

    try:
        is_manual = event_config.getboolean('trail', 'manual')
        if is_manual:
            return False

        videostart_ts_str = event_config.get('video', 'start')
        sun_alt, sun_az = get_sun_position(station_config, videostart_ts_str)

        if sun_alt > Settings.MIN_SUN_ALTITUDE_FOR_CHECK:
            midpoint_az, midpoint_alt = map(float, event_config.get('trail', 'midpoint').split(','))
            separation = calculate_angular_distance(sun_az, sun_alt, midpoint_az, midpoint_alt)
            if separation < Settings.MIN_SUN_SEPARATION_ARC:
                print(f"Discarding: Event is too close to the sun ({separation:.1f}°).")
                return True
    except (configparser.NoOptionError, ValueError):
        pass

    return False


def cleanup_directory_and_exit(event_dir: Path):
    """Removes the event directory and its contents, then exits."""
    print(f"Cleaning up and removing directory: {event_dir}")
    for f in event_dir.glob('*'):
        f.unlink()
    event_dir.rmdir()
    try:
        event_dir.parent.rmdir()
    except OSError:
        pass
    sys.exit(0)


def acquire_processing_slot(timeout: int) -> posix_ipc.Semaphore:
    """Acquires a slot from a system-wide counting semaphore to limit concurrency."""
    semaphore = None
    try:
        semaphore = posix_ipc.Semaphore(
            Settings.SEMAPHORE_NAME,
            posix_ipc.O_CREX,
            initial_value=Settings.MAX_CONCURRENT_JOBS
        )
        print(f"Semaphore created. Max concurrent jobs set to: {Settings.MAX_CONCURRENT_JOBS}")
    except posix_ipc.ExistentialError:
        semaphore = posix_ipc.Semaphore(Settings.SEMAPHORE_NAME)

    try:
        print(f"Process {os.getpid()} waiting for a processing slot...")
        semaphore.acquire(timeout=timeout)
        print(f"Process {os.getpid()} acquired a slot. Starting job.")
        return semaphore
    except posix_ipc.BusyError:
        print(f"Error: Could not acquire a processing slot within {timeout} seconds.")
        print("The system might be overloaded with other processing jobs.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while acquiring semaphore: {e}")
        sys.exit(1)


def run_command(command: list, cwd: Path) -> subprocess.CompletedProcess:
    """Helper function to run an external command."""
    print(f"Running command: {' '.join(map(str, command))} in {cwd}")
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.cmd}")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(e.returncode)


def run_video_processing(event_config, station_config, event_dir: Path) -> tuple:
    """Runs the makevideos.py script and calculates corrected duration."""
    videostart_str_full = event_config.get('video', 'start')
    cleaned_videostart_str = videostart_str_full.rsplit('(', 1)[0].strip()
    videostart_ts = dt_parse(cleaned_videostart_str)

    station_name = station_config.get('station', 'name')
    base_name = f"{station_name}-{videostart_ts.strftime('%Y%m%d%H%M%S')}"

    proc = run_command([Settings.MAKEVIDEOS_SCRIPT, base_name], cwd=event_dir)
    output = proc.stdout.split()

    start_az, start_alt = float(output[-5]), float(output[-4])
    end_az, end_alt = float(output[-2]), float(output[-1])

    original_arc = event_config.getfloat('trail', 'arc')
    original_duration = event_config.getfloat('trail', 'duration')

    if original_arc > 0:
        calibrated_arc = calculate_angular_distance(start_az, start_alt, end_az, end_alt)
        corrected_duration = original_duration * (calibrated_arc / original_arc)
    else:
        corrected_duration = original_duration

    return start_az, start_alt, end_az, end_alt, corrected_duration


def update_event_summary(event_config, station_config, proc_results: tuple):
    """Adds or updates the [summary] section in the event config."""
    start_az, start_alt, end_az, end_alt, duration = proc_results
    videostart_ts_str = event_config.get('video', 'start')
    sun_alt, _ = get_sun_position(station_config, videostart_ts_str)

    if not event_config.has_section('summary'):
        event_config.add_section('summary')

    event_config.set('summary', 'latitude', station_config.get('astronomy', 'latitude'))
    event_config.set('summary', 'longitude', station_config.get('astronomy', 'longitude'))
    event_config.set('summary', 'elevation', station_config.get('astronomy', 'elevation'))
    event_config.set('summary', 'timestamp', videostart_ts_str)
    event_config.set('summary', 'startpos', f"{start_az} {start_alt}")
    event_config.set('summary', 'endpos', f"{end_az} {end_alt}")
    event_config.set('summary', 'duration', f"{duration:.2f}")
    event_config.set('summary', 'sunalt', f"{sun_alt:.1f}")
    event_config.set('summary', 'recalibrated', "0" if sun_alt > -10 else "1")


def run_classification(event_config, event_dir: Path):
    """Crops meteor image and cleans up directory if classification is low."""
    fireball_jpg = event_dir.resolve() / 'fireball.jpg'

    run_command([sys.executable, Settings.CROP_SCRIPT, '--mode', 'both', str(event_dir.resolve())], cwd=event_dir)

    if event_config.has_option('summary', 'meteor_probability'):
        return

    proc = run_command([Settings.METEOR_TEST_SCRIPT, fireball_jpg], cwd=event_dir)
    probability = float(proc.stdout.strip())

    event_config.set('summary', 'meteor_probability', str(probability))

    is_ams = event_config.getboolean('trail', 'ams', fallback=False)
    is_manual = event_config.getboolean('trail', 'manual', fallback=False)

    threshold = Settings.CLASSIFICATION_THRESHOLD_AMS if is_ams else Settings.CLASSIFICATION_THRESHOLD_DEFAULT

    if probability < threshold and not is_manual:
        print(f"Meteor probability ({probability:.2f}) is below threshold ({threshold}). Deleting directory.")
        cleanup_directory_and_exit(event_dir)


def write_data_files(event_config, station_config, event_dir: Path):
    """Writes the machine-readable data files (metrack, centroid, light)."""
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

    metrack_file = event_dir / f"{base_name}.txt"
    with metrack_file.open('w', encoding='utf-8') as f:
        lon = station_config.get('astronomy', 'longitude')
        lat = station_config.get('astronomy', 'latitude')
        elev = station_config.get('astronomy', 'elevation')
        code = station_config.get('station', 'code')
        start_az, start_alt = event_config.get('summary', 'startpos').split()
        end_az, end_alt = event_config.get('summary', 'endpos').split()
        duration = event_config.getfloat('summary', 'duration')
        start_time_unix = time.mktime(videostart_ts.utctimetuple()) + videostart_ts.microsecond / 1e6

        line = (f"{lon} {lat} {start_az} {end_az} {start_alt} {end_alt} 1 "
                f"{duration:.2f} 400 128 255 255 {code} {start_time_unix:.2f} {elev}\n")
        f.write(line)

    centroid_file = event_dir / 'centroid.txt'
    with centroid_file.open('w', encoding='utf-8') as f:
        for i, (ts, coord) in enumerate(zip(timestamps, coordinates)):
            az, alt = coord.split(',')
            dt = ts - timestamps[0]
            gm_time = time.gmtime(ts)
            frac_sec = f"{(ts % 1):.2f}"[2:]
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', gm_time)

            line = f"{i} {dt:.2f} {alt} {az} 1.0 {code} {ts_str}.{frac_sec} UTC\n"
            f.write(line)

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
    event_config, station_config, event_dir = load_configs(args.event_file)

    if should_discard_event(event_config, station_config):
        cleanup_directory_and_exit(event_dir)

    semaphore = acquire_processing_slot(args.timeout)

    try:
        proc_results = run_video_processing(event_config, station_config, event_dir)

        event_config, station_config, event_dir = load_configs(args.event_file)

        update_event_summary(event_config, station_config, proc_results)
        run_classification(event_config, event_dir)
        with args.event_file.open('w', encoding='utf-8') as f:
            event_config.write(f)

        write_data_files(event_config, station_config, event_dir)
        
        timestamps = [float(t) for t in event_config.get('trail', 'timestamps').split()]
        brightness = [float(b) for b in event_config.get('trail', 'brightness').split()]
        generate_brightness_plots(event_dir, timestamps, brightness)

    finally:
        if semaphore:
            semaphore.release()
            semaphore.close()
            print(f"Process {os.getpid()} released its processing slot.")

    sys.exit(1)


if __name__ == "__main__":
    main()
