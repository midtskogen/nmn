#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes a single meteor event detection.

This script takes an event.txt file as input and performs the following steps:
1.  Validates the event against various criteria (e.g., movement direction,
    proximity to the sun).
2.  Acquires a system-wide lock to prevent concurrent processing.
3.  Calls an external script to process video, correct for lens distortion,
    and generate gnomonic projection data.
4.  Generates a Metrack-compatible observation file.
5.  Updates the event.txt file with a summary section.
6.  Optionally runs a classification script to determine the probability of
    the event being a meteor and cleans up the directory if it falls below
    a threshold.
7.  Generates centroid and brightness data files and a brightness plot.

This script requires several files to be present in the event directory,
including meteor.cfg, lens.pto, and the event video file.
"""

import argparse
import configparser
import datetime
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


class Settings:
    """Configuration constants for the script."""
    # Path to the script that processes video files
    MAKEVIDEOS_SCRIPT = Path('/home/httpd/norskmeteornettverk.no/bin/makevideos.sh')
    CLASSIFY_SCRIPT = Path('/home/httpd/norskmeteornettverk.no/bin/classify.py')
    METEOR_TEST_SCRIPT = Path('/home/httpd/norskmeteornettverk.no/bin/meteor_test.sh')

    # Timeouts and thresholds
    LOCK_TIMEOUT_SECONDS = 900  # 15 minutes
    LOCK_RETRY_DELAY_SECONDS = 3
    MIN_SUN_ALTITUDE_FOR_CHECK = 1  # degrees
    MIN_SUN_SEPARATION_ARC = 20  # degrees
    CLASSIFICATION_THRESHOLD_DEFAULT = 0.25
    CLASSIFICATION_THRESHOLD_AMS = 0.05


def get_args():
    """Parses and returns command-line arguments."""
    # Corrected argument parsing to match the original script's behavior.
    parser = argparse.ArgumentParser(
        description="Processes a single meteor event from an event.txt file."
    )
    parser.add_argument(
        "event_file",
        type=Path,
        help="Path to the event.txt file for the detection."
    )
    return parser.parse_args()


def load_configs(event_file: Path) -> tuple:
    """
    Loads configuration from the event file and the station's meteor.cfg.

    Args:
        event_file: Path to the event.txt file.

    Returns:
        A tuple containing:
        - event_config (ConfigParser): Parsed event data.
        - station_config (ConfigParser): Parsed station data.
        - event_dir (Path): The directory containing the event file.
    """
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
    
    # Haversine formula
    a = math.sin((y2 - y1) / 2)**2 + math.cos(y1) * math.cos(y2) * math.sin((x2 - x1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return math.degrees(c)


def get_sun_position(station_config: configparser.ConfigParser, timestamp: str) -> tuple:
    """Calculates the sun's altitude and azimuth for a given time and location."""
    # Clean the timestamp string to remove the parenthesized Unix timestamp if present
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
    """
    Performs pre-checks to determine if an event should be discarded.

    Returns:
        True if the event should be discarded, False otherwise.
    """
    # Check 1: Discard objects moving upwards near the horizon
    try:
        is_flash = event_config.getboolean('video', 'flash')
        coords = event_config.get('trail', 'coordinates').split()
        start_alt = float(coords[0].split(',')[1])
        end_alt = float(coords[-1].split(',')[1])

        if end_alt > start_alt and end_alt < 20 and not is_flash:
            print("Discarding: Event is moving upwards near the horizon.")
            return True
    except (configparser.NoOptionError, IndexError):
        pass  # Ignore if data is missing

    # Check 2: Discard objects too close to the sun
    try:
        is_manual = event_config.getboolean('trail', 'manual')
        if is_manual:
            return False # Keep all manually verified events

        videostart_ts_str = event_config.get('video', 'start')
        sun_alt, sun_az = get_sun_position(station_config, videostart_ts_str)

        if sun_alt > Settings.MIN_SUN_ALTITUDE_FOR_CHECK:
            midpoint_az, midpoint_alt = map(float, event_config.get('trail', 'midpoint').split(','))
            separation = calculate_angular_distance(sun_az, sun_alt, midpoint_az, midpoint_alt)
            if separation < Settings.MIN_SUN_SEPARATION_ARC:
                print(f"Discarding: Event is too close to the sun ({separation:.1f}°).")
                return True
    except (configparser.NoOptionError, ValueError):
        pass # Ignore if data is missing

    return False


def acquire_process_lock():
    """
    Acquires a system-wide lock using a Unix domain socket.
    Exits if the lock cannot be acquired within the timeout.
    """
    # The original script used sys.argv[0] for the lock name. We replicate that.
    lock_name = os.path.basename(sys.argv[0])
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    lock_address = b'\0' + lock_name.encode('utf-8')
    start_time = time.time()

    while time.time() - start_time < Settings.LOCK_TIMEOUT_SECONDS:
        try:
            lock_socket.bind(lock_address)
            print("Lock acquired successfully.")
            return lock_socket
        except socket.error:
            time.sleep(Settings.LOCK_RETRY_DELAY_SECONDS)

    print(f"Error: Failed to acquire lock within {Settings.LOCK_TIMEOUT_SECONDS} seconds.")
    # Find which process is holding the lock
    for conn in psutil.net_connections(kind='unix'):
        if conn.laddr and conn.laddr.endswith(lock_name):
            print(f"Error: Another process (PID {conn.pid}) is already holding the lock.")
            break
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
    """
    Runs the makevideos.sh script and calculates corrected duration.

    Returns:
        A tuple containing (start_az, start_alt, end_az, end_alt, duration).
    """
    videostart_str_full = event_config.get('video', 'start')
    cleaned_videostart_str = videostart_str_full.rsplit('(', 1)[0].strip()
    videostart_ts = dt_parse(cleaned_videostart_str)
    
    station_name = station_config.get('station', 'name')
    base_name = f"{station_name}-{videostart_ts.strftime('%Y%m%d%H%M%S')}"

    proc = run_command([Settings.MAKEVIDEOS_SCRIPT, base_name], cwd=event_dir)
    output = proc.stdout.split()

    start_az, start_alt = float(output[-5]), float(output[-4])
    end_az, end_alt = float(output[-2]), float(output[-1])

    # Correct the duration based on the arc length of the calibrated path
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
    """
    Runs classification scripts and cleans up the directory if the event
    is likely not a meteor.
    """
    if event_config.has_option('summary', 'meteor_probability'):
        return

    # Run scripts to get probability
    run_command([sys.executable, Settings.CLASSIFY_SCRIPT, str(event_dir)], cwd=event_dir)
    fireball_jpg = event_dir / 'fireball.jpg'
    proc = run_command([Settings.METEOR_TEST_SCRIPT, fireball_jpg], cwd=event_dir)
    probability = float(proc.stdout.strip())
    
    event_config.set('summary', 'meteor_probability', str(probability))

    # Check against threshold and delete if necessary
    is_ams = event_config.getboolean('trail', 'ams', fallback=False)
    is_manual = event_config.getboolean('trail', 'manual', fallback=False)
    
    threshold = Settings.CLASSIFICATION_THRESHOLD_AMS if is_ams else Settings.CLASSIFICATION_THRESHOLD_DEFAULT
    
    if probability < threshold and not is_manual:
        print(f"Meteor probability ({probability:.2f}) is below threshold ({threshold}). Deleting directory.")
        for f in event_dir.glob('*'):
            f.unlink()
        event_dir.rmdir()
        try:
            # Attempt to remove the parent time directory if it's empty
            event_dir.parent.rmdir()
        except OSError:
            pass # Directory not empty, which is fine
        sys.exit(0)


def write_output_files(event_config, station_config, event_dir: Path):
    """Writes the metrack, centroid, and light data files."""
    videostart_str_full = event_config.get('video', 'start')
    cleaned_videostart_str = videostart_str_full.rsplit('(', 1)[0].strip()
    videostart_ts = dt_parse(cleaned_videostart_str)

    station_name = station_config.get('station', 'name')
    base_name = f"{station_name}-{videostart_ts.strftime('%Y%m%d%H%M%S')}"
    
    # Get data from config
    timestamps = [float(t) for t in event_config.get('trail', 'timestamps').split()]
    coordinates = event_config.get('trail', 'coordinates').split()
    brightness = [float(b) for b in event_config.get('trail', 'brightness').split()]
    size = [float(s) for s in event_config.get('trail', 'size').split()]
    frame_brightness = [float(f) for f in event_config.get('trail', 'frame_brightness').split()]

    # --- Write Metrack file (.txt) ---
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

    # --- Write Centroid file (centroid.txt) ---
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

    # --- Write Light curve file (light.txt) ---
    light_file = event_dir / 'light.txt'
    with light_file.open('w', encoding='utf-8') as f:
        for ts, b, s, fb in zip(timestamps, brightness, size, frame_brightness):
            dt = ts - timestamps[0]
            f.write(f"{dt:.2f} {b} {s} {fb}\n")

    # --- Generate Brightness Plot ---
    time_axis = [t - timestamps[0] for t in timestamps]
    plt.figure()
    plt.plot(time_axis, brightness)
    plt.xlabel('Time [s]')
    plt.ylabel('Brightness')
    plt.savefig(event_dir / 'brightness.svg')
    plt.savefig(event_dir / 'brightness.jpg')
    plt.close()


def main():
    """Main execution flow of the script."""
    args = get_args()
    event_config, station_config, event_dir = load_configs(args.event_file)

    if should_discard_event(event_config, station_config):
        sys.exit(0)

    # Acquire lock and ensure it's released on exit
    lock_socket = acquire_process_lock()
    
    try:
        proc_results = run_video_processing(event_config, station_config, event_dir)
        update_event_summary(event_config, station_config, proc_results)
        run_classification(event_config, event_dir)

        # Save the updated summary back to the event file
        with args.event_file.open('w', encoding='utf-8') as f:
            event_config.write(f)

        write_output_files(event_config, station_config, event_dir)
        
    finally:
        # Release the lock by closing the socket
        lock_socket.close()
        print("Lock released.")

    # The original script exits with 1 on success, so we preserve that.
    sys.exit(1)


if __name__ == "__main__":
    main()
