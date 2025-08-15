#!/usr/bin/env python3
"""
Processes a meteor event, classifies it using a CNN, and reports it to the
Norsk Meteornettverk server if it meets the probability threshold.

Usage:
    python report.py <event.txt>
"""

import configparser
import datetime
import calendar
import math
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

# Third-party libraries
try:
    import ephem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dateutil import parser
except ImportError as e:
    print(f"Error: Missing required library. {e}", file=sys.stderr)
    print("Please install it using: pip install pyephem matplotlib python-dateutil", file=sys.stderr)
    sys.exit(1)


# --- Constants ---
METEOR_PROBABILITY_THRESHOLD = 0.5
SSH_TUNNEL_CONFIG_PATH = '/etc/default/ssh_tunnel'
REMOTE_REPORT_URL = "http://norskmeteornettverk.no/ssh/report.php"
# Assume processing scripts are in the user's bin directory
METEORCROP_PATH = Path.home() / "bin" / "meteorcrop.py"
CLASSIFY_PATH = Path.home() / "bin" / "classify.py"


def load_config(event_file_path: Path) -> configparser.ConfigParser:
    """Loads configuration from the event file and system/user config files."""
    config = configparser.ConfigParser()
    config.read(event_file_path)
    station_config_paths = ['/etc/meteor.cfg', Path.home() / 'meteor.cfg']
    config.read(station_config_paths)
    return config


def haversine_arc(az1: float, alt1: float, az2: float, alt2: float) -> float:
    """Calculates the angular separation (arc) in degrees between two points."""
    x1, x2 = math.radians(az1), math.radians(az2)
    y1, y2 = math.radians(alt1), math.radians(alt2)
    delta_lat = (y2 - y1) / 2
    delta_lon = (x2 - x1) / 2
    a = math.sin(delta_lat)**2 + math.cos(y1) * math.cos(y2) * math.sin(delta_lon)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return math.degrees(c)


def get_sun_position(config: configparser.ConfigParser) -> Tuple[float, float]:
    """Computes the sun's altitude and azimuth for the event time."""
    video_start_str = config.get('video', 'start')
    observer = ephem.Observer()
    observer.lat = config.get('astronomy', 'latitude')
    observer.lon = config.get('astronomy', 'longitude')
    observer.elevation = config.getfloat('astronomy', 'elevation')
    observer.date = parser.parse(video_start_str)
    sun = ephem.Sun()
    sun.compute(observer)
    return math.degrees(sun.alt), math.degrees(sun.az)


@contextmanager
def acquire_lock():
    """Acquires a system-wide lock to prevent multiple instances from running."""
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    lock_name = f'\0{Path(__file__).name}'
    try:
        while True:
            try:
                lock_socket.bind(lock_name)
                print("Lock acquired.")
                yield
                break
            except socket.error:
                print("Another instance is running. Waiting...")
                time.sleep(5)
    finally:
        lock_socket.close()
        print("Lock released.")


def run_video_creation(config: configparser.ConfigParser, event_file_path: Path) -> Optional[list]:
    """Runs makevideos.py script and returns its output."""
    video_start_str = config.get('video', 'start')
    start_dt = parser.parse(video_start_str)
    start_timestamp = calendar.timegm(start_dt.utctimetuple()) + start_dt.microsecond / 1_000_000.0
    duration = math.ceil(config.getfloat('trail', 'duration'))
    station_name = config.get('station', 'name')
    event_timestamp_str = start_dt.strftime('%Y%m%d%H%M%S')
    video_name = f"{station_name}-{event_timestamp_str}"
    video_dir = event_file_path.parents[3]
    event_dir = event_file_path.parent
    makevideos_script = Path.home() / 'bin' / 'makevideos.py'
    command = [
        str(makevideos_script), "--client",
        "--video-dir", str(video_dir),
        "--start", str(int(round(start_timestamp))),
        "--length", str(duration),
        video_name
    ]
    print(f"Running command: {' '.join(command)}")
    try:
        proc = subprocess.run(command, cwd=event_dir, capture_output=True, text=True, check=True)
        return proc.stdout.split()
    except subprocess.CalledProcessError as e:
        print(f"Error running makevideos.py: {e}\nStderr: {e.stderr}", file=sys.stderr)
        return None


def generate_reports(config: configparser.ConfigParser, video_output: list, event_dir: Path, video_name: str):
    """Generates metrack, centroid, light data files and a brightness plot."""
    video_start_str = config.get('video', 'start')
    start_dt = parser.parse(video_start_str)
    start_timestamp = calendar.timegm(start_dt.utctimetuple()) + start_dt.microsecond / 1_000_000.0
    original_arc = config.getfloat('trail', 'arc')
    original_duration = config.getfloat('trail', 'duration')
    start_az, start_alt = float(video_output[-5]), float(video_output[-4])
    end_az, end_alt = float(video_output[-2]), float(video_output[-1])
    recalibrated_arc = haversine_arc(start_az, start_alt, end_az, end_alt)
    duration = original_duration * (recalibrated_arc / original_arc) if original_arc > 0 else 0
    lon = config.get('astronomy', 'longitude')
    lat = config.get('astronomy', 'latitude')
    elevation = config.get('astronomy', 'elevation')
    station_code = config.get('station', 'code')
    metrack_data = [
        lon, lat, str(start_az), str(end_az), str(start_alt), str(end_alt),
        '1', str(round(duration, 2)), '400', '128', '255', '255',
        station_code, str(round(start_timestamp, 2)), elevation
    ]
    (event_dir / f"{video_name}.txt").write_text(' '.join(metrack_data) + '\n')
    print(f"Generated {video_name}.txt")

    timestamps = [float(t) for t in config.get('trail', 'timestamps').split()]
    coordinates = config.get('trail', 'coordinates').split()
    brightness = config.get('trail', 'brightness').split()
    frame_brightness = config.get('trail', 'frame_brightness').split()
    size = config.get('trail', 'size').split()

    with open(event_dir / 'centroid.txt', 'w') as f_centroid, \
         open(event_dir / 'light.txt', 'w') as f_light:
        for i, t in enumerate(timestamps):
            time_offset = round(t - timestamps[0], 2)
            az, alt = coordinates[i].split(',')
            gm_time = time.gmtime(t)
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', gm_time)
            ms = f"{round(t - math.floor(t), 2):.2f}"[2:]
            f_centroid.write(f"{i} {time_offset} {alt} {az} 1.0 {station_code} {time_str}.{ms} UTC\n")
            f_light.write(f"{time_offset} {brightness[i]} {size[i]} {frame_brightness[i]}\n")
    print("Generated centroid.txt and light.txt")

    time_points = [t - timestamps[0] for t in timestamps]
    plt.figure()
    plt.plot(time_points, list(map(float, brightness)))
    plt.xlabel('Time [s]')
    plt.ylabel('Brightness')
    plt.savefig(event_dir / 'brightness.svg')
    plt.savefig(event_dir / 'brightness.jpg')
    print("Generated brightness plot.")


def get_meteor_probability(event_dir: Path) -> float:
    """
    Runs meteorcrop and classify scripts to determine the meteor probability.
    Returns the probability score as a float.
    """
    print("\n--- Starting Meteor Classification ---")
    # 1. Run meteorcrop to generate fireball.jpg and other files
    try:
        print(f"Running meteorcrop in '{event_dir}'...")
        crop_cmd = [sys.executable, str(METEORCROP_PATH), "--mode", "both", str(event_dir)]
        subprocess.run(crop_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running meteorcrop.py: {e}\nStderr: {e.stderr}", file=sys.stderr)
        return 0.0

    # 2. Run classify on the resulting fireball.jpg
    fireball_jpg_path = event_dir / "fireball.jpg"
    if not fireball_jpg_path.is_file():
        print(f"Error: meteorcrop did not produce '{fireball_jpg_path.name}'.", file=sys.stderr)
        return 0.0

    try:
        print(f"Running classify on '{fireball_jpg_path.name}'...")
        classify_cmd = [sys.executable, str(CLASSIFY_PATH), "predict", str(fireball_jpg_path)]
        result = subprocess.run(classify_cmd, check=True, capture_output=True, text=True)
        
        # 3. Parse the output (e.g., "fireball.jpg (image): 0.999766")
        output_line = result.stdout.strip()
        probability_str = output_line.split(' ')[-1]
        probability = float(probability_str)
        print(f"Meteor probability: {probability:.4f}")
        return probability
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Error getting probability from classify.py: {e}", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        return 0.0


def update_event_file(config: configparser.ConfigParser, video_output: list, event_file_path: Path, probability: float):
    """Adds a [summary] section to the event file with key results."""
    original_duration = config.getfloat('trail', 'duration')
    original_arc = config.getfloat('trail', 'arc')
    start_az, start_alt = float(video_output[-5]), float(video_output[-4])
    end_az, end_alt = float(video_output[-2]), float(video_output[-1])
    recalibrated_arc = haversine_arc(start_az, start_alt, end_az, end_alt)
    duration = original_duration * (recalibrated_arc / original_arc) if original_arc > 0 else 0
    sun_alt, _ = get_sun_position(config)

    if not config.has_section('summary'):
        config.add_section('summary')

    config.set('summary', 'latitude', config.get('astronomy', 'latitude'))
    config.set('summary', 'longitude', config.get('astronomy', 'longitude'))
    config.set('summary', 'elevation', config.get('astronomy', 'elevation'))
    config.set('summary', 'timestamp', config.get('video', 'start'))
    config.set('summary', 'startpos', f"{start_az} {start_alt}")
    config.set('summary', 'endpos', f"{end_az} {end_alt}")
    config.set('summary', 'duration', str(round(duration, 2)))
    config.set('summary', 'sunalt', str(round(sun_alt, 1)))
    config.set('summary', 'recalibrated', "0" if sun_alt > -10 else "1")
    config.set('summary', 'meteor_probability', f"{probability:.6f}")

    with open(event_file_path, 'w') as configfile:
        config.write(configfile)
    print(f"Updated event file with summary and probability: {event_file_path.name}")


def upload_results(config: configparser.ConfigParser, event_dir: Path):
    """Uploads the event directory to the server and pings the report URL."""
    try:
        with open(SSH_TUNNEL_CONFIG_PATH) as f:
            port = next((line.strip().split('=')[1] for line in f if line.startswith('PORT=')), '0')
    except FileNotFoundError:
        port = '0'

    station_name = config.get('station', 'name')
    if int(port) == 0:
        print("SSH tunnel port is 0, using lftp to upload...")
        remote_path = f"upload/meteor/{station_name}/"
        command = ['lftp', '-e', f'mirror -R {event_dir} {remote_path}', 'norskmeteornettverk.no']
        subprocess.run(command)
    else:
        print(f"SSH tunnel is active on port {port}. Not using lftp.")

    print("Pinging report URL...")
    report_command = [
        'curl', '-s', '-o', '/dev/null',
        f'{REMOTE_REPORT_URL}?station={station_name}&port={port}&dir={event_dir.name}'
    ]
    subprocess.run(report_command)
    print("Upload and reporting complete.")


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <event.txt>")
        sys.exit(1)

    event_file_path = Path(sys.argv[1]).resolve()
    if not event_file_path.is_file():
        print(f"Error: File not found at {event_file_path}", file=sys.stderr)
        sys.exit(1)

    event_dir = event_file_path.parent
    config = load_config(event_file_path)

    with acquire_lock():
        video_output = run_video_creation(config, event_file_path)
        if not video_output:
            sys.exit(1)

        video_start_str = config.get('video', 'start')
        start_dt = parser.parse(video_start_str)
        station_name = config.get('station', 'name')
        event_timestamp_str = start_dt.strftime('%Y%m%d%H%M%S')
        video_name = f"{station_name}-{event_timestamp_str}"

        generate_reports(config, video_output, event_dir, video_name)
        
        # New classification and reporting logic
        probability = get_meteor_probability(event_dir)
        
        update_event_file(config, video_output, event_file_path, probability)

        if probability >= METEOR_PROBABILITY_THRESHOLD:
            print(f"Probability ({probability:.4f}) is >= {METEOR_PROBABILITY_THRESHOLD}. Reporting to server.")
            upload_results(config, event_dir)
        else:
            print(f"Probability ({probability:.4f}) is < {METEOR_PROBABILITY_THRESHOLD}. Not reporting to server.")

if __name__ == "__main__":
    main()
