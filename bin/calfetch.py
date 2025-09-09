#!/usr/bin/env python3
"""
Connects to a list of meteor stations via SSH to fetch camera calibration
files (lens.pto), parses them, and compiles the data into a single
JSON file (cameras.json). It preserves existing data for any stations
or cameras that cannot be reached during the run.
"""

import json
import re
import os
import sys
import argparse
import concurrent.futures
import warnings
from threading import Lock

import paramiko
from tqdm import tqdm

# --- Constants ---
STATIONS_FILENAME = "stations.json"
CAMERAS_FILENAME = "cameras.json"
REMOTE_CAM_PATH_TPL = "/meteor/cam{}/lens.pto"
MAX_CAMERAS = 7

# Suppress paramiko's cryptography warnings which are not relevant here
warnings.filterwarnings(action='ignore', module='.*paramiko.*')

# --- Globals for Thread-Safe UI ---
print_lock = Lock()
status_lines = ["", ""]

# --- Core Logic ---

def log_error(*args, **kwargs):
    """Logs an error message to stderr, compatible with tqdm."""
    tqdm.write(*args, file=sys.stderr, **kwargs)

def parse_i_line(i_line: str) -> dict:
    """
    Parses a single 'i' line from a .pto file to extract calibration parameters.
    Ensures keys are strings and values are JSON-serializable.

    Args:
        i_line: The calibration line string, starting with 'i '.

    Returns:
        A dictionary of the parsed key-value parameters.
    """
    parameters = {}
    # This regex is designed to split by space while correctly handling
    # quoted strings (for the 'n' parameter) and various number formats.
    # It captures: 1. Quoted content | 2. Key-value pairs | 3. Other space-separated parts
    parts = re.findall(r'n\"(.*?)\"|([a-zA-Z]+-?\d*\.?\d*[eE]?-?\d+)|(\S+)', i_line)

    flat_parts = [p for group in parts for p in group if p]

    # Skip the initial 'i'
    for part in flat_parts[1:]:
        # Find the split point between the alphabetic key and the numeric/quoted value.
        match = re.search(r"[-+]?\d|n\"\"", part)
        if not match:
            continue  # Skip malformed parts

        key = part[:match.start()]
        value_str = part[match.start():]

        # Sanitize key to ensure it's a valid string
        key = re.sub(r'\W+', '', key)
        if not key:
            continue

        # Sanitize value
        if key == 'n' and value_str == '""':
            parameters[key] = ""
        else:
            try:
                if '.' in value_str or 'e' in value_str.lower():
                    parameters[key] = float(value_str)
                else:
                    parameters[key] = int(value_str)
            except (ValueError, TypeError):
                # If conversion fails, it's a malformed value. Skip it.
                continue

    return parameters

def process_station(station_id: str, quiet: bool, ssh_user: str = 'root', ssh_timeout: int = 10) -> tuple:
    """
    Connects to a station, fetches all lens.pto files, and parses them.
    Only returns data for successfully fetched and parsed files.

    Returns:
        A tuple: (station_id, dict_of_new_cam_data, stats_dict).
    """
    new_cam_data = {}
    stats = {'found': 0, 'missing': 0, 'errors': 0, 'failed_to_connect': False}

    try:
        ssh_config = paramiko.SSHConfig()
        config_path = os.path.expanduser("~/.ssh/config")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                ssh_config.parse(f)

        host_config = ssh_config.lookup(station_id)

        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=host_config.get('hostname', station_id),
                username=host_config.get('user', ssh_user),
                port=int(host_config.get('port', 22)),
                timeout=ssh_timeout
            )

            with ssh.open_sftp() as sftp:
                for i in range(1, MAX_CAMERAS + 1):
                    cam_name = f"cam{i}"
                    remote_path = REMOTE_CAM_PATH_TPL.format(i)
                    try:
                        with sftp.open(remote_path, 'r') as f:
                            i_line_found = next((line.strip() for line in f if line.strip().startswith('i ')), None)

                        if i_line_found:
                            params = parse_i_line(i_line_found)
                            new_cam_data[cam_name] = {"calibration": i_line_found, "parameters": params}
                            stats['found'] += 1
                            if not quiet:
                                update_status_line(1, f"[{station_id}] OK: {cam_name}")
                        else:
                            stats['errors'] += 1
                            if not quiet:
                                update_status_line(1, f"[{station_id}] No 'i' line in {cam_name}")
                    except FileNotFoundError:
                        stats['missing'] += 1
                        if not quiet:
                            update_status_line(1, f"[{station_id}] Not found: {cam_name}")
                    except Exception as e:
                        stats['errors'] += 1
                        if not quiet:
                            update_status_line(1, f"[{station_id}] Read error on {cam_name}: {e}")

    except (paramiko.AuthenticationException, paramiko.SSHException) as e:
        log_error(f"--> SSH AUTH/CONFIG ERROR for {station_id}: {e}")
        stats.update({'missing': MAX_CAMERAS, 'failed_to_connect': True})
    except Exception as e:
        log_error(f"--> FATAL ERROR connecting to {station_id}: {e}")
        stats.update({'missing': MAX_CAMERAS, 'failed_to_connect': True})

    return station_id, new_cam_data, stats

# --- UI and File I/O ---

def update_status_line(line_index: int, message: str):
    """Thread-safe update of a status line above the tqdm progress bar."""
    with print_lock:
        status_lines[line_index] = message
        sys.stdout.write("\x1b[s")  # Save cursor position
        sys.stdout.write(f"\x1b[{len(status_lines) + 1}A")  # Move cursor up
        for line in status_lines:
            sys.stdout.write(f"\r\x1b[K{line}\n") # Clear and write line
        sys.stdout.write("\x1b[u")  # Restore cursor position
        sys.stdout.flush()

def validate_stations_data(data: dict) -> dict:
    """Filters out malformed station entries from the input data."""
    validated_data = {}
    for station_id, info in data.items():
        if isinstance(info, dict) and 'astronomy' in info and 'station' in info:
            validated_data[station_id] = info
        else:
            log_error(f"Warning: Skipping malformed station entry: '{station_id}'")
    return validated_data

def print_final_summary(stats: dict, stations_data: dict, processed_this_run: dict, quiet: bool):
    """Prints a detailed summary of the script's execution."""
    if quiet:
        return
        
    print("\n" + "---" * 10)
    print("Download and Update Summary")
    print("---" * 10)
    print(f"Stations Processed: {stats['complete']}/{stats['total']}")
    print(f"New Calibration Files Found: {stats['found']}")
    print(f"Files Not Found (Missing): {stats['missing']}")
    print(f"File Read/Parse Errors: {stats['errors']}")

    missing_items = []
    for station_id in sorted(stations_data.keys()):
        run_info = processed_this_run.get(station_id, {})
        if run_info.get('failed_to_connect'):
            missing_items.append(f"  - {station_id} (Connection failed, using old data)")
            continue

        # Check each camera for this station
        for i in range(1, MAX_CAMERAS + 1):
            cam_name = f"cam{i}"
            # A cam is missing for this run if it wasn't successfully found
            if cam_name not in run_info.get('found_cams', []):
                missing_items.append(f"  - {station_id}/{cam_name}")

    if missing_items:
        print("\nThe following files could not be updated in this run (old data preserved):")
        for item in missing_items:
            print(item)
    elif stats['errors'] == 0 and stats['missing'] == 0:
        print("\nSuccess! All files from all stations were processed and updated.")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Fetch camera calibration data from meteor stations.")
    parser.add_argument(
        "-d", "--directory", default=".",
        help="Directory for stations.json (input) and cameras.json (output)."
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Silence all non-error output."
    )
    args = parser.parse_args()

    stations_json_path = os.path.join(args.directory, STATIONS_FILENAME)
    cameras_json_path = os.path.join(args.directory, CAMERAS_FILENAME)

    try:
        with open(stations_json_path, 'r', encoding='utf-8') as f:
            stations_data = validate_stations_data(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        log_error(f"Error loading {stations_json_path}: {e}")
        return

    try:
        with open(cameras_json_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        if not args.quiet:
            print(f"Loaded existing data from {cameras_json_path}.")
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        output_data = {}
        if not args.quiet:
            print(f"{cameras_json_path} not found or invalid. A new file will be created.")
    
    # Merge station info into output data
    for sid, s_info in stations_data.items():
        output_data.setdefault(sid, {}).update(s_info)

    total_stations = len(stations_data)
    summary_stats = {'complete': 0, 'found': 0, 'missing': 0, 'errors': 0, 'total': total_stations}
    processed_this_run = {}
    
    if not args.quiet:
        sys.stdout.write("\n" * (len(status_lines)))
        sys.stdout.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_stations) as executor:
        future_to_station = {executor.submit(process_station, sid, args.quiet): sid for sid in stations_data.keys()}
        
        pbar_desc = "Fetching Calibrations"
        pbar = tqdm(concurrent.futures.as_completed(future_to_station), total=total_stations, desc=pbar_desc, disable=args.quiet)
        for future in pbar:
            try:
                sid, new_data, stats = future.result()
                output_data[sid].update(new_data)
                processed_this_run[sid] = {
                    'found_cams': list(new_data.keys()),
                    'failed_to_connect': stats['failed_to_connect']
                }
                
                with print_lock:
                    summary_stats['complete'] += 1
                    for key in ['found', 'missing', 'errors']:
                        summary_stats[key] += stats[key]
                
                if not args.quiet:
                    msg = (f"Progress: {summary_stats['complete']}/{total_stations} | "
                           f"Found: {summary_stats['found']} | Missing: {summary_stats['missing']} | "
                           f"Errors: {summary_stats['errors']}")
                    update_status_line(0, msg)

            except Exception as exc:
                sid = future_to_station[future]
                log_error(f"  -> Critical error in thread for station {sid}: {exc}")

    if not args.quiet:
        update_status_line(0, "")
        update_status_line(1, "")

    try:
        with open(cameras_json_path, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted(output_data.items())), f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nSuccessfully created/updated {cameras_json_path}.")
    except IOError as e:
        log_error(f"\nError writing to {cameras_json_path}: {e}")

    print_final_summary(summary_stats, stations_data, processed_this_run, args.quiet)


if __name__ == "__main__":
    main()

