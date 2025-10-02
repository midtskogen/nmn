#!/usr/bin/env python3

import sys
import json
import os
import logging
import urllib.request
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Try to import third-party libraries ---
# This script requires the 'skyfield' library for orbital mechanics calculations
# and 'numpy' for efficient vector and matrix operations.
try:
    from skyfield.api import load, EarthSatellite, wgs84
    import numpy as np
except ImportError as e:
    print(json.dumps({"error": f"A server-side dependency is missing. Please check logs. Missing: {e}"}))
    logging.exception(f"A critical dependency is missing: {e}")
    sys.exit(1)

# --- Import from our new shared utility library ---
# Imports utility functions shared across multiple backend prediction scripts.
from prediction_utils import (
    update_status,
    is_sky_coord_in_view,
    PTO_MAPPER_AVAILABLE,
    BASE_DIR, LOG_DIR, LOCK_DIR, CACHE_DIR, STATIONS_FILE, CAMERAS_FILE
)

# --- Try to import pto_mapper.py ---
# The PTO_MAPPER_AVAILABLE flag, imported from prediction_utils, determines if
# camera calibration features can be used.
if PTO_MAPPER_AVAILABLE:
    try:
        from pto_mapper import get_pto_data_from_json
    except ImportError:
        # This case is handled by the flag, but prevents a crash if pto_mapper is missing.
        pass

# --- Configuration specific to this script ---
TLE_FILE = os.path.join(BASE_DIR, 'tle.json')
PASS_CACHE_FILE = os.path.join(CACHE_DIR, 'pass_cache.json')
LOG_FILE = os.path.join(LOG_DIR, 'predict_sat.log')

# --- Script settings ---
MAX_LOG_LINES = 10000
TLE_UPDATE_INTERVAL_HOURS = 4 # How often to fetch fresh TLE data.
PASS_CACHE_LIFETIME_MINUTES = 235 # How long to use cached pass predictions. Set just under TLE interval.
SEARCH_DAYS = 7 # How many days into the past to search for passes.
MAX_VISIBLE_MAGNITUDE = 5.0 # The faintest satellite magnitude to consider. Lower is brighter.
MAXIMUM_SUN_ALT = -9 # The maximum sun altitude for a pass to be considered "dark enough". (-6 is civil twilight)

# A dictionary of specific satellites to track by name and NORAD catalog number.
SATELLITES_OF_INTEREST = {
    "ISS (ZARYA)": 25544, "LACROSSE 5": 28646, "PAZ": 43215, "CSG-1": 47219,
    "TERRA": 25994, "AQUA": 27424, "GENESIS 2": 31820, "TANDEM-X": 36605,
    "SARAL": 39086, "OCEANSAT-2": 35931,
}
# A dictionary of estimated absolute magnitudes for the satellites of interest.
SATELLITE_MAGNITUDES = {
    "ISS (ZARYA)": -2.0, "LACROSSE 5": 1.5, "PAZ": 2.0, "CSG-1": 2.5, "TERRA": 2.8,
    "AQUA": 2.8, "GENESIS 2": 3.0, "TANDEM-X": 3.0, "SARAL": 3.0, "OCEANSAT-2": 3.0,
}


# --- Worker Initialization ---
# These global variables are populated once per worker process to avoid redundant loading.
worker_ts = None
worker_eph = None

def init_worker():
    """
    Initializer for each worker process in the ProcessPoolExecutor.
    It loads Skyfield's timescale and ephemeris data into the process's global memory.
    """
    global worker_ts, worker_eph
    logging.info(f"Initializing worker process {os.getpid()}...")
    worker_ts = load.timescale()
    eph_data = load('de421.bsp') # Planetary ephemeris data file.
    worker_eph = {'sun': eph_data['sun'], 'earth': eph_data['earth']}


# --- Helper Functions ---
def trim_log_file(log_path, max_lines):
    """Trims a log file to a maximum number of lines, keeping the most recent ones."""
    try:
        if not os.path.exists(log_path): return
        with open(log_path, 'r') as f: lines = f.readlines()
        if len(lines) > max_lines:
            logging.info(f"Trimming log file {os.path.basename(log_path)} from {len(lines)} to {max_lines} lines.")
            with open(log_path, 'w') as f: f.writelines(lines[-max_lines:])
    except Exception as e:
        logging.error(f"Could not trim log file {log_path}: {e}")

def get_tle_data(ts):
    """
    Fetches and caches Two-Line Element (TLE) data for satellites from CelesTrak.
    TLE data describes the orbits of satellites and is required for position prediction.
    """
    cached_data = None
    # Check if a recent and complete TLE cache file exists.
    if os.path.exists(TLE_FILE) and (datetime.now().timestamp() - os.path.getmtime(TLE_FILE)) / 3600 < TLE_UPDATE_INTERVAL_HOURS:
        with open(TLE_FILE, 'r') as f: cached_data = json.load(f)
        if all(sat_name in cached_data for sat_name in SATELLITES_OF_INTEREST): return cached_data
    
    logging.info("Cache is stale or incomplete. Forcing fresh TLE download.")
    tle_data = {}
    # List of CelesTrak TLE sources to query.
    sources = [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle", "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle", "https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=radar&FORMAT=tle", "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    ]
    for source_url in sources:
        try:
            req = urllib.request.Request(source_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                lines = response.read().decode('utf-8').strip().splitlines()
            # TLE data comes in 3-line sets (Name, Line 1, Line 2).
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines): continue
                name, line1, line2 = lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()
                try: satnum = int(line1[2:7])
                except (ValueError, IndexError): continue
                # Keep the TLE if it's one of the satellites we are interested in.
                if name in SATELLITES_OF_INTEREST or satnum in SATELLITES_OF_INTEREST.values():
                    temp_sat = EarthSatellite(line1, line2, name, ts)
                    tle_data[name] = {'satnum': satnum, 'line1': line1, 'line2': line2, 'inclination': temp_sat.model.inclo}
        except Exception as e:
            logging.error(f"Could not process TLE from {source_url}: {e}")
            if cached_data: return cached_data # Return stale data if fresh download fails.
            else: return {"error": f"Failed to download TLE data: {e}"}
    with open(TLE_FILE, 'w') as f: json.dump(tle_data, f)
    return tle_data


# --- Worker and Main Logic ---
def process_station(args):
    """
    Worker function for the process pool. Calculates all visible satellite passes
    for a single ground station over the defined search period.
    """
    station_id, station_info, tle_data = args
    ts, eph = worker_ts, worker_eph

    sun, earth = eph['sun'], eph['earth']
    station_passes = []
    location = wgs84.latlon(station_info['astronomy']['latitude'], station_info['astronomy']['longitude'], elevation_m=station_info['astronomy']['elevation'])
    
    # Define the time window for the search.
    utc_now = datetime.now(timezone.utc)
    end_dt = utc_now.replace(minute=0, second=0, microsecond=0)
    start_dt = (end_dt - timedelta(days=SEARCH_DAYS)).replace(hour=12, minute=0, second=0, microsecond=0)

    end_time = ts.utc(end_dt)
    start_time = ts.utc(start_dt)

    for name, data in tle_data.items():
        if not all(k in data for k in ['line1', 'line2', 'inclination']): continue
        satellite = EarthSatellite(data['line1'], data['line2'], name, ts)
        # A satellite can't be seen from a latitude higher than its inclination. This is a quick filter.
        if abs(np.deg2rad(station_info['astronomy']['latitude'])) > data['inclination']: continue

        # Use skyfield's `find_events` to find when the satellite rises, culminates, and sets.
        times, events = satellite.find_events(location, start_time, end_time, altitude_degrees=20.0)

        for i, event_type in enumerate(events):
            # A pass is a sequence of rise (0), culmination (1), and set (2).
            if event_type == 0 and i + 2 < len(events) and np.all(events[i+1:i+3] == [1, 2]):
                rise_time, culminate_time, set_time = times[i : i+3]

                # --- Visibility Checks ---
                # 1. The satellite must be illuminated by the sun.
                if not satellite.at(culminate_time).is_sunlit(eph): continue
                # 2. The observer on the ground must be in darkness.
                if (earth + location).at(culminate_time).observe(sun).apparent().altaz()[0].degrees > MAXIMUM_SUN_ALT: continue
                
                # 3. Estimate the satellite's apparent magnitude (brightness).
                abs_mag = SATELLITE_MAGNITUDES.get(name, 99.0)
                topocentric_culminate = (satellite - location).at(culminate_time)
                r_km = topocentric_culminate.distance().km
                # The following calculates the phase angle to adjust brightness.
                sat_pos_au = satellite.at(culminate_time).position.au
                obs_pos_au = (earth + location).at(culminate_time).position.au
                sun_pos_au = sun.at(culminate_time).position.au
                vec_obs_sat, vec_sun_sat = sat_pos_au - obs_pos_au, sat_pos_au - sun_pos_au
                dot_product = np.dot(vec_obs_sat, vec_sun_sat)
                mag_obs_sat, mag_sun_sat = np.linalg.norm(vec_obs_sat), np.linalg.norm(vec_sun_sat)
                phi = np.arccos(dot_product / (mag_obs_sat * mag_sun_sat))
                phase_factor = (np.sin(phi) + (np.pi - phi) * np.cos(phi)) / np.pi
                est_mag = abs_mag + 5 * np.log10(r_km / 1000.0)
                if phase_factor > 0: est_mag += -2.5 * np.log10(phase_factor)
                if est_mag >= MAX_VISIBLE_MAGNITUDE: continue

                # If all checks pass, calculate the detailed track for the pass.
                pass_duration_seconds = (set_time - rise_time) * 86400.0
                num_steps = int(pass_duration_seconds / 5) or 2 # One point every ~5 seconds.
                pass_times = ts.linspace(rise_time, set_time, num_steps)
                
                # Calculate the satellite's position as seen from the station (az, alt).
                topocentric_pass = (satellite - location).at(pass_times)
                alt, az, _ = topocentric_pass.altaz()
                alt_degs, az_degs = alt.degrees, az.degrees
                
                # Calculate the satellite's ground track (subpoint on Earth's surface).
                subpoints = wgs84.subpoint(satellite.at(pass_times))
                ground_lat, ground_lon = subpoints.latitude.degrees, subpoints.longitude.degrees
                
                camera_views = []
                # Check visibility for each of the station's cameras.
                for cam_num in range(1, 8):
                    pto_data = None
                    if PTO_MAPPER_AVAILABLE:
                        try: pto_data = get_pto_data_from_json(CAMERAS_FILE, f"{station_id.replace('ams', '')}:{cam_num}")
                        except Exception: continue
                    
                    in_view_start, in_view_end = None, None
                    for j, t in enumerate(pass_times):
                        # Use prediction_utils to check if the (az, alt) point is within the camera's FoV.
                        is_in_view, _ = is_sky_coord_in_view(pto_data, az_degs[j], alt_degs[j]) if PTO_MAPPER_AVAILABLE else (True, None)
                        if is_in_view:
                            if in_view_start is None: in_view_start = t
                            in_view_end = t
                        elif in_view_start is not None:
                            # If the satellite leaves the view, finalize the camera view event.
                            camera_views.append({"camera": cam_num, "station_code": station_info['station']['code'], "station_id": station_id, "start_utc": in_view_start.utc_iso(), "end_utc": in_view_end.utc_iso()})
                            in_view_start = None
                    if in_view_start is not None:
                        camera_views.append({"camera": cam_num, "station_code": station_info['station']['code'], "station_id": station_id, "start_utc": in_view_start.utc_iso(), "end_utc": in_view_end.utc_iso()})

                if camera_views:
                    # If the pass was visible to at least one camera, store its full data.
                    ground_track = [{'lat': lat, 'lon': lon, 'time': t.utc_iso()} for lat, lon, t in zip(ground_lat, ground_lon, pass_times)]
                    sky_track = [{'alt': round(alt, 2), 'az': round(az, 2), 'time': t.utc_iso()} for alt, az, t in zip(alt_degs, az_degs, pass_times)]
                    station_passes.append({
                        "pass_group_id": f"{name}-{round(culminate_time.tt * 24 * 4)}",
                        "satellite": name, "magnitude": est_mag, "ground_track": ground_track,
                        "sky_track": sky_track, "camera_views": camera_views
                    })
    return station_passes

def _group_and_finalize_passes(all_passes_found):
    """
    Groups passes of the same satellite that occur at the same time across different stations.
    It creates a single "pass" entry that contains all camera views from all involved stations.
    """
    grouped_by_id = {}
    for p in all_passes_found:
        group_id = p['pass_group_id']
        if group_id not in grouped_by_id:
            grouped_by_id[group_id] = []
        grouped_by_id[group_id].append(p)

    final_passes = []
    for group_id, passes_in_group in grouped_by_id.items():
        passes_in_group.sort(key=lambda p: (p['magnitude'], p['camera_views'][0]['station_id']))
        master_pass = passes_in_group[0]

        all_camera_views = []
        min_magnitude = master_pass['magnitude']
        station_sky_tracks = {}

        # Consolidate data from all stations in the group.
        for p in passes_in_group:
            all_camera_views.extend(p['camera_views'])
            if p['magnitude'] < min_magnitude:
                min_magnitude = p['magnitude']
            
            station_id = p['camera_views'][0]['station_id']
            station_sky_tracks[station_id] = p['sky_track'] # Store the station-specific sky track.

        rounded_ground_track = [
            {'lat': round(p['lat'], 5), 'lon': round(p['lon'], 5), 'time': p['time']}
            for p in master_pass['ground_track']
        ]

        final_pass = {
            "pass_id": group_id,
            "satellite": master_pass['satellite'],
            "magnitude": round(min_magnitude, 1),
            "ground_track": rounded_ground_track,
            "station_sky_tracks": station_sky_tracks,
            "camera_views": all_camera_views,
            "earliest_camera_utc": min(cv['start_utc'] for cv in all_camera_views)
        }
        final_passes.append(final_pass)

    final_passes.sort(key=lambda p: p['earliest_camera_utc'], reverse=True)
    return final_passes

def find_all_passes_for_cron():
    """
    A special version of the prediction function intended to be run by a cron job.
    It runs silently and its only purpose is to update the main pass cache file.
    """
    if os.path.exists(PASS_CACHE_FILE):
        try:
            mod_time = os.path.getmtime(PASS_CACHE_FILE)
            lifetime_seconds = PASS_CACHE_LIFETIME_MINUTES * 60
            if (datetime.now().timestamp() - mod_time) < lifetime_seconds:
                with open(PASS_CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                # Exit early if the cache is still valid and contains the correct set of satellites.
                if set(SATELLITES_OF_INTEREST.keys()) == set(cached_data.get("satellites_in_cache", [])):
                    logging.info("Cron run: Cache is still valid. Exiting.")
                    return
        except (IOError, json.JSONDecodeError):
             logging.warning("Cache file exists but is invalid. Proceeding with recalculation.")

    logging.info("--- Starting cron pass prediction (cache invalid or missing) ---")
    try:
        ts = load.timescale()
        with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)
        tle_data = get_tle_data(ts)
        if "error" in tle_data:
            logging.error(f"Failed to get TLE data in cron mode: {tle_data['error']}")
            return
        all_passes_found = []
        
        tasks = [(sid, sinfo, tle_data) for sid, sinfo in stations_data.items()]
        with ProcessPoolExecutor(initializer=init_worker) as executor:
            results = executor.map(process_station, tasks)
            for station_result in results:
                all_passes_found.extend(station_result)
        
        final_passes = _group_and_finalize_passes(all_passes_found)
        result_data = {"passes": final_passes}
        with open(PASS_CACHE_FILE, 'w') as f:
            json.dump({"satellites_in_cache": list(SATELLITES_OF_INTEREST.keys()), "data": result_data}, f, indent=2)
        logging.info(f"Cron prediction finished. Found {len(final_passes)} passes and updated cache.")
    except Exception as e:
        logging.exception(f"An unhandled error occurred during cron pass prediction")

def find_all_passes(task_id):
    """
    Main orchestrator function for an on-demand pass prediction request.
    It first tries to serve from cache, and if the cache is invalid, it recalculates
    all passes, reports progress, and updates the cache.
    """
    if os.path.exists(PASS_CACHE_FILE):
        try:
            mod_time = os.path.getmtime(PASS_CACHE_FILE)
            lifetime_seconds = PASS_CACHE_LIFETIME_MINUTES * 60
            if (datetime.now().timestamp() - mod_time) < lifetime_seconds:
                with open(PASS_CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                if set(SATELLITES_OF_INTEREST.keys()) == set(cached_data.get("satellites_in_cache", [])):
                    logging.info(f"Task {task_id}: Found valid cache. Serving from cache.")
                    update_status(os.path.join(LOCK_DIR, f"{task_id}.json"), "complete", {"data": cached_data.get("data", {})})
                    return
        except (IOError, json.JSONDecodeError):
            logging.warning(f"Task {task_id}: Cache file is invalid. Recalculating.")

    logging.info(f"--- Starting pass prediction for task {task_id} (cache invalid or missing) ---")
    try:
        status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
        ts = load.timescale()
        with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)
        tle_data = get_tle_data(ts)
        if "error" in tle_data:
            update_status(status_file, "error", {"message": tle_data["error"]})
            return
        all_passes_found = []
        
        tasks = [(sid, sinfo, tle_data) for sid, sinfo in stations_data.items()]
        total_stations = len(tasks)
        stations_processed = 0
        update_status(status_file, "progress", {"step": 5, "total": 100, "message": "Beregner..."})
        with ProcessPoolExecutor(initializer=init_worker) as executor:
            futures = [executor.submit(process_station, task) for task in tasks]
            # As each worker process completes, update the overall progress.
            for future in as_completed(futures):
                try:
                    station_result = future.result()
                    all_passes_found.extend(station_result)
                except Exception as exc:
                    logging.error(f'A station task generated an exception: {exc}')
                stations_processed += 1
                progress = 5 + int((stations_processed / total_stations) * 90)
                message = f"Beregner for stasjon {stations_processed}/{total_stations}..."
                update_status(status_file, "progress", {"step": progress, "total": 100, "message": message})

        update_status(status_file, "progress", {"step": 95, "total": 100, "message": "Grupperer og ferdigstiller resultatet..."})
        final_passes = _group_and_finalize_passes(all_passes_found)
        result_data = {"passes": final_passes}
        with open(PASS_CACHE_FILE, 'w') as f:
            json.dump({"satellites_in_cache": list(SATELLITES_OF_INTEREST.keys()), "data": result_data}, f, indent=2)
        logging.info(f"Finished prediction for task {task_id}. Found {len(final_passes)} passes.")
        update_status(status_file, "complete", {"data": result_data})
    except Exception as e:
        logging.exception(f"An unhandled error occurred during pass prediction for task {task_id}")
        update_status(status_file, "error", {"message": "En intern feil oppstod."})

def main():
    """Parses command-line arguments to run in either on-demand or cron mode."""
    script_path = os.path.abspath(__file__)
    epilog_text = f"""
How to run:
  1. For web integration (called from PHP):
     python3 predict_sat.py <task_id>
     - The script will report progress to a status file named <task_id>.json.
  2. For cron jobs (silent execution to update cache):
     python3 predict_sat.py --cron
     - This will run silently and update the pass_cache.json file.
Example cron job to run every 4 hours:
       0 */4 * * * /usr/bin/python3 {script_path} --cron
"""
    parser = argparse.ArgumentParser(
        description="Calculate satellite passes and cache the results.",
        formatter_class=argparse.RawTextHelpFormatter, epilog=epilog_text
    )
    parser.add_argument("task_id", nargs='?', default=None, help="The task ID provided by the PHP script for progress tracking.")
    parser.add_argument("--cron", action="store_true", help="Run in cron mode to silently update the cache file. This implies --quiet.")
    parser.add_argument("--quiet", action="store_true", help="Suppress all logging to files and the terminal.")
    args = parser.parse_args()

    is_quiet = args.quiet or args.cron
    if not is_quiet:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(LOG_FILE)]
        )
        logging.info("--- Script execution started ---")
        trim_log_file(LOG_FILE, MAX_LOG_LINES)
    else:
        logging.basicConfig(level=logging.CRITICAL + 1)

    if args.cron:
        find_all_passes_for_cron()
    elif args.task_id:
        status_file = os.path.join(LOCK_DIR, f"{args.task_id}.json")
        try:
            find_all_passes(args.task_id)
        except Exception as e:
            logging.exception("A fatal error occurred at the top level of the script!")
            update_status(status_file, "error", {"message": "En kritisk feil oppstod."})
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
