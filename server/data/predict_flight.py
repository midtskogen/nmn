#!/usr/bin/env python3

import sys
import json
import os
import logging
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import time
import io
import csv

# --- Try to import third-party libraries ---
# This script requires the 'requests' library to make API calls.
try:
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError as e:
    # This will log if the script is run directly and requests is missing
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"FATAL: A server-side dependency is missing: {e}")
    print(json.dumps({"status": "error", "message": f"Server-side dependency missing: {e}"}))
    sys.exit(1)

# --- Import from our new shared utility library ---
# Imports utility functions shared across multiple backend prediction scripts.
from prediction_utils import (
    update_status,
    haversine_distance,
    cross_track_distance,
    is_sky_coord_in_view,
    PTO_MAPPER_AVAILABLE,
    BASE_DIR, LOG_DIR, LOCK_DIR, CACHE_DIR, STATIONS_FILE, CAMERAS_FILE
)
from data_fetchers import get_air_pressure

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
TRACK_CACHE_DIR = os.path.join(CACHE_DIR, 'flight_tracks')
CREDENTIALS_FILE = os.path.join(BASE_DIR, 'credentials.json')
LOG_FILE = os.path.join(LOG_DIR, 'find_aircraft.log')
TOKEN_CACHE_FILE = os.path.join(CACHE_DIR, 'opensky_token.json')
AIRPORT_DB_FILE = os.path.join(CACHE_DIR, 'airports.json')
FLIGHT_DB_FILE = os.path.join(CACHE_DIR, 'flight_database.json')

# --- Script settings ---
MAX_LOG_LINES = 10000
# Distance from a station to consider a flight path for further processing.
SHORTLIST_VICINITY_KM = 50
DETAILED_VICINITY_KM = 25 # Not currently used, but intended for more precise filtering.
PROCESSING_CHUNK_SIZE = 50 # Limits how many flights are processed in a single run to manage load.
CHUNK_DELAY_SECONDS = 5 # Not currently used.
TRACK_CACHE_HOURS = 24 # How long to keep cached flight track data.
REQUEST_RETRY_COUNT = 3 # Number of retries for failed API requests.
REQUEST_RETRY_DELAY_S = 2 # Delay between retries.
# --- WGS84 Ellipsoid Constants for ECEF coordinate conversion ---
WGS84_A = 6378137.0 # Major axis (radius)
WGS84_E2 = 0.00669437999014 # Eccentricity squared

# --- Setup Directories ---
for d in [TRACK_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


def get_airport_db(task_id):
    """
    Downloads and caches a CSV of global airports for resolving airport ICAO codes to names and locations.
    """
    if os.path.exists(AIRPORT_DB_FILE): return json.load(open(AIRPORT_DB_FILE))
    logging.info(f"Task {task_id}: Airport database not found, downloading...")
    url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    db = {}
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_file = io.StringIO(response.text)
        csv_reader = csv.reader(csv_file)
        next(csv_reader) # Skip header row
        for row in csv_reader:
            try:
                if len(row) > 13:
                    ident, iata, lat, lon = row[1], row[13], row[4], row[5]
                    if ident and lat and lon: db[ident] = {'lat': float(lat), 'lon': float(lon), 'iata': iata}
            except (ValueError, IndexError): continue
        with open(AIRPORT_DB_FILE, 'w') as f: json.dump(db, f, indent=2)
        logging.info(f"Task {task_id}: Successfully built airport database with {len(db)} entries.")
        return db
    except requests.exceptions.RequestException as e:
        logging.error(f"Task {task_id}: Failed to download airport database: {e}")
        return {}

def get_opensky_token(task_id, client_id, client_secret):
    """
    Retrieves an OAuth access token from the OpenSky Network API, caching it for reuse.
    """
    if os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, 'r') as f: token_data = json.load(f)
            # Use cached token if it's not
            # close to expiring.
            if time.time() < token_data.get('expires_at', 0) - 60: return token_data.get('access_token')
        except (json.JSONDecodeError, KeyError): pass
    logging.info(f"Task {task_id}: Fetching new OpenSky access token.")
    try:
        response = requests.post("https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token", data={"grant_type": "client_credentials"}, headers={"Content-Type": "application/x-www-form-urlencoded"}, auth=HTTPBasicAuth(client_id, client_secret), timeout=20)
        response.raise_for_status()
        data = response.json()
        data['expires_at'] = time.time() + data.get('expires_in', 300)
        with open(TOKEN_CACHE_FILE, 'w') as f: json.dump(data, f)
        return data.get('access_token')
    except requests.exceptions.RequestException as e:
        logging.error(f"Task {task_id}: Failed to get access token from OpenSky. Error: {e}")
        return None

def update_and_get_flight_database(task_id, access_token):
    """
    Maintains a local JSON database of flights from the last 24 hours.
    It prunes old entries and fetches new data since the last update.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    full_24h_ago = now_ts - (24 * 3600)
    flight_db = {}
    if os.path.exists(FLIGHT_DB_FILE):
        try:
            with open(FLIGHT_DB_FILE, 'r') as f: db_data = json.load(f)
            if isinstance(db_data, dict) and 'flights' in db_data: flight_db = db_data.get('flights', {})
        except (IOError, json.JSONDecodeError): pass
    # Prune flights that were last seen more than
    # 24 hours ago.
    pruned_db = {k: v for k, v in flight_db.items() if isinstance(v, dict) and v.get('lastSeen', 0) >= full_24h_ago}
    # Determine the start time for fetching new data to fill in gaps.
    fetch_start_ts = max(full_24h_ago, max((v['lastSeen'] for v in pruned_db.values()), default=0) - 3600) if pruned_db else full_24h_ago
    logging.info(f"Task {task_id}: Fetching flight data from {datetime.fromtimestamp(fetch_start_ts, tz=timezone.utc)} to {datetime.fromtimestamp(now_ts, tz=timezone.utc)}")
    all_new_flights = []
    # Fetch data in 2-hour chunks to stay within API limits.
    for chunk_start in range(fetch_start_ts, now_ts, 7200):
        chunk_end = min(chunk_start + 7200, now_ts)
        if chunk_end - chunk_start < 60: continue
        try:
            response = requests.get(f"https://opensky-network.org/api/flights/all?begin={chunk_start}&end={chunk_end}", headers={"Authorization": f"Bearer {access_token}"}, timeout=90)
            response.raise_for_status()
            all_new_flights.extend(response.json())
            time.sleep(2) # Be courteous to the
            # API.
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Task {task_id}: Could not fetch flight data chunk. Error: {e}")
            return list(pruned_db.values()), fetch_start_ts
    for flight in all_new_flights: pruned_db[f"{flight['icao24']}-{flight['firstSeen']}"] = flight
    with open(FLIGHT_DB_FILE, 'w') as f: json.dump({"last_update_ts": now_ts, "flights": pruned_db}, f, indent=2)
    logging.info(f"Task {task_id}: Flight database updated. Total flights in window: {len(pruned_db)}.")
    return list(pruned_db.values()), fetch_start_ts

def get_flight_track(task_id, icao24, time_within_flight, access_token):
    """Fetches the full trajectory (track) for a specific flight,
    using a local cache."""
    cache_file = os.path.join(TRACK_CACHE_DIR, f"{icao24}-{time_within_flight}.json")
    if os.path.exists(cache_file):
        if (time.time() - os.path.getmtime(cache_file)) < (TRACK_CACHE_HOURS * 3600):
            with open(cache_file, 'r') as f: return json.load(f)
    url = f"https://opensky-network.org/api/tracks/all?icao24={icao24}&time={time_within_flight}"
    headers = {"Authorization": f"Bearer {access_token}"}
    for attempt in range(REQUEST_RETRY_COUNT):
        try:
            response = requests.get(url, headers=headers, timeout=45)
            if response.status_code == 404: return None # No track available for this flight.
            response.raise_for_status()
            data = response.json()
            with open(cache_file, 'w') as f: json.dump(data, f)
            return data
        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code == 500 and attempt < REQUEST_RETRY_COUNT - 1:
                logging.warning(f"Task {task_id}: Server error for {icao24}. Retrying in {REQUEST_RETRY_DELAY_S}s...")
                time.sleep(REQUEST_RETRY_DELAY_S)
            else:
                logging.error(f"Task {task_id}: Failed to get track for {icao24}. Error: {e}")
                return None
    return None

def enu_to_az_alt(e, n, u):
    """Converts East, North, Up (ENU) coordinates to Azimuth and Altitude."""
    az = math.degrees(math.atan2(e, n)) % 360
    alt = math.degrees(math.asin(u / math.sqrt(e**2 + n**2 + u**2)))
    return az, alt

def interpolate_raw_track(path, max_interval_sec):
    """Fills in gaps in a raw flight track from OpenSky by linear interpolation."""
    if not path or len(path) < 2:
        return path
    new_path = [path[0]]
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        time_diff = p2[0] - p1[0]
        if time_diff > max_interval_sec:
            num_segments = math.ceil(time_diff / max_interval_sec)
            for j in range(1, int(num_segments)):
                fraction = j / num_segments
                interp_point = [None] * len(p1)
                interp_point[0] = p1[0] + fraction * time_diff
                interp_point[1] = p1[1] + fraction * (p2[1] - p1[1])
                interp_point[2] = p1[2] + fraction * (p2[2] - p1[2])
                
                if p1[3] is not None and p2[3] is not None:
                    interp_point[3] = p1[3] + fraction * (p2[3] - p1[3])
                else:
                    interp_point[3] = p1[3] or p2[3]
                
                if len(p1) > 9 and len(p2) > 9:
                    if p1[9] is not None and p2[9] is not None:
                        interp_point[9] = p1[9] + fraction * (p2[9] - p1[9])
                    else:
                        interp_point[9] = p1[9] or p2[9]
                
                new_path.append(interp_point)
        new_path.append(p2)
    return new_path

def fetch_and_process_track(args):
    """
    Worker function for the process pool.
    Takes a flight, fetches its full track,
    and calculates if/when it was visible to any of the camera stations.
    """
    flight_info, stations_data, task_id, access_token = args
    logging.info(f"Worker {os.getpid()}: Starting processing for flight ICAO {flight_info['icao24']}.")

    track_data = get_flight_track(task_id, flight_info['icao24'], flight_info['firstSeen'], access_token)
    if not track_data or not track_data.get('path'):
        logging.warning(f"Worker {os.getpid()}: No track data found for {flight_info['icao24']}.")
        return None

    pressure_hpa = None
    needs_correction = any(len(p) > 3 and p[3] is not None for p in track_data['path']) and \
                       not all(len(p) > 9 and p[9] is not None for p in track_data['path'])
    
    if needs_correction:
        mid_point = track_data['path'][len(track_data['path']) // 2]
        p_time, p_lat, p_lon = mid_point[0], mid_point[1], mid_point[2]
        if p_lat and p_lon:
            dt_utc = datetime.fromtimestamp(p_time, tz=timezone.utc)
            logging.info(f"Worker {os.getpid()}: Calling get_air_pressure for lat={p_lat}, lon={p_lon}, time={dt_utc}")
            pressure_hpa = get_air_pressure(p_lat, p_lon, dt_utc)
            logging.info(f"Worker {os.getpid()}: get_air_pressure returned: {pressure_hpa}")

    if needs_correction and pressure_hpa is None:
        logging.warning(f"Worker {os.getpid()}: Using standard pressure 1013.25 hPa as fallback for {flight_info['icao24']}.")

    interpolated_path = interpolate_raw_track(track_data['path'], 15)
    all_visible_points = []
    
    for station_id, station_info in stations_data.items():
        lat_ref, lon_ref, alt_ref = station_info['astronomy']['latitude'], station_info['astronomy']['longitude'], station_info['astronomy']['elevation']
        lat_rad_ref, lon_rad_ref = map(math.radians, [lat_ref, lon_ref])
        sin_lat_ref, cos_lat_ref = math.sin(lat_rad_ref), math.cos(lat_rad_ref)
        sin_lon_ref, cos_lon_ref = math.sin(lon_rad_ref), math.cos(lon_rad_ref)
        N_ref = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat_ref**2)
        x0, y0, z0 = (N_ref + alt_ref) * cos_lat_ref * cos_lon_ref, (N_ref + alt_ref) * cos_lat_ref * sin_lon_ref, (N_ref * (1 - WGS84_E2) + alt_ref) * sin_lat_ref

        for point in interpolated_path:
            p_time, p_lat, p_lon, baro_alt = point[0], point[1], point[2], point[3]
            geo_alt = point[9] if len(point) > 9 and point[9] is not None else None
            p_alt_uncorrected = geo_alt if geo_alt is not None else baro_alt

            if not all([p_lat, p_lon, p_alt_uncorrected]): continue
            
            lat_rad_ac_u, lon_rad_ac_u = map(math.radians, [p_lat, p_lon])
            sin_lat_ac_u = math.sin(lat_rad_ac_u)
            N_ac_u = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat_ac_u**2)
            x_ac_u = (N_ac_u + p_alt_uncorrected) * math.cos(lat_rad_ac_u) * math.cos(lon_rad_ac_u)
            y_ac_u = (N_ac_u + p_alt_uncorrected) * math.cos(lat_rad_ac_u) * math.sin(lon_rad_ac_u)
            z_ac_u = (N_ac_u * (1 - WGS84_E2) + p_alt_uncorrected) * sin_lat_ac_u
            
            dx_u, dy_u, dz_u = x_ac_u - x0, y_ac_u - y0, z_ac_u - z0
            e_u, n_u, u_u = -sin_lon_ref*dx_u + cos_lon_ref*dy_u, -sin_lat_ref*cos_lon_ref*dx_u - sin_lat_ref*sin_lon_ref*dy_u + cos_lat_ref*dz_u, cos_lat_ref*cos_lon_ref*dx_u + cos_lat_ref*sin_lon_ref*dy_u + sin_lat_ref*dz_u
            _, alt_deg_uncorrected = enu_to_az_alt(e_u, n_u, u_u)
        
            if alt_deg_uncorrected > 10:
                p_alt_for_plot = p_alt_uncorrected
                if geo_alt is None:
                    # If the API call failed, use standard pressure (1013.25 hPa).
                    # This results in a zero correction, a safe fallback.
                    pressure_to_use = pressure_hpa if pressure_hpa is not None else 1013.25

                    pressure_deviation = 1013.25 - pressure_to_use
                    altitude_correction = pressure_deviation * 8.23
                    p_alt_for_plot = baro_alt + altitude_correction
                    
                    if pressure_hpa is not None:
                         logging.info(f"Worker {os.getpid()}: Correcting altitude for {flight_info['icao24']}. Original: {baro_alt:.1f}m, Corrected: {p_alt_for_plot:.1f}m")

                lat_rad_ac, lon_rad_ac = map(math.radians, [p_lat, p_lon])
                sin_lat_ac = math.sin(lat_rad_ac)
                N_ac = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat_ac**2)
                x_ac = (N_ac + p_alt_for_plot) * math.cos(lat_rad_ac) * math.cos(lon_rad_ac)
                y_ac = (N_ac + p_alt_for_plot) * math.cos(lat_rad_ac) * math.sin(lon_rad_ac)
                z_ac = (N_ac * (1 - WGS84_E2) + p_alt_for_plot) * sin_lat_ac
                
                dx, dy, dz = x_ac - x0, y_ac - y0, z_ac - z0
                e, n, u = -sin_lon_ref * dx + cos_lon_ref * dy, -sin_lat_ref * cos_lon_ref * dx - sin_lat_ref * sin_lon_ref * dy + cos_lat_ref * dz, cos_lat_ref * cos_lon_ref * dx + cos_lat_ref * sin_lon_ref * dy + sin_lat_ref * dz
                az_deg, alt_deg = enu_to_az_alt(e, n, u)

                for cam_num in range(1, 8):
                    pto_data = get_pto_data_from_json(CAMERAS_FILE, f"{station_id.replace('ams', '')}:{cam_num}") if PTO_MAPPER_AVAILABLE else None
                    is_in_view, _ = is_sky_coord_in_view(pto_data, az_deg, alt_deg)
                    if is_in_view:
                        all_visible_points.append({"time": p_time, "lat": p_lat, "lon": p_lon, "az": az_deg, "alt": alt_deg, "station_id": station_id, "camera": cam_num, "station_code": station_info['station']['code']})

    if not all_visible_points:
        logging.info(f"Worker {os.getpid()}: Finished {flight_info['icao24']}. No visible points found.")
        return None

    all_visible_points.sort(key=lambda x: x['time'])

    if len(all_visible_points) > 1:
        filtered_points = []
        for i, p in enumerate(all_visible_points):
            prev_diff = float('inf') if i == 0 else p['time'] - all_visible_points[i-1]['time']
            next_diff = float('inf') if i == len(all_visible_points) - 1 else all_visible_points[i+1]['time'] - p['time']
            if min(prev_diff, next_diff) <= 300: # 5 minutes
                filtered_points.append(p)
        all_visible_points = filtered_points

    if not all_visible_points:
        logging.info(f"Worker {os.getpid()}: Finished {flight_info['icao24']}. No continuous segments found after filtering.")
        return None

    ground_track = []
    seen_coords = set()
    for p in all_visible_points:
        coord_tuple = (p['lat'], p['lon'])
        if coord_tuple not in seen_coords:
            ground_track.append({'lat': p['lat'], 'lon': p['lon'], 'time': datetime.fromtimestamp(p['time'], tz=timezone.utc).isoformat()})
            seen_coords.add(coord_tuple)

    station_sky_tracks = {}
    for p in all_visible_points:
        station_track = station_sky_tracks.setdefault(p['station_id'], [])
        p_time_iso = datetime.fromtimestamp(p['time'], tz=timezone.utc).isoformat()
        if not station_track or station_track[-1]['time'] != p_time_iso:
             station_track.append({'alt': p['alt'], 'az': p['az'], 'time': p_time_iso})

    all_visible_points.sort(key=lambda x: (x['station_id'], x['camera'], x['time']))
    camera_views = []
    current_view = None
    for p in all_visible_points:
        if current_view and p['station_id'] == current_view['station_id'] and p['camera'] == current_view['camera'] and (p['time'] - current_view['end_time']) <= 120:
            current_view.update(end_utc=datetime.fromtimestamp(p['time'], tz=timezone.utc).isoformat(), end_time=p['time'])
        else:
            if current_view: camera_views.append(current_view)
            current_view = {"station_id": p['station_id'], "camera": p['camera'], "station_code": p['station_code'], "start_utc": datetime.fromtimestamp(p['time'], tz=timezone.utc).isoformat(), "end_utc": datetime.fromtimestamp(p['time'], tz=timezone.utc).isoformat(), "start_time": p['time'], "end_time": p['time']}
    if current_view: camera_views.append(current_view)

    if not camera_views or len(ground_track) <= 1:
        logging.info(f"Worker {os.getpid()}: Finished {flight_info['icao24']}. No valid camera views constructed.")
        return None

    logging.info(f"Worker {os.getpid()}: Successfully finished processing for {flight_info['icao24']}. Found crossing.")
    return {"crossing_id": f"{flight_info['icao24']}-{flight_info['firstSeen']}", "flight_info": {"callsign": (flight_info.get('callsign') or '????').strip(), "origin": (flight_info.get('displayOriginCode') or '????'), "destination": (flight_info.get('displayDestinationCode') or '????')}, "ground_track": ground_track, "station_sky_tracks": station_sky_tracks, "camera_views": camera_views, "earliest_camera_utc": min(cv['start_utc'] for cv in camera_views)}

def cleanup_old_cache(directory, max_age_hours):
    """Deletes files in a directory that are older than a specified age."""
    if not os.path.isdir(directory): return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                if (time.time() - os.path.getmtime(file_path)) > (max_age_hours * 3600):
                    os.remove(file_path)
        except OSError:
            continue

def find_all_crossings(task_id):
    """
    Main orchestrator function. Fetches flight data, filters for candidates,
    and uses a process pool to calculate visibility for each candidate flight.
    """
    status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    try:
        cleanup_old_cache(TRACK_CACHE_DIR, TRACK_CACHE_HOURS)
        with open(CREDENTIALS_FILE, 'r') as f: creds = json.load(f)
        access_token = get_opensky_token(task_id, creds['clientId'], creds['clientSecret'])
        if not access_token: raise ValueError("Failed to obtain OpenSky access token.")
        with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)
        airport_db = get_airport_db(task_id)
        flight_list, _ = update_and_get_flight_database(task_id, access_token)
        if not flight_list:
            update_status(status_file, "complete", {"data": {"crossings": []}}); return

        station_coords = [(s['astronomy']['latitude'], s['astronomy']['longitude']) for s in stations_data.values()]
        flights_after_validation = []
        for flight in flight_list:
            dep_icao, arr_icao = flight.get('estDepartureAirport'), flight.get('estArrivalAirport')
    
            if not dep_icao or not arr_icao or dep_icao == arr_icao: continue
            dep_airport, arr_airport = airport_db.get(dep_icao), airport_db.get(arr_icao)
            if not dep_airport or not arr_airport: continue
            dep_code = dep_airport.get('iata') or (dep_icao if any(haversine_distance(dep_airport['lat'], dep_airport['lon'], s_lat, s_lon) < SHORTLIST_VICINITY_KM for s_lat, s_lon in station_coords) else None)
            arr_code = arr_airport.get('iata') or (arr_icao if any(haversine_distance(arr_airport['lat'], arr_airport['lon'], s_lat, s_lon) < SHORTLIST_VICINITY_KM for s_lat, s_lon in station_coords) else None)
            if dep_code and arr_code:
                flight.update({'dep_airport_data': dep_airport, 'arr_airport_data': arr_airport, 'displayOriginCode': dep_code, 'displayDestinationCode': arr_code})
                flights_after_validation.append(flight)

        candidate_flights = [f for f in flights_after_validation if any(cross_track_distance(s_lat, s_lon, f['dep_airport_data']['lat'], f['dep_airport_data']['lon'], f['arr_airport_data']['lat'], f['arr_airport_data']['lon']) < SHORTLIST_VICINITY_KM for s_lat, s_lon in station_coords)]
        logging.info(f"Task {task_id}: Filtered down to {len(candidate_flights)} candidates (full shortlist).")
        candidate_flights.sort(key=lambda f: f['lastSeen'], reverse=True)
        if len(candidate_flights) > PROCESSING_CHUNK_SIZE:
             logging.warning(f"Task {task_id}: Shortlist > {PROCESSING_CHUNK_SIZE}. Processing {PROCESSING_CHUNK_SIZE} most recent flights.")
             candidate_flights = candidate_flights[:PROCESSING_CHUNK_SIZE]

        final_crossings, total_tasks = [], len(candidate_flights)
        tasks = [(flight, stations_data, task_id, access_token) for flight in candidate_flights]
        with ProcessPoolExecutor() as executor:
            futures = []
            for task in tasks:
                futures.append(executor.submit(fetch_and_process_track, task))
                time.sleep(1) # Add a 1-second delay to stagger API requests
            for i, future in enumerate(as_completed(futures)):
                if result := future.result(): final_crossings.append(result)
                update_status(status_file, "progress", {"step": 25 + int(((i + 1) / total_tasks) * 70), "message": f"status_fetching_aircraft_track|i={i + 1},total={total_tasks}"})

        final_crossings.sort(key=lambda p: p['earliest_camera_utc'], reverse=True)

        time_window_hours = 24
        if final_crossings:
            oldest_pass_time_str = final_crossings[-1]['earliest_camera_utc']
            oldest_pass_time = datetime.fromisoformat(oldest_pass_time_str)
            now = datetime.now(timezone.utc)
            duration_seconds = (now - oldest_pass_time).total_seconds()
            time_window_hours = round(duration_seconds / 3600)

        update_status(status_file, "complete", {"data": {"crossings": final_crossings, "time_window_hours": time_window_hours}})
 
    except Exception as e:
        logging.exception(f"An unhandled error occurred for task {task_id}")
        update_status(status_file, "error", {"message": "error_internal"})

def main():
    """Parses command-line arguments and initiates the aircraft finding process."""
    parser = argparse.ArgumentParser(description="Finds aircraft visible to camera stations.")
    parser.add_argument("task_id", help="The task ID for progress tracking.")
    args = parser.parse_args()
    # Ensure log file directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(process)d - %(message)s', handlers=[logging.FileHandler(LOG_FILE)])
    logging.info(f"--- Script execution started for task {args.task_id} ---")
    
    find_all_crossings(args.task_id)

if __name__ == "__main__":
    main()

