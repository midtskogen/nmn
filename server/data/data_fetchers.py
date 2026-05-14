#!/usr/bin/env python3

import os
import re
import json
import logging
import time
import math
import base64
import statistics
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

# --- Try to import from pto_mapper.py ---
# This attempts to import the necessary functions for camera calibration transformations.
# The PTO_MAPPER_AVAILABLE flag is used by other modules to conditionally enable features
# that rely on this functionality.
try:
    from pto_mapper import get_pto_data_from_json, map_image_to_pano
    PTO_MAPPER_AVAILABLE = True
except ImportError:
    PTO_MAPPER_AVAILABLE = False

# --- Configuration ---
# Establishes base paths for all necessary directories and configuration files.
BASE_DIR = os.environ.get('NMN_DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
STATIONS_FILE = os.path.join(BASE_DIR, 'stations.json')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
CAMERAS_FILE = os.path.join(BASE_DIR, 'cameras.json')
CAMERA_FOV_CACHE_FILE = os.path.join(CACHE_DIR, 'camera_fov_cache.json')
# Defines the path to the meteor data directory, which is located outside the web application's root.
METEOR_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'meteor'))

# A cache for API credentials and the nearest weather station lookup to avoid repeated API calls
FROST_API_CREDS = None
NEAREST_STATION_CACHE = {}

def _load_frost_api_creds():
    """Loads and caches Frost API credentials from config.json."""
    global FROST_API_CREDS
    if FROST_API_CREDS is not None:
        return FROST_API_CREDS
    try:
        with open(os.path.join(BASE_DIR, 'config.json'), 'r') as f:
            config = json.load(f)
        FROST_API_CREDS = (config['frost_api']['client_id'], config['frost_api']['client_secret'])
        return FROST_API_CREDS
    except (IOError, KeyError, json.JSONDecodeError) as e:
        logging.error(f"CRITICAL: Could not load Frost API credentials from config.json: {e}")
        return None

def get_kp_data():
    """
    Fetches the planetary Kp-index data from the NOAA Space Weather Prediction Center.
    The Kp-index is a measure of global geomagnetic activity, which correlates with aurora visibility.
    """
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    try:
        # A User-Agent is specified to identify the application making the request.
        req = urllib.request.Request(url, headers={'User-Agent': 'NorskMeteornettverk-Interface/1.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        logging.error(f"Error fetching Kp data: {e}")
        return json.dumps({"error": f"Could not fetch Kp data: {e}"})


def get_meteor_data():
    """
    Scans the local meteor data directory for observation files from the last 7 days.
    It parses the trajectory result files to extract the start and
    end points of observed meteors.
    """
    logging.info("Starting meteor data scan...")
    meteors = []
    counts_by_station = {}
    # Aborts if the necessary data directories or files are missing.
    if not os.path.exists(METEOR_DIR) or not os.path.exists(STATIONS_FILE):
        logging.warning(f"Meteor directory ({METEOR_DIR}) or stations file ({STATIONS_FILE}) not found. Aborting scan.")
        return {"tracks": [], "counts": {}}
    
    with open(STATIONS_FILE, 'r') as f:
        stations_data = json.load(f)
    # Creates a mapping from station name to station ID for easy lookup.
    name_to_id = {s['station']['name']: sid for sid, s in stations_data.items()}

    today = datetime.now(timezone.utc)
    
    # Iterates through the directories for each of the last 7 days.
    for i in range(7):
        current_date = today - timedelta(days=i)
        date_dir = current_date.strftime('%Y%m%d')
        full_date_path = os.path.join(METEOR_DIR, date_dir)

        if not os.path.isdir(full_date_path):
            continue

        # Each subdirectory within a date directory represents a single meteor event.
        for time_dir in os.listdir(full_date_path):
            event_path = os.path.join(full_date_path, time_dir)
            if not os.path.isdir(event_path): continue

            observing_station_ids = [name_to_id.get(d) for d in os.listdir(event_path)
                                     if os.path.isdir(os.path.join(event_path, d)) and name_to_id.get(d)
                                     and _has_data(os.path.join(event_path, d))]
            observing_station_ids = list(set(observing_station_ids))
            for sid in observing_station_ids:
                if not sid:
                    continue
                counts_by_station.setdefault(sid, {'total': 0, 'multi': 0})
                counts_by_station[sid]['total'] += 1
            
            meteor_trajectory = None
            # Finds the trajectory result file (.res) within the event directory.
            for filename in os.listdir(event_path):
                if filename.startswith('obs') and filename.endswith('.res'):
                    res_file_path = os.path.join(event_path, filename)
                    try:
                        with open(res_file_path, 'r') as f: lines = f.readlines()
    
                        # The .res file format contains start and end coordinates on the first two lines.
                        if len(lines) >= 2:
                            start_parts, end_parts = lines[0].split(), lines[1].split()
                            
                            # Parse start and end altitudes in km
                            start_altitude_km = float(start_parts[4])
                            end_altitude_km = float(end_parts[4])

                            # Apply the altitude filter
                            if start_altitude_km <= 150 and end_altitude_km >= 10:
                                timestamp_str = f"{date_dir[0:4]}-{date_dir[4:6]}-{date_dir[6:8]}T{time_dir[0:2]}:{time_dir[2:4]}:{time_dir[4:6]}Z"
                                meteor_trajectory = {
                                    'timestamp': timestamp_str,
                                    'lat1': float(start_parts[1]), 'lon1': float(start_parts[0]), 'h1': start_altitude_km,
                                    'lat2': float(end_parts[1]), 'lon2': float(end_parts[0]), 'h2': end_altitude_km
                                }
                        break 
                    except (IOError, IndexError, ValueError) as e:
                
                         logging.warning(f"Could not parse meteor file {res_file_path}: {e}")
            
            if not meteor_trajectory: continue

            meteor_trajectory['station_ids'] = observing_station_ids
            if len(observing_station_ids) > 1:
                for sid in observing_station_ids:
                    if not sid:
                        continue
                    counts_by_station.setdefault(sid, {'total': 0, 'multi': 0})
                    counts_by_station[sid]['multi'] += 1
            meteors.append(meteor_trajectory)

    logging.info(f"Meteor scan finished. Found {len(meteors)} tracks.")
    return {"tracks": meteors, "counts": counts_by_station}


def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth."""
    R = 6371  # Earth radius in kilometers
    dLat, dLon, lat1, lat2 = map(math.radians, [lat2 - lat1, lon2 - lon1, lat1, lat2])
    
    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def get_air_pressure(lat, lon, dt_utc):
    """
    Fetches the air pressure at mean sea level for a given location and time from the Frost API.
    Returns pressure in hPa.
    """
    creds = _load_frost_api_creds()
    if not creds:
        return None

    client_id, client_secret = creds
    auth = f'{client_id}:{client_secret}'
    headers = {'Authorization': f'Basic {base64.b64encode(auth.encode()).decode()}', 'User-Agent': 'NorskMeteornettverk-Interface/1.0'}
    
    # Get up to three closest stations in one request
    sources_url = f"https://frost.met.no/sources/v0.jsonld?types=SensorSystem&geometry=nearest(POINT({lon}%20{lat}))&nearestmaxcount=3&elements=air_pressure_at_sea_level"
    logging.info(f"Frost API URL for stations: {sources_url}")

    nearby_stations = []
    try:
        req = urllib.request.Request(sources_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            sources_data = json.load(response)
        if sources_data.get('data'):
            nearby_stations = [s['id'] for s in sources_data['data']]
    except Exception as e:
        logging.error(f"API error finding nearest stations: {e}")
        return None

    if not nearby_stations:
        logging.warning(f"No nearby weather stations found for ({lat}, {lon}).")
        return None

    # Iterate through the fetched stations and try to get data
    for station_id in nearby_stations:
        # Try a narrow 2-hour window, then a wider 12-hour window
        for hours in [2, 12]:
            start_time = dt_utc - timedelta(hours=hours/2)
            end_time = dt_utc + timedelta(hours=hours/2)
            time_range = f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            
            obs_url = f"https://frost.met.no/observations/v0.jsonld?sources={station_id}&referencetime={time_range}&elements=air_pressure_at_sea_level"
            try:
                req = urllib.request.Request(obs_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    obs_data = json.load(response)
                if obs_data.get('data'):
                    pressure = float(obs_data['data'][0]['observations'][0]['value'])
                    logging.info(f"Successfully fetched pressure {pressure} hPa from {station_id} using {hours}h window.")
                    return pressure
            except Exception as e:
                logging.warning(f"Failed to fetch pressure for {station_id} with {hours}h window: {e}")
                continue
    
    logging.warning(f"All attempts to fetch pressure for ({lat}, {lon}) failed for stations {nearby_stations}.")
    return None


def get_lightning_data(end_date_str):
    """
    Fetches, de-duplicates, filters, and caches lightning data for a 7-day period.
    Data is sourced from the MET Norway Frost API.
    """
    all_raw_strikes_parts = []
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # Loads API credentials from a configuration file.
    creds = _load_frost_api_creds()
    if not creds:
        logging.error("CRITICAL: Could not load Frost API credentials from config.json.")
        return {"error": "Server API credentials are not configured."}

    CLIENT_ID, CLIENT_SECRET = creds
    auth = f'{CLIENT_ID}:{CLIENT_SECRET}'
    headers = {'Authorization': f'Basic {base64.b64encode(auth.encode()).decode()}', 'User-Agent': 'NorskMeteornettverk-Interface/1.0'}

    # Iterates through the last 7 days, fetching data for each.
    for i in range(7):
        current_date = end_date - timedelta(days=i)
        current_date_str = current_date.strftime('%Y-%m-%d')
        raw_cache_file = os.path.join(CACHE_DIR, f"lightning_raw_cache_{current_date_str}.json")
        is_today = current_date_str == datetime.utcnow().strftime('%Y-%m-%d')
        
        # Determines if a fresh fetch is needed based on cache existence and age.
        should_fetch = not os.path.exists(raw_cache_file) or (is_today and (time.time() - os.path.getmtime(raw_cache_file) > 3600))

        if should_fetch:
            time_param = f"latest&maxage=P1D" if is_today else f"{current_date_str}T00:00:00Z/{current_date_str}T23:59:59Z"
            url = f"https://frost.met.no/lightning/v0.ualf?referencetime={time_param}"
            try:
                logging.info(f"Fetching lightning data from: {url}")
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    lines = response.read().decode('utf-8').strip().splitlines()
                    day_strikes_parts = [line.split() for line in lines]
                    # Caches the raw fetched data to a file.
                    with open(raw_cache_file, 'w') as f: json.dump(day_strikes_parts, f)
                    all_raw_strikes_parts.extend(day_strikes_parts)
            except Exception as e:
                logging.error(f"Failed to fetch lightning data for {current_date_str}: {e}")
        else:
            # If cache is valid, load data from the file.
            with open(raw_cache_file, 'r') as f: all_raw_strikes_parts.extend(json.load(f))

    # Parse and de-duplicate the raw strike data.
    parsed_strikes = []
    for parts in all_raw_strikes_parts:
        try:
            if len(parts) < 21: continue
            dt = datetime(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), tzinfo=timezone.utc)
            parsed_strikes.append({'dt': dt, 'lat': float(parts[8]), 'lon': float(parts[9]), 'type': 'ic' if int(float(parts[20])) == 1 else 'cg', 'full_time': f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{int(parts[7]):09d}Z"})
        except (ValueError, IndexError): continue
    
    # The API can return multiple reports for a single lightning event.
    # This logic
    # groups strikes by the second they occurred and removes any that are within
    # 0.5 km of an already-kept strike in that same second, effectively de-duplicating the data.
    grouped_by_second = {}
    for strike in parsed_strikes:
        timestamp_sec_str = strike['dt'].strftime('%Y-%m-%dT%H:%M:%S')
        if timestamp_sec_str not in grouped_by_second:
            grouped_by_second[timestamp_sec_str] = []
        grouped_by_second[timestamp_sec_str].append(strike)

    unique_events = []
    for _, strikes_in_second in grouped_by_second.items():
        kept_strikes = []
        for strike in strikes_in_second:
            
            is_duplicate = any(haversine(strike['lat'], strike['lon'], kept['lat'], kept['lon']) < 0.5 for kept in kept_strikes)
            if not is_duplicate:
                kept_strikes.append(strike)
        unique_events.extend(kept_strikes)

    # Loads station locations to filter lightning strikes to only those near a station.
    try:
        with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)
        station_coords = [(s['astronomy']['latitude'], s['astronomy']['longitude']) for s in stations_data.values()]
    except Exception as e:
        logging.error(f"Could not load stations file for lightning filtering: {e}")
        return {"error": "Station data is unavailable for filtering."}

    # Filters the unique strikes, keeping only those within 30 km of any station.
    final_strikes = []
    for strike in unique_events:
        min_dist = min(haversine(strike['lat'], strike['lon'], s_lat, s_lon) for s_lat, s_lon in station_coords)
        if min_dist <= 30:
            final_strikes.append({"time": strike['full_time'], "lat": strike['lat'], "lon": strike['lon'], "type": strike['type'], "dist": min_dist})
            
    logging.info(f"Processed {len(all_raw_strikes_parts)} raw strikes into {len(final_strikes)} unique, filtered events.")
    return final_strikes

_RE_DIRECTION = re.compile(r'Direction:</td><td>\s*([\d.]+)')
_RE_SPEED = re.compile(r'Entry speed:</td><td>\s*([\d.]+)')
_RE_SHOWER = re.compile(r'Meteor Shower:</td><td[^>]*>\s*(\S[^<]*\S)\s*</td>')
_RE_START_H = re.compile(r'Start height:</td><td>\s*(-?[\d.]+)')
_RE_END_H = re.compile(r'End height:</td><td>\s*(-?[\d.]+)')
_RE_START_POS = re.compile(r'Start position:</td><td>\s*([\d.]+)([NS])\s+([\d.]+)([EW])')
_RE_END_POS = re.compile(r'End position:</td><td>\s*([\d.]+)([NS])\s+([\d.]+)([EW])')

def _parse_latlon(match):
    try:
        lat = float(match.group(1)) * (-1 if match.group(2) == 'S' else 1)
        lon = float(match.group(3)) * (-1 if match.group(4) == 'W' else 1)
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None, None
        return lat, lon
    except (ValueError, IndexError):
        return None, None

def _parse_trajectory_details(event_path):
    """Parses trajectory details from en_tables.html if present."""
    tables_file = os.path.join(event_path, 'en_tables.html')
    if not os.path.isfile(tables_file):
        return {}
    try:
        with open(tables_file, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()
    except (OSError, IOError):
        return {}
    result = {}
    try:
        m = _RE_DIRECTION.search(html)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 360:
                result['direction'] = val
        m = _RE_SPEED.search(html)
        if m:
            val = float(m.group(1))
            if 0 <= val < 100:  # reasonable speed range km/s
                result['speed'] = val
        m = _RE_SHOWER.search(html)
        if m:
            shower = m.group(1).strip()
            if shower:
                result['shower'] = shower
        m = _RE_START_H.search(html)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 150:  # reasonable altitude range km
                result['start_alt'] = val
        m = _RE_END_H.search(html)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 150:
                result['end_alt'] = val
        m = _RE_START_POS.search(html)
        if m:
            lat, lon = _parse_latlon(m)
            if lat is not None:
                result['start_lat'] = lat
                result['start_lon'] = lon
        m = _RE_END_POS.search(html)
        if m:
            lat, lon = _parse_latlon(m)
            if lat is not None:
                result['end_lat'] = lat
                result['end_lon'] = lon
    except (ValueError, TypeError, IndexError):
        # Return partial results if parsing fails mid-way
        pass
    return result


def _has_data(station_dir):
    """Return True if the station dir has at least one camera subdir with event.txt."""
    for cam in os.listdir(station_dir):
        if os.path.exists(os.path.join(station_dir, cam, 'event.txt')):
            return True
    return False


def get_station_stats(station_id, start_date=None, end_date=None):
    """
    Returns observation statistics for a single station over a date range.
    Accepts start_date and end_date as YYYY-MM-DD strings.
    If omitted, defaults to the last 7 days.
    station_id can be 'all' for network-wide totals (no per-event list).
    """
    if not os.path.exists(METEOR_DIR) or not os.path.exists(STATIONS_FILE):
        return {"error": "Data directories not found."}

    try:
        with open(STATIONS_FILE, 'r') as f:
            stations_data = json.load(f)
    except (json.JSONDecodeError, IOError, OSError) as e:
        return {"error": f"Failed to load stations data: {e}"}

    is_all = (station_id == 'all')
    if not is_all and station_id not in stations_data:
        return {"error": f"Unknown station: {station_id}"}

    try:
        station_name = None if is_all else stations_data[station_id]['station']['name']
        station_code = 'all' if is_all else stations_data[station_id]['station']['code']
        name_to_id = {s['station']['name']: sid for sid, s in stations_data.items() if 'station' in s and 'name' in s['station']}
        id_to_code = {sid: s['station']['code'] for sid, s in stations_data.items() if 'station' in s and 'code' in s['station']}
    except (KeyError, TypeError):
        return {"error": "Corrupted station data format."}

    today = datetime.now(timezone.utc).date()
    try:
        d_end = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else today
        d_start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else (d_end - timedelta(days=6))
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    if d_start > d_end:
        d_start, d_end = d_end, d_start
    span = (d_end - d_start).days + 1
    if span > 730:
        return {"error": "stats_range_too_large"}

    parse_trajectory = True
    total = 0
    multi = 0
    shower_count = 0
    sporadic_count = 0
    speeds = []
    start_alts = []
    end_alts = []
    events = []

    current = d_start
    while current <= d_end:
        date_dir = current.strftime('%Y%m%d')
        full_date_path = os.path.join(METEOR_DIR, date_dir)
        current += timedelta(days=1)

        if not os.path.isdir(full_date_path):
            continue

        try:
            time_entries = os.listdir(full_date_path)
        except (OSError, IOError):
            continue
        for time_dir in time_entries:
            event_path = os.path.join(full_date_path, time_dir)
            if not os.path.isdir(event_path):
                continue

            if not is_all:
                station_dir = os.path.join(event_path, station_name)
                if not os.path.isdir(station_dir) or not _has_data(station_dir):
                    continue

            total += 1

            try:
                event_contents = os.listdir(event_path)
            except (OSError, IOError):
                event_contents = []
            observing_ids = [
                name_to_id[d] for d in event_contents
                if os.path.isdir(os.path.join(event_path, d))
                and name_to_id.get(d)
                and _has_data(os.path.join(event_path, d))
            ]
            observing_ids = list(set(observing_ids))
            num_stations = len(observing_ids)
            is_multi = num_stations > 1
            if is_multi:
                multi += 1

            try:
                traj = _parse_trajectory_details(event_path) if parse_trajectory else {}
            except Exception:
                traj = {}
            shower = traj.get('shower')
            if shower:
                if shower.lower() == 'sporadic':
                    sporadic_count += 1
                else:
                    shower_count += 1
            if 'speed' in traj:
                speeds.append(traj['speed'])
            if 'start_alt' in traj and traj['start_alt'] > 0:
                start_alts.append(traj['start_alt'])
            if 'end_alt' in traj and traj['end_alt'] > 0:
                end_alts.append(traj['end_alt'])

            if not is_all:
                other_ids = [sid for sid in observing_ids if sid != station_id]
                try:
                    has_trajectory = any(
                        fn.startswith('obs') and fn.endswith('.res')
                        for fn in event_contents
                    )

                    timestamp_str = (
                        f"{date_dir[0:4]}-{date_dir[4:6]}-{date_dir[6:8]}"
                        f"T{time_dir[0:2]}:{time_dir[2:4]}:{time_dir[4:6]}Z"
                    )

                    evt = {
                        'timestamp': timestamp_str,
                        'date_dir': date_dir,
                        'time_dir': time_dir,
                        'num_stations': num_stations,
                        'other_stations': [id_to_code.get(sid, sid) for sid in other_ids],
                        'has_trajectory': has_trajectory,
                    }
                    if shower:
                        evt['shower'] = shower
                    if 'speed' in traj:
                        evt['speed'] = traj['speed']
                    if 'direction' in traj:
                        evt['direction'] = traj['direction']
                    if 'start_alt' in traj:
                        evt['start_alt'] = traj['start_alt']
                    if 'end_alt' in traj:
                        evt['end_alt'] = traj['end_alt']
                    if 'start_lat' in traj:
                        evt['start_lat'] = traj['start_lat']
                        evt['start_lon'] = traj['start_lon']
                    if 'end_lat' in traj:
                        evt['end_lat'] = traj['end_lat']
                        evt['end_lon'] = traj['end_lon']
                    events.append(evt)
                except (KeyError, TypeError, ValueError):
                    # Skip malformed event data but continue processing others
                    continue

    events.sort(key=lambda e: e['timestamp'], reverse=True)

    traj_summary = {}
    if parse_trajectory and speeds:
        traj_summary['avg_speed'] = round(sum(speeds) / len(speeds), 1)
        traj_summary['median_speed'] = round(statistics.median(speeds), 1)
    if parse_trajectory and start_alts:
        traj_summary['avg_start_alt'] = round(sum(start_alts) / len(start_alts), 1)
        traj_summary['median_start_alt'] = round(statistics.median(start_alts), 1)
    if parse_trajectory and end_alts:
        traj_summary['avg_end_alt'] = round(sum(end_alts) / len(end_alts), 1)
        traj_summary['median_end_alt'] = round(statistics.median(end_alts), 1)

    return {
        'station_id': station_id,
        'station_code': station_code,
        'start_date': d_start.strftime('%Y-%m-%d'),
        'end_date': d_end.strftime('%Y-%m-%d'),
        'total': total,
        'multi': multi,
        'shower_count': shower_count,
        'sporadic_count': sporadic_count,
        'has_trajectory_details': parse_trajectory,
        **traj_summary,
        'events': events,
    }


def get_camera_fovs():
    """
    Calculates and caches
    the center azimuth and horizontal Field of View (FOV) for all cameras.
    It uses the pto_mapper library to transform pixel coordinates at the edges of an image
    to panoramic (azimuth/altitude) coordinates, thereby determining the FOV.
    """
    # Checks if the cache file is newer than the source camera calibration file.
    if os.path.exists(CAMERA_FOV_CACHE_FILE) and os.path.exists(CAMERAS_FILE):
        if os.path.getmtime(CAMERA_FOV_CACHE_FILE) >= os.path.getmtime(CAMERAS_FILE):
             with open(CAMERA_FOV_CACHE_FILE, 'r') as f: return json.load(f)

    # Returns empty if the required pto_mapper library is not available.
    if not PTO_MAPPER_AVAILABLE:
        logging.warning("pto_mapper is not available. Cannot calculate camera FOVs.")
        return {}
    
    logging.info("Generating camera FOV cache...")
    fov_data = {}
    try:
        with open(CAMERAS_FILE, 'r') as f: cameras_data = json.load(f)
    except Exception as e:
        logging.error(f"Could not load cameras.json to calculate FOVs: {e}")
        return {}

    # Iterates through 
    # each camera defined in the calibration file.
    for station_id, station_cams in cameras_data.items():
        station_num = station_id.replace('ams', '')
        fov_data[station_id] = {}
        for cam_name, cam_info in station_cams.items():
            if not cam_name.startswith('cam') or 'calibration' not in cam_info: continue
            try:
                cam_num = cam_name.replace('cam', '')
     
                pto_data = get_pto_data_from_json(CAMERAS_FILE, f"{station_num}:{cam_num}")
                
                img_params = pto_data[1][0]
                sw, sh = img_params.get('w'), img_params.get('h')
                if not sw or not sh: continue

          
                # Maps the center, left-horizon, and right-horizon points of the image to the panorama.
                center_pano, left_pano, right_pano = map_image_to_pano(pto_data, 0, sw / 2, sh / 2), map_image_to_pano(pto_data, 0, 0, sh / 2), map_image_to_pano(pto_data, 0, sw, sh / 2)

                if center_pano and left_pano and right_pano:
                    pano_w = pto_data[0]['w']
                    # Converts panoramic x-coordinates to azimuth in degrees.
                    center_az, left_az, right_az = (p[0] / pano_w * 360 for p in [center_pano, left_pano, right_pano])
                    # Calculates the horizontal FOV, handling the 360/0 degree wrap-around.
                    h_fov = right_az - left_az
                    if h_fov < -180: h_fov += 360
                    if h_fov > 180: h_fov -= 360
                    fov_data[station_id][cam_name] = {"centerAzimuth": (center_az + 360) % 360, "hFov": abs(h_fov)}
            except Exception as e:
                logging.warning(f"Could not calculate FOV for {station_id}/{cam_name}: {e}")
                continue
    
    # Writes the calculated data to the cache file.
    with open(CAMERA_FOV_CACHE_FILE, 'w') as f: json.dump(fov_data, f)
    logging.info("Finished generating camera FOV cache.")
    return fov_data

