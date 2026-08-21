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
from shared_utils import atomic_json_write, read_json_file

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
# Per-station cache directory. Each station's passes (over the full
# SEARCH_DAYS window) are cached independently, so an on-demand request for
# one or a few stations doesn't have to (re)compute passes for every station
# in the network -- this used to be the dominant cost once the satellite
# catalog was broadened (245 satellites x ~90 stations x 7 days took several
# minutes and produced a 58 MB response). The 'days' filter requested by the
# client is applied when *serving* results from these caches, not when
# computing them, so it never triggers recomputation.
PASS_CACHE_DIR = os.path.join(CACHE_DIR, 'passes')
LOG_FILE = os.path.join(LOG_DIR, 'predict_sat.log')

# --- Script settings ---
MAX_LOG_LINES = 20000
TLE_UPDATE_INTERVAL_HOURS = 4 # How often to fetch fresh TLE data.
PASS_CACHE_LIFETIME_MINUTES = 235 # How long to use cached pass predictions. Set just under TLE interval.
SEARCH_DAYS = 7 # How many days into the past to search for passes.
MAX_VISIBLE_MAGNITUDE = 6.0 # The faintest satellite magnitude to consider. Lower is brighter.
MAXIMUM_SUN_ALT = -9 # The maximum sun altitude for a pass to be considered "dark enough".
# (-6 is civil twilight)

# A dictionary of specific satellites to track by name and NORAD catalog number.
# These are always kept regardless of the generic brightness pre-filter below,
# and use a hand-tuned absolute magnitude instead of the per-category default.
SATELLITES_OF_INTEREST = {
    "ISS (ZARYA)": 25544, "LACROSSE 5": 28646, "PAZ": 43215, "CSG-1": 47219,
    "TERRA": 25994, "AQUA": 27424, "GENESIS 2": 31820, "TANDEM-X": 36605,
    "SARAL": 39086, "OCEANSAT-2": 35931,
    # Identified 2026-08-21 as the object triangulated by 6 NMN stations on
    # 2026-08-17 ~20:52 UTC (event 20260817/205203): propagating this TLE
    # matches the triangulated start point to ~12 km and, 206s later, the
    # triangulated end point to ~1.9 km (course/speed/altitude all consistent
    # with the fbspd/metrack fit). NORAD 25406 is the satellite itself
    # (1998-045A); its rocket body is a separate object, NORAD 25407.
    "COSMOS 2360": 25406,
}
# A dictionary of estimated absolute magnitudes for the satellites of interest.
SATELLITE_MAGNITUDES = {
    "ISS (ZARYA)": -2.0, "LACROSSE 5": 1.5, "PAZ": 2.0, "CSG-1": 2.5, "TERRA": 2.8,
    "AQUA": 2.8, "GENESIS 2": 3.0, "TANDEM-X": 3.0, "SARAL": 3.0, "OCEANSAT-2": 3.0,
    "COSMOS 2360": 3.5,  # not in CelesTrak's top-100 visual list; rough estimate
}

# --- Generic catalogue coverage (beyond the curated SATELLITES_OF_INTEREST) ---
# Historically this script discarded every downloaded TLE that wasn't one of
# the ~10 named satellites above, even though it downloads whole CelesTrak
# groups (stations, starlink, weather, resource, radar, active). That made
# the "satellite passes" feature far too narrow.
#
# We now auto-include satellites from the small, purpose-built groups below
# (a few hundred objects total) using a per-category default absolute
# magnitude, since those groups consist of comparably large/bright
# satellites where a rough shared brightness assumption is reasonable.
#
# 'starlink' and 'active' are deliberately EXCLUDED from this auto-include:
# they contain many thousands of wildly different-sized objects (from
# defunct rocket stages down to cubesats), so no single default magnitude is
# meaningful, and at typical LEO altitudes almost any plausible default
# clears MAX_VISIBLE_MAGNITUDE anyway -- auto-including them would balloon
# the tracked set to nearly the entire catalogue and make the per-station
# pass search far too slow. Those two groups are still downloaded (so
# specific named SATELLITES_OF_INTEREST entries living only in those groups
# are still found), but non-curated members of them are skipped.
CATEGORY_DEFAULT_ABS_MAG = {
    'stations': 0.0,    # crewed/large stations (ISS/CSS-class)
    'weather': 3.0,      # large weather satellites (NOAA, MetOp, Meteor-M, ...)
    'resource': 3.0,     # large earth-observation satellites (Landsat, Sentinel, ...)
    'radar': 3.5,        # radar imaging satellites (TerraSAR-X, RADARSAT, ICEYE, ...)
}
BROADENED_CATEGORIES = frozenset(CATEGORY_DEFAULT_ABS_MAG.keys())
BEST_CASE_MAGNITUDE_MARGIN = 1.0  # extra slack for the optimistic pre-filter above
WGS72_EARTH_RADIUS_KM = 6378.135  # radius used internally by the SGP4 propagator


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
    # Check if a recent TLE cache file exists. We intentionally do NOT also
    # require every curated SATELLITES_OF_INTEREST name to be present here:
    # a satellite that has permanently decayed/deorbited (no source ever has
    # a current TLE for it again) would otherwise make this check fail
    # forever, forcing a full re-fetch -- and the CelesTrak rate-limit risk
    # that comes with it -- on every single call regardless of freshness.
    if os.path.exists(TLE_FILE) and (datetime.now().timestamp() - os.path.getmtime(TLE_FILE)) / 3600 < TLE_UPDATE_INTERVAL_HOURS:
        cached_data = read_json_file(TLE_FILE, default={})
        if cached_data: return cached_data
    
    logging.info("Cache is stale or missing. Forcing fresh TLE download.")
    tle_data = {}
    last_error = None
    # List of CelesTrak TLE sources to query, tagged with a category used to
    # assign a default absolute magnitude to satellites we don't have a
    # hand-curated estimate for (see CATEGORY_DEFAULT_ABS_MAG).
    sources = [
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle", 'stations'),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle", 'starlink'),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle", 'weather'),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=tle", 'resource'),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=radar&FORMAT=tle", 'radar'),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle", 'active'),
    ]
    for source_url, category in sources:
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

                is_curated = name in SATELLITES_OF_INTEREST or satnum in SATELLITES_OF_INTEREST.values()
                try:
                    temp_sat = EarthSatellite(line1, line2, name, ts)
                except Exception:
                    continue

                if is_curated:
                    abs_mag = SATELLITE_MAGNITUDES.get(name, 3.0)
                elif category in BROADENED_CATEGORIES:
                    # Cheap pre-filter: estimate this satellite's best-possible
                    # brightness (directly overhead, best phase) and skip it
                    # if that's already fainter than we could ever detect.
                    default_abs_mag = CATEGORY_DEFAULT_ABS_MAG[category]
                    altitude_km = max(100.0, (temp_sat.model.a * WGS72_EARTH_RADIUS_KM) - WGS72_EARTH_RADIUS_KM)
                    best_case_mag = default_abs_mag + 5 * np.log10(altitude_km / 1000.0)
                    if best_case_mag >= MAX_VISIBLE_MAGNITUDE + BEST_CASE_MAGNITUDE_MARGIN:
                        continue
                    abs_mag = default_abs_mag
                else:
                    # 'starlink' / 'active': too large and size-heterogeneous
                    # to safely auto-include (see BROADENED_CATEGORIES note
                    # above). Only keep named SATELLITES_OF_INTEREST matches
                    # from these groups, which is_curated already handled.
                    continue

                # Keep the most optimistic (brightest) entry if the same
                # satellite shows up in more than one downloaded group.
                existing = tle_data.get(name)
                if existing is not None and existing.get('abs_mag', 99.0) <= abs_mag:
                    continue
                tle_data[name] = {'satnum': satnum, 'line1': line1, 'line2': line2, 'inclination': temp_sat.model.inclo, 'abs_mag': abs_mag}

        except Exception as e:
            # CelesTrak rate-limits individual GROUP endpoints (e.g. it returns
            # HTTP 403 with "GP data has not updated since your last successful
            # download" if the same group is requested again within its update
            # window). That must not abort the whole fetch: skip this source,
            # keep whatever TLEs we already collected from other sources, and
            # fall back to any previously cached entry for the satellites we
            # could not refresh.
            logging.error(f"Could not process TLE from {source_url}: {e}")
            last_error = e

    # Some curated satellites aren't members of any of the group downloads
    # above (e.g. CelesTrak's GROUP=active only lists satellites it
    # considers currently "active", which excludes some trackable objects
    # like COSMOS 2360). Fetch those individually by NORAD catalog number so
    # a curated entry doesn't silently go missing just because it fell out
    # of (or never belonged to) one of the bulk group downloads.
    missing_curated = {name: satnum for name, satnum in SATELLITES_OF_INTEREST.items() if name not in tle_data}
    for name, satnum in missing_curated.items():
        try:
            url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={satnum}&FORMAT=tle"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                lines = response.read().decode('utf-8').strip().splitlines()
            if len(lines) < 3: continue
            fetched_name, line1, line2 = lines[0].strip(), lines[1].strip(), lines[2].strip()
            temp_sat = EarthSatellite(line1, line2, fetched_name, ts)
            tle_data[name] = {'satnum': satnum, 'line1': line1, 'line2': line2, 'inclination': temp_sat.model.inclo, 'abs_mag': SATELLITE_MAGNITUDES.get(name, 3.0)}
        except Exception as e:
            logging.error(f"Could not fetch individual TLE for curated satellite '{name}' (NORAD {satnum}): {e}")
            last_error = e

    if cached_data:
        # Fill in any satellites we failed to refresh this round with their
        # last known-good TLE, rather than dropping them entirely.
        for name, data in cached_data.items():
            tle_data.setdefault(name, data)

    if not tle_data:
        if cached_data: return cached_data # Return stale data if fresh download fails.
        return {"error": f"Failed to download TLE data: {last_error}"}

    atomic_json_write(TLE_FILE, tle_data)
    return tle_data


# --- Worker and Main Logic ---
def process_station(args):
    """
    Worker function for the process pool. Calculates all visible satellite passes
    for a single ground station over the defined search period.
    """
    station_id, station_info, tle_data = args
    # Add a log message here to know which station the worker is handling
    logging.info(f"Worker {os.getpid()}: Processing station {station_id} ({station_info.get('station', {}).get('code', 'N/A')})")
    ts, eph = worker_ts, worker_eph

    sun, earth = eph['sun'], eph['earth']
    station_passes = []
    location = wgs84.latlon(station_info['astronomy']['latitude'], station_info['astronomy']['longitude'], elevation_m=station_info['astronomy']['elevation']) # 

    # Define the time window for the search.
    utc_now = datetime.now(timezone.utc) # 
    end_dt = utc_now.replace(minute=0, second=0, microsecond=0) # 
    start_dt = (end_dt - timedelta(days=SEARCH_DAYS)).replace(hour=12, minute=0, second=0, microsecond=0) # 

    end_time = ts.utc(end_dt) # 
    start_time = ts.utc(start_dt) # 

    # Camera calibration (PTO) data only depends on the station and camera
    # number, not on the satellite/pass being evaluated, so look it up once
    # per station instead of once per (satellite, pass, camera) combination.
    # With hundreds of passes x up to 7 cameras this used to mean re-reading
    # and re-parsing cameras.json from disk thousands of times per station.
    station_pto_data = {}
    if PTO_MAPPER_AVAILABLE:
        for cam_num in range(1, 8):
            try:
                station_pto_data[cam_num] = get_pto_data_from_json(CAMERAS_FILE, f"{station_id.replace('ams', '')}:{cam_num}")
            except Exception:
                station_pto_data[cam_num] = None

    for name, data in tle_data.items():
        # Log which satellite is being checked
        logging.debug(f"Worker {os.getpid()}: Checking satellite {name} for station {station_id}") # <-- ADD DEBUG LOG

        if not all(k in data for k in ['line1', 'line2', 'inclination']): continue # 
        satellite = EarthSatellite(data['line1'], data['line2'], name, ts) # 
        # A satellite can't be seen from a latitude higher than its inclination.
        # This is a quick filter.
        if abs(np.deg2rad(station_info['astronomy']['latitude'])) > data['inclination']: # 
            # Log if filtered by inclination
            logging.debug(f"Worker {os.getpid()}: Satellite {name} filtered out for {station_id} by inclination ({np.rad2deg(data['inclination']):.1f} deg vs station lat {station_info['astronomy']['latitude']:.1f} deg)") # <-- ADD DEBUG LOG
            continue # 

        # Use skyfield's `find_events` to find when the satellite rises, culminates, and sets.
        try: # Add try/except around find_events
            times, events = satellite.find_events(location, start_time, end_time, altitude_degrees=20.0) # 
            # Log the number of events found
            logging.debug(f"Worker {os.getpid()}: Found {len(events)} rise/cul/set events for {name} at {station_id}") # <-- ADD DEBUG LOG
        except Exception as e:
            logging.error(f"Worker {os.getpid()}: Error during find_events for {name} at {station_id}: {e}") # <-- ADD ERROR LOG
            continue # Skip this satellite if find_events fails

        for i, event_type in enumerate(events): # 
            # A pass is a sequence of rise (0), culmination (1), and set (2).
            if event_type == 0 and i + 2 < len(events) and np.all(events[i+1:i+3] == [1, 2]): # 
                rise_time, culminate_time, set_time = times[i : i+3] # 

                # Log the potential pass time
                logging.debug(f"Worker {os.getpid()}: Potential pass for {name} at {station_id}, culmination: {culminate_time.utc_iso()}") # <-- ADD DEBUG LOG

                # --- Visibility Checks ---
                # 1. The satellite must be illuminated by the sun.
                is_sunlit = satellite.at(culminate_time).is_sunlit(eph) # 
                if not is_sunlit: # 
                    logging.debug(f"Worker {os.getpid()}: Pass {name} rejected: Satellite not sunlit at culmination.") # <-- ADD DEBUG LOG
                    continue # 
                # 2. The observer on the ground must be in darkness.
                observer_alt = (earth + location).at(culminate_time).observe(sun).apparent().altaz()[0].degrees # 
                if observer_alt > MAXIMUM_SUN_ALT: # 
                    logging.debug(f"Worker {os.getpid()}: Pass {name} rejected: Observer sun altitude {observer_alt:.1f} > {MAXIMUM_SUN_ALT:.1f}") # <-- ADD DEBUG LOG
                    continue # 

                # 3. Estimate the satellite's apparent magnitude (brightness).
                # 'abs_mag' is stored per-satellite in tle_data (hand-curated
                # for SATELLITES_OF_INTEREST, or a per-category default for
                # everything else picked up from the broader catalogue).
                abs_mag = data.get('abs_mag', SATELLITE_MAGNITUDES.get(name, 99.0)) # 
                topocentric_culminate = (satellite - location).at(culminate_time) # 
                r_km = topocentric_culminate.distance().km # 
                # The following calculates the phase angle to adjust brightness.
                sat_pos_au = satellite.at(culminate_time).position.au # 
                obs_pos_au = (earth + location).at(culminate_time).position.au # 
                sun_pos_au = sun.at(culminate_time).position.au # 
                vec_obs_sat, vec_sun_sat = sat_pos_au - obs_pos_au, sat_pos_au - sun_pos_au # 
                dot_product = np.dot(vec_obs_sat, vec_sun_sat) # 
                mag_obs_sat, mag_sun_sat = np.linalg.norm(vec_obs_sat), np.linalg.norm(vec_sun_sat) # 
                phi = np.arccos(np.clip(dot_product / (mag_obs_sat * mag_sun_sat), -1.0, 1.0)) # Add clip for safety # 
                phase_factor = (np.sin(phi) + (np.pi - phi) * np.cos(phi)) / np.pi # 
                est_mag = abs_mag + 5 * np.log10(r_km / 1000.0) # 
                if phase_factor > 1e-6: # Avoid log10(0)
                   est_mag += -2.5 * np.log10(phase_factor) # 

                if est_mag >= MAX_VISIBLE_MAGNITUDE: # 
                    logging.debug(f"Worker {os.getpid()}: Pass {name} rejected: Estimated magnitude {est_mag:.1f} >= {MAX_VISIBLE_MAGNITUDE:.1f}") # <-- ADD DEBUG LOG
                    continue # 

                # If all checks pass, calculate the detailed track for the pass.
                # Log that a pass passed filters
                logging.info(f"Worker {os.getpid()}: Pass found for {name} at {station_id}, culmination: {culminate_time.utc_iso()}, Mag: {est_mag:.1f}, Sun Alt: {observer_alt:.1f}")

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
                    pto_data = station_pto_data.get(cam_num)
                    if PTO_MAPPER_AVAILABLE and pto_data is None:
                        continue

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
            "earliest_camera_utc": min(cv['start_utc'] for cv in all_camera_views),
            # Needed (together with earliest_camera_utc) to test time-range
            # filters by *overlap* rather than just the start time: a pass
            # can span several minutes across multiple stations/cameras, so a
            # narrow requested window can fall inside that span without
            # containing its earliest view.
            "latest_camera_utc": max(cv['end_utc'] for cv in all_camera_views)
        }
        final_passes.append(final_pass)

    final_passes.sort(key=lambda p: p['earliest_camera_utc'], reverse=True)
    return final_passes

def _station_cache_path(station_id):
    return os.path.join(PASS_CACHE_DIR, f"{station_id}.json")

def _load_station_cache(station_id, current_tle_names):
    """Returns the cached raw (ungrouped) passes for a station if the cache
    exists, is within PASS_CACHE_LIFETIME_MINUTES, and was built from the
    current TLE set. Returns None if a fresh computation is needed."""
    path = _station_cache_path(station_id)
    if not os.path.exists(path):
        return None
    try:
        mod_time = os.path.getmtime(path)
        if (datetime.now().timestamp() - mod_time) >= PASS_CACHE_LIFETIME_MINUTES * 60:
            return None
        cached = read_json_file(path, default=None)
        if not cached or set(cached.get("satellites_in_cache", [])) != current_tle_names:
            return None
        return cached.get("passes", [])
    except (IOError, json.JSONDecodeError):
        return None

def _save_station_cache(station_id, tle_names, station_passes):
    os.makedirs(PASS_CACHE_DIR, exist_ok=True)
    atomic_json_write(_station_cache_path(station_id), {"satellites_in_cache": list(tle_names), "passes": station_passes}, indent=2)

def _filter_by_days(final_passes, days):
    """Keeps only passes that overlap the last `days` days (i.e. any part of
    the pass -- not just its earliest camera view -- falls at/after the
    cutoff). `days` of None or <= 0 disables filtering (full SEARCH_DAYS)."""
    if not days or days <= 0:
        return final_passes
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    return [p for p in final_passes if p.get('latest_camera_utc', p['earliest_camera_utc']) >= cutoff_iso]

def _parse_iso(value):
    """Parses an ISO8601 timestamp (with optional trailing 'Z') into a
    'YYYY-MM-DDTHH:MM:SSZ' string comparable with skyfield's utc_iso() output.
    Returns None if `value` is falsy or unparseable."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        return None

def _filter_by_range(final_passes, start_iso, end_iso):
    """Keeps only passes that *overlap* [start_iso, end_iso] (either bound
    may be omitted), i.e. any part of the pass's span -- from its earliest to
    its latest camera view across all involved stations -- intersects the
    requested window. A pure "earliest_camera_utc within bounds" test would
    wrongly drop passes whose earliest view starts just before a narrow
    window even though most of the pass (as seen by other stations/cameras)
    falls inside it. Takes precedence over `_filter_by_days` when explicit
    bounds are supplied (e.g. from the satellite panel's drag-to-select time
    range slider)."""
    start_cutoff = _parse_iso(start_iso)
    end_cutoff = _parse_iso(end_iso)
    if not start_cutoff and not end_cutoff:
        return final_passes
    result = final_passes
    if start_cutoff:
        result = [p for p in result if p.get('latest_camera_utc', p['earliest_camera_utc']) >= start_cutoff]
    if end_cutoff:
        result = [p for p in result if p['earliest_camera_utc'] <= end_cutoff]
    return result

def _compute_and_cache_stations(station_items, tle_data, tle_names, status_file=None):
    """Computes passes for the given [(station_id, station_info), ...] list
    (in parallel across stations), caching each station's result as it
    completes. Returns the combined raw (ungrouped) passes for all of them."""
    all_passes_found = []
    if not station_items:
        return all_passes_found
    tasks = [(sid, sinfo, tle_data) for sid, sinfo in station_items]
    total = len(tasks)
    done = 0
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        futures = {executor.submit(process_station, task): task[0] for task in tasks}
        for future in as_completed(futures):
            station_id = futures[future]
            try:
                station_result = future.result()
                all_passes_found.extend(station_result)
                _save_station_cache(station_id, tle_names, station_result)
            except Exception as exc:
                logging.error(f"A station task generated an exception for {station_id}: {exc}")
            done += 1
            if status_file:
                progress = 5 + int((done / total) * 90)
                message = f"status_calculating_for_station|processed={done},total={total}"
                update_status(status_file, "progress", {"step": progress, "total": 100, "message": message})
    return all_passes_found

def find_all_passes_for_cron():
    """
    A special version of the prediction function intended to be run by a cron
    job. It runs silently and its only purpose is to keep every station's
    per-station pass cache warm, so on-demand requests (for any station
    selection) can be served quickly.
    """
    try:
        ts = load.timescale()
        with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)
        tle_data = get_tle_data(ts)
        if "error" in tle_data:
            logging.error(f"Failed to get TLE data in cron mode: {tle_data['error']}")
            return
        tle_names = set(tle_data.keys())

        stale_items = [(sid, sinfo) for sid, sinfo in stations_data.items() if _load_station_cache(sid, tle_names) is None]
        if not stale_items:
            logging.info("Cron run: All station caches are still valid. Exiting.")
            return

        logging.info(f"--- Starting cron pass prediction for {len(stale_items)} stale/missing station caches ---")
        _compute_and_cache_stations(stale_items, tle_data, tle_names)
        logging.info("Cron prediction finished; per-station caches updated.")
    except Exception as e:
        logging.exception(f"An unhandled error occurred during cron pass prediction")

def find_all_passes(task_id, station_ids=None, days=None, start_iso=None, end_iso=None):
    """
    Main orchestrator function for an on-demand pass prediction request.

    station_ids: optional list of station IDs to restrict the search to. This
    is the main lever for keeping requests fast: each station's raw passes
    (over the full SEARCH_DAYS window) are cached independently, so asking
    for one or two stations only computes (or reuses the cache for) those,
    instead of every station in the network.
    days: optional number of days (<= SEARCH_DAYS) to limit the *returned*
    results to (from now, going back). Ignored if start_iso/end_iso is given.
    start_iso/end_iso: optional explicit ISO8601 UTC bounds (e.g. from the
    satellite panel's drag-to-select time range slider), used instead of
    `days` when either is supplied. Both filters are cheap post-filters over
    the (fully cached) results and never trigger recomputation.
    """
    status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    try:
        ts = load.timescale()
        with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)

        if station_ids:
            target_items = [(sid, stations_data[sid]) for sid in station_ids if sid in stations_data]
            if not target_items:
                update_status(status_file, "error", {"message": "error_invalid_station"})
                return
        else:
            target_items = list(stations_data.items())

        tle_data = get_tle_data(ts)
        if "error" in tle_data:
            update_status(status_file, "error", {"message": tle_data["error"]})
            return
        tle_names = set(tle_data.keys())

        update_status(status_file, "progress", {"step": 5, "total": 100, "message": "status_calculating"})

        cached_passes, to_compute = [], []
        for sid, sinfo in target_items:
            station_cached = _load_station_cache(sid, tle_names)
            if station_cached is None:
                to_compute.append((sid, sinfo))
            else:
                cached_passes.extend(station_cached)

        logging.info(f"Task {task_id}: {len(target_items) - len(to_compute)} station(s) served from cache, {len(to_compute)} need computation.")
        computed_passes = _compute_and_cache_stations(to_compute, tle_data, tle_names, status_file=status_file)

        update_status(status_file, "progress", {"step": 95, "total": 100, "message": "status_grouping_results"})
        final_passes = _group_and_finalize_passes(cached_passes + computed_passes)
        if start_iso or end_iso:
            final_passes = _filter_by_range(final_passes, start_iso, end_iso)
        else:
            final_passes = _filter_by_days(final_passes, days)
        result_data = {"passes": final_passes}
        logging.info(f"Finished prediction for task {task_id}. Found {len(final_passes)} passes (range={start_iso}..{end_iso}, days={days or SEARCH_DAYS}, stations={len(target_items)}).")
        update_status(status_file, "complete", {"data": result_data})
    except Exception as e:
        logging.exception(f"An unhandled error occurred during pass prediction for task {task_id}")
        update_status(status_file, "error", {"message": "error_internal", "task_id": task_id, "debug": "logs/predict_sat.log"})

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
    parser.add_argument("--station", default=None, help="Comma-separated list of station IDs to restrict the search to (default: all stations).")
    parser.add_argument("--days", type=int, default=None, help=f"Only return passes from the last N days (default: full {SEARCH_DAYS}-day search window). Ignored if --start/--end is given.")
    parser.add_argument("--start", default=None, help="Only return passes at/after this ISO8601 UTC timestamp. Takes precedence over --days.")
    parser.add_argument("--end", default=None, help="Only return passes at/before this ISO8601 UTC timestamp. Takes precedence over --days.")
    args = parser.parse_args()
    station_ids = [s.strip() for s in args.station.split(',') if s.strip()] if args.station else None

    is_quiet = args.quiet or args.cron
    if not is_quiet:
        logging.basicConfig(
            level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
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
            find_all_passes(args.task_id, station_ids=station_ids, days=args.days, start_iso=args.start, end_iso=args.end)
        except Exception as e:
            logging.exception("A fatal error occurred at the top level of the script!")
            update_status(status_file, "error", {"message": "error_internal", "task_id": args.task_id, "debug": "logs/predict_sat.log"})
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
