#!/usr/bin/env python3

import os
import json
import subprocess
import logging
import time
import shutil
import signal
import socket
import re
import shlex
import threading
from datetime import datetime, timezone
from PIL import Image

# Import from our new shared utility library
# Imports utility functions shared across multiple backend scripts.
from shared_utils import atomic_json_rw, update_status, uniqid, read_json_file, pid_cmdline_matches

# --- Configuration (specific to streaming) ---
# Establishes base paths for all necessary directories and configuration files.
BASE_DIR = os.environ.get('NMN_DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, 'locks')
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'download')
STREAM_DIR = os.path.join(BASE_DIR, 'streams')
STATIONS_FILE = os.path.join(BASE_DIR, 'stations.json')
STREAM_TIME_TRACKER_FILE = os.path.join(BASE_DIR, 'stream_time_tracker.json')

GRID_CACHE_DIR = DOWNLOAD_DIR

# Defines the total daily streaming time allowed per user IP, per station.
# Stations with the 'quota' flag have more restrictive limits.
STREAM_TIME_LIMITS_SECONDS = {
    'normal': {'lowres': 2 * 3600, 'hires': 30 * 60}, # Normal stations: 2hr low-res, 30min high-res
    'quota': {'lowres': 30 * 60, 'hires': 10 * 60}    # Quota stations: 30min low-res, 10min high-res
}


def fetch_grid_file(stream_task_id, station_id, camera_num):
    """
    Fetches the calibration grid image for a specific camera from a remote station.
    This allows the user to overlay a grid on the live video stream for reference.
    """
    log_prefix = f"GridFetch for {stream_task_id} -"
   
    logging.info(f"{log_prefix} Request for {station_id} cam {camera_num}.")
    status_file = os.path.join(LOCK_DIR, f"{stream_task_id}.json")
    
    # Waits for the main stream task's status file to be created.
    for _ in range(50): # Wait up to 10 seconds for status file
        if os.path.exists(status_file): break
        time.sleep(0.2)
    else:
        logging.error(f"{log_prefix} Status file not found after waiting.")
        return {"success": False, "error": "Stream task not found."}

    try:
        os.makedirs(GRID_CACHE_DIR, exist_ok=True)

        cached_filename = f"grid_{station_id}_cam{camera_num}.png"
        cached_filepath = os.path.join(GRID_CACHE_DIR, cached_filename)
        if os.path.exists(cached_filepath) and os.path.getsize(cached_filepath) > 0:
            try:
                age_seconds = time.time() - os.path.getmtime(cached_filepath)
            except OSError:
                age_seconds = 10**9

            if age_seconds < 86400:
                logging.info(f"{log_prefix} Using cached grid: {cached_filepath}")
                return {"success": True, "grid_url": f"download/{cached_filename}"}
            logging.info(f"{log_prefix} Cached grid is stale ({age_seconds:.0f}s). Refetching.")

        # Securely copies the grid.png file from the remote station.
        tmp_filename = f"grid_{station_id}_cam{camera_num}_{uniqid()}.png"
        tmp_filepath = os.path.join(DOWNLOAD_DIR, tmp_filename)
        command = ["scp", "-B", "-o", "ConnectTimeout=10", f"{station_id}:/meteor/cam{camera_num}/grid.png", tmp_filepath]
        subprocess.run(command, check=True, timeout=40, capture_output=True)
        logging.info(f"{log_prefix} Fetched grid to {tmp_filepath}")

        try:
            os.replace(tmp_filepath, cached_filepath)
        except OSError:
            # If atomic replace fails, keep the tmp file and serve it.
            cached_filepath = tmp_filepath
            cached_filename = os.path.basename(cached_filepath)

        # Updates the stream's status file with the path to the downloaded grid.
        with atomic_json_rw(status_file, stream_task_id) as data:
            data['grid_local_path'] = cached_filepath
            data['grid_cached'] = (cached_filepath == os.path.join(GRID_CACHE_DIR, f"grid_{station_id}_cam{camera_num}.png"))
        
        return {"success": True, "grid_url": f"download/{cached_filename}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} SCP timed out.")
        return {"success": False, "error": "error_grid_fetch_timeout"}
    except subprocess.CalledProcessError as e:
        logging.error(f"{log_prefix} SCP failed. Stderr: {e.stderr.decode()}")
        return {"success": False, "error": "error_grid_not_found"}
    except Exception as e:
        logging.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
        return {"success": False, "error": "error_internal"}


DRAWGRID_SCRIPT = os.path.join(os.path.dirname(BASE_DIR), 'bin', 'drawgrid.py')
PTO_CACHE_DIR = DOWNLOAD_DIR


def fetch_annotation_file(stream_task_id, station_id, camera_num):
    """
    Generates a star annotation overlay for the live stream.
    Uses the cached grid PNG as a base and draws star positions on top using drawgrid.py.
    Requires the lens.pto calibration file from the remote station.
    """
    log_prefix = f"AnnotationFetch for {stream_task_id} -"
    logging.info(f"{log_prefix} Request for {station_id} cam {camera_num}.")

    status_file = os.path.join(LOCK_DIR, f"{stream_task_id}.json")
    for _ in range(50):
        if os.path.exists(status_file):
            break
        time.sleep(0.2)
    else:
        logging.error(f"{log_prefix} Status file not found after waiting.")
        return {"success": False, "error": "Stream task not found."}

    try:
        os.makedirs(PTO_CACHE_DIR, exist_ok=True)

        # 1. Fetch lens.pto from the remote station (cache for 24h)
        pto_filename = f"lens_{station_id}_cam{camera_num}.pto"
        pto_filepath = os.path.join(PTO_CACHE_DIR, pto_filename)
        pto_fresh = False
        if os.path.exists(pto_filepath) and os.path.getsize(pto_filepath) > 0:
            try:
                age_seconds = time.time() - os.path.getmtime(pto_filepath)
            except OSError:
                age_seconds = 10**9
            if age_seconds < 86400:
                pto_fresh = True
                logging.info(f"{log_prefix} Using cached lens.pto: {pto_filepath}")

        if not pto_fresh:
            tmp_pto = os.path.join(DOWNLOAD_DIR, f"lens_{station_id}_cam{camera_num}_{uniqid()}.pto")
            command = ["scp", "-B", "-o", "ConnectTimeout=10",
                       f"{station_id}:/meteor/cam{camera_num}/lens.pto", tmp_pto]
            subprocess.run(command, check=True, timeout=40, capture_output=True)
            logging.info(f"{log_prefix} Fetched lens.pto to {tmp_pto}")
            try:
                os.replace(tmp_pto, pto_filepath)
            except OSError:
                pto_filepath = tmp_pto

        # 2. Get station latitude/longitude from stations.json
        with open(STATIONS_FILE, 'r') as f:
            stations_data = json.load(f)
        station = stations_data.get(station_id, {})
        lat = station.get('astronomy', {}).get('latitude')
        lon = station.get('astronomy', {}).get('longitude')
        if lat is None or lon is None:
            logging.error(f"{log_prefix} Station {station_id} missing lat/lon.")
            return {"success": False, "error": "error_station_not_found"}

        # 3. Run drawgrid.py with --annotations-only for star-only transparent overlay
        # Add 15s to compensate for video delay (~8s) and refresh interval
        timestamp = int(time.time()) + 15
        annotation_filename = f"annotation_{station_id}_cam{camera_num}.png"
        annotation_filepath = os.path.join(DOWNLOAD_DIR, annotation_filename)

        cmd = [
            "python3", DRAWGRID_SCRIPT,
            "--annotations-only",
            "-Y", str(lat), "-X", str(lon),
            "-d", str(timestamp),
            pto_filepath, annotation_filepath
        ]
        logging.info(f"{log_prefix} Running drawgrid: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logging.error(f"{log_prefix} drawgrid failed: {result.stderr}")
            return {"success": False, "error": "error_annotation_generation_failed"}

        logging.info(f"{log_prefix} Annotation generated: {annotation_filepath}")
        return {"success": True, "annotation_url": f"download/{annotation_filename}?t={timestamp}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} Operation timed out.")
        return {"success": False, "error": "error_grid_fetch_timeout"}
    except subprocess.CalledProcessError as e:
        logging.error(f"{log_prefix} SCP failed. Stderr: {e.stderr.decode()}")
        return {"success": False, "error": "error_annotation_pto_not_found"}
    except Exception as e:
        logging.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
        return {"success": False, "error": "error_internal"}


# --- Archive Video Overlay Functions (separate from live stream) ---

def get_archive_grid_overlay(station_code, cam_num, timestamp, stations_data):
    """
    Fetches or generates a grid overlay for an archive video.
    For archive videos, we use the current grid.png (calibration doesn't change often).
    Returns a web-accessible URL path.
    """
    # Map station code to station ID
    station_id = None
    for sid, s in stations_data.items():
        if s.get('station', {}).get('code', '').upper() == station_code.upper():
            station_id = sid
            break

    if not station_id:
        logging.error(f"[ArchiveGrid] Station code {station_code} not found in stations data")
        return {"success": False, "error": "error_station_not_found"}

    log_prefix = f"[ArchiveGrid {station_id}_cam{cam_num}]"
    logging.info(f"{log_prefix} Request for timestamp={timestamp}")

    # Extract date from timestamp for caching
    try:
        date_str = timestamp[:10].replace('-', '')  # YYYY-MM-DD -> YYYYMMDD
    except:
        date_str = datetime.now().strftime('%Y%m%d')

    cached_filename = f"grid_{station_id}_cam{cam_num}_{date_str}.png"
    cached_filepath = os.path.join(DOWNLOAD_DIR, cached_filename)

    # Check if already cached
    if os.path.exists(cached_filepath) and os.path.getsize(cached_filepath) > 0:
        logging.info(f"{log_prefix} Using cached grid: {cached_filename}")
        return {"success": True, "grid_url": f"download/{cached_filename}"}

    # Need to fetch from station
    try:
        hostname = station_id  # Use station_id directly as hostname

        # Try to fetch dated grid file first
        remote_dated = f"/meteor/cam{cam_num}/grid-{date_str}.png"
        tmp_path = os.path.join(DOWNLOAD_DIR, f"grid_{station_id}_cam{cam_num}_{uniqid()}.png")

        scp_cmd = ["scp", "-B", "-o", "ConnectTimeout=10",
                   f"{hostname}:{remote_dated}", tmp_path]
        logging.info(f"{log_prefix} Trying dated grid: {remote_dated}")
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0 or not os.path.exists(tmp_path):
            # Fall back to current grid.png
            logging.info(f"{log_prefix} Dated grid not found, falling back to grid.png")
            remote_current = f"/meteor/cam{cam_num}/grid.png"
            scp_cmd = ["scp", "-B", "-o", "ConnectTimeout=10",
                       f"{hostname}:{remote_current}", tmp_path]
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0 or not os.path.exists(tmp_path):
                logging.error(f"{log_prefix} SCP failed: {result.stderr}")
                return {"success": False, "error": "error_grid_not_found"}

        # Move to cache location
        try:
            os.replace(tmp_path, cached_filepath)
        except OSError:
            cached_filepath = tmp_path
            cached_filename = os.path.basename(tmp_path)

        logging.info(f"{log_prefix} Grid fetched: {cached_filename}")
        return {"success": True, "grid_url": f"download/{cached_filename}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} SCP timeout")
        return {"success": False, "error": "error_grid_fetch_timeout"}
    except Exception as e:
        logging.error(f"{log_prefix} Error: {e}")
        return {"success": False, "error": "error_internal"}


def _equirect_map_h(stitch_w: int, stitch_h: int = 0) -> int:
    """Return the equirect PTO canvas height (orig_h) for a given output width.

    Two canvas aspect ratios are in use:
      - stitch_latest.sh scales the default 4096×2160 canvas proportionally
        (h/w = 2160/4096 ≈ 0.5273).
      - server stitch.py passes explicit 5120×3392 or 1280×848
        (h/w = 3392/5120 = 0.6625).

    """
    # server stitch.py uses exactly two fixed canvas widths defined in stitch.py:
    #   EQUIRECT_HIRES_W=5120, EQUIRECT_LOWRES_W=1280
    # Everything else comes from stitch_latest.sh which scales the 4096x2160 default.
    if stitch_w in (5120, 1280):
        return round(3392 * stitch_w / 5120) & ~1
    return round(2160 * stitch_w / 4096) & ~1


def get_stitch_cam_boundaries(station_id_arg: str, projection: str, stations_data: dict, resolution: str = 'hires') -> dict:
    """
    Generate (or return cached) a camera-boundary overlay PNG for a stitched
    equirectangular or fisheye panorama.

    Fetches /meteor/camN/lens.pto for every camera of the station via SSH,
    then runs draw_camera_boundaries.py to produce a transparent PNG.

    Args:
        station_id_arg: Station ID string (e.g. 'ams173') used as SSH host.
        projection:     'eq' or 'fe'
        stations_data:  Full stations.json dict.

    Returns:
        {'success': True,  'grid_url': 'download/cam_bounds_ams173_eq.png'}
        {'success': False, 'error': '...'}
    """
    if projection not in ('eq', 'fe'):
        return {"success": False, "error": "error_invalid_projection"}

    # Accept either station SSH ID (e.g. 'ams173') or 3-letter station code (e.g. 'GAU')
    station_id = station_id_arg
    station_data = stations_data.get(station_id_arg)
    if not station_data:
        # Try resolving by 3-letter code
        for sid, s in stations_data.items():
            if s.get('station', {}).get('code', '').upper() == station_id_arg.upper():
                station_id = sid
                station_data = s
                break
    if not station_data:
        return {"success": False, "error": "error_station_not_found"}

    resolution = resolution if resolution in ('hires', 'lowres') else 'hires'
    log_prefix = f"[CamBounds {station_id} {projection} {resolution}]"

    cached_name = f"cam_bounds_{station_id}_{projection}_{resolution}.png"
    cached_path = os.path.join(DOWNLOAD_DIR, cached_name)
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
        logging.info(f"{log_prefix} Using cached: {cached_name}")
        return {"success": True, "grid_url": f"download/{cached_name}"}

    # Determine which camera numbers to fetch based on projection.
    # equirect uses cams 1-7 (not 8/9 which are stitched outputs),
    # fisheye likewise.  We fetch all available.
    # The station's cameras list comes from stations.json where available.
    cameras_info = station_data.get('cameras', {})
    cam_nums = sorted(int(k.replace('cam', '')) for k in cameras_info if k.startswith('cam') and k.replace('cam', '').isdigit())
    # Exclude virtual stitch cameras (8, 9)
    cam_nums = [c for c in cam_nums if c < 8]
    if not cam_nums:
        # Fallback: assume cams 1-7
        cam_nums = list(range(1, 8))

    logging.info(f"{log_prefix} Fetching lens.pto for cams {cam_nums}")

    # Fetch lens.pto files (SSH tar) and stitch dimensions (SSH identify) in parallel.
    import tempfile, threading
    workdir = tempfile.mkdtemp(prefix="cam_bounds_")
    try:
        stitch_cam = "cam8" if projection == "eq" else "cam9"
        pat = 'mini' if resolution == 'lowres' else 'full'

        tar_result   = [None]
        id_result    = [None]
        tar_exc      = [None]
        id_exc       = [None]

        def _fetch_tar():
            try:
                remote_files = " ".join(f"/meteor/cam{c}/lens.pto" for c in cam_nums)
                remote_cmd = f"tar -c -h --ignore-failed-read -f - {remote_files} 2>/dev/null"
                tar_result[0] = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes",
                     station_id, remote_cmd],
                    capture_output=True, timeout=60
                )
            except Exception as e:
                tar_exc[0] = e

        def _fetch_size():
            try:
                id_result[0] = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                     station_id,
                     f"find /meteor/{stitch_cam}/ -name '{pat}_*.jpg' | head -1 | xargs -r identify -format '%w %h\\n' 2>/dev/null"],
                    capture_output=True, timeout=30, text=True
                )
            except Exception as e:
                id_exc[0] = e

        t_tar  = threading.Thread(target=_fetch_tar,  daemon=True)
        t_size = threading.Thread(target=_fetch_size, daemon=True)
        t_tar.start(); t_size.start()
        t_tar.join();  t_size.join()

        if tar_exc[0]:
            logging.warning(f"{log_prefix} SSH tar error: {tar_exc[0]}")
            return {"success": False, "error": "error_lens_fetch_failed"}
        result = tar_result[0]
        if result is None or result.returncode not in (0, 1) or not result.stdout:
            logging.warning(f"{log_prefix} SSH tar failed, rc={result.returncode if result else 'None'}")
            return {"success": False, "error": "error_lens_fetch_failed"}

        tar_dir = os.path.join(workdir, "tar")
        os.makedirs(tar_dir, exist_ok=True)
        subprocess.run(["tar", "-x", "-f", "-", "-C", tar_dir],
                       input=result.stdout, capture_output=True, timeout=30)

        lens_files = []
        for c in cam_nums:
            src = os.path.join(tar_dir, "meteor", f"cam{c}", "lens.pto")
            dest = os.path.join(workdir, f"lens_cam{c}.pto")
            if os.path.exists(src) and os.path.getsize(src) > 0:
                shutil.move(src, dest)
                lens_files.append(dest)
                logging.info(f"{log_prefix} lens.pto cam{c} OK")
            else:
                logging.warning(f"{log_prefix} lens.pto cam{c} not found on station")

        if not lens_files:
            return {"success": False, "error": "error_no_lens_files"}

        # Locate draw_camera_boundaries.py and the panorama PTO.
        # Use __file__'s realpath (the repo copy of live_streamer.py) so that
        # symlinked or NFS-mounted BASE_DIR paths don't lead to wrong locations.
        _here = os.path.realpath(os.path.dirname(__file__))
        draw_script = os.path.normpath(os.path.join(_here, '..', '..', 'bin', 'draw_camera_boundaries.py'))
        if not os.path.exists(draw_script):
            # Fallback: search upward from __file__ for a bin/ sibling
            p = _here
            draw_script = None
            for _ in range(5):
                candidate = os.path.join(p, 'bin', 'draw_camera_boundaries.py')
                if os.path.exists(candidate):
                    draw_script = candidate
                    break
                p = os.path.dirname(p)
            if not draw_script:
                logging.error(f"{log_prefix} draw_camera_boundaries.py not found (searched from {_here})")
                return {"success": False, "error": "error_script_not_found"}

        # Panorama PTO lives alongside live_streamer.py in the repo data dir
        pano_pto_name = f"grid_{'eq' if projection == 'eq' else 'fe'}_hd.pto"
        pano_pto = os.path.join(_here, pano_pto_name)
        if not os.path.exists(pano_pto):
            logging.error(f"{log_prefix} Pano PTO not found: {pano_pto}")
            return {"success": False, "error": "error_pano_pto_not_found"}

        # Collect stitch dimensions from the parallel identify result
        stitch_w, stitch_h = None, None
        if id_exc[0]:
            logging.warning(f"{log_prefix} Could not get stitch dimensions: {id_exc[0]}")
        elif id_result[0] is not None and id_result[0].returncode == 0 and id_result[0].stdout.strip():
            try:
                parts = id_result[0].stdout.strip().split()
                stitch_w, stitch_h = int(parts[0]), int(parts[1])
                logging.info(f"{log_prefix} Stitch size from station: {stitch_w}x{stitch_h}")
            except Exception as e:
                logging.warning(f"{log_prefix} Could not parse stitch dimensions: {e}")

        # Defaults if station query failed
        if not stitch_w or not stitch_h:
            if projection == "eq":
                stitch_w, stitch_h = (1280, 448) if resolution == 'lowres' else (4096, 1168)
            else:
                stitch_w, stitch_h = (4096, 4096)
            logging.info(f"{log_prefix} Using default stitch size: {stitch_w}x{stitch_h}")

        # For equirect: compute crop_top — the y-offset of the stitched strip
        # within the full equirect sphere.  The stitcher crops from the first row
        # with any camera coverage.  We replicate that by projecting all camera
        # sensor corners through the same mapping and taking the minimum y.
        crop_top = 0
        if projection == "eq":
            try:
                import sys as _sys
                _bin = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..', '..', 'bin'))
                if _bin not in _sys.path:
                    _sys.path.insert(0, _bin)
                import pto_mapper as _pto_mapper
                _pano_global, _ = _pto_mapper.parse_pto_file(pano_pto)
                _pano_v = float(_pano_global.get('v', 360))
                # Use the exact same canvas as stitcher.py's calculate_source_coords.
                # Two canvas aspect ratios exist:
                #   stitch_latest.sh  -> scaled from 4096x2160  (h/w = 2160/4096)
                #   server stitch.py  -> explicit 5120x3392 or 1280x848  (h/w = 3392/5120)
                # Detect by checking which base width the stitch_w scales from.
                _map_w = stitch_w
                _map_h = _equirect_map_h(stitch_w, stitch_h)  # even, matches stitcher's canvas
                _mapping = {'f': 2, 'v': _pano_v, 'w': _map_w, 'h': _map_h, 'r': 0.0, 's': 1.0}
                _min_y = float(_map_h)  # start high
                for lf in lens_files:
                    _, _imgs = _pto_mapper.parse_pto_file(lf)
                    if not _imgs:
                        continue
                    _img = _imgs[0]
                    _w, _h = float(_img.get('w', 1920)), float(_img.get('h', 1080))
                    _pto = (_mapping, [_img])
                    _N = 100
                    for _i in range(_N + 1):
                        _t = _i / _N
                        for _ex, _ey in [(_t*_w, 0), (_t*_w, _h), (0, _t*_h), (_w, _t*_h)]:
                            _r = _pto_mapper.map_image_to_pano(_pto, 0, _ex, _ey)
                            if _r is not None:
                                _min_y = min(_min_y, _r[1])
                # crop_top is in PTO canvas units (unscaled); the script subtracts
                # it before applying x_scale, so do NOT pre-scale here.
                crop_top = int(_min_y)
                logging.info(f"{log_prefix} Equirect crop_top={crop_top} (min_y={_min_y:.0f}, map={_map_w}x{_map_h})")
            except Exception as e:
                logging.warning(f"{log_prefix} Could not compute crop_top: {e}")
                crop_top = 0

        import uuid
        tmp_out = os.path.join(DOWNLOAD_DIR, f"cam_bounds_tmp_{uuid.uuid4().hex}.png")
        map_h_arg = str(_equirect_map_h(stitch_w, stitch_h)) if projection == "eq" else str(stitch_h)
        cmd = [
            "python3", draw_script,
            "--pano", pano_pto,
            "--lens"] + lens_files + [
            "--output", tmp_out,
            "--width", str(stitch_w),
            "--height", str(stitch_h),
            "--map-height", map_h_arg,
            "--crop-top", str(crop_top),
            "--samples", "400",
        ]
        logging.info(f"{log_prefix} Running draw_camera_boundaries.py")
        proc = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if proc.returncode != 0 or not os.path.exists(tmp_out):
            logging.error(f"{log_prefix} draw_camera_boundaries failed: {proc.stderr[:500]}")
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            return {"success": False, "error": "error_boundary_generation_failed"}

        os.replace(tmp_out, cached_path)
        logging.info(f"{log_prefix} Saved: {cached_name}")
        return {"success": True, "grid_url": f"download/{cached_name}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} Timeout")
        return {"success": False, "error": "error_timeout"}
    except Exception as e:
        logging.exception(f"{log_prefix} Exception: {e}")
        return {"success": False, "error": "error_internal"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def get_archive_annotation_overlay(station_code, cam_num, timestamp, stations_data):
    """
    Generates a star annotation overlay for an archive video using drawgrid.py.
    Uses the video's timestamp to calculate star positions.
    Returns a web-accessible URL path.
    """
    # Map station code to station ID
    station_id = None
    for sid, s in stations_data.items():
        if s.get('station', {}).get('code', '').upper() == station_code.upper():
            station_id = sid
            break

    if not station_id:
        logging.error(f"[ArchiveAnnotation] Station code {station_code} not found in stations data")
        return {"success": False, "error": "error_station_not_found"}

    log_prefix = f"[ArchiveAnnotation {station_id}_cam{cam_num}]"
    logging.info(f"{log_prefix} Request for timestamp={timestamp}")

    try:
        # Extract timestamp for caching (include hour and minute since stars move)
        date_str = timestamp[:10].replace('-', '')  # YYYY-MM-DD -> YYYYMMDD
        hour_min = timestamp[11:13] + timestamp[14:16]  # HH:MM -> HHMM
        output_filename = f"annotation_{station_id}_cam{cam_num}_{date_str}_{hour_min}_labels.png"
        output_path = os.path.join(DOWNLOAD_DIR, output_filename)

        # Check cache (valid for 1 hour since stars don't move much in that time)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_age = time.time() - os.path.getmtime(output_path)
            if file_age < 3600:  # 1 hour cache
                logging.info(f"{log_prefix} Using cached annotation (age={file_age:.0f}s)")
                return {"success": True, "annotation_url": f"download/{output_filename}"}
            logging.info(f"{log_prefix} Cache expired (age={file_age:.0f}s), regenerating")

        # Get station lat/lon
        station = stations_data.get(station_id, {})
        lat = station.get('astronomy', {}).get('latitude')
        lon = station.get('astronomy', {}).get('longitude')
        if lat is None or lon is None:
            logging.error(f"{log_prefix} Station missing lat/lon")
            return {"success": False, "error": "error_station_not_found"}

        # Fetch lens.pto if needed
        pto_filename = f"lens_{station_id}_cam{cam_num}.pto"
        pto_path = os.path.join(DOWNLOAD_DIR, pto_filename)

        if not os.path.exists(pto_path) or os.path.getsize(pto_path) == 0:
            hostname = station_id
            remote_pto = f"/meteor/cam{cam_num}/lens.pto"
            scp_cmd = ["scp", "-B", "-o", "ConnectTimeout=10",
                       f"{hostname}:{remote_pto}", pto_path]
            logging.info(f"{log_prefix} Fetching lens.pto")
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logging.error(f"{log_prefix} Failed to fetch lens.pto: {result.stderr}")
                return {"success": False, "error": "error_annotation_pto_not_found"}

        # Parse timestamp to Unix epoch. Keep the datetime timezone-aware so
        # .timestamp() is interpreted as UTC regardless of server local time.
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            epoch_time = int(dt.timestamp())
        except:
            epoch_time = int(time.time())

        # Generate annotation with drawgrid.py
        cmd = [
            "python3", DRAWGRID_SCRIPT,
            "--annotations-only",
            "-Y", str(lat), "-X", str(lon),
            "-d", str(epoch_time),
            pto_path, output_path
        ]
        logging.info(f"{log_prefix} Running drawgrid: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logging.error(f"{log_prefix} drawgrid failed: {result.stderr}")
            return {"success": False, "error": "error_annotation_generation_failed"}

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logging.error(f"{log_prefix} Output file not created")
            return {"success": False, "error": "error_annotation_generation_failed"}

        logging.info(f"{log_prefix} Annotation generated: {output_filename}")
        return {"success": True, "annotation_url": f"download/{output_filename}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} Operation timeout")
        return {"success": False, "error": "error_annotation_timeout"}
    except Exception as e:
        logging.error(f"{log_prefix} Error: {e}")
        return {"success": False, "error": "error_internal"}


def get_archive_mask_overlay(station_code, cam_num, stations_data):
    """
    Fetches a camera's foreground mask (/meteor/camN/mask.png on the station,
    white=non-sky/foreground, black=sky - see automask.py) and turns it into
    a transparent "Show mask" overlay: sky (black) becomes fully transparent
    and foreground (white) becomes opaque black, so overlaying it on the
    image/video visually masks out everything but the sky.
    Returns a web-accessible URL path.
    """
    # Map station code to station ID
    station_id = None
    for sid, s in stations_data.items():
        if s.get('station', {}).get('code', '').upper() == station_code.upper():
            station_id = sid
            break

    if not station_id:
        logging.error(f"[ArchiveMask] Station code {station_code} not found in stations data")
        return {"success": False, "error": "error_station_not_found"}

    log_prefix = f"[ArchiveMask {station_id}_cam{cam_num}]"
    logging.info(f"{log_prefix} Request")

    try:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        cached_filename = f"mask_{station_id}_cam{cam_num}.png"
        cached_filepath = os.path.join(DOWNLOAD_DIR, cached_filename)

        # The station-side mask only changes when automask.py is re-run
        # there, so a day-long cache (matching fetch_archive_grid) is fine.
        if os.path.exists(cached_filepath) and os.path.getsize(cached_filepath) > 0:
            try:
                age_seconds = time.time() - os.path.getmtime(cached_filepath)
            except OSError:
                age_seconds = 10**9
            if age_seconds < 86400:
                logging.info(f"{log_prefix} Using cached mask overlay (age={age_seconds:.0f}s)")
                return {"success": True, "mask_url": f"download/{cached_filename}"}
            logging.info(f"{log_prefix} Cached mask overlay is stale ({age_seconds:.0f}s), refetching")

        tmp_raw = os.path.join(DOWNLOAD_DIR, f"mask_raw_{station_id}_cam{cam_num}_{uniqid()}.png")
        scp_cmd = ["scp", "-B", "-o", "ConnectTimeout=10",
                   f"{station_id}:/meteor/cam{cam_num}/mask.png", tmp_raw]
        logging.info(f"{log_prefix} Fetching mask.png")
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not os.path.exists(tmp_raw):
            logging.error(f"{log_prefix} SCP failed: {result.stderr}")
            return {"success": False, "error": "error_mask_not_found"}

        try:
            # Black (sky, 0) in the source mask -> alpha 0 (transparent).
            # White (foreground, 255) -> alpha 255, drawn as opaque black.
            with Image.open(tmp_raw) as src:
                alpha = src.convert('L')
                black = Image.new('L', alpha.size, 0)
                overlay = Image.merge('RGBA', (black, black, black, alpha))
                tmp_out = cached_filepath + f".{uniqid()}.tmp"
                overlay.save(tmp_out, 'PNG')
            os.replace(tmp_out, cached_filepath)
        finally:
            try:
                os.remove(tmp_raw)
            except OSError:
                pass

        logging.info(f"{log_prefix} Mask overlay generated: {cached_filename}")
        return {"success": True, "mask_url": f"download/{cached_filename}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} SCP timeout")
        return {"success": False, "error": "error_grid_fetch_timeout"}
    except Exception as e:
        logging.error(f"{log_prefix} Error: {e}")
        return {"success": False, "error": "error_internal"}


def _get_timeout_for_station(station_id, resolution, stations_data):
    """Determines the maximum duration for a single streaming session."""
    has_quota = stations_data.get(station_id, {}).get("station", {}).get("quota", False)
    if resolution == 'hires':
        return 60 if has_quota else 3 * 60 # 1 minute for quota, 3 for normal
    else: # lowres
 
        return 5 * 60 if has_quota else 15 * 60 # 5 minutes for quota, 15 for normal


def _check_stream_time_quota(user_ip, station_id, resolution, stations_data):
    """Checks if a user has exceeded their daily streaming time quota for a station."""
    try:
        # Read-only check: no need to take the write lock or rewrite the tracker.
        tracker = read_json_file(STREAM_TIME_TRACKER_FILE, default={}) or {}
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        station_usage = tracker.get(today_str, {}).get(user_ip, {}).get(station_id, {})

        lowres_used = station_usage.get('total_lowres_seconds', 0)
        hires_used = station_usage.get('total_hires_seconds', 0)
    except Exception as e:
        logging.error(f"Failed to check stream time quota for IP {user_ip}: {e}")
        return True, "" # Fail open (allow stream if quota check fails)

    station_type = 'quota' if stations_data.get(station_id, {}).get("station", {}).get("quota", False) else 'normal'
    limit_lowres = STREAM_TIME_LIMITS_SECONDS[station_type]['lowres']
    limit_hires = STREAM_TIME_LIMITS_SECONDS[station_type]['hires']

    if resolution == 'lowres' and lowres_used >= limit_lowres:
        return False, f"error_stream_quota_lowres|limit={limit_lowres // 60}"
    if resolution == 'hires' and hires_used >= limit_hires:
        return False, f"error_stream_quota_hires|limit={limit_hires // 60}"
    return True, ""


def _update_stream_time_tracker(user_ip, station_id, resolution, duration_seconds):
    """Logs the duration of a completed streaming session to the quota tracker file."""
    if duration_seconds <= 0: return
    with atomic_json_rw(STREAM_TIME_TRACKER_FILE) as tracker:
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        user_day = tracker.setdefault(today_str, {}).setdefault(user_ip, {})
        station_day = user_day.setdefault(station_id, {'total_lowres_seconds': 0, 'total_hires_seconds': 0})
        
        if resolution == 'lowres':
        
            station_day['total_lowres_seconds'] += duration_seconds
        elif resolution == 'hires':
            station_day['total_hires_seconds'] += duration_seconds
    logging.info(f"Logged {duration_seconds:.1f}s of {resolution} streaming for IP {user_ip} on station {station_id}.")


# Expected cmdline markers for processes recorded in stream state files. Used
# to verify a PID has not been reused by an unrelated process before killing it.
_PID_CMDLINE_MARKERS = {'ssh_pid': ('ssh',), 'ffmpeg_pid': ('ffmpeg',)}

def _kill_recorded_pid(pid_name, pid, log_prefix):
    """Kills a PID recorded in a state file only if it still looks like the expected process."""
    markers = _PID_CMDLINE_MARKERS.get(pid_name)
    if not markers or not pid_cmdline_matches(pid, markers):
        logging.warning(f"{log_prefix} Skipping kill of {pid_name} (PID: {pid}): process identity mismatch (stale or reused PID).")
        return
    try:
        os.kill(pid, signal.SIGKILL)
        logging.info(f"{log_prefix} Killed {pid_name} (PID: {pid}).")
    except OSError:
        logging.info(f"{log_prefix} {pid_name} (PID: {pid}) was already gone.")


def request_stream_transcode(task_id):
    """
    Sets the hot-swap command in an active stream's status file, asking the
    stream's monitor loop to restart FFmpeg with H.264 transcoding.
    """
    status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    if not os.path.exists(status_file):
        return {"success": False, "error": "stream_not_found"}
    try:
        with atomic_json_rw(status_file) as s_data:
            s_data['command'] = 'switch_to_h264'
        logging.info(f"Transcode to H.264 requested for stream task {task_id}.")
        return {"success": True}
    except Exception as e:
        logging.error(f"Failed to request transcode for task {task_id}: {e}")
        return {"success": False, "error": "error_internal"}


def stop_stream_relay(task_id):
    """
    Stops all processes associated with a stream task and cleans up all related files.
    """
    logging.info(f"Stopping stream task: {task_id}")
    status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    
    data = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f: data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error reading status file for stopping task {task_id}: {e}")
    
    # Kills the SSH tunnel and ffmpeg processes by their PIDs, after verifying
    # each PID still belongs to the expected process (guard against PID reuse).
    for pid_name, pid in data.get("pids", {}).items():
        _kill_recorded_pid(pid_name, pid, f"Task {task_id} -")
            
    # Deletes the temporary grid file (but never delete the cached grid).
    if grid_path := data.get("grid_local_path"):
        is_cached = bool(data.get("grid_cached"))
        if (not is_cached) and os.path.exists(grid_path):
            try:
                os.remove(grid_path)
                logging.info(f"Removed grid file: {grid_path}")
            except OSError as e:
                logging.error(f"Error removing grid file {grid_path}: {e}")

    # Deletes the stream directory and all control/lock files.
    if stream_dir := data.get("stream_dir"):
        stream_identity = os.path.basename(stream_dir)
        for f in [status_file, f"{status_file}.lock", os.path.join(STREAM_DIR, f"{stream_identity}.lock")]:
            if os.path.exists(f):
                try: os.remove(f)
                except OSError as e: logging.error(f"Error removing control file {f}: {e}")
        if os.path.exists(stream_dir):
         
            shutil.rmtree(stream_dir, ignore_errors=True)
            logging.info(f"Removed stream directory: {stream_dir}")


def _cleanup_stale_stream_locks(log_prefix):
    """
    Finds and cleans up lock files from previous streaming sessions that may have crashed.
    """
    now = time.time()
    for filename in os.listdir(STREAM_DIR):
        if filename.endswith(".lock"):
            file_path = os.path.join(STREAM_DIR, filename)
            try:
      
                # If a lock file is older than one hour, it's considered stale.
                if (now - os.path.getmtime(file_path)) > 3600: # 1 hour
                    logging.warning(f"{log_prefix} Removing stale stream lock: {filename}")
                    with open(file_path, 'r') as f: lock_data = json.load(f)
                    # Attempts to kill any lingering processes associated with the stale lock.
                    for pid_name, pid in lock_data.get("pids", {}).items():
                        _kill_recorded_pid(pid_name, pid, log_prefix)
                    os.remove(file_path)
            except (IOError, OSError, json.JSONDecodeError) as e:
                logging.error(f"{log_prefix} Error during stale lock cleanup for {filename}: {e}")


def _get_free_port():
    """Finds and returns an available ephemeral port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _start_ssh_tunnel(station_id, camera_num):
    """
    Establishes an SSH tunnel to a remote station.
    This forwards a local port to the camera's RTSP stream port on the station's internal network.
    """
    log_prefix = f"SSH-Tunnel {station_id}-{camera_num} -"
    local_port = _get_free_port()
    # The command forwards the local port to the camera's fixed IP and port.
    ssh_command = ["ssh", "-o", "RequestTTY=no", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-N", "-L", f"{local_port}:192.168.76.7{camera_num}:554", station_id]
    
    logging.info(f"{log_prefix} Attempting to establish tunnel on port {local_port} with command: {' '.join(ssh_command)}")
    process = subprocess.Popen(ssh_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Avoid a fixed sleep here; instead, quickly detect failure or readiness.
    start = time.time()
    while True:
        if process.poll() is not None:
            stderr_output = process.stderr.read().decode('utf-8', errors='ignore').strip()
            error_message = f"error_ssh_tunnel_failed_with_msg|error={stderr_output}" if stderr_output else "error_ssh_process_terminated"
            logging.error(f"{log_prefix} FAILED. {error_message}")
            raise RuntimeError(error_message)
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=0.3):
                break
        except OSError:
            if (time.time() - start) > 4:
                raise RuntimeError(f"error_ssh_tunnel_timeout|port={local_port}")
            time.sleep(0.1)

    logging.info(f"{log_prefix} Tunnel is ready (PID: {process.pid}) on port {local_port}.")
    return process, local_port


def _start_ffmpeg_relay(local_port, resolution, stream_dir, hevc_supported, log_prefix):
    """
    Starts an ffmpeg process. Uses robust probing to default to transcoding if codec detection fails.
    """
    stream_index = '1' if resolution == 'lowres' else '0' 
    rtsp_url = f"rtsp://127.0.0.1:{local_port}/user=admin&password=&channel=1&stream={stream_index}.sdp"
    playlist_path = os.path.join(stream_dir, 'playlist.m3u8')

    try:
        with socket.create_connection(("127.0.0.1", local_port), timeout=3):
            logging.info(f"{log_prefix} Port {local_port} is open. SSH tunnel is confirmed active.")
    except (socket.timeout, ConnectionRefusedError) as e:
        raise RuntimeError(f"error_local_tunnel_inactive|port={local_port}")

    # --- Codec Detection (time-bounded) ---
    codec_name = None
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-analyzeduration", "0", "-probesize", "32768",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            "-rtsp_transport", "tcp",
            rtsp_url
        ]
        probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=2, check=True)
        detected = probe_result.stdout.strip().lower()
        if detected:
            codec_name = 'hevc' if detected == 'h265' else detected
            logging.info(f"{log_prefix} Detected video codec: {codec_name}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        logging.warning(f"{log_prefix} Probe failed quickly; defaulting to HEVC for safety.")
            
    if not codec_name:
        logging.warning(f"{log_prefix} Probing failed. Defaulting to HEVC->Transcode for safety.")
        codec_name = 'hevc'

    should_copy = (codec_name == 'h264') or (codec_name == 'hevc' and hevc_supported)
    output_codec = codec_name if should_copy else 'h264'
    transcoding = not should_copy
  
    video_opts = ["-c:v"]
    if should_copy:
        logging.info(f"{log_prefix} Copying stream (codec: {codec_name})")
        video_opts.append("copy")
    else:
        logging.info(f"{log_prefix} Transcoding stream to H.264")
        video_opts.extend(["libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-force_key_frames", "expr:gte(t,n_forced*1)"])
        
    ffmpeg_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "0",
        "-probesize", "32768",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        *video_opts,
        "-an",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "2",
        "-hls_flags", "delete_segments",
        playlist_path
    ]
    
    logging.info(f"{log_prefix} Starting ffmpeg with command: {' '.join(ffmpeg_command)}")
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return process, playlist_path, codec_name, output_codec, transcoding


def _wait_for_playlist(process, playlist_path, log_prefix, timeout_seconds=10):
    """Waits for ffmpeg to create a non-empty playlist file."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            err_out = process.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg process died immediately. Error: {err_out}")
        if os.path.exists(playlist_path) and os.path.getsize(playlist_path) > 0:
            return
        time.sleep(0.1)
    raise RuntimeError("FFmpeg is running but failed to create a valid playlist file.")


def _onvif_request_keyframe_via_station(station_id, camera_num, log_prefix, status_file=None, resolution=None):
    """Best-effort ONVIF request for an IDR/keyframe.

    This runs *from the station* (which has access to the camera LAN) by executing
    curl over SSH. Many embedded ONVIF stacks expose gSOAP on port 8899.

    If anything fails, it logs and returns without raising.
    """
    cam_ip = f"192.168.76.7{camera_num}"
    media_service = f"http://{cam_ip}:8899/onvif/Media"

    # NOTE: We intentionally do not write ONVIF diagnostics into the stream status file.
    # The frontend only needs readiness states; detailed ONVIF payloads were temporary debug.

    # Important: ssh executes the *remote* command via a shell string. If we pass an argv list
    # with arguments containing spaces (like the Content-Type header), they will be split.
    # Build a single shell-escaped command string instead.
    def _ssh_curl(url, soap_xml, timeout_s=2):
        # Build a single remote shell command string (ssh runs through a remote shell).
        remote_cmd = [
            "curl",
            "-sS",
            "--max-time", str(timeout_s),
            "-w", "\\nHTTP_CODE:%{http_code}\\n",
            "-X", "POST",
            url,
            "-H", "Content-Type: application/soap+xml; charset=utf-8",
            "--data-binary", "@-",
        ]
        cmd = ["ssh", station_id, " ".join(shlex.quote(x) for x in remote_cmd)]
        return subprocess.run(cmd, input=soap_xml.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s + 1)

    def _set_sync(token):
        sync_xml = (
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\">"
            "<s:Body>"
            "<trt:SetSynchronizationPoint xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\">"
            f"<trt:ProfileToken>{token}</trt:ProfileToken>"
            "</trt:SetSynchronizationPoint>"
            "</s:Body>"
            "</s:Envelope>"
        )
        t1 = time.time()
        res2 = _ssh_curl(media_service, sync_xml)
        dt2 = time.time() - t1
        out2 = (res2.stdout or "")
        err2 = (res2.stderr or "")
        body2 = out2 + "\n" + err2
        http_m2 = re.search(r"HTTP_CODE:(\d{3})", body2)
        http_code2 = int(http_m2.group(1)) if http_m2 else None
        soap_fault2 = ("<SOAP-ENV:Fault" in body2) or ("<SOAP-ENV:Fault" in out2) or ("<SOAP-ENV:Fault" in err2)
        return {
            "ok": bool(res2.returncode == 0) and not soap_fault2,
            "returncode": res2.returncode,
            "http_code": http_code2,
            "soap_fault": bool(soap_fault2),
            "seconds": round(dt2, 3),
            "profile_token": token,
        }

    # Request keyframes for the known encoder tokens.
    # On these modules, tokens are typically stable (000=hires, 001=lowres).
    try:
        results = []
        for tok in ("000", "001"):
            results.append(_set_sync(tok))
        ok_count = sum(1 for r in results if r.get('ok'))
        logging.info(f"{log_prefix} ONVIF: requested keyframe for {len(results)} tokens on {cam_ip} (ok={ok_count}).")
    except Exception as e:
        logging.info(f"{log_prefix} ONVIF: SetSynchronizationPoint failed: {e}")
        return None

    return None


def start_stream_relay(task_id, station_id, camera_num, resolution, user_ip, hevc_supported=False):
    """
    Main function to orchestrate the entire live stream setup process.
    Supports dynamic switching to H.264 transcoding without dropping the SSH tunnel.
    """
    stream_start_time = time.time()
    status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    log_prefix = f"StreamRelay {task_id} -"
    logging.info(f"{log_prefix} Request for {station_id}/{camera_num}/{resolution}. HEVC support: {hevc_supported}")

    timings = {
        "t0": stream_start_time,
        "ssh_tunnel_seconds": None,
        "ffmpeg_to_playlist_seconds": None,
        "setup_seconds": None,
    }
    
    _cleanup_stale_stream_locks(log_prefix)

    with open(STATIONS_FILE, 'r') as f: stations_data = json.load(f)
    # Checks if the user has exceeded their daily streaming quota.
    allowed, message = _check_stream_time_quota(user_ip, station_id, resolution, stations_data)
    if not allowed:
        logging.warning(f"Stream rejected for IP {user_ip} ({station_id}/{resolution}): {message}")
        update_status(status_file, "error", {"message": message})
        return

    stream_identity = f"{station_id}_{camera_num}_{resolution}"
    stream_dir = os.path.join(STREAM_DIR, stream_identity)
    if os.path.exists(stream_dir): shutil.rmtree(stream_dir)
    os.makedirs(stream_dir, exist_ok=True)
    
    ssh_process, ffmpeg_process = None, None
    try:
        update_status(status_file, "establishing_tunnel", {"message": "status_contacting_station"})
   
        t_ssh0 = time.time()
        ssh_process, local_port = _start_ssh_tunnel(station_id, camera_num)
        timings["ssh_tunnel_seconds"] = round(time.time() - t_ssh0, 3)

        update_status(status_file, "connecting_camera", {"message": "status_connecting_camera"})

        # Start ffmpeg first, then request keyframe (avoids race: we are ready to receive).
        t_ff0 = time.time()
        ffmpeg_process, playlist_path, input_codec, output_codec, transcoding = _start_ffmpeg_relay(local_port, resolution, stream_dir, hevc_supported, log_prefix)

        # Wait for playlist in parallel so we can see whether ONVIF actually changes time-to-playlist.
        # HD streams require transcoding and take longer to produce the first playlist segment.
        playlist_timeout = 30 if transcoding else 20
        playlist_ready = {"ts": None, "error": None}
        def _playlist_waiter():
            try:
                _wait_for_playlist(ffmpeg_process, playlist_path, log_prefix, timeout_seconds=playlist_timeout)
                playlist_ready["ts"] = time.time()
            except Exception as e:
                playlist_ready["error"] = e

        waiter_thread = threading.Thread(target=_playlist_waiter, daemon=True)
        waiter_thread.start()

        # ONVIF keyframe request is done asynchronously so it cannot delay readiness.
        def _onvif_worker():
            try:
                # Cross-process rate-limit to avoid spamming the camera on rapid restarts.
                # (Each stream start is a separate Python process.)
                rate_file = os.path.join(LOCK_DIR, f"onvif_rate_{station_id}_{camera_num}.json")
                now = time.time()
                try:
                    with atomic_json_rw(rate_file) as r:
                        last = float(r.get('last_ts') or 0.0)
                        if (now - last) < 2.0:
                            return
                        r['last_ts'] = now
                except Exception:
                    pass

                _onvif_request_keyframe_via_station(
                    station_id,
                    camera_num,
                    log_prefix,
                    status_file=status_file,
                    resolution=resolution,
                )
            except Exception:
                pass

        threading.Thread(target=_onvif_worker, daemon=True).start()

        # Ensure playlist is ready (or error) before marking ready.
        waiter_thread.join(timeout=12)
        if playlist_ready["error"]:
            raise playlist_ready["error"]
        if not playlist_ready["ts"]:
            raise RuntimeError("Playlist waiter did not finish in time.")

        timings["ffmpeg_to_playlist_seconds"] = round(playlist_ready["ts"] - t_ff0, 3)
        
        timeout_seconds = _get_timeout_for_station(station_id, resolution, stations_data)
        update_data = {
            "message": "status_stream_ready", "pids": {"ssh_pid": ssh_process.pid, "ffmpeg_pid": ffmpeg_process.pid},
            "stream_dir": stream_dir, "station_id": station_id, "timeout_seconds": timeout_seconds, "resolution": resolution
        }
        timings["setup_seconds"] = round(time.time() - stream_start_time, 3)
        update_data["input_codec"] = input_codec
        update_data["output_codec"] = output_codec
        update_data["transcoding"] = bool(transcoding)
        update_status(status_file, "ready", update_data)

        logging.info(
            f"{log_prefix} Timing: ssh={timings['ssh_tunnel_seconds']}s, ffmpeg->playlist={timings['ffmpeg_to_playlist_seconds']}s, "
            f"setup={timings['setup_seconds']}s"
        )
        
        # This loop keeps the script alive, monitoring the stream processes until the timeout is reached
        # or a process dies or is stopped externally.
        end_time = time.time() + timeout_seconds
        while time.time() < end_time:
            # 1. Check for Hot-Swap Command. Read without the write lock on
            # every poll; only rewrite the status file when a command is found.
            should_switch = False
            try:
                if (read_json_file(status_file, default={}) or {}).get('command') == 'switch_to_h264':
                    with atomic_json_rw(status_file) as s_data:
                        if s_data.get('command') == 'switch_to_h264':
                            logging.info(f"{log_prefix} Hot-swap requested. Restarting FFmpeg...")
                            del s_data['command']
                            should_switch = True
            except Exception: pass

            if should_switch:
                # Terminate old FFmpeg politely, then forcefully
                if ffmpeg_process and ffmpeg_process.poll() is None:
                    try:
                        os.kill(ffmpeg_process.pid, signal.SIGTERM)
                        for _ in range(20): # Wait 2s
                            if ffmpeg_process.poll() is not None: break
                            time.sleep(0.1)
                        if ffmpeg_process.poll() is None:
                            os.kill(ffmpeg_process.pid, signal.SIGKILL)
                            ffmpeg_process.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired): pass
                
                time.sleep(1) # Cooldown
                
                # Cleanup playlist
                plist = os.path.join(stream_dir, 'playlist.m3u8')
                if os.path.exists(plist): 
                    try: os.remove(plist)
                    except OSError: pass

                # Restart
                try:
                    ffmpeg_process, _, input_codec, output_codec, transcoding = _start_ffmpeg_relay(local_port, resolution, stream_dir, False, log_prefix)
                    with atomic_json_rw(status_file) as s_data:
                        s_data['pids']['ffmpeg_pid'] = ffmpeg_process.pid
                        s_data['input_codec'] = input_codec
                        s_data['output_codec'] = output_codec
                        s_data['transcoding'] = bool(transcoding)
                except Exception as e:
                    logging.error(f"{log_prefix} Failed to restart FFmpeg: {e}")
                    break

            if not os.path.exists(status_file):
                logging.warning(f"Status file gone for {task_id}. Stopping.")
                break
            if ssh_process.poll() is not None:
                logging.warning(f"SSH process terminated for {task_id}.")
                break
            if ffmpeg_process.poll() is not None and not should_switch:
                logging.warning(f"FFmpeg process terminated for {task_id}.")
                break
            time.sleep(0.5)

    except Exception as e:
 
        logging.error(f"Error in stream task {task_id}: {e}")
        update_status(status_file, "camera_failed", {"message": str(e)})
    finally:
        # Ensures all spawned processes are killed on exit.
        if ssh_process and ssh_process.poll() is None:
            try: os.kill(ssh_process.pid, signal.SIGKILL)
            except OSError: pass
        if ffmpeg_process and ffmpeg_process.poll() is None:
            try: os.kill(ffmpeg_process.pid, signal.SIGKILL)
            except OSError: pass

        # Logs the stream duration for quota tracking and performs final cleanup.
        duration = time.time() - stream_start_time
        _update_stream_time_tracker(user_ip, station_id, resolution, duration)
        logging.info(f"Cleaning up resources for stream task {task_id}")
        stop_stream_relay(task_id)
