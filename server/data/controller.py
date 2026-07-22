#!/usr/bin/env python3

import sys
import json
import os
import io
import tarfile
import subprocess
import logging
import time
import shutil
import re
import signal
import threading
from datetime import datetime, timedelta, timezone
from PIL import Image
import numpy as np
import numba

# --- Configuration (Defined first to avoid circular dependency issues) ---
# NMN_DATA_DIR is set by index.php (the entry point) so all subprocesses
# inherit the correct runtime data directory regardless of __file__ resolution.
BASE_DIR = os.environ.get('NMN_DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
STATIONS_FILE = os.path.join(BASE_DIR, 'stations.json')
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'download')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOCK_DIR = os.path.join(BASE_DIR, 'locks')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
STREAM_DIR = os.path.join(BASE_DIR, 'streams')
LOG_FILE = os.path.join(LOG_DIR, 'activity.log')
CAMERAS_FILE = os.path.join(BASE_DIR, 'cameras.json')
PASS_CACHE_FILE = os.path.join(CACHE_DIR, 'pass_cache.json')
QUOTA_TRACKER_FILE = os.path.join(BASE_DIR, 'quota_tracker.json')

CLEANUP_AGE_DAYS = 7 
MAX_LOG_LINES = 10000 
MAX_STATIONS_PER_REQUEST = 10 
MAX_SEQUENCE_LENGTH = 60 
MAX_SEQUENCE_INTERVAL = 60 
MAX_FILE_SIZE_FOR_THUMBNAIL_MB = 200 
FILE_TYPE_LIMITS = {'lowres': 300, 'hires': 100, 'image': 300, 'image_lowres': 600, 'image_long': 100, 'image_lowres_long': 300, 'timelapse': 50, 'timelapse_hires': 20}
AVG_FILE_SIZES_MB = {'lowres': 2, 'hires': 15, 'image': 1, 'image_lowres': 0.2, 'image_long': 1, 'image_lowres_long': 0.2, 'timelapse': 100, 'timelapse_hires': 400}
STITCH_SCRIPT = os.path.join(BASE_DIR, 'stitch.py')
STACK_SCRIPT = os.path.join(BASE_DIR, 'stack.py')
TOTAL_QUOTA_LIMIT_MB = 2048 
PER_SITE_QUOTA_LIMIT_MB = 1024 

# Ensure critical directories exist
for d in [LOG_DIR, LOCK_DIR, DOWNLOAD_DIR, CACHE_DIR, STREAM_DIR]: os.makedirs(d, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(LOG_FILE)])

# --- Inline Helper: Robust Video Probing ---
# Defined here to prevent ImportError crashes if media_processor.py is out of sync.
def internal_probe_duration(filepath):
    """Return video duration in seconds, or None if it cannot be determined."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None

def internal_probe_start_time(filepath):
    """Return container start_time in seconds, or None if it cannot be determined."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None

def internal_probe_codec(filepath):
    try:
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip().lower()
    except Exception:
        pass
    return 'unknown'

# --- Enhance Filter Functions ---
@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_enhance_filter_core(plane: np.ndarray, t: int, log2sizex: int, log2sizey: int) -> np.ndarray:
    height, width = plane.shape
    tmp_h = np.zeros(plane.shape, dtype=np.int16)
    final_f = np.zeros(plane.shape, dtype=np.int16)
    shiftx, shifty = 6 - log2sizex, 6 - log2sizey
    indices = np.array([-31, -23, -14, -5, 5, 14, 23, 31], dtype=np.int32)
    indices_x, indices_y = indices // (1 << shiftx), indices // (1 << shifty)

    for i in numba.prange(height):
        for j in range(width):
            center_val, h_sum = plane[i, j], 0
            for l in indices_x:
                sample_j = max(0, min(width - 1, j + l))
                sample_val = plane[i, sample_j]
                h_sum += center_val if abs(int(sample_val) - int(center_val)) > t else sample_val
            tmp_h[i, j] = h_sum

    for i in numba.prange(height):
        for j in range(width):
            center_val_h, v_sum = tmp_h[i, j], 0
            for l in indices_y:
                sample_i = max(0, min(height - 1, i + l))
                sample_val_h = tmp_h[sample_i, j]
                v_sum += center_val_h if abs(int(sample_val_h) - int(center_val_h)) > t * 4 else sample_val_h
            final_f[i, j] = v_sum
    return final_f

@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_enhance_filter_core_rgb(image: np.ndarray, t: int, log2sizex: int, log2sizey: int) -> np.ndarray:
    """Single-pass RGB filter for better performance."""
    height, width, channels = image.shape
    tmp_h = np.zeros(image.shape, dtype=np.int16)
    final_f = np.zeros(image.shape, dtype=np.int16)
    shiftx, shifty = 6 - log2sizex, 6 - log2sizey
    indices = np.array([-31, -23, -14, -5, 5, 14, 23, 31], dtype=np.int32)
    indices_x, indices_y = indices // (1 << shiftx), indices // (1 << shifty)

    # Horizontal pass
    for i in numba.prange(height):
        for j in range(width):
            for c in range(channels):
                center_val, h_sum = image[i, j, c], 0
                for l in indices_x:
                    sample_j = max(0, min(width - 1, j + l))
                    sample_val = image[i, sample_j, c]
                    h_sum += center_val if abs(int(sample_val) - int(center_val)) > t else sample_val
                tmp_h[i, j, c] = h_sum

    # Vertical pass
    for i in numba.prange(height):
        for j in range(width):
            for c in range(channels):
                center_val_h, v_sum = tmp_h[i, j, c], 0
                for l in indices_y:
                    sample_i = max(0, min(height - 1, i + l))
                    sample_val_h = tmp_h[sample_i, j, c]
                    v_sum += center_val_h if abs(int(sample_val_h) - int(center_val_h)) > t * 4 else sample_val_h
                final_f[i, j, c] = v_sum
    return final_f

def enhance_filter(plane: np.ndarray, t: int, log2sizex: int, log2sizey: int,
                   dither: int, seed: int,
                   num_workers: int = None) -> np.ndarray:
    if num_workers is None: num_workers = os.cpu_count() or 1
    numba.set_num_threads(num_workers)
    log2sizex, log2sizey = np.clip(log2sizex, 3, 6), np.clip(log2sizey, 3, 6)

    # Use RGB version for 3D arrays, single-channel for 2D
    if len(plane.shape) == 3:
        final_f = _jit_enhance_filter_core_rgb(plane, t, log2sizex, log2sizey)
    else:
        final_f = _jit_enhance_filter_core(plane, t, log2sizex, log2sizey)

    c_dither = np.clip(dither, 2, 11) - 2
    dmask = (1 << c_dither) - 1
    doffset = (1 << (c_dither - 1)) - 8 if c_dither > 0 else -8
    if dmask > 0 and seed != 0:
        rng = np.random.default_rng(seed)
        noise = rng.integers(0, dmask + 1, size=plane.shape, dtype=np.int16)
        final_f += noise + doffset
    else:
        final_f += doffset
    final_f //= (1 << 6)
    return np.clip(final_f, 0, 255).astype(np.uint8)

def apply_enhance_filter(image_path: str, threshold: int) -> str:
    """Apply enhance filter to an image and return base64 encoded result."""
    # Convert URL path to file system path
    # The image_path is expected to be like "download/some_file.jpg"
    # We need to convert it to an absolute path relative to BASE_DIR
    if image_path.startswith('download/'):
        image_path = os.path.join(BASE_DIR, image_path)
    elif not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if threshold == 0:
        # No filter, return empty string to indicate original should be used
        return ""

    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        # Convert to grayscale if needed, or process each channel
        if len(img_array.shape) == 2:
            # Grayscale
            enhanced = enhance_filter(img_array, t=threshold, log2sizex=6, log2sizey=6, dither=1, seed=np.random.randint(0, 2**32))
            result_img = Image.fromarray(enhanced)
        elif len(img_array.shape) == 3:
            # RGB - single-pass filter
            enhanced = enhance_filter(img_array, t=threshold, log2sizex=6, log2sizey=6, dither=1, seed=np.random.randint(0, 2**32))
            result_img = Image.fromarray(enhanced)
        else:
            raise ValueError("Unsupported image format")

        # Convert to base64
        import base64
        from io import BytesIO
        buffer = BytesIO()
        result_img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error applying enhance filter: {e}")
        raise

# --- Imports with Error Catching ---
try:
    from shared_utils import (
        uniqid, update_status, update_quota_tracker, trim_log_file, cleanup_old_files
    )
    # Import media_processor but safeguard against missing probe function
    import media_processor
    from media_processor import (
        stack_images, draw_track_on_image, apply_ffmpeg_overlay, create_thumbnail,
        PTO_MAPPER_AVAILABLE
    )
    from live_streamer import (
        start_stream_relay, stop_stream_relay, fetch_grid_file, fetch_annotation_file,
        get_archive_grid_overlay, get_archive_annotation_overlay,
        get_stitch_cam_boundaries
    )
    from data_fetchers import (
        get_kp_data, get_lightning_data, get_meteor_data, get_camera_fovs, get_station_stats
    )
except ImportError as e:
    # Print JSON error so the frontend displays it instead of "Unexpected end of JSON"
    print(json.dumps({"error": f"ImportError in controller: {e}"}))
    sys.exit(1)
except SyntaxError as e:
    print(json.dumps({"error": f"SyntaxError in backend scripts: {e}"}))
    sys.exit(1)


HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><title>__{{html_title}}__</title>
<link rel="stylesheet" href="//unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="//unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="//cdn.jsdelivr.net/npm/chart.js"></script>
<script src="//cdn.jsdelivr.net/npm/hls.js@latest"></script>
<link rel="stylesheet" href="style.css?v=20260614e">
</head><body>
<div class="container">
    <header>
        <div id="language-selector">
            <span data-lang="nb_NO" title="Norsk">🇳🇴</span>
            <span data-lang="en_GB" title="English">🇬🇧</span>
            <span data-lang="de_DE" title="Deutsch">🇩🇪</span>
            <span data-lang="cs_CZ" title="Čeština">🇨🇿</span>
            <span data-lang="fi_FI" title="Suomi">🇫🇮</span>
            <span data-lang="lv_LV" title="Latviešu">🇱🇻</span>
        </div>
        <h1>__{{page_title}}__</h1>
    </header>
    <main class="main-content">
        <div id="map-panel">
            <h2>__{{map_panel_title}}__</h2>
            <div id="map"></div>
            <div class="map-controls">
                <button id="map-snapshot-btn" class="btn btn-secondary">__{{snapshot_button}}__</button>
            </div>
            <p class="map-description">
                 __{{map_description_archive}}__ __{{map_description_lightning_source}}__
            </p>
        </div>
        <div id="control-panel">
            <h2>__{{control_panel_title}}__</h2>
            <form id="download-form">
                <div class="form-group">
                    <h6>__{{selected_stations_label}}__</h6>
                     <div id="station-list-container">
                        <p style="color: #6c757d; margin: 0;">__{{no_station_selected}}__</p>
                        <ul id="station-list" style="display: none;"></ul>
                    </div>
                </div>
                <div class="form-group">
                    <div class="toggle-wrapper"><label class="checkbox-label-inline"><input type="checkbox" id="cloud-toggle"> __{{toggle_clouds}}__</label></div>
                    <div class="toggle-wrapper"><label class="checkbox-label-inline"><input type="checkbox" id="aurora-toggle"> __{{toggle_aurora}}__</label></div>
                    <div class="toggle-wrapper"><label class="checkbox-label-inline"><input type="checkbox" id="terminator-toggle"> __{{toggle_terminator}}__</label></div>
                    <div class="toggle-wrapper"><label class="checkbox-label-inline"><input type="checkbox" id="satellite-toggle"> __{{toggle_satellites}}__</label></div>
                    <div class="toggle-wrapper"><label class="checkbox-label-inline"><input type="checkbox" id="aircraft-toggle"> __{{toggle_aircraft}}__</label></div>
                    <div class="toggle-wrapper">
                        <label class="checkbox-label-inline"><input type="checkbox" id="lightning-toggle"> __{{toggle_lightning}}__</label>
                        <label class="checkbox-label-inline" id="lightning-filter-label" style="display: none; margin-left: 15px;">
                            <input type="checkbox" id="lightning-24h-toggle"> __{{toggle_lightning_24h}}__
                        </label>
                    </div>
                    <div class="toggle-wrapper"><label class="checkbox-label-inline"><input type="checkbox" id="meteor-toggle"> __{{toggle_meteors}}__</label></div>
                </div>
                <div class="form-group date-group-flex">
                   <label for="date-display">__{{date_label}}__</label>
                   <div class="date-input-wrapper">
                        <button type="button" id="date-prev-btn" class="date-nav-btn" aria-label="Previous day">‹</button>
                        <input type="text" id="date-display" placeholder="YYYY-MM-DD" readonly>
                        <input type="date" id="date" name="date" required>
                        <button type="button" id="date-next-btn" class="date-nav-btn" aria-label="Next day">›</button>
                   </div>
                   <button type="button" id="last-night-btn" title="__{{last_night_button_title}}__">__{{last_night_button}}__</button>
                   <button type="button" id="now-button" title="__{{now_button_title}}__">__{{now_button}}__</button>
                </div>
                <div class="form-group time-group"><div><label for="hour">__{{hour_label}}__</label><div class="select-stepper"><button type="button" id="hour-prev-btn" class="date-nav-btn" aria-label="Previous hour">‹</button><select id="hour" name="hour" required><option value="" disabled selected>--</option></select><button type="button" id="hour-next-btn" class="date-nav-btn" aria-label="Next hour">›</button></div></div>
                <div><label for="minute">__{{minute_label}}__</label><div class="select-stepper"><button type="button" id="minute-prev-btn" class="date-nav-btn" aria-label="Previous minute">‹</button><select id="minute" name="minute" required><option value="" disabled selected>--</option></select><button type="button" id="minute-next-btn" class="date-nav-btn" aria-label="Next minute">›</button></div></div></div>
                <div class="form-group time-group" id="length-interval-group">
                    <div><label for="length">__{{length_label}}__</label><div class="select-stepper"><button type="button" id="length-prev-btn" class="date-nav-btn" aria-label="Previous length">‹</button><select id="length" name="length" required><option value="" disabled selected>--</option></select><button type="button" id="length-next-btn" class="date-nav-btn" aria-label="Next length">›</button></div></div>
                    <div><label for="interval" id="interval-label">__{{interval_label}}__</label><div class="select-stepper"><button type="button" id="interval-prev-btn" class="date-nav-btn" aria-label="Previous interval">‹</button><select id="interval" name="interval" required><option value="" disabled selected>--</option></select><button type="button" id="interval-next-btn" class="date-nav-btn" aria-label="Next interval">›</button></div></div>
                </div>
                <div class="form-group time-group" id="duration-group" style="display: none;">
                    <div><label for="duration">__{{duration_label}}__</label><select id="duration" name="duration" required><option value="1" selected>1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option><option value="6">6</option><option value="7">7</option><option value="8">8</option><option value="9">9</option><option value="10">10</option><option value="11">11</option><option value="12">12</option><option value="13">13</option><option value="14">14</option><option value="15">15</option></select></div>
                </div>
                <fieldset class="form-group">
                    <legend>__{{camera_legend}}__</legend>
                    <div class="checkbox-group" id="camera-checkbox-group">
                        <label><input type="checkbox" name="cameras" value="1" checked> 1</label>
                        <label><input type="checkbox" name="cameras" value="2" checked> 2</label>
                        <label><input type="checkbox" name="cameras" value="3" checked> 3</label>
                        <label><input type="checkbox" name="cameras" value="4" checked> 4</label>
                        <label><input type="checkbox" name="cameras" value="5" checked> 5</label>
                        <label><input type="checkbox" name="cameras" value="6" checked> 6</label>
                        <label><input type="checkbox" name="cameras" value="7" checked> 7</label>
                    </div>
                    <div class="stitch-options" id="stitch-options">
                        <label class="checkbox-label-inline" id="fisheye-label"><input type="checkbox" id="fisheye-switch"> __{{fisheye_checkbox}}__</label>
                        <label class="checkbox-label-inline" id="equirect-label"><input type="checkbox" id="equirect-switch"> __{{equirect_checkbox}}__</label>
                    </div>
                </fieldset>
                <fieldset class="form-group combined-file-type">
                    <div class="primary-type-group">
                         <label><input type="radio" name="primary_file_type" value="video"> __{{video_radio}}__</label>
                         <label><input type="radio" name="primary_file_type" value="image" checked> __{{image_radio}}__</label>
                         <label><input type="radio" name="primary_file_type" value="timelapse"> __{{timelapse_radio}}__</label>
                    </div>
                    <div class="options-group" id="file-options-group">
                         <label class="checkbox-label-inline"><input type="checkbox" id="high-resolution-switch"> __{{high_res_checkbox}}__</label>
                         <label class="checkbox-label-inline" id="long-integration-label"><input type="checkbox" id="long-integration-switch">__{{long_int_checkbox}}__</label>
                    </div>
                </fieldset>
                <div class="button-group">
                    <button type="submit" id="download-button">__{{download_button_start}}__</button>
                    <button type="button" id="cancel-button" style="display: none;">__{{cancel_button}}__</button>
                </div>
                <div id="form-error" class="error-msg" style="margin-top: 10px;"></div>
            </form>
        </div>
    </main>
</div>
<div class="full-width-container" id="satellite-panel-container"><div class="container" id="satellite-panel"><h2>__{{satellite_panel_title}}__</h2><div id="satellite-list"><p style="color: #6c757d; margin: 0;">__{{loading_passes}}__</p></div></div></div>
<div class="full-width-container" id="aircraft-panel-container"><div class="container" id="aircraft-panel"><h2>__{{aircraft_panel_title}}__</h2><div id="aircraft-list"><p style="color: #6c757d; margin: 0;">__{{loading_aircraft}}__</p></div></div></div>
<div class="full-width-container" id="lightning-panel-container"><div class="container" id="lightning-panel"><h2>__{{lightning_panel_title}}__</h2><div style="margin-bottom: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: nowrap;">__{{sort_by_label}}__ <label class="checkbox-label-inline" style="display: inline-flex; gap: 2px; align-items: center;"><input type="radio" name="lightning-sort" value="time" checked style="margin: 0;"> __{{sort_by_time}}__</label><label class="checkbox-label-inline" style="display: inline-flex; gap: 2px; align-items: center;"><input type="radio" name="lightning-sort" value="station" style="margin: 0;"> __{{sort_by_station}}__</label><label class="checkbox-label-inline" style="display: inline-flex; gap: 2px; align-items: center;"><input type="radio" name="lightning-sort" value="distance" style="margin: 0;"> __{{sort_by_distance}}__</label></div><div id="lightning-station-subsort" style="margin-bottom: 10px; display: none; padding-left: 20px;">__{{then_sort_by_label}}__ <label class="checkbox-label-inline" style="display: inline-flex; gap: 2px; align-items: center; margin-right: 12px;"><input type="radio" name="lightning-station-subsort" value="time" checked style="margin: 0;"> __{{sort_by_time}}__</label><label class="checkbox-label-inline" style="display: inline-flex; gap: 2px; align-items: center;"><input type="radio" name="lightning-station-subsort" value="distance" style="margin: 0;"> __{{sort_by_distance}}__</label></div><div id="lightning-list"><p style="color: #6c757d; margin: 0;">__{{loading_lightning}}__</p></div></div></div>
<div class="full-width-container" id="meteor-panel-container"><div class="container" id="meteor-panel"><h2>__{{toggle_meteors}}__</h2><div id="meteor-list"><p style="color: #6c757d; margin: 0;"></p></div></div></div>
<div class="full-width-container" id="station-stats-panel-container" style="display: none;"><div class="container" id="station-stats-panel"><h2>__{{stats_panel_title_default}}__</h2><div id="station-stats-list"></div></div></div>
<div class="full-width-container" id="aurora-plot-container"><div class="chart-container"><canvas id="aurora-chart"></canvas></div></div>
<div class="container">
    <footer id="results-panel">
        <h2>__{{results_panel_title}}__</h2>
        <div id="progress-container" style="display: none;"><p>__{{status_label}}__<span id="progress-text">__{{status_starting}}__</span></p>
        <div class="progress-bar-outline"><div id="progress-bar-inner" class="progress-bar-inner"></div></div></div>
        <div id="results-log"></div>
    </footer>
</div>
<script src="main.js?v=20260614e" type="module"></script>
</body></html>
"""

def _interpolate_track(track_points, max_interval_sec):
    if not track_points or len(track_points) < 2: return track_points
    new_track = [track_points[0]]
 
    keys_to_interpolate = [k for k, v in track_points[0].items() if isinstance(v, (int, float))]
    for i in range(len(track_points) - 1):
        p1, p2 = track_points[i], track_points[i+1]
        t1 = datetime.fromisoformat(p1['time'].replace('Z', '+00:00'))
        t2 = datetime.fromisoformat(p2['time'].replace('Z', '+00:00'))
        time_diff_sec = (t2 - t1).total_seconds()
        
        if time_diff_sec > max_interval_sec:
            num_new_points = int(time_diff_sec // max_interval_sec)
            for j in range(1, num_new_points + 1):
                interp_factor = j / (num_new_points + 1)
                new_point_time = t1 + timedelta(seconds=time_diff_sec * interp_factor)
                
                new_point = {'time': new_point_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}
                for key in keys_to_interpolate:
                    val1, val2 = p1.get(key, 0), p2.get(key, 0)
                    new_point[key] = val1 + (val2 - val1) * interp_factor
                new_track.append(new_point)
        
        new_track.append(p2)
    return new_track


class FileProcessor:
    def __init__(self, task_id, station_id, station_code, cam, time_utc, file_type, all_pass_data, pto_data_cache, hevc_supported=False, translations=None, ssh_control_socket=None, status_file=None, current_step=0, total_steps=1, results_ref=None, errors_ref=None, duration=1):
        self.task_id, self.station_id, self.station_code, self.cam, self.time_utc, self.file_type = task_id, station_id, station_code, cam, time_utc, file_type
        self.duration = max(1, int(duration))
        self.all_pass_data, self.pto_data_cache, self.hevc_supported = all_pass_data, pto_data_cache, hevc_supported
        self.translations = translations or {}
        self.ssh_control_socket = ssh_control_socket
        self.status_file = status_file
        self.current_step = current_step
        self.current_sub_step = 0
        self.total_steps = total_steps
        self.results_ref = results_ref
        self.errors_ref = errors_ref
        self.errors, self.total_bytes_downloaded, self.is_blending_job = [], 0, False
        self.relevant_pass = self._find_relevant_pass()
        self._determine_paths_and_types()

    def _find_relevant_pass(self):
        if not self.all_pass_data or not PTO_MAPPER_AVAILABLE: return None
        for p in self.all_pass_data:
            for cv in p.get('camera_views', []):
                if cv['station_id'] == self.station_id and cv['camera'] == self.cam and datetime.fromisoformat(cv['start_utc']).replace(second=0, microsecond=0) <= self.time_utc < datetime.fromisoformat(cv['end_utc']):
                    return p
        return None

    def _determine_paths_and_types(self):
        is_flight = self.relevant_pass and 'flight_info' in self.relevant_pass
        overlay_suffix = "_flight_overlay" if is_flight else "_overlay"
  
        self.is_image, self.is_long_integration, self.is_low_res = self.file_type.startswith('image'), self.file_type.endswith('_long'), 'lowres' in self.file_type
        t = self.time_utc
        base_name = f"{self.station_code}_cam{self.cam}_{t.strftime('%Y%m%d')}_{t.strftime('%H%M')}"
        duration_suffix = f"_dur{self.duration}" if self.duration > 1 and (not self.is_image or self.is_long_integration) else ""
        base_name_with_type = f"{base_name}{duration_suffix}_{self.file_type}"
        if self.is_image:
            self.output_filepath = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}.jpg")
            self.overlay_filepath = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}{overlay_suffix}.jpg")
        else:
            self.output_filepath_h264 = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}.mp4")
            self.overlay_filepath_h264 = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}{overlay_suffix}.mp4")
            self.output_filepath_hevc = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}_hevc.mp4")
            self.overlay_filepath_hevc = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}{overlay_suffix}_hevc.mp4")
        self.track_filepath = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}_{self.task_id}_track.png")

    def _scp_file(self, remote_path, local_path):
        temp_path = local_path + ".part"
        command = ["scp", "-B", "-o", "ConnectTimeout=300"]
        if self.ssh_control_socket and os.path.exists(self.ssh_control_socket):
            command += ["-o", f"ControlPath={self.ssh_control_socket}"]
        command += [f"{self.station_id}:{remote_path}", temp_path]
        # Best-effort: get remote file size for progress reporting
        remote_size = 0
        if self.status_file:
            try:
                sz_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
                if self.ssh_control_socket and os.path.exists(self.ssh_control_socket):
                    sz_cmd += ["-o", f"ControlPath={self.ssh_control_socket}"]
                sz_cmd += [self.station_id, f"stat -c%s {remote_path}"]
                sz_out = subprocess.run(sz_cmd, capture_output=True, text=True, timeout=15)
                remote_size = int(sz_out.stdout.strip())
            except Exception:
                pass
        scp_exc = [None]
        def _run():
            try:
                subprocess.run(command, check=True, timeout=360, capture_output=True)
            except Exception as e:
                scp_exc[0] = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        msg = f"status_processing_file_of_total|i={self.current_step+1},total={self.total_steps}"
        while t.is_alive():
            time.sleep(1)
            if self.status_file and remote_size > 0:
                try:
                    part_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                    file_pct = min(part_size / remote_size, 0.99)
                    update_status(self.status_file, "progress", {
                        "step": self.current_step * self.duration + self.current_sub_step + file_pct,
                        "total": self.total_steps,
                        "message": msg,
                        "files": self.results_ref if self.results_ref is not None else {},
                        "errors": self.errors_ref if self.errors_ref is not None else [],
                    })
                except Exception:
                    pass
        t.join()
        if scp_exc[0] is not None:
            e = scp_exc[0]
            if os.path.exists(temp_path): os.remove(temp_path)
            stderr = e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') and e.stderr else ''
            raise FileNotFoundError(f"Remote file not found: {self.station_id}:{remote_path} ({stderr.strip()})") from None
        os.rename(temp_path, local_path)
        self.total_bytes_downloaded += os.path.getsize(local_path)

    def _transcode_to_h264_blocking(self, input_hevc_path, output_h264_path):
        logging.info(f"Task {self.task_id} - Transcoding {os.path.basename(input_hevc_path)} to H.264...")
        temp_output = output_h264_path + ".part"
        try:
            # Probe source start_time (Unix PTS) so the transcoded timeline starts
            # at the same absolute timestamp. Also preserve it as creation_time metadata.
            creation_time_arg = []
            output_ts_offset_arg = []
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_hevc_path],
                    capture_output=True, text=True, timeout=15
                )
                fmt = json.loads(probe.stdout).get("format", {})
                start_time_str = fmt.get("start_time") or fmt.get("tags", {}).get("creation_time")
                if start_time_str:
                    try:
                        # start_time is a float Unix timestamp string like "1783641601.236016"
                        ts = float(start_time_str)
                        output_ts_offset_arg = ["-output_ts_offset", str(ts)]
                        creation_time_arg = ["-metadata", f"creation_time={datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z"]
                    except ValueError:
                        # Already an ISO string (from tags.creation_time)
                        creation_time_arg = ["-metadata", f"creation_time={start_time_str}"]
            except Exception:
                pass
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", input_hevc_path,
            ] + output_ts_offset_arg + [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                # One-second GOP / fixed keyframe interval makes scrubbing responsive.
                "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                # Keep a fine timescale so the fractional start_time survives the muxer.
                "-video_track_timescale", "12800",
                "-c:a", "copy", "-map_metadata", "0", "-movflags", "+faststart"
            ] + creation_time_arg + ["-f", "mp4", "-y", temp_output]
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            os.rename(temp_output, output_h264_path)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logging.error(f"Task {self.task_id} - H.264 transcoding failed: {e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') else e}")
            if os.path.exists(temp_output): os.remove(temp_output)
            return False

    def _ensure_faststart(self, input_path):
        """Remux an MP4 so its metadata (moov atom) is at the start of the file.

        Browsers need the moov atom up front to determine duration and allow
        seeking/scrubbing before the entire file has downloaded. The input file
        is replaced in-place. Returns True on success, False if remux failed.
        """
        temp_path = input_path + ".faststart.tmp"
        try:
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", input_path,
                # Preserve input timestamps so the absolute start_time from station
                # files is not reset to zero during the faststart remux.
                "-copyts",
                "-c", "copy",
                "-map_metadata", "0",
                "-movflags", "+faststart",
                "-f", "mp4",
                temp_path,
            ]
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            os.replace(temp_path, input_path)
            return True
        except Exception as e:
            logging.warning(f"Task {self.task_id} - Faststart remux failed for {input_path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def _download_minute_video(self, t, video_path, hevc_path, force_h264=False):
        """Download a single minute's source video and return an H.264 path."""
        source_prefix = 'mini' if self.is_low_res else 'full'
        remote_dir = f"/meteor/cam{self.cam}/{t.strftime('%Y%m%d')}/{t.strftime('%H')}"
        temp_path = video_path + ".tmp"
        self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.mp4", temp_path)
        codec = internal_probe_codec(temp_path)
        if codec == 'hevc':
            os.rename(temp_path, hevc_path)
            if (force_h264 or not self.hevc_supported) and not os.path.exists(video_path):
                self._transcode_to_h264_blocking(hevc_path, video_path)
            return video_path if os.path.exists(video_path) else (hevc_path if os.path.exists(hevc_path) else None)
        else:
            os.rename(temp_path, video_path)
            return video_path

    def _concat_videos(self, input_paths, output_path):
        """Concatenate a list of MP4s (copy streams) into a single continuous MP4."""
        list_path = output_path + ".concat_list.txt"
        with open(list_path, 'w') as f:
            for p in input_paths:
                f.write(f"file '{p}'\n")
        temp_path = output_path + ".part"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            "-f", "mp4", temp_path
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            os.replace(temp_path, output_path)
        finally:
            if os.path.exists(list_path):
                os.remove(list_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _apply_absolute_timeline(self, input_path, output_path, start_ts):
        """Shift a relative-timeline MP4 so its start time is the Unix-epoch start_ts."""
        if not start_ts:
            os.replace(input_path, output_path)
            return
        temp_path = output_path + ".abs.part"
        creation_time = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c", "copy",
            "-copyts",
            "-output_ts_offset", str(start_ts),
            "-video_track_timescale", "12800",
            "-movflags", "+faststart",
            "-metadata", f"creation_time={creation_time}",
            "-f", "mp4", temp_path
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            os.replace(temp_path, output_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def process_multi_minute(self):
        """Download consecutive minute videos and produce one continuous output."""
        if self.duration <= 1:
            return self.process()
        if self.is_image and not self.is_long_integration:
            self.errors.append(f"error_for_camera|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
            return None

        times = [self.time_utc + timedelta(minutes=i) for i in range(self.duration)]
        temp_dir = os.path.join(DOWNLOAD_DIR, f"{self.station_code}_cam{self.cam}_{self.time_utc.strftime('%Y%m%d')}_{self.time_utc.strftime('%H%M')}_{self.file_type}_dur{self.duration}_tmp")
        os.makedirs(temp_dir, exist_ok=True)
        minute_clips = []
        try:
            for i, t in enumerate(times):
                self.current_sub_step = i
                base = os.path.join(temp_dir, f"min_{t.strftime('%H%M')}")
                video_path = base + ".mp4"
                hevc_path = base + "_hevc.mp4"
                try:
                    final_source = self._download_minute_video(t, video_path, hevc_path, force_h264=True)
                except Exception as e:
                    logging.warning(f"Task {self.task_id} - Could not download minute {t.strftime('%H:%M')} for cam {self.cam}: {e}")
                    final_source = None
                if not final_source:
                    self.errors.append(f"error_source_file|cam={self.cam},time={t.strftime('%H:%M')}")
                    continue
                minute_clips.append({'time': t, 'path': final_source})
                if self.status_file:
                    update_status(self.status_file, "progress", {
                        "step": self.current_step * self.duration + i + 1,
                        "total": self.total_steps,
                        "message": f"status_processing_file_of_total|i={self.current_step+1},total={self.total_steps}",
                        "files": self.results_ref if self.results_ref is not None else {},
                        "errors": self.errors_ref if self.errors_ref is not None else []
                    })

            if not minute_clips:
                self.errors.append(f"error_source_file|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
                return None

            actual_duration = len(minute_clips)
            if actual_duration < self.duration:
                logging.warning(f"Task {self.task_id} - Only {actual_duration}/{self.duration} minute files available for cam {self.cam}; producing dur{actual_duration} output.")
                self.duration = actual_duration
                self._determine_paths_and_types()

            source_paths = [c['path'] for c in minute_clips]
            first_start_ts = internal_probe_start_time(source_paths[0]) or self.time_utc.timestamp()
            concat_path = os.path.join(temp_dir, "concat.mp4")
            self._concat_videos(source_paths, concat_path)
            if self.is_image:
                if os.path.exists(STACK_SCRIPT):
                    command = [sys.executable, STACK_SCRIPT, concat_path, "-o", self.output_filepath]
                    subprocess.run(command, check=True, capture_output=True, text=True, timeout=1200)
                else:
                    logging.error(f"Stack script not found at {STACK_SCRIPT}")
                    return None
                final_path = self.output_filepath
            else:
                self._apply_absolute_timeline(concat_path, self.output_filepath_h264, first_start_ts)
                final_path = self.output_filepath_h264

            if not os.path.exists(final_path):
                return None
            return self.get_final_result(final_path, is_flight=self.relevant_pass and 'flight_info' in self.relevant_pass)
        except Exception as e:
            logging.error(f"Task {self.task_id} - Multi-minute processing failed for cam {self.cam}: {e}", exc_info=True)
            self.errors.append(f"error_for_camera|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _ensure_base_media_exists(self):
        if self.is_image:
            if os.path.exists(self.output_filepath) or os.path.exists(self.overlay_filepath): return self.output_filepath
        else:
            # 1. Sanity Check for Self-Healing:
            # If the user needs H.264, but we have a file named .mp4 that is actually HEVC (from a previous failed probe),
            # we must detect this. Otherwise we return the "fake" H.264 file, browser fails, user retries, we loop.
            if not self.hevc_supported and os.path.exists(self.output_filepath_h264) and not os.path.exists(self.output_filepath_hevc):
                actual_codec = internal_probe_codec(self.output_filepath_h264)
                if actual_codec == 'hevc':
                    logging.warning(f"Task {self.task_id} - Found .mp4 file is actually HEVC. Renaming to _hevc.mp4 and triggering transcode.")
                    os.rename(self.output_filepath_h264, self.output_filepath_hevc)
                    # Now the logic below will see _hevc exists but _h264 does not, and trigger transcode.
            
            # 2. Transcode Check (Hot-Request):
            # Browsers scrub/seek H.264 far more reliably than raw-station HEVC,
            # especially with long GOPs. Always generate a web-optimized H.264
            # copy when we have an HEVC source but no H.264 yet.
            if os.path.exists(self.output_filepath_hevc) and not os.path.exists(self.output_filepath_h264):
                if self._transcode_to_h264_blocking(self.output_filepath_hevc, self.output_filepath_h264):
                    return self.output_filepath_h264
                else:
                    # Transcode failed. Fall back to HEVC so the client gets something.
                    if os.path.exists(self.output_filepath_hevc): return self.output_filepath_hevc

            # 3. Standard Return:
            if os.path.exists(self.output_filepath_h264): return self.output_filepath_h264
            # If only HEVC exists (e.g. transcode not yet run), return it as fallback.
            if os.path.exists(self.output_filepath_hevc): return self.output_filepath_hevc

        # 4. Download from Station (if local file missing):
        t, source_prefix = self.time_utc, 'mini' if self.is_low_res else 'full'
        remote_dir = f"/meteor/cam{self.cam}/{t.strftime('%Y%m%d')}/{t.strftime('%H')}"
        
        if self.is_image:
            if self.is_long_integration:
                base_video_name = f"{self.station_code}_cam{self.cam}_{t.strftime('%Y%m%d')}_{t.strftime('%H%M')}_{'lowres' if self.is_low_res else 'hires'}"
                video_filepath = os.path.join(DOWNLOAD_DIR, f"{base_video_name}.mp4")
                hevc_filepath = os.path.join(DOWNLOAD_DIR, f"{base_video_name}_hevc.mp4")

                # Try to download if neither exists
                if not os.path.exists(video_filepath) and not os.path.exists(hevc_filepath):
                    self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.mp4", video_filepath)
                    codec = internal_probe_codec(video_filepath)
                    if codec == 'hevc':
                        os.rename(video_filepath, hevc_filepath)
                        # Transcode to H.264 for browsers that don't support HEVC
                        if not self.hevc_supported and not os.path.exists(video_filepath):
                            self._transcode_to_h264_blocking(hevc_filepath, video_filepath)

                # Ensure H.264 copy exists for non-HEVC browsers (may have been missed on a prior download)
                if not self.hevc_supported and os.path.exists(hevc_filepath) and not os.path.exists(video_filepath):
                    self._transcode_to_h264_blocking(hevc_filepath, video_filepath)
                final_source_video = video_filepath if os.path.exists(video_filepath) else (hevc_filepath if os.path.exists(hevc_filepath) else None)
                if final_source_video:
                    if os.path.exists(STACK_SCRIPT):
                        command = [sys.executable, STACK_SCRIPT, final_source_video, "-o", self.output_filepath]
                        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
                    else:
                        logging.error(f"Stack script not found at {STACK_SCRIPT}")
                        return None
                else:
                    return None
            else:
                self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.jpg", self.output_filepath)
            return self.output_filepath if os.path.exists(self.output_filepath) else None
        else: # Video Download
            temp_download_path = self.output_filepath_h264 + ".tmp"
            self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.mp4", temp_download_path)
            
            codec = internal_probe_codec(temp_download_path)
            final_path = self.output_filepath_hevc if codec == 'hevc' else self.output_filepath_h264
            os.rename(temp_download_path, final_path)
            # Make sure downloaded MP4s are web-optimized (moov atom at start).
            self._ensure_faststart(final_path)

            # Always produce a web-optimized H.264 copy for reliable scrubbing,
            # regardless of whether the client reports HEVC support.
            if codec == 'hevc':
                self._transcode_to_h264_blocking(final_path, self.output_filepath_h264)

            return self.output_filepath_h264 if os.path.exists(self.output_filepath_h264) else final_path

    def process(self):
        if self.duration > 1 and (not self.is_image or self.is_long_integration):
            return self.process_multi_minute()
        try:
            base_media_path = self._ensure_base_media_exists()
            if not base_media_path or not os.path.exists(base_media_path):
                self.errors.append(f"error_source_file|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
                return
    
            if self.relevant_pass:
                is_flight = 'flight_info' in self.relevant_pass
                if is_flight:
                    self.relevant_pass['ground_track'] = _interpolate_track(self.relevant_pass.get('ground_track', []), 15)
                for station_id, track in self.relevant_pass.get('station_sky_tracks', {}).items():
                    self.relevant_pass['station_sky_tracks'][station_id] = _interpolate_track(track, 15)
                is_hevc = '_hevc.mp4' in os.path.basename(base_media_path)
           
                overlay_filepath = (self.overlay_filepath_hevc if is_hevc else self.overlay_filepath_h264) if not self.is_image else self.overlay_filepath
                
                if self.relevant_pass.get('station_sky_tracks', {}).get(self.station_id) and not os.path.exists(overlay_filepath):
                    pass_info = self.relevant_pass.copy()
                    if is_flight: pass_info['satellite'] = pass_info.get('flight_info', {}).get('callsign', 'Flight').strip()
                    pass_info['sky_track'] = self.relevant_pass['station_sky_tracks'][self.station_id]
                    
                    if PTO_MAPPER_AVAILABLE:
                        from pto_mapper import get_pto_data_from_json
                        selector = f"{self.station_id.replace('ams', '')}:{self.cam}"
                        if selector not in self.pto_data_cache: self.pto_data_cache[selector] = get_pto_data_from_json(CAMERAS_FILE, selector)
                        
                        if self.is_image:
                            with Image.open(base_media_path) as img: w, h = img.size
                        else: w, h = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", base_media_path]).decode().strip().split('x'))
                        
                        draw_track_on_image(self.pto_data_cache[selector], pass_info, self.track_filepath, target_w=w, target_h=h, is_flight=is_flight)
                        if os.path.exists(self.track_filepath):
                            p = subprocess.Popen([sys.executable, __file__, "_internal_blend_overlay", base_media_path, self.track_filepath, overlay_filepath])
                            self.is_blending_job = True
                            return {"process": p, "overlay_filepath": overlay_filepath, "track_filepath": self.track_filepath}

            final_overlay = (self.overlay_filepath_hevc if '_hevc' in base_media_path else self.overlay_filepath_h264) if not self.is_image else self.overlay_filepath
            final_path = final_overlay if os.path.exists(final_overlay) else base_media_path
            return self.get_final_result(final_path, is_flight=self.relevant_pass and 'flight_info' in self.relevant_pass)
        except FileNotFoundError as e:
            logging.debug(f"Skipping missing file: {e}")
        except Exception as e:
            self.errors.append(f"error_for_camera|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
            logging.error(f"Processing error: {e}", exc_info=True)

    def get_final_result(self, final_filepath, is_flight=False):
        if not os.path.exists(final_filepath): return None
        final_filename = os.path.basename(final_filepath)
        has_overlay = "_overlay" in final_filename
        base_name_part = '_'.join(final_filename.split('_')[:4])
        
        def _is_alt_file(f):
            if f == final_filename: return False
            if not f.startswith(base_name_part): return False
            if '_thumb.' in f or '_track.png' in f: return False
            if f.endswith('.part') or f.endswith('.concat.mp4') or f.endswith('.concat_list.txt') or f.endswith('.faststart.tmp'): return False
            if os.path.isdir(os.path.join(DOWNLOAD_DIR, f)): return False
            return True
        alternatives = [{"url": f"download/{f}", "name": f} for f in os.listdir(DOWNLOAD_DIR) if _is_alt_file(f)]
        result = {"url": f"download/{final_filename}", "name": final_filename, "utc_time_iso": self.time_utc.isoformat(), "alternatives": alternatives}
        # Include the real media duration and start_time so the frontend
        # scrubber/timestamp can work even when the browser reports absolute
        # Unix-epoch timestamps for the timeline.
        if not self.is_image:
            duration = internal_probe_duration(final_filepath)
            if duration is not None:
                result["duration"] = duration
            start_ts = internal_probe_start_time(final_filepath)
            if start_ts is not None:
                result["start_time"] = start_ts
        
        thumb_kwargs = {
            "task_id": self.task_id, "path": final_filepath, "file_type": self.file_type, "station_code": self.station_code, "cam_num": self.cam,
            "has_overlay": has_overlay, "is_flight": is_flight, "max_file_size_mb": MAX_FILE_SIZE_FOR_THUMBNAIL_MB
        }
        if self.translations.get('flight_track_text'): thumb_kwargs['flight_track_text'] = self.translations['flight_track_text']
        if self.translations.get('sat_track_text'): thumb_kwargs['sat_track_text'] = self.translations['sat_track_text']

        if thumb := create_thumbnail(**thumb_kwargs): result["thumb_url"] = f"download/{thumb}"
        return result

def download_for_single_station(task_id, station_id, json_payload_str, master_task_id):
    station_status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    try:
        data = json.loads(json_payload_str)
        logging.info(f"Worker {task_id} for '{station_id}' Started, part of master {master_task_id}.")
        
        with open(STATIONS_FILE, 'r') as f: stations = json.load(f)

        # --- Timelapse early-return path ---
        if data.get('file_type') in ('timelapse', 'timelapse_hires'):
            station_code = stations[station_id]['station']['code']
            start_date_str = data.get('date', '')  # YYYY-MM-DD
            num_days = max(1, int(data.get('length', 1)))
            day_interval = max(1, int(data.get('interval', 1)))
            projections = []
            if data.get('stitch_equirect'): projections.append(('equirect', 8))
            if data.get('stitch_fisheye'):  projections.append(('fisheye', 9))
            # Build list of (date_str, date_compact) for each day
            from datetime import date as date_cls
            start_d = date_cls.fromisoformat(start_date_str)
            dates = [(str(start_d + timedelta(days=i * day_interval)),
                      (start_d + timedelta(days=i * day_interval)).strftime('%Y%m%d'))
                     for i in range(num_days)]
            total_items = len(dates) * len(projections)
            results, errors, total_bytes = {}, [], 0
            update_status(station_status_file, "progress", {"step": 0, "total": total_items, "message": f"status_fetching_timelapse|station={station_code}", "files": results, "errors": errors})
            ssh_control_socket = os.path.join(LOCK_DIR, f"ssh_ctl_{task_id}_{station_id}")
            try:
                ssh_master_proc = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=60",
                     "-o", f"ControlPath={ssh_control_socket}",
                     "-o", "ControlMaster=yes", "-o", "ControlPersist=60",
                     "-N", station_id],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)
            except OSError:
                ssh_master_proc = None
            step = 0
            for date_str, date_compact in dates:
                for proj, cam_num in projections:
                    is_hires_tl = data.get('file_type') == 'timelapse_hires'
                    proj_short = ('teqh' if proj == 'equirect' else 'tfeh') if is_hires_tl else ('teq' if proj == 'equirect' else 'tfe')
                    local_name = f"{station_code}_{date_compact}_{proj_short}.mp4"
                    local_path = os.path.join(DOWNLOAD_DIR, local_name)
                    remote_path = f"/meteor/cam{cam_num}/{date_compact}/timelapse{'_hires' if is_hires_tl else ''}.mp4"
                    if not os.path.exists(local_path):
                        temp_path = local_path + ".part"
                        cmd = ["scp", "-B", "-o", "ConnectTimeout=300"]
                        if ssh_control_socket and os.path.exists(ssh_control_socket):
                            cmd += ["-o", f"ControlPath={ssh_control_socket}"]
                        cmd += [f"{station_id}:{remote_path}", temp_path]
                        # Get remote file size for progress reporting (best-effort)
                        remote_size = 0
                        try:
                            sz_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
                            if ssh_control_socket and os.path.exists(ssh_control_socket):
                                sz_cmd += ["-o", f"ControlPath={ssh_control_socket}"]
                            sz_cmd += [station_id, f"stat -c%s {remote_path}"]
                            sz_out = subprocess.run(sz_cmd, capture_output=True, text=True, timeout=15)
                            remote_size = int(sz_out.stdout.strip())
                        except Exception:
                            pass
                        scp_result = [None]
                        scp_exc = [None]
                        def _run_scp():
                            try:
                                scp_result[0] = subprocess.run(cmd, check=True, timeout=360, capture_output=True)
                            except Exception as e:
                                scp_exc[0] = e
                        scp_thread = threading.Thread(target=_run_scp, daemon=True)
                        scp_thread.start()
                        # Poll .part file size and emit intermediate progress
                        while scp_thread.is_alive():
                            time.sleep(1)
                            if remote_size > 0:
                                try:
                                    part_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                                    file_pct = min(part_size / remote_size, 0.99)
                                    update_status(station_status_file, "progress", {
                                        "step": step + file_pct,
                                        "total": total_items,
                                        "message": f"status_fetching_timelapse|station={station_code}",
                                        "files": results, "errors": errors,
                                    })
                                except Exception:
                                    pass
                        scp_thread.join()
                        if scp_exc[0] is not None:
                            if os.path.exists(temp_path): os.remove(temp_path)
                            errors.append(f"error_timelapse_not_found|date={date_str}")
                            logging.warning(f"Worker {task_id} - Timelapse not found: {station_id}:{remote_path}")
                            step += 1
                            continue
                        os.rename(temp_path, local_path)
                        total_bytes += os.path.getsize(local_path)
                    if os.path.exists(local_path):
                        thumb_kwargs = {"task_id": task_id, "path": local_path, "file_type": "lowres", "station_code": station_code, "cam_num": cam_num}
                        entry = {"url": f"download/{local_name}", "name": local_name, "utc_time_iso": f"{date_str}T00:00:00+00:00", "alternatives": []}
                        if thumb := create_thumbnail(**thumb_kwargs): entry["thumb_url"] = f"download/{thumb}"
                        results.setdefault(date_compact, []).append(entry)
                    step += 1
                    update_status(station_status_file, "progress", {"step": step, "total": total_items, "message": f"status_fetching_timelapse|station={station_code}", "files": results, "errors": errors})
            if ssh_master_proc and ssh_master_proc.poll() is None:
                try:
                    subprocess.run(["ssh", "-o", f"ControlPath={ssh_control_socket}", "-O", "exit", station_id], capture_output=True, timeout=10)
                except Exception: ssh_master_proc.terminate()
            if os.path.exists(ssh_control_socket): os.remove(ssh_control_socket)
            update_status(station_status_file, "complete", {"files": results, "errors": errors, "total_bytes_downloaded": total_bytes})
            logging.info(f"Worker {task_id} for '{station_id}' (timelapse) Completed.")
            return
        # --- End timelapse path ---

        pass_data_list = [data[p] for p in ['pass_data', 'flight_pass_data'] if p in data and data[p]]
        if not pass_data_list and data.get('satellite_panel_enabled', False) and os.path.exists(PASS_CACHE_FILE):
            with open(PASS_CACHE_FILE, 'r') as f: pass_data_list = json.load(f).get("data", {}).get("passes", [])

        results, errors, blending_jobs, pto_data_cache, total_bytes = {}, [], [], {}, 0
        files_to_process = []
        
        translations = {}
        lang_code = data.get('lang')
        if lang_code:
            try:
                lang_file = os.path.join(BASE_DIR, 'lang', f"{lang_code}.json")
                if os.path.exists(lang_file):
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        lang_data = json.load(f)
                    translations['flight_track_text'] = lang_data.get('thumb_flight_track')
                    translations['sat_track_text'] = lang_data.get('thumb_satellite_track')
            except Exception as e:
                logging.warning(f"Worker {task_id} - Could not load translations for lang '{lang_code}': {e}")

        duration = max(1, int(data.get('duration', 1)))
        file_type = data['file_type']
        is_video_multi = duration > 1 and file_type in ('lowres', 'hires')
        is_long = file_type.endswith('_long')

        if 'camera_views' in data and data['camera_views']:
            for view in data['camera_views']:
                start = datetime.fromisoformat(view['start_utc']).replace(second=0, microsecond=0)
                end = datetime.fromisoformat(view['end_utc'])
                while start <= end:
                    files_to_process.append({'time': start, 'cam': view['camera']})
                    start += timedelta(minutes=1)
        elif is_video_multi:
            start_time = datetime.strptime(f"{data['date']} {data['hour']}:{data['minute']}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
            for cam in data['cameras']:
                files_to_process.append({'time': start_time, 'cam': int(cam), 'duration': duration})
        else:
            start_time = datetime.strptime(f"{data['date']} {data['hour']}:{data['minute']}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
            seq_length = int(data['length'])
            seq_interval = int(data['interval'])
            for i in range(seq_length):
                for cam in data['cameras']:
                    item = {'time': start_time + timedelta(minutes=i*seq_interval), 'cam': int(cam)}
                    if is_long:
                        item['duration'] = duration
                    files_to_process.append(item)

        station_code = stations[station_id]['station']['code']

        # Open an SSH ControlMaster only when at least one required file is missing
        # from disk.  For image file types we can cheaply pre-check all expected
        # local paths, saving ~1 s on a fully-cached request.
        ssh_control_socket = os.path.join(LOCK_DIR, f"ssh_ctl_{task_id}_{station_id}")
        ssh_master_proc = None
        def _expected_image_path(fi):
            dur = fi.get('duration', 1)
            suffix = f"_dur{dur}" if dur > 1 and is_long else ""
            return os.path.join(DOWNLOAD_DIR,
                f"{station_code}_cam{fi['cam']}_{fi['time'].strftime('%Y%m%d')}"
                f"_{fi['time'].strftime('%H%M')}{suffix}_{data['file_type']}.jpg")
        need_ssh = not data['file_type'].startswith('image') or any(
            not os.path.exists(_expected_image_path(fi))
            for fi in files_to_process
        )
        if need_ssh:
            try:
                ssh_master_proc = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=60",
                     "-o", f"ControlPath={ssh_control_socket}",
                     "-o", "ControlMaster=yes", "-o", "ControlPersist=120",
                     "-N", station_id],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                import time as _time; _time.sleep(1)  # give master time to establish
                logging.info(f"Worker {task_id} - SSH ControlMaster opened for {station_id} (socket={ssh_control_socket})")
            except OSError as e:
                logging.warning(f"Worker {task_id} - SSH ControlMaster failed to start: {e}")
        else:
            logging.info(f"Worker {task_id} - All files cached for {station_id}, skipping SSH ControlMaster")

        # --- Batch prefetch images via tar pipe (one SSH exec per remote dir) ---
        is_image_dl = data['file_type'].startswith('image') and not data['file_type'].endswith('_long')
        if is_image_dl:
            source_prefix = 'mini' if 'lowres' in data['file_type'] else 'full'
            # Group files by remote directory
            by_dir = {}
            for fi in files_to_process:
                t = fi['time']
                remote_dir = f"/meteor/cam{fi['cam']}/{t.strftime('%Y%m%d')}/{t.strftime('%H')}"
                remote_file = f"{source_prefix}_{t.strftime('%M')}.jpg"
                local_name = f"{station_code}_cam{fi['cam']}_{t.strftime('%Y%m%d')}_{t.strftime('%H%M')}_{data['file_type']}.jpg"
                local_path = os.path.join(DOWNLOAD_DIR, local_name)
                if not os.path.exists(local_path):
                    by_dir.setdefault(remote_dir, []).append((remote_file, local_path))
            for remote_dir, file_list in by_dir.items():
                filenames = [rf for rf, _ in file_list]
                local_map = {rf: lp for rf, lp in file_list}
                ssh_cmd = ["ssh"]
                if ssh_control_socket and os.path.exists(ssh_control_socket):
                    ssh_cmd += ["-o", f"ControlPath={ssh_control_socket}"]
                ssh_cmd += [station_id, f"cd {remote_dir} && tar -czf - {' '.join(filenames)} 2>/dev/null"]
                try:
                    result = subprocess.run(ssh_cmd, capture_output=True, timeout=120)
                    if result.returncode == 0 and result.stdout:
                        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
                            for member in tar.getmembers():
                                bname = os.path.basename(member.name)
                                if bname in local_map:
                                    f = tar.extractfile(member)
                                    if f:
                                        tmp = local_map[bname] + ".part"
                                        with open(tmp, 'wb') as out:
                                            out.write(f.read())
                                        os.rename(tmp, local_map[bname])
                        logging.info(f"Worker {task_id} - batch prefetch done for {remote_dir} ({len(filenames)} files)")
                except Exception as e:
                    logging.warning(f"Worker {task_id} - batch prefetch failed for {remote_dir}: {e}")

        files_iterator = iter(enumerate(files_to_process))
        current_file_idx, current_file_item = next(files_iterator, (None, None))
        processing_done = False

        do_fisheye  = data.get('stitch_fisheye',  False)
        do_equirect = data.get('stitch_equirect', False)
        is_hires    = data['file_type'] in ('image', 'image_long')
        is_long     = data['file_type'].endswith('_long')
        do_stitch   = (do_fisheye or do_equirect) and os.path.exists(STITCH_SCRIPT)

        # Pre-compute expected camera count per timestamp for stitch triggering
        stitch_cams_expected = {}  # t_key -> set of cam nums
        if do_stitch:
            for fi in files_to_process:
                t_key = fi['time'].strftime('%H:%M')
                stitch_cams_expected.setdefault(t_key, set()).add(int(fi['cam']))

        stitch_ready     = {}   # t_key -> {cam: abs_path}  (downloaded so far)
        stitch_launched  = set()  # t_keys already handed to stitch.py
        stitch_jobs      = []   # [{process, t_key, t_iso, stdout_path}]
        stitch_done      = 0    # stitch outputs added to results so far
        stitch_output_count = (int(do_fisheye) + int(do_equirect)) * len(stitch_cams_expected) if do_stitch else 0
        total_steps = sum(fi.get('duration', 1) for fi in files_to_process) + stitch_output_count

        master_lock_file = os.path.join(LOCK_DIR, f"{master_task_id}.lock")
        step_offset = 0
        while not processing_done:
            if not os.path.exists(master_lock_file):
                logging.warning(f"Worker {task_id} - Master task lock file not found. Terminating.")
                break

            if current_file_item:
                item_duration = current_file_item.get('duration', 1)
                start_step = step_offset
                update_status(station_status_file, "progress", {"step": start_step, "total": total_steps, "message": f"status_processing_file_of_total|i={current_file_idx+1},total={total_steps}", "files": results, "errors": errors})

                proc = FileProcessor(task_id, station_id, station_code, current_file_item['cam'], current_file_item['time'], data['file_type'], pass_data_list, pto_data_cache, data.get('hevc_supported', False), translations=translations, ssh_control_socket=ssh_control_socket, status_file=station_status_file, current_step=current_file_idx, total_steps=total_steps, results_ref=results, errors_ref=errors, duration=item_duration)
                job = proc.process()

                errors.extend(proc.errors)
                total_bytes += proc.total_bytes_downloaded
                if proc.is_blending_job and job:
                    job['time_key'], job['processor'] = current_file_item['time'].strftime('%H:%M'), proc
                    blending_jobs.append(job)
                elif job:
                    t_key_dl = current_file_item['time'].strftime('%H:%M')
                    results.setdefault(t_key_dl, []).append(job)
                    end_step = step_offset + item_duration
                    update_status(station_status_file, "progress", {"step": end_step, "total": total_steps, "message": f"status_processing_file_of_total|i={current_file_idx+1},total={total_steps}", "files": results, "errors": errors})

                    # Track downloaded image for stitch readiness check
                    if do_stitch:
                        fname = job.get('name', '')
                        try:
                            parts = fname.split('_')
                            cam_part = next(p for p in parts if p.startswith('cam'))
                            cam_num = int(cam_part.replace('cam', ''))
                            hhmm = parts[3]
                            t_key_img = f"{hhmm[:2]}:{hhmm[2:]}"
                            abs_path = os.path.join(DOWNLOAD_DIR, fname)
                            if os.path.exists(abs_path):
                                stitch_ready.setdefault(t_key_img, {})[cam_num] = abs_path
                        except (StopIteration, ValueError, IndexError):
                            pass

                        # Launch stitch as soon as all cameras for a timestamp are ready
                        for t_key_s, expected_cams in stitch_cams_expected.items():
                            if t_key_s in stitch_launched:
                                continue
                            ready_cams = stitch_ready.get(t_key_s, {})
                            if expected_cams.issubset(ready_cams.keys()):
                                stitch_launched.add(t_key_s)
                                t_obj = datetime.strptime(f"{data['date']} {t_key_s}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
                                base_name = f"{station_code}_{t_obj.strftime('%Y%m%d')}_{t_obj.strftime('%H%M')}"
                                res_suffix = ("hires" if is_hires else "lowres") + ("_long" if is_long else "")

                                # Register any already-existing stitch outputs immediately.
                                stitch_hit = {}
                                for proj, flag in [('fisheye', do_fisheye), ('equirect', do_equirect)]:
                                    if flag:
                                        fname = f"{base_name}_{res_suffix}_{proj}.jpg"
                                        fpath = os.path.join(DOWNLOAD_DIR, fname)
                                        if os.path.exists(fpath):
                                            stitch_hit[proj] = {'path': fpath, 'name': fname}
                                _STITCH_CAM = {'equirect': 8, 'fisheye': 9}

                                # Try to fetch pre-stitched files from station (not available for long-integration)
                                if not is_long:
                                    _PRESTITCH_PREFIX = {'equirect': 'mini' if not is_hires else 'full',
                                                         'fisheye':  'mini' if not is_hires else 'full'}
                                    for proj, cam_num in [('equirect', 8), ('fisheye', 9)]:
                                        if not (proj == 'equirect' and do_equirect or proj == 'fisheye' and do_fisheye):
                                            continue
                                        if proj in stitch_hit:
                                            continue
                                        fname = f"{base_name}_{res_suffix}_{proj}.jpg"
                                        fpath = os.path.join(DOWNLOAD_DIR, fname)
                                        remote_dir = f"/meteor/cam{cam_num}/{t_obj.strftime('%Y%m%d')}/{t_obj.strftime('%H')}"
                                        remote_file = f"{'mini' if not is_hires else 'full'}_{t_obj.strftime('%M')}.jpg"
                                        remote_path = f"{remote_dir}/{remote_file}"
                                        temp_path = fpath + ".part"
                                        cmd_scp = ["scp", "-B", "-o", "ConnectTimeout=15"]
                                        if ssh_control_socket and os.path.exists(ssh_control_socket):
                                            cmd_scp += ["-o", f"ControlPath={ssh_control_socket}"]
                                        cmd_scp += [f"{station_id}:{remote_path}", temp_path]
                                        try:
                                            subprocess.run(cmd_scp, check=True, timeout=60, capture_output=True)
                                            os.rename(temp_path, fpath)
                                            stitch_hit[proj] = {'path': fpath, 'name': fname}
                                            logging.info(f"Worker {task_id} - Pre-stitched {proj} fetched from station: {remote_path}")
                                        except subprocess.CalledProcessError:
                                            if os.path.exists(temp_path): os.remove(temp_path)
                                            logging.info(f"Worker {task_id} - Pre-stitched {proj} not found on station: {remote_path}, will stitch locally")

                                def get_stitch_display_name(proj, is_hires, is_long):
                                    """Generate display name for stitch output based on projection, resolution, and exposure."""
                                    prefix = 'eq' if proj == 'equirect' else 'fe'
                                    res = 'h' if is_hires else 'l'
                                    long = 'l' if is_long else ''
                                    return f"{prefix}{res}{long}"

                                for proj, info in stitch_hit.items():
                                    thumb_kwargs = {"task_id": task_id, "path": info['path'], "file_type": 'image', "station_code": station_code, "cam_num": _STITCH_CAM.get(proj, 0)}
                                    display_name = get_stitch_display_name(proj, is_hires, is_long)
                                    stitch_entry = {"url": f"download/{info['name']}", "name": display_name, "utc_time_iso": t_obj.isoformat(), "alternatives": []}
                                    if thumb := create_thumbnail(**thumb_kwargs): stitch_entry["thumb_url"] = f"download/{thumb}"
                                    results.setdefault(t_key_s, []).append(stitch_entry)
                                stitch_done += len(stitch_hit)

                                need_fisheye  = do_fisheye  and 'fisheye'  not in stitch_hit
                                need_equirect = do_equirect and 'equirect' not in stitch_hit
                                if need_fisheye or need_equirect:
                                    cmd = [
                                        sys.executable, STITCH_SCRIPT,
                                        '--station-id', station_id,
                                        '--station-code', station_code,
                                        '--image-paths'] + list(ready_cams.values()) + [
                                        '--cameras'] + [str(c) for c in ready_cams.keys()] + [
                                        '--output-dir', DOWNLOAD_DIR,
                                        '--base-name', base_name,
                                    ]
                                    if is_hires:       cmd.append('--hires')
                                    if is_long:        cmd.append('--long')
                                    if need_fisheye:   cmd.append('--fisheye')
                                    if need_equirect:  cmd.append('--equirect')
                                    stdout_path = os.path.join(LOCK_DIR, f"{task_id}_stitch_{t_key_s.replace(':', '')}.json")
                                    logging.info(f"Worker {task_id} STITCH-DEBUG: launching background stitch for {t_key_s}, cams={sorted(ready_cams.keys())}")
                                    with open(stdout_path, 'w') as sf:
                                        sp = subprocess.Popen(cmd, stdout=sf, stderr=subprocess.PIPE)
                                    stitch_jobs.append({
                                        'process': sp, 't_key': t_key_s, 't_iso': t_obj.isoformat(),
                                        'stdout_path': stdout_path,
                                        'station_code': station_code,
                                        'base_name': base_name,
                                        'res_suffix': res_suffix,
                                        'do_fisheye': need_fisheye,
                                        'do_equirect': need_equirect,
                                        'reported': set(),
                                    })
                                else:
                                    logging.info(f"Worker {task_id} STITCH-DEBUG: all stitch outputs cached for {t_key_s}, skipping subprocess")

                step_offset += item_duration
                current_file_idx, current_file_item = next(files_iterator, (None, None))

            remaining_blending_jobs = []
            blends_finished_this_cycle = False
            for job in blending_jobs:
                if job['process'].poll() is not None:
                    blends_finished_this_cycle = True
                    if job['process'].returncode == 0:
                        if res := job['processor'].get_final_result(job['overlay_filepath'], 'flight_info' in job['processor'].relevant_pass):
                            results.setdefault(job['time_key'], []).append(res)
                    else:
                        errors.append(f"error_blending_track|filename={os.path.basename(job['overlay_filepath'])}")
                    if os.path.exists(job['track_filepath']): os.remove(job['track_filepath'])
                else:
                    remaining_blending_jobs.append(job)
            blending_jobs = remaining_blending_jobs

            # Poll background stitch jobs
            remaining_stitch_jobs = []
            _STITCH_CAM = {'equirect': 8, 'fisheye': 9}

            def get_stitch_display_name(proj, is_hires, is_long):
                """Generate display name for stitch output based on projection, resolution, and exposure."""
                prefix = 'eq' if proj == 'equirect' else 'fe'
                res = 'h' if is_hires else 'l'
                long = 'l' if is_long else ''
                return f"{prefix}{res}{long}"

            for sjob in stitch_jobs:
                # Parse res_suffix to determine is_hires and is_long
                is_hires = 'hires' in sjob['res_suffix']
                is_long = 'long' in sjob['res_suffix']

                # Check for outputs that are already on disk while the subprocess is still running.
                for proj, flag in [('fisheye', sjob['do_fisheye']), ('equirect', sjob['do_equirect'])]:
                    if flag and proj not in sjob['reported']:
                        fname = f"{sjob['base_name']}_{sjob['res_suffix']}_{proj}.jpg"
                        fpath = os.path.join(DOWNLOAD_DIR, fname)
                        if os.path.exists(fpath):
                            thumb_kwargs = {"task_id": task_id, "path": fpath, "file_type": 'image', "station_code": sjob['station_code'], "cam_num": _STITCH_CAM.get(proj, 0)}
                            display_name = get_stitch_display_name(proj, is_hires, is_long)
                            stitch_entry = {"url": f"download/{fname}", "name": display_name, "utc_time_iso": sjob['t_iso'], "alternatives": []}
                            if thumb := create_thumbnail(**thumb_kwargs): stitch_entry["thumb_url"] = f"download/{thumb}"
                            results.setdefault(sjob['t_key'], []).append(stitch_entry)
                            logging.info(f"Worker {task_id} STITCH-DEBUG: added early result {fname}")
                            stitch_done += 1
                            sjob['reported'].add(proj)
                            blends_finished_this_cycle = True

                if sjob['process'].poll() is not None:
                    blends_finished_this_cycle = True  # trigger a status update
                    rc = sjob['process'].returncode
                    logging.info(f"Worker {task_id} STITCH-DEBUG: stitch finished for {sjob['t_key']} rc={rc}")
                    if rc == 0:
                        try:
                            with open(sjob['stdout_path']) as sf:
                                stitch_results = json.loads(sf.read().strip())
                            for key, info in stitch_results.items():
                                if key in sjob['reported']:
                                    continue
                                if info and os.path.exists(info['path']):
                                    thumb_kwargs = {"task_id": task_id, "path": info['path'], "file_type": 'image', "station_code": sjob['station_code'], "cam_num": _STITCH_CAM.get(key, 0)}
                                    display_name = get_stitch_display_name(key, is_hires, is_long)
                                    stitch_entry = {"url": f"download/{info['name']}", "name": display_name, "utc_time_iso": sjob['t_iso'], "alternatives": []}
                                    if thumb := create_thumbnail(**thumb_kwargs): stitch_entry["thumb_url"] = f"download/{thumb}"
                                    results.setdefault(sjob['t_key'], []).append(stitch_entry)
                                    logging.info(f"Worker {task_id} STITCH-DEBUG: added result {info['name']}")
                                    stitch_done += 1
                                    sjob['reported'].add(key)
                        except Exception as e:
                            logging.error(f"Worker {task_id} STITCH-DEBUG: failed to read stitch output: {e}")
                            errors.append("error_internal")
                    else:
                        stderr_out = sjob['process'].stderr.read().decode('utf-8', errors='ignore') if sjob['process'].stderr else ''
                        logging.error(f"Worker {task_id} STITCH-DEBUG: stitch failed for {sjob['t_key']}: {stderr_out[:500]}")
                        errors.append("error_internal")
                    if os.path.exists(sjob['stdout_path']):
                        os.remove(sjob['stdout_path'])
                else:
                    remaining_stitch_jobs.append(sjob)
            stitch_jobs = remaining_stitch_jobs
            
            if blends_finished_this_cycle:
                update_status(station_status_file, "progress", {"step": len(files_to_process) + stitch_done, "total": total_steps, "message": f"status_waiting_for_blend|count={len(blending_jobs)}", "files": results, "errors": errors})

            if current_file_item is None and not blending_jobs and not stitch_jobs:
                processing_done = True
            else:
                time.sleep(0.5)

        is_stackable_request = (data['file_type'] in ['image_long', 'image_lowres_long'] and int(data.get('length', 0)) > 1 and int(data.get('interval', 0)) == 1 and not any(k in data for k in ['pass_data', 'flight_pass_data']))
        
        if is_stackable_request:
            logging.info(f"Worker {task_id} - Stackable request detected. Creating combined images.")
            images_by_camera = {}
            for time_key, files in results.items():
                for file_info in files:
                    try:
                        parts = file_info['name'].split('_')
                        cam_num = int(parts[1].replace('cam', ''))
                        images_by_camera.setdefault(cam_num, []).append({'path': os.path.join(DOWNLOAD_DIR, file_info['name']), 'time': datetime.fromisoformat(file_info['utc_time_iso'])})
                    except (IndexError, ValueError): continue
            
            for cam_num, images in images_by_camera.items():
                if len(images) > 1:
                    images.sort(key=lambda x: x['time'])
                    time_range_label = f"{images[0]['time'].strftime('%H:%M')} - {images[-1]['time'].strftime('%H:%M')}"
                    
                    parts = os.path.basename(images[0]['path']).split('_')
                    time_fn_part = f"{images[0]['time'].strftime('%H%M')}-{images[-1]['time'].strftime('%M')}"
                    output_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{time_fn_part}_{data['file_type']}_stacked.jpg"
                    output_filepath = os.path.join(DOWNLOAD_DIR, output_filename)
                    
                    thumb_kwargs = {"task_id": task_id, "path": output_filepath, "file_type": 'image', "station_code": stations[station_id]['station']['code'], "cam_num": cam_num}
                    if translations.get('flight_track_text'): thumb_kwargs['flight_track_text'] = translations['flight_track_text']
                    if translations.get('sat_track_text'): thumb_kwargs['sat_track_text'] = translations['sat_track_text']

                    if stack_images([img['path'] for img in images], output_filepath, task_id):
                        stacked_result = {"url": f"download/{output_filename}", "name": output_filename, "utc_time_iso": images[0]['time'].isoformat(), "alternatives": []}
                        if thumb := create_thumbnail(**thumb_kwargs): stacked_result["thumb_url"] = f"download/{thumb}"
                        results.setdefault(time_range_label, []).append(stacked_result)
                    else:
                        errors.append(f"error_stacking_image|cam_num={cam_num}")

        if ssh_master_proc and ssh_master_proc.poll() is None:
            try:
                subprocess.run(["ssh", "-o", f"ControlPath={ssh_control_socket}",
                                "-O", "exit", station_id],
                               capture_output=True, timeout=10)
            except Exception:
                ssh_master_proc.terminate()
            logging.info(f"Worker {task_id} - SSH ControlMaster closed for {station_id}")
        if os.path.exists(ssh_control_socket):
            os.remove(ssh_control_socket)

        update_status(station_status_file, "complete", {"files": results, "errors": errors, "total_bytes_downloaded": total_bytes})
        logging.info(f"Worker {task_id} for '{station_id}' Completed.")
    except Exception as e:
        error_msg = f"Worker crashed: {str(e)}"
        logging.exception(f"Worker {task_id} crashed.")
        update_status(station_status_file, "error", {"message": error_msg})

def main_download_coordinator(master_task_id, json_payload, user_ip):
    status_file = os.path.join(LOCK_DIR, f"{master_task_id}.json")
    pid_file = os.path.join(LOCK_DIR, f"{master_task_id}.pid")
    lock_file = os.path.join(LOCK_DIR, f"{master_task_id}.lock")
    sub_tasks = {}
    
    try:
        with open(pid_file, 'w') as f: f.write(str(os.getpid()))
    except IOError:
        update_status(status_file, "error", {"message": "error_internal"}); return

    try:
        logging.info(f"Coordinator {master_task_id} Started for IP {user_ip} (PID: {os.getpid()}).")
        open(lock_file, 'w').close()
        trim_log_file(LOG_FILE, MAX_LOG_LINES, master_task_id)
        for d in [DOWNLOAD_DIR, LOG_DIR, CACHE_DIR]: cleanup_old_files(d, CLEANUP_AGE_DAYS, master_task_id, [os.path.basename(LOG_FILE)] if d == LOG_DIR else [])
        
        data = json.loads(json_payload)
        if 'crossing_data' in data: data['flight_pass_data'] = data.pop('crossing_data')
        active_pass_data = data.get('pass_data') or data.get('flight_pass_data')

        if active_pass_data:
            user_selected_stations = data.get("stations", [])
            try: user_selected_cameras = [int(c) for c in data.get('cameras', [])]
            except (ValueError, TypeError): user_selected_cameras = []
            if user_selected_stations and user_selected_cameras:
                active_pass_data['camera_views'] = [v for v in active_pass_data.get('camera_views', []) if v.get('station_id') in user_selected_stations and v.get('camera') in user_selected_cameras]

        station_ids = list(set(v['station_id'] for v in active_pass_data.get('camera_views', []))) if active_pass_data else data.get("stations", [])
        aggregated_errors, stations_to_process = {}, []
        
        try:
            if not station_ids: raise ValueError("error_no_station_selected")
            if len(station_ids) > MAX_STATIONS_PER_REQUEST: raise ValueError(f"error_too_many_stations|max={MAX_STATIONS_PER_REQUEST}")
            
            with open(STATIONS_FILE, 'r') as f: valid_stations = json.load(f)
            for sid in station_ids:
                if sid not in valid_stations: raise ValueError(f"error_invalid_station_id|sid={sid}")
            
            file_type = data.get('file_type', 'lowres')
            if file_type not in FILE_TYPE_LIMITS: raise ValueError(f"error_invalid_file_type|file_type={file_type}")
            
            limit = FILE_TYPE_LIMITS[file_type]
            if file_type in ('timelapse', 'timelapse_hires'):
                num_days = max(1, int(data.get('length', 1)))
                num_files = len(station_ids) * num_days * (int(bool(data.get('stitch_fisheye'))) + int(bool(data.get('stitch_equirect'))))
            elif active_pass_data:
                num_files = sum(round((datetime.fromisoformat(v['end_utc']) - datetime.fromisoformat(v['start_utc'])).total_seconds() / 60) + 1 for v in active_pass_data.get('camera_views', []))
            else:
                duration = int(data.get('duration', 1))
                is_video_multi = duration > 1 and file_type in ('lowres', 'hires')
                if is_video_multi:
                    num_files = len(station_ids) * len(data.get('cameras', []))
                else:
                    num_files = len(station_ids) * len(data.get('cameras', [])) * int(data.get('length', 1))
            if num_files > limit: raise ValueError(f"error_too_many_files|num_files={num_files},limit={limit}")
            
            if not active_pass_data and file_type in ('timelapse', 'timelapse_hires'):
                if not (1 <= int(data.get('length', 0)) <= 100): raise ValueError(f"error_invalid_length|max=100")
                if not (1 <= int(data.get('interval', 0)) <= 365): raise ValueError(f"error_invalid_interval|max=365")
            elif not active_pass_data:
                if not (1 <= int(data.get('length', 0)) <= MAX_SEQUENCE_LENGTH): raise ValueError(f"error_invalid_length|max={MAX_SEQUENCE_LENGTH}")
                if not (1 <= int(data.get('interval', 0)) <= MAX_SEQUENCE_INTERVAL): raise ValueError(f"error_invalid_interval|max={MAX_SEQUENCE_INTERVAL}")
          
            quota_tracker = {}
            if os.path.exists(QUOTA_TRACKER_FILE):
                with open(QUOTA_TRACKER_FILE, 'r') as f:
                    try: quota_tracker = json.load(f)
                    except json.JSONDecodeError: logging.warning(f"Task {master_task_id} - Could not parse quota_tracker.json.")
            
            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            todays_usage = quota_tracker.get(today_str, {})
            total_quota_bytes, per_site_quota_bytes = TOTAL_QUOTA_LIMIT_MB * 1024 * 1024, PER_SITE_QUOTA_LIMIT_MB * 1024 * 1024

            for station_id in station_ids:
                station_info = valid_stations.get(station_id, {})
                station_code = station_info.get("station", {}).get("code", station_id)
                if station_info.get("station", {}).get("quota"):
                    station_usage_today = todays_usage.get(station_id, {"total": 0, "sites": {}})
                    if isinstance(station_usage_today, int): station_usage_today = {"total": station_usage_today, "sites": {}}
                    total_usage_bytes = station_usage_today.get("total", 0)
                    site_usage_bytes = station_usage_today.get("sites", {}).get(user_ip, 0)
                    avg_size_bytes = AVG_FILE_SIZES_MB.get(file_type, 2) * 1024 * 1024
                    if active_pass_data:
                        num_files_for_station = sum(round((datetime.fromisoformat(v['end_utc']) - datetime.fromisoformat(v['start_utc'])).total_seconds() / 60) + 1 for v in active_pass_data.get('camera_views', []) if v['station_id'] == station_id)
                    else:
                        duration_q = int(data.get('duration', 1))
                        is_video_multi_q = duration_q > 1 and file_type in ('lowres', 'hires')
                        num_files_for_station = len(data.get('cameras', [])) * (duration_q if is_video_multi_q else int(data.get('length', 1)))
                    estimated_request_size = num_files_for_station * avg_size_bytes
                    if site_usage_bytes + estimated_request_size > per_site_quota_bytes:
                        aggregated_errors.setdefault(station_code, []).append(f"error_user_quota_exceeded|limit={PER_SITE_QUOTA_LIMIT_MB},station_code={station_code}"); continue
                    if total_usage_bytes + estimated_request_size > total_quota_bytes:
                        aggregated_errors.setdefault(station_code, []).append(f"error_total_quota_exceeded|limit={TOTAL_QUOTA_LIMIT_MB},station_code={station_code}"); continue
            
                stations_to_process.append(station_id)
        except (ValueError, TypeError) as e:
            update_status(status_file, "error", {"message": str(e)}); return

        if not stations_to_process:
            update_status(status_file, "complete", {"files": {}, "errors": aggregated_errors})
            logging.info(f"Coordinator {master_task_id} - No stations to process after quota check."); return

        for station_id in stations_to_process:
            sub_task_id = uniqid('task_')
            worker_payload = data.copy()
            if active_pass_data: worker_payload['camera_views'] = [v for v in active_pass_data.get('camera_views', []) if v['station_id'] == station_id]
            command = [sys.executable, __file__, '_internal_download_station', sub_task_id, station_id, json.dumps(worker_payload), master_task_id]
            sub_tasks[sub_task_id] = {"station_id": station_id, "process": subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)}
        
        start_time = time.time()
        while time.time() - start_time < 1800:
            all_done, total_steps_done, total_steps_overall = True, 0, 0
            aggregated_files_so_far, aggregated_errors_so_far = {}, {}
            for task_id, task_info in sub_tasks.items():
                s_file = os.path.join(LOCK_DIR, f"{task_id}.json")
                station_code = valid_stations.get(task_info['station_id'], {}).get('station', {}).get('code', 'UNKNOWN')
                if not os.path.exists(s_file): all_done = False; continue
                try:
                     with open(s_file, 'r') as f: s_data = json.load(f)
                except (json.JSONDecodeError, IOError): all_done = False; continue
                
                if s_data.get("status") != "complete" and s_data.get("status") != "error": all_done = False
                total_steps_done += s_data.get("step", 0)
                total_steps_overall += s_data.get("total", 1)
                if s_data.get("files"): aggregated_files_so_far[station_code] = s_data["files"]
                if s_data.get("errors"): aggregated_errors_so_far.setdefault(station_code, []).extend(s_data["errors"])
                if s_data.get("status") == "error": pass

            percentage_done = (total_steps_done / total_steps_overall) * 100 if total_steps_overall > 0 else (100 if all_done else 0)
            update_status(status_file, "progress", {"step": percentage_done, "total": 100, "message": f"status_processing_files|done={int(total_steps_done)},total={int(total_steps_overall)}", "files": aggregated_files_so_far, "errors": aggregated_errors_so_far})
            if all_done: break
            time.sleep(1)
        
        aggregated_files, quota_updates = {}, {}
        with open(STATIONS_FILE, 'r') as f: all_stations_data = json.load(f)
        for task_id, task_info in sub_tasks.items():
            s_file = os.path.join(LOCK_DIR, f"{task_id}.json")
            if os.path.exists(s_file):
                with open(s_file, 'r') as f: s_data = json.load(f)
                station_code = all_stations_data.get(task_info['station_id'], {}).get('station', {}).get('code', 'UNKNOWN')
                if s_data.get("files"): aggregated_files[station_code] = s_data["files"]
                if s_data.get("errors"): aggregated_errors.setdefault(station_code, []).extend(s_data["errors"])
                if s_data.get("message") and s_data.get("status") == "error": aggregated_errors.setdefault(station_code, []).append(f"error_worker_crash|msg={s_data.get('message')}")
                if s_data.get("total_bytes_downloaded", 0) > 0: quota_updates[task_info['station_id']] = quota_updates.get(task_info['station_id'], 0) + s_data["total_bytes_downloaded"]

        if quota_updates: update_quota_tracker(quota_updates, master_task_id, user_ip, QUOTA_TRACKER_FILE)
        update_status(status_file, "complete", {"files": aggregated_files, "errors": aggregated_errors})
        logging.info(f"Coordinator {master_task_id} finished successfully.")

    except Exception as e:
        logging.exception(f"Coordinator {master_task_id} crashed with an unhandled exception.")
        update_status(status_file, "error", {"message": "error_internal"})
    finally:
        logging.info(f"Coordinator {master_task_id} entering cleanup.")
        for task_id, task in sub_tasks.items():
            if task['process'].poll() is None:
                logging.info(f"Coordinator {master_task_id} - Terminating worker PID {task['process'].pid}")
                try:
                    task['process'].terminate()
                    task['process'].wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    task['process'].kill()
                except Exception as e:
                    logging.error(f"Coordinator {master_task_id} - Error killing worker {task_id}: {e}")
        for f in [pid_file, lock_file]:
            if os.path.exists(f):
                try: os.remove(f)
                except OSError as e: logging.error(f"Coordinator {master_task_id} - Could not remove control file {f}: {e}")
        logging.info(f"Coordinator {master_task_id} finished cleanup.")


def render_template(template, lang_data):
    def replace_match(match):
        key = match.group(1)
        return str(lang_data.get(key, match.group(0)))
    return re.sub(r'__\{\{([a-zA-Z0-9_]+)\}\}__', replace_match, template)

def main():
    if len(sys.argv) < 2: sys.exit("Usage: controller.py <action> [args...]")
    action = sys.argv[1]

    # Load stations data for archive overlay functions
    try:
        with open(STATIONS_FILE, 'r') as f:
            stations_data = json.load(f)
    except Exception as e:
        stations_data = {}
        logging.warning(f"Could not load stations data: {e}")

    # Helper function to prevent FFmpeg crashes by ensuring stream directory exists
    def handle_start_stream():
        task_id = sys.argv[2]
        station_id = sys.argv[3]
        cam_num = sys.argv[4]
        resolution = sys.argv[5]
        hevc_supported = sys.argv[6].lower() == 'true'
        user_ip = sys.argv[7]
        
        # Explicitly create the stream subdirectory (e.g., streams/ams173_1_hires)
        stream_subdir = os.path.join(STREAM_DIR, f"{station_id}_{cam_num}_{resolution}")
        os.makedirs(stream_subdir, exist_ok=True)
        
        start_stream_relay(task_id, station_id, cam_num, resolution, user_ip, hevc_supported)

    # --- Global Exception Handling Wrapper ---
    try:
        actions = {
            "get_stations": lambda: print(open(STATIONS_FILE).read()),
            "get_camera_fovs": lambda: print(json.dumps(get_camera_fovs())),
            "get_kp_data": lambda: print(get_kp_data()),
            "get_lightning_data": lambda: print(json.dumps(get_lightning_data(sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().strftime('%Y-%m-%d')))),
            "get_meteor_data": lambda: print(json.dumps(get_meteor_data())),
            "get_station_stats": lambda: print(json.dumps(get_station_stats(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else None))),
            "fetch_grid": lambda: print(json.dumps(fetch_grid_file(sys.argv[2], sys.argv[3], sys.argv[4]))),
            "fetch_annotation": lambda: print(json.dumps(fetch_annotation_file(sys.argv[2], sys.argv[3], sys.argv[4]))),
            "fetch_archive_grid": lambda: print(json.dumps(get_archive_grid_overlay(
                sys.argv[2], sys.argv[3], sys.argv[4], stations_data))),
            "fetch_stitch_cam_boundaries": lambda: print(json.dumps(get_stitch_cam_boundaries(
                sys.argv[2], sys.argv[3], stations_data,
                resolution=sys.argv[4] if len(sys.argv) > 4 else 'hires'))),
            "fetch_archive_annotation": lambda: print(json.dumps(get_archive_annotation_overlay(
                sys.argv[2], sys.argv[3], sys.argv[4], stations_data))),
            "enhance_filter": lambda: print(json.dumps({"image": apply_enhance_filter(sys.argv[2], int(sys.argv[3]))})),
            "download": lambda: main_download_coordinator(sys.argv[2], sys.argv[3], sys.argv[4]),
            "_internal_download_station": lambda: download_for_single_station(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]),
            "_internal_start_stream": handle_start_stream,
            "stop_stream": lambda: stop_stream_relay(sys.argv[2]),
            "_internal_blend_overlay": lambda: sys.exit(0 if apply_ffmpeg_overlay(sys.argv[2], sys.argv[3], sys.argv[4]) else 1)
        }

        if action in actions:
            actions[action]()
        elif action == "get_page":
            lang_data = json.loads(sys.argv[2])
            print(render_template(HTML_TEMPLATE, lang_data))
        elif action in ["cancel", "cleanup"]:
            master_task_id = sys.argv[2]
            if action == "cancel":
                pid_file = os.path.join(LOCK_DIR, f"{master_task_id}.pid")
                if os.path.exists(pid_file):
                    try:
                        with open(pid_file, 'r') as f: pid = int(f.read().strip())
                        logging.info(f"Task {master_task_id} - Cancellation requested. Killing coordinator PID: {pid}")
                        os.kill(pid, signal.SIGTERM)
                    except (IOError, ValueError, ProcessLookupError) as e:
                        logging.warning(f"Task {master_task_id} - Could not kill coordinator process: {e}")
            
            time.sleep(1) 
            for f in os.listdir(LOCK_DIR):
                if f.startswith(master_task_id):
                    try: os.remove(os.path.join(LOCK_DIR, f))
                    except OSError: pass
            for f in os.listdir(DOWNLOAD_DIR):
                if f.endswith(".part") or ".tmp" in f:
                    try: os.remove(os.path.join(DOWNLOAD_DIR, f))
                    except OSError: pass
    except Exception as e:
        # Prevent "Unexpected end of JSON input" by ensuring valid JSON error is printed
        # Note: Do not print for 'download' action as it runs in background/detached
        if action != 'download' and action != '_internal_download_station':
            error_json = json.dumps({"error": f"Internal Server Error: {str(e)}"})
            print(error_json)
        sys.exit(1)

if __name__ == "__main__":
    main()
