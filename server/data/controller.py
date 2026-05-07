#!/usr/bin/env python3

import sys
import json
import os
import subprocess
import logging
import time
import shutil
import re
import signal
from datetime import datetime, timedelta, timezone
from PIL import Image

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
FILE_TYPE_LIMITS = {'lowres': 300, 'hires': 100, 'image': 300, 'image_lowres': 600, 'image_long': 100, 'image_lowres_long': 300}
AVG_FILE_SIZES_MB = {'lowres': 2, 'hires': 15, 'image': 1, 'image_lowres': 0.2, 'image_long': 1, 'image_lowres_long': 0.2}
TOTAL_QUOTA_LIMIT_MB = 2048 
PER_SITE_QUOTA_LIMIT_MB = 1024 

# Ensure critical directories exist
for d in [LOG_DIR, LOCK_DIR, DOWNLOAD_DIR, CACHE_DIR, STREAM_DIR]: os.makedirs(d, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(LOG_FILE)])

# --- Inline Helper: Robust Video Probing ---
# Defined here to prevent ImportError crashes if media_processor.py is out of sync.
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
        get_archive_grid_overlay, get_archive_annotation_overlay
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
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<link rel="stylesheet" href="style.css?v=20260412">
</head><body>
<div class="container">
    <header>
        <div id="language-selector">
            <span data-lang="nb_NO" title="Norsk">🇳🇴</span>
            <span data-lang="en_GB" title="English">🇬🇧</span>
            <span data-lang="de_DE" title="Deutsch">🇩🇪</span>
            <span data-lang="cs_CZ" title="Čeština">🇨🇿</span>
            <span data-lang="fi_FI" title="Suomi">🇫🇮</span>
        </div>
        <h1>__{{page_title}}__</h1>
    </header>
    <main class="main-content">
        <div id="map-panel">
            <h2>__{{map_panel_title}}__</h2>
            <div id="map"></div>
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
                <div class="form-group time-group">
                    <div><label for="length">__{{length_label}}__</label><div class="select-stepper"><button type="button" id="length-prev-btn" class="date-nav-btn" aria-label="Previous length">‹</button><select id="length" name="length" required><option value="" disabled selected>--</option></select><button type="button" id="length-next-btn" class="date-nav-btn" aria-label="Next length">›</button></div></div>
                    <div><label for="interval">__{{interval_label}}__</label><div class="select-stepper"><button type="button" id="interval-prev-btn" class="date-nav-btn" aria-label="Previous interval">‹</button><select id="interval" name="interval" required><option value="" disabled selected>--</option></select><button type="button" id="interval-next-btn" class="date-nav-btn" aria-label="Next interval">›</button></div></div>
                </div>
                <fieldset class="form-group">
                    <legend>__{{camera_legend}}__</legend>
                    <div class="checkbox-group">
                        <label><input type="checkbox" name="cameras" value="1" checked> 1</label>
                        <label><input type="checkbox" name="cameras" value="2" checked> 2</label>
                        <label><input type="checkbox" name="cameras" value="3" checked> 3</label>
                        <label><input type="checkbox" name="cameras" value="4" checked> 4</label>
                        <label><input type="checkbox" name="cameras" value="5" checked> 5</label>
                        <label><input type="checkbox" name="cameras" value="6" checked> 6</label>
                        <label><input type="checkbox" name="cameras" value="7" checked> 7</label>
                    </div>
                </fieldset>
                <fieldset class="form-group combined-file-type">
                    <div class="primary-type-group">
                         <label><input type="radio" name="primary_file_type" value="video"> __{{video_radio}}__</label>
                         <label><input type="radio" name="primary_file_type" value="image" checked> __{{image_radio}}__</label>
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
<div class="full-width-container" id="lightning-panel-container"><div class="container" id="lightning-panel"><h2>__{{lightning_panel_title}}__</h2><div id="lightning-list"><p style="color: #6c757d; margin: 0;">__{{loading_lightning}}__</p></div></div></div>
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
<script src="main.js" type="module"></script>
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
    def __init__(self, task_id, station_id, station_code, cam, time_utc, file_type, all_pass_data, pto_data_cache, hevc_supported=False, translations=None):
        self.task_id, self.station_id, self.station_code, self.cam, self.time_utc, self.file_type = task_id, station_id, station_code, cam, time_utc, file_type
        self.all_pass_data, self.pto_data_cache, self.hevc_supported = all_pass_data, pto_data_cache, hevc_supported
        self.translations = translations or {}
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
        base_name_with_type = f"{base_name}_{self.file_type}"
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
        command = ["scp", "-B", "-o", "ConnectTimeout=300", f"{self.station_id}:{remote_path}", temp_path]
        subprocess.run(command, check=True, timeout=360, capture_output=True)
        os.rename(temp_path, local_path)
        self.total_bytes_downloaded += os.path.getsize(local_path)

    def _transcode_to_h264_blocking(self, input_hevc_path, output_h264_path):
        logging.info(f"Task {self.task_id} - Transcoding {os.path.basename(input_hevc_path)} to H.264...")
        temp_output = output_h264_path + ".part"
        try:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", input_hevc_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-f", "mp4", "-y", temp_output]
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            os.rename(temp_output, output_h264_path)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logging.error(f"Task {self.task_id} - H.264 transcoding failed: {e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') else e}")
            if os.path.exists(temp_output): os.remove(temp_output)
            return False

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
            # If we need H.264, it's missing, but HEVC source exists: Transcode locally.
            if not self.hevc_supported and os.path.exists(self.output_filepath_hevc) and not os.path.exists(self.output_filepath_h264):
                if self._transcode_to_h264_blocking(self.output_filepath_hevc, self.output_filepath_h264):
                    return self.output_filepath_h264
                else:
                    # Transcode failed. Return HEVC if available so client gets something.
                    if os.path.exists(self.output_filepath_hevc): return self.output_filepath_hevc

            # 3. Standard Return:
            if self.hevc_supported and os.path.exists(self.output_filepath_hevc): return self.output_filepath_hevc
            if os.path.exists(self.output_filepath_h264): return self.output_filepath_h264

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

                final_source_video = video_filepath if os.path.exists(video_filepath) else (hevc_filepath if os.path.exists(hevc_filepath) else None)
                if final_source_video:
                    stack_script = os.path.join(BASE_DIR, 'stack.py')
                    if os.path.exists(stack_script):
                        command = [sys.executable, stack_script, final_source_video, "-o", self.output_filepath]
                        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
                    else:
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
        
            if codec == 'hevc' and not self.hevc_supported:
                self._transcode_to_h264_blocking(final_path, self.output_filepath_h264)
            
            return self.output_filepath_hevc if self.hevc_supported and os.path.exists(self.output_filepath_hevc) else self.output_filepath_h264

    def process(self):
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
        except Exception as e:
            self.errors.append(f"error_for_camera|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
            logging.error(f"Processing error: {e}", exc_info=True)

    def get_final_result(self, final_filepath, is_flight=False):
        if not os.path.exists(final_filepath): return None
        final_filename = os.path.basename(final_filepath)
        has_overlay = "_overlay" in final_filename
        base_name_part = '_'.join(final_filename.split('_')[:4])
        
        alternatives = [{"url": f"download/{f}", "name": f} for f in os.listdir(DOWNLOAD_DIR) if f.startswith(base_name_part) and f != final_filename and '_thumb.' not in f and '_track.png' not in f]
        result = {"url": f"download/{final_filename}", "name": final_filename, "utc_time_iso": self.time_utc.isoformat(), "alternatives": alternatives}
        
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

        if 'camera_views' in data and data['camera_views']:
            for view in data['camera_views']:
                start = datetime.fromisoformat(view['start_utc']).replace(second=0, microsecond=0)
                end = datetime.fromisoformat(view['end_utc'])
                while start <= end:
                    files_to_process.append({'time': start, 'cam': view['camera']})
                    start += timedelta(minutes=1)
        else:
            start_time = datetime.strptime(f"{data['date']} {data['hour']}:{data['minute']}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
            for i in range(int(data['length'])):
                for cam in data['cameras']:
                    files_to_process.append({'time': start_time + timedelta(minutes=i*int(data['interval'])), 'cam': int(cam)})

        files_iterator = iter(enumerate(files_to_process))
        current_file_idx, current_file_item = next(files_iterator, (None, None))
        processing_done = False

        master_lock_file = os.path.join(LOCK_DIR, f"{master_task_id}.lock")
        while not processing_done:
            if not os.path.exists(master_lock_file):
                logging.warning(f"Worker {task_id} - Master task lock file not found. Terminating.")
                break
                
            if current_file_item:
                update_status(station_status_file, "progress", {"step": current_file_idx, "total": len(files_to_process), "message": f"status_processing_file_of_total|i={current_file_idx+1},total={len(files_to_process)}", "files": results, "errors": errors})
                
                proc = FileProcessor(task_id, station_id, stations[station_id]['station']['code'], current_file_item['cam'], current_file_item['time'], data['file_type'], pass_data_list, pto_data_cache, data.get('hevc_supported', False), translations=translations)
                job = proc.process()
                
                errors.extend(proc.errors)
                total_bytes += proc.total_bytes_downloaded
                if proc.is_blending_job and job:
                    job['time_key'], job['processor'] = current_file_item['time'].strftime('%H:%M'), proc
                    blending_jobs.append(job)
                elif job:
                    results.setdefault(current_file_item['time'].strftime('%H:%M'), []).append(job)
                    update_status(station_status_file, "progress", {"step": current_file_idx + 1, "total": len(files_to_process), "message": f"status_processing_file_of_total|i={current_file_idx+1},total={len(files_to_process)}", "files": results, "errors": errors})
                
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
            
            if blends_finished_this_cycle:
                update_status(station_status_file, "progress", {"step": current_file_idx or len(files_to_process), "total": len(files_to_process), "message": f"status_waiting_for_blend|count={len(blending_jobs)}", "files": results, "errors": errors})

            if current_file_item is None and not blending_jobs:
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
            num_files = sum(round((datetime.fromisoformat(v['end_utc']) - datetime.fromisoformat(v['start_utc'])).total_seconds() / 60) + 1 for v in active_pass_data.get('camera_views', [])) if active_pass_data else len(station_ids) * len(data.get('cameras', [])) * int(data.get('length', 1))
            if num_files > limit: raise ValueError(f"error_too_many_files|num_files={num_files},limit={limit}")
            
            if not active_pass_data:
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
                    num_files_for_station = sum(round((datetime.fromisoformat(v['end_utc']) - datetime.fromisoformat(v['start_utc'])).total_seconds() / 60) + 1 for v in active_pass_data.get('camera_views', []) if v['station_id'] == station_id) if active_pass_data else len(data.get('cameras', [])) * int(data.get('length', 1))
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
            "fetch_archive_annotation": lambda: print(json.dumps(get_archive_annotation_overlay(
                sys.argv[2], sys.argv[3], sys.argv[4], stations_data))),
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
