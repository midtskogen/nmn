#!/Usr/bin/env python3

import sys
import json
import os
import subprocess
import logging
import time
import shutil
import re
from datetime import datetime, timedelta, timezone
from PIL import Image

# --- Import from our new refactored modules ---
# Imports utility functions shared across multiple backend scripts.
from shared_utils import (
    uniqid, update_status, update_quota_tracker, trim_log_file, cleanup_old_files
)
# Imports functions for media processing like image stacking, overlays, and thumbnails.
from media_processor import (
    stack_images, draw_track_on_image, apply_ffmpeg_overlay, create_thumbnail,
    probe_video_codec, PTO_MAPPER_AVAILABLE
)
# Imports functions for managing live video stream relays.
from live_streamer import (
    start_stream_relay, stop_stream_relay, fetch_grid_file
)
# Imports functions to fetch external data sources like Kp-index, lightning, etc.
from data_fetchers import (
    get_kp_data, get_lightning_data, get_meteor_data, get_camera_fovs
)

# --- Configuration & Setup ---
# Establishes base paths for all necessary directories and configuration files.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
CLEANUP_AGE_DAYS = 7 # Age in days for cleaning up old downloaded files.
MAX_LOG_LINES = 10000 # Maximum number of lines to keep in the activity log.
# --- Security & Download Configuration ---
# Defines limits to prevent abuse and manage server resources.
MAX_STATIONS_PER_REQUEST = 10 # Max stations in a single download request.
MAX_SEQUENCE_LENGTH = 60 # Max number of files in a time-sequence download.
MAX_SEQUENCE_INTERVAL = 60 # Max interval in minutes for a time-sequence download.
MIN_FREE_DISK_SPACE_GB = 1 # Not currently implemented, but planned.
MAX_FILE_SIZE_FOR_THUMBNAIL_MB = 200 # Skips thumbnail generation for very large files.
# Defines the maximum number of files that can be requested for each file type.
FILE_TYPE_LIMITS = {'lowres': 300, 'hires': 100, 'image': 300, 'image_lowres': 600, 'image_long': 100, 'image_lowres_long': 300}
# Defines average file sizes used for estimating quota usage.
AVG_FILE_SIZES_MB = {'lowres': 2, 'hires': 15, 'image': 1, 'image_lowres': 0.2, 'image_long': 1, 'image_lowres_long': 0.2}
# Defines total download quotas per day for stations that have quotas enabled.
TOTAL_QUOTA_LIMIT_MB = 2048 # Per-station quota.
PER_SITE_QUOTA_LIMIT_MB = 1024 # Per-user-IP, per-station quota.
# --- Setup Directories & Logging ---
# Creates necessary directories if they don't exist and configures the logging system.
for d in [LOG_DIR, LOCK_DIR, DOWNLOAD_DIR, CACHE_DIR, STREAM_DIR]: os.makedirs(d, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(LOG_FILE)])


HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><title>__{{html_title}}__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<link rel="stylesheet" href="style.css">
</head><body>
<div id="language-selector">
    <span data-lang="nb_NO" title="Norsk">🇳🇴</span>
    <span data-lang="en_GB" title="English">🇬🇧</span>
    <span data-lang="de_DE" title="Deutsch">🇩🇪</span>
</div><div class="container">
    <header><h1>__{{page_title}}__</h1></header>
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
                        <input type="text" id="date-display" placeholder="YYYY-MM-DD" readonly>
                        <input type="date" id="date" name="date" required>
                   </div>
                   <button type="button" id="last-night-btn" title="__{{last_night_button_title}}__">__{{last_night_button}}__</button>
                   <button type="button" id="now-button" title="__{{now_button_title}}__">__{{now_button}}__</button>
               </div>
                <div class="form-group time-group"><div><label for="hour">__{{hour_label}}__</label><select id="hour" name="hour" required><option value="" disabled selected>--</option></select></div>
                <div><label for="minute">__{{minute_label}}__</label><select id="minute" name="minute" required><option value="" disabled selected>--</option></select></div></div>
                <div class="form-group time-group">
                    <div><label for="length">__{{length_label}}__</label><select id="length" name="length" required><option value="" disabled selected>--</option></select></div>
                    <div><label for="interval">__{{interval_label}}__</label><select id="interval" name="interval" required><option value="" disabled selected>--</option></select></div>
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
    """
    Fills in gaps in a satellite or aircraft track if the time between points is too large.
    This creates a smoother path for overlay generation by ensuring a minimum temporal resolution.
    """
    if not track_points or len(track_points) < 2: return track_points
    new_track = [track_points[0]]
 
    # Identify all numeric keys (like lat, lon, alt, az) that should be interpolated.
    keys_to_interpolate = [k for k, v in track_points[0].items() if isinstance(v, (int, float))]
    for i in range(len(track_points) - 1):
        p1, p2 = track_points[i], track_points[i+1]
        t1 = datetime.fromisoformat(p1['time'].replace('Z', '+00:00'))
        t2 = datetime.fromisoformat(p2['time'].replace('Z', '+00:00'))
        time_diff_sec = (t2 - t1).total_seconds()
        
        # If the time gap exceeds the maximum allowed interval, insert new points.
        if time_diff_sec > max_interval_sec:
            num_new_points = int(time_diff_sec // max_interval_sec)
            for j in range(1, num_new_points + 1):
                interp_factor = j / (num_new_points + 1)
                new_point_time = t1 + timedelta(seconds=time_diff_sec * interp_factor)
                
   
                new_point = {'time': new_point_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}
                # Perform linear interpolation for each numeric key.
                for key in keys_to_interpolate:
                    val1, val2 = p1.get(key, 0), p2.get(key, 0)
                    new_point[key] = val1 + (val2 - val1) * interp_factor
                new_track.append(new_point)
        
        new_track.append(p2)
    return new_track


class FileProcessor:
    """Handles the processing of a single file for a specific camera and time."""
    def __init__(self, task_id, station_id, station_code, cam, time_utc, file_type, all_pass_data, pto_data_cache, hevc_supported=False):
        self.task_id, self.station_id, self.station_code, self.cam, self.time_utc, self.file_type = task_id, station_id, station_code, cam, time_utc, file_type
        self.all_pass_data, self.pto_data_cache, self.hevc_supported = all_pass_data, pto_data_cache, hevc_supported
        self.errors, self.total_bytes_downloaded, self.is_blending_job = [], 0, False
        # Check if this file corresponds to a known satellite or aircraft pass.
        self.relevant_pass = self._find_relevant_pass()
        # Determine all necessary file paths based on the request.
        self._determine_paths_and_types()

    def _find_relevant_pass(self):
        """Cross-references the file's time and camera with cached pass data."""
        if not self.all_pass_data or not PTO_MAPPER_AVAILABLE: return None
        for p in self.all_pass_data:
            for cv in p.get('camera_views', []):
                # Check if the station, camera, and time match a known camera view event.
                if cv['station_id'] == self.station_id and cv['camera'] == self.cam and datetime.fromisoformat(cv['start_utc']).replace(second=0, microsecond=0) <= self.time_utc < datetime.fromisoformat(cv['end_utc']):
                    return p
        return None

    def _determine_paths_and_types(self):
        """Constructs all potential output file paths based on the file type and request details."""
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
         
            # Define paths for both H.264 and HEVC video formats.
            self.output_filepath_h264 = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}.mp4")
            self.overlay_filepath_h264 = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}{overlay_suffix}.mp4")
            self.output_filepath_hevc = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}_hevc.mp4")
            self.overlay_filepath_hevc = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}{overlay_suffix}_hevc.mp4")
        # Path for the temporary transparent PNG track overlay.
        self.track_filepath = os.path.join(DOWNLOAD_DIR, f"{base_name_with_type}_{self.task_id}_track.png")

    def _scp_file(self, remote_path, local_path):
        """Securely copies a file from a remote station to a local temporary path."""
        temp_path = local_path + ".part"
        command = ["scp", "-B", "-o", "ConnectTimeout=300", f"{self.station_id}:{remote_path}", temp_path]
        subprocess.run(command, check=True, timeout=360, capture_output=True)
        os.rename(temp_path, local_path)
        self.total_bytes_downloaded += os.path.getsize(local_path)

    def _transcode_to_h264_blocking(self, input_hevc_path, output_h264_path):
      
        """Converts an HEVC (H.265) video file to the more widely supported H.264 format."""
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
        """
       Ensures the requested media file exists locally, downloading it if necessary.
        Handles transcoding from HEVC to H.264 if the user's browser lacks HEVC support.
        For "long integration" images, it downloads the source video and stacks it.
        """
        # For images, the logic is simple: just check if the file or its overlay version exists.
        if self.is_image:
            if os.path.exists(self.output_filepath) or os.path.exists(self.overlay_filepath): return self.output_filepath
        else: # For video, the logic is more complex due to multiple codecs.
            # Determine the preferred path based on browser support.
          
            preferred_path = self.output_filepath_hevc if self.hevc_supported else self.output_filepath_h264
            if os.path.exists(preferred_path): return preferred_path
            # If preferred (H.264) doesn't exist but the HEVC version does, transcode it.
            if not self.hevc_supported and os.path.exists(self.output_filepath_hevc) and not os.path.exists(self.output_filepath_h264):
                 if self._transcode_to_h264_blocking(self.output_filepath_hevc, self.output_filepath_h264):
               
                    return self.output_filepath_h264
            # If any version exists, return its path. The overlay logic will handle the rest.
            if os.path.exists(self.output_filepath_h264): return self.output_filepath_h264
            if os.path.exists(self.output_filepath_hevc): return self.output_filepath_hevc

        # If no local file exists, proceed to download from the remote station.
        t, source_prefix = self.time_utc, 'mini' if self.is_low_res else 'full'
        remote_dir = f"/meteor/cam{self.cam}/{t.strftime('%Y%m%d')}/{t.strftime('%H')}"
        if self.is_image:
            if self.is_long_integration:
                # For long integrations, first download the corresponding video.
                video_filename = f"{self.station_code}_cam{self.cam}_{t.strftime('%Y%m%d')}_{t.strftime('%H%M')}_{'lowres' if self.is_low_res else 'hires'}.mp4"
                video_filepath = os.path.join(DOWNLOAD_DIR, video_filename)
                if not os.path.exists(video_filepath): self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.mp4", video_filepath)
                # Then, call the stack.py script to create the stacked image.
                command = [sys.executable, os.path.join(BASE_DIR, 'stack.py'), video_filepath, "-o", self.output_filepath]
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
            else: # For regular images, just download the JPG.
                self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.jpg", self.output_filepath)
            return self.output_filepath if os.path.exists(self.output_filepath) else None
        else: # For video downloads
            temp_download_path = self.output_filepath_h264 + ".tmp"
            self._scp_file(f"{remote_dir}/{source_prefix}_{t.strftime('%M')}.mp4", temp_download_path)
            # Probe the downloaded video to determine if it's HEVC or H.264.
            codec = probe_video_codec(temp_download_path, self.task_id)
            final_path = self.output_filepath_hevc if codec == 'hevc' else self.output_filepath_h264
            os.rename(temp_download_path, final_path)
        
            
            # After downloading, if the file is HEVC but the browser doesn't support it,
            # we must transcode it to H.264 so the user can play it.
            if codec == 'hevc' and not self.hevc_supported:
                self._transcode_to_h264_blocking(final_path, self.output_filepath_h264)
            
            # Return the path to the format the browser prefers.
            return self.output_filepath_hevc if self.hevc_supported and os.path.exists(self.output_filepath_hevc) else self.output_filepath_h264

    def process(self):
        """Main processing logic for the file."""
        try:
            base_media_path = self._ensure_base_media_exists()
            if not base_media_path or not os.path.exists(base_media_path):
                self.errors.append(f"error_source_file|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
                return
            # If the file corresponds to a pass, prepare to generate an overlay.
            if self.relevant_pass:
                is_flight = 'flight_info' in self.relevant_pass
                # Interpolate track data for a smoother overlay appearance.
                if is_flight:
                    self.relevant_pass['ground_track'] = _interpolate_track(self.relevant_pass.get('ground_track', []), 15)
                for station_id, track in self.relevant_pass.get('station_sky_tracks', {}).items():
                    self.relevant_pass['station_sky_tracks'][station_id] = _interpolate_track(track, 15)
                is_hevc = '_hevc.mp4' in os.path.basename(base_media_path)
           
                overlay_filepath = (self.overlay_filepath_hevc if is_hevc else self.overlay_filepath_h264) if not self.is_image else self.overlay_filepath
                
                # Check if this station has a sky track for this pass and the overlay doesn't already exist.
                if self.relevant_pass.get('station_sky_tracks', {}).get(self.station_id) and not os.path.exists(overlay_filepath):
                    pass_info = self.relevant_pass.copy()
                    if is_flight: pass_info['satellite'] = pass_info.get('flight_info', {}).get('callsign', 'Flight').strip()
                    pass_info['sky_track'] = self.relevant_pass['station_sky_tracks'][self.station_id]
                    
     
                    if PTO_MAPPER_AVAILABLE:
                        from pto_mapper import get_pto_data_from_json
                        selector = f"{self.station_id.replace('ams', '')}:{self.cam}"
                        # Use a cache for PTO data to avoid re-reading the JSON file.
                        if selector not in self.pto_data_cache: self.pto_data_cache[selector] = get_pto_data_from_json(CAMERAS_FILE, selector)
                        
                        # Get media dimensions for the overlay.
                        if self.is_image:
                            with Image.open(base_media_path) as img: w, h = img.size
                        else: w, h = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", base_media_path]).decode().strip().split('x'))
                        
  
                        # Draw the track onto a transparent PNG image.
                        draw_track_on_image(self.pto_data_cache[selector], pass_info, self.track_filepath, target_w=w, target_h=h, is_flight=is_flight)
                        if os.path.exists(self.track_filepath):
                            # Launch the ffmpeg overlay process as a separate, non-blocking subprocess.
                            p = subprocess.Popen([sys.executable, __file__, "_internal_blend_overlay", base_media_path, self.track_filepath, overlay_filepath])
                            self.is_blending_job = True
                            return {"process": p, "overlay_filepath": overlay_filepath, "track_filepath": self.track_filepath}

            # If no overlay is needed or it already exists, proceed to finalize the result.
            final_overlay = (self.overlay_filepath_hevc if '_hevc' in base_media_path else self.overlay_filepath_h264) if not self.is_image else self.overlay_filepath
            final_path = final_overlay if os.path.exists(final_overlay) else base_media_path
            return self.get_final_result(final_path, is_flight=self.relevant_pass and 'flight_info' in self.relevant_pass)
        except Exception as e:
            self.errors.append(f"error_for_camera|cam={self.cam},time={self.time_utc.strftime('%H:%M')}")
            logging.error(f"Processing error: {e}", exc_info=True)

    def get_final_result(self, final_filepath, is_flight=False):
 
        """Gathers the final result dictionary for a processed file, including thumbnails and alternatives."""
        if not os.path.exists(final_filepath): return None
        final_filename = os.path.basename(final_filepath)
        has_overlay = "_overlay" in final_filename
        base_name_part = '_'.join(final_filename.split('_')[:4])
        
        # Find all other files with the same base name to list as alternatives (e.g., HEVC vs H264).
        alternatives = [{"url": f"download/{f}", "name": f} for f in os.listdir(DOWNLOAD_DIR) if f.startswith(base_name_part) and f != final_filename and '_thumb.' not in f and '_track.png' not in f]
        result = {"url": f"download/{final_filename}", "name": final_filename, "utc_time_iso": self.time_utc.isoformat(), "alternatives": alternatives}
        # Create a thumbnail for the file.
        if thumb := create_thumbnail(self.task_id, final_filepath, self.file_type, self.station_code, self.cam, has_overlay, is_flight, MAX_FILE_SIZE_FOR_THUMBNAIL_MB):
            result["thumb_url"] = f"download/{thumb}"
        return result

def download_for_single_station(task_id, station_id, json_payload_str):
    """
    Worker process that handles all file processing for a single station in a download request.
    This function is spawned as a subprocess by the main coordinator.
    """
    data = json.loads(json_payload_str)
    logging.info(f"Worker {task_id} for '{station_id}' Started.")
    station_status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
   
    with open(STATIONS_FILE, 'r') as f: stations = json.load(f)
    
    # Load satellite/aircraft pass data if available in the request or from the main cache.
    pass_data_list = [data[p] for p in ['pass_data', 'flight_pass_data'] if p in data and data[p]]
    if not pass_data_list and data.get('satellite_panel_enabled', False) and os.path.exists(PASS_CACHE_FILE):
        with open(PASS_CACHE_FILE, 'r') as f: pass_data_list = json.load(f).get("data", {}).get("passes", [])

    results, errors, blending_jobs, pto_data_cache, total_bytes = {}, [], [], {}, 0
    files_to_process = []
    # Determine the list of individual files to process based on the request type.
    if 'camera_views' in data and data['camera_views']:
        # If it's a pass download, create a file list from the camera view time ranges.
        for view in data['camera_views']:
            start = datetime.fromisoformat(view['start_utc']).replace(second=0, microsecond=0)
            end = datetime.fromisoformat(view['end_utc'])
            while start <= end:
                files_to_process.append({'time': start, 'cam': view['camera']})
                start += timedelta(minutes=1)
    else:
        # If it's a manual time range download, create the list from form inputs.
        start_time = datetime.strptime(f"{data['date']} {data['hour']}:{data['minute']}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
        for i in range(int(data['length'])):
            for cam in data['cameras']:
                files_to_process.append({'time': start_time + timedelta(minutes=i*int(data['interval'])), 'cam': int(cam)})

    # Process each file sequentially.
    for i, item in enumerate(files_to_process):
        update_status(station_status_file, "progress", {"step": i, "total": len(files_to_process), "message": f"status_processing_file_of_total|i={i+1},total={len(files_to_process)}", "files": results, "errors": errors})
        proc = FileProcessor(task_id, station_id, stations[station_id]['station']['code'], item['cam'], item['time'], data['file_type'], pass_data_list, pto_data_cache, data.get('hevc_supported', False))
        job = proc.process()
        
        errors.extend(proc.errors)
        total_bytes += proc.total_bytes_downloaded
        if proc.is_blending_job and job:
         
            # If the job is a non-blocking overlay blend, add it to a list to wait for later.
            job['time_key'], job['processor'] = item['time'].strftime('%H:%M'), proc
            blending_jobs.append(job)
        elif job:
            results.setdefault(item['time'].strftime('%H:%M'), []).append(job)

    # Wait for all non-blocking overlay blending jobs to complete.
    for i, job in enumerate(blending_jobs):
        update_status(station_status_file, "progress", {"step": len(files_to_process)-len(blending_jobs)+i, "total": len(files_to_process), "message": f"status_waiting_for_blend|count={len(blending_jobs)-i}", "files": results, "errors": errors})
        job['process'].wait() # Wait for the ffmpeg subprocess to finish.
        if job['process'].returncode == 0:
            if res := job['processor'].get_final_result(job['overlay_filepath'], 'flight_info' in job['processor'].relevant_pass):
                results.setdefault(job['time_key'], []).append(res)
        else: errors.append(f"error_blending_track|filename={os.path.basename(job['overlay_filepath'])}")
        if os.path.exists(job['track_filepath']): os.remove(job['track_filepath'])
    
    # Check if the request is for a stackable sequence (long integration images over consecutive minutes).
    is_stackable_request = (
        data['file_type'] in ['image_long', 'image_lowres_long'] and
        'length' in data and int(data['length']) > 1 and
        'interval' in data and int(data['interval']) == 1 and
        not any(k in data for k in ['pass_data', 'flight_pass_data'])
    )
    if is_stackable_request:
        logging.info(f"Worker {task_id} - Stackable request detected. Creating combined images.")
        images_by_camera = {}
     
        # Group the downloaded images by camera number.
        for time_key, files in results.items():
            for file_info in files:
                try:
                    parts = file_info['name'].split('_')
                    cam_num = int(parts[1].replace('cam', ''))
                    
                    images_by_camera.setdefault(cam_num, []).append({
                        'path': os.path.join(DOWNLOAD_DIR, file_info['name']),
                        'time': datetime.fromisoformat(file_info['utc_time_iso'])
                    })
                except (IndexError, ValueError): continue
        
 
        # For each camera with multiple images, create a combined stacked image.
        for cam_num, images in images_by_camera.items():
            if len(images) > 1:
                images.sort(key=lambda x: x['time'])
                time_range_label = f"{images[0]['time'].strftime('%H:%M')} - {images[-1]['time'].strftime('%H:%M')}"
                
                parts = os.path.basename(images[0]['path']).split('_')
         
                time_fn_part = f"{images[0]['time'].strftime('%H%M')}-{images[-1]['time'].strftime('%M')}"
                output_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{time_fn_part}_{data['file_type']}_stacked.jpg"
                output_filepath = os.path.join(DOWNLOAD_DIR, output_filename)
                
                if stack_images([img['path'] for img in images], output_filepath, task_id):
               
                    stacked_result = {
                        "url": f"download/{output_filename}", "name": output_filename,
                        "utc_time_iso": images[0]['time'].isoformat(), "alternatives": []
                    }
                   
                    if thumb := create_thumbnail(task_id, output_filepath, 'image', stations[station_id]['station']['code'], cam_num):
                        stacked_result["thumb_url"] = f"download/{thumb}"
                    # Add the new stacked image to the results under a time-range key.
                    results.setdefault(time_range_label, []).append(stacked_result)
                else:
                    errors.append(f"error_stacking_image|cam_num={cam_num}")

    # Final status update for this worker process.
    update_status(station_status_file, "complete", {"files": results, "errors": errors, "total_bytes_downloaded": total_bytes})
    logging.info(f"Worker {task_id} for '{station_id}' Completed.")

def main_download_coordinator(master_task_id, json_payload, user_ip):
    """
    Main coordinator for a download request. It validates the request, checks quotas,
    spawns worker subprocesses for each station, and aggregates the final results.
    """
    status_file = os.path.join(LOCK_DIR, f"{master_task_id}.json")
    logging.info(f"Coordinator {master_task_id} Started for IP {user_ip}.")
    open(os.path.join(LOCK_DIR, f"{master_task_id}.lock"), 'w').close()
    # Perform routine cleanup of logs and old downloaded files.
    trim_log_file(LOG_FILE, MAX_LOG_LINES, master_task_id)
    for d in [DOWNLOAD_DIR, LOG_DIR, CACHE_DIR]: cleanup_old_files(d, CLEANUP_AGE_DAYS, master_task_id, [os.path.basename(LOG_FILE)] if d == LOG_DIR else [])
    
    data = json.loads(json_payload)
    if 'crossing_data' in data: data['flight_pass_data'] = data.pop('crossing_data')
    active_pass_data = data.get('pass_data') or data.get('flight_pass_data')

    # When downloading a specific pass, the pass data contains all possible camera views.
    # This code filters those views down to only the ones the user actually selected in the form.
    # This prevents downloading files for cameras the user did not request.
    if active_pass_data:
        user_selected_stations = data.get("stations", [])
        try:
            user_selected_cameras = [int(c) for c in data.get('cameras', [])]
        except (ValueError, TypeError):
            user_selected_cameras = []
        
        if user_selected_stations and user_selected_cameras:
            original_views = active_pass_data.get('camera_views', [])
   
            filtered_views = [
                view for view in original_views 
                if view.get('station_id') in user_selected_stations and view.get('camera') in user_selected_cameras
            ]
            active_pass_data['camera_views'] = filtered_views

    # Determine the final list of station IDs to process.
    station_ids = list(set(v['station_id'] for v in active_pass_data.get('camera_views', []))) if active_pass_data else data.get("stations", [])
    
    aggregated_errors = []
    stations_to_process = []
    
    try:
        # --- Validation of the user's request against defined limits.
        if not station_ids: raise ValueError("error_no_station_selected")
        if len(station_ids) > MAX_STATIONS_PER_REQUEST: raise ValueError(f"error_too_many_stations|max={MAX_STATIONS_PER_REQUEST}")
        
        with open(STATIONS_FILE, 'r') as f: valid_stations = json.load(f)
        for sid in station_ids:
            if sid not in valid_stations: raise ValueError(f"error_invalid_station_id|sid={sid}")
        
      
        file_type = data.get('file_type', 'lowres')
        if file_type not in FILE_TYPE_LIMITS: raise ValueError(f"error_invalid_file_type|file_type={file_type}")
        
        limit = FILE_TYPE_LIMITS[file_type]
        num_files = 0
        if active_pass_data and 'camera_views' in active_pass_data:
            # For pass downloads, calculate the number of files based on the duration of camera views.
            num_files = sum(round((datetime.fromisoformat(v['end_utc']) - datetime.fromisoformat(v['start_utc'])).total_seconds() / 60) + 1 for v in active_pass_data.get('camera_views', []))
        else:
             # For manual downloads, calculate based on form inputs.
            num_files = len(station_ids) * len(data.get('cameras', [])) * int(data.get('length', 1))

        if num_files > limit: raise ValueError(f"error_too_many_files|num_files={num_files},limit={limit}")
        
        if not active_pass_data:
            if not (1 <= int(data.get('length', 0)) <= MAX_SEQUENCE_LENGTH): raise ValueError(f"error_invalid_length|max={MAX_SEQUENCE_LENGTH}")
            if not (1 <= int(data.get('interval', 0)) <= MAX_SEQUENCE_INTERVAL): raise ValueError(f"error_invalid_interval|max={MAX_SEQUENCE_INTERVAL}")
      
        # --- Quota check logic for stations with quotas enabled.
        quota_tracker = {}
        if os.path.exists(QUOTA_TRACKER_FILE):
            with open(QUOTA_TRACKER_FILE, 'r') as f:
                try: quota_tracker = json.load(f)
                except json.JSONDecodeError: logging.warning(f"Task {master_task_id} - Could not parse quota_tracker.json.")
        
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
        todays_usage = quota_tracker.get(today_str, {})
        total_quota_bytes = TOTAL_QUOTA_LIMIT_MB * 1024 * 1024
        per_site_quota_bytes = PER_SITE_QUOTA_LIMIT_MB * 1024 * 1024

        for station_id in station_ids:
            station_info = valid_stations.get(station_id, {})
            station_code = station_info.get("station", {}).get("code", station_id)

            if station_info.get("station", {}).get("quota"):
         
                station_usage_today = todays_usage.get(station_id, {"total": 0, "sites": {}})
                if isinstance(station_usage_today, int): station_usage_today = {"total": station_usage_today, "sites": {}}
                total_usage_bytes = station_usage_today.get("total", 0)
                site_usage_bytes = station_usage_today.get("sites", {}).get(user_ip, 0)
            
            
                # Estimate the size of the current request.
                avg_size_bytes = AVG_FILE_SIZES_MB.get(file_type, 2) * 1024 * 1024
                num_files_for_station = sum(round((datetime.fromisoformat(v['end_utc']) - datetime.fromisoformat(v['start_utc'])).total_seconds() / 60) + 1 for v in active_pass_data.get('camera_views', []) if v['station_id'] == station_id) if active_pass_data else len(data.get('cameras', [])) * int(data.get('length', 1))
                estimated_request_size = num_files_for_station * avg_size_bytes

                # Check against both per-user and total station quotas.
                if site_usage_bytes + estimated_request_size > per_site_quota_bytes:
                    aggregated_errors.append(f"error_user_quota_exceeded|limit={PER_SITE_QUOTA_LIMIT_MB},station_code={station_code}")
                    continue
                if total_usage_bytes + estimated_request_size > total_quota_bytes:
                    aggregated_errors.append(f"error_total_quota_exceeded|limit={TOTAL_QUOTA_LIMIT_MB},station_code={station_code}")
                    continue
        
            stations_to_process.append(station_id)

    except (ValueError, TypeError) as e:
        update_status(status_file, "error", {"message": str(e)})
        return

    if not stations_to_process:
        update_status(status_file, "complete", {"files": {}, "errors": aggregated_errors})
        
        logging.info(f"Coordinator {master_task_id} - No stations to process after quota check.")
        return

    sub_tasks = {}
    aggregated_files, quota_updates = {}, {}
    try:
        # For each station that passed validation, spawn a worker subprocess.
        for station_id in stations_to_process:
            sub_task_id = uniqid('task_')
            worker_payload = data.copy()
            # If it's a pass download, filter the camera views for this specific station worker.
            if active_pass_data: 
                worker_payload['camera_views'] = [v for v in active_pass_data.get('camera_views', []) if v['station_id'] == station_id]
            command = [sys.executable, __file__, '_internal_download_station', sub_task_id, station_id, json.dumps(worker_payload)]
            sub_tasks[sub_task_id] = {"station_id": station_id, "process": subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)}
        
        # Monitor the progress of all worker subprocesses.
        start_time = time.time()
        while time.time() - start_time < 1800: # 30 min timeout
            all_done, total_steps_done, total_steps_overall = True, 0, 0
            for task_id, task_info in sub_tasks.items():
                s_file = os.path.join(LOCK_DIR, f"{task_id}.json")
                if not os.path.exists(s_file):
                    all_done = False
                    continue
                with open(s_file, 'r') as f: s_data = json.load(f)
                if s_data.get("status") != "complete": all_done = False
                total_steps_done += s_data.get("step", 0);
                total_steps_overall += s_data.get("total", 1)
            
            # Update the main task's progress based on the aggregated progress of all workers.
            percentage_done = (total_steps_done / total_steps_overall) * 100 if total_steps_overall > 0 else (100 if all_done else 0)
            update_status(status_file, "progress", {"step": percentage_done, "total": 100, "message": f"status_processing_files|done={int(total_steps_done)},total={int(total_steps_overall)}"})
            
            if all_done: break
            time.sleep(1)
        
        # Once all workers are finished, collect their results.
        with open(STATIONS_FILE, 'r') as f: all_stations_data = json.load(f)
        for task_id, task_info in sub_tasks.items():
            s_file = os.path.join(LOCK_DIR, f"{task_id}.json")
            if os.path.exists(s_file):
                with open(s_file, 'r') as f: s_data = json.load(f)
                station_code = all_stations_data.get(task_info['station_id'], {}).get('station', {}).get('code', 'UNKNOWN')
         
                if s_data.get("files"): aggregated_files[station_code] = s_data["files"]
                if s_data.get("errors"): aggregated_errors.extend(f"{station_code}: {e}" for e in s_data["errors"])
                if s_data.get("total_bytes_downloaded", 0) > 0: quota_updates[task_info['station_id']] = quota_updates.get(task_info['station_id'], 0) + s_data["total_bytes_downloaded"]
    finally:
        # Final cleanup and status update.
        for task in sub_tasks.values():
            if task['process'].poll() is None: task['process'].kill()
        if quota_updates: update_quota_tracker(quota_updates, master_task_id, user_ip, QUOTA_TRACKER_FILE)
        update_status(status_file, "complete", {"files": aggregated_files, "errors": aggregated_errors})
        logging.info(f"Coordinator {master_task_id} finished.")

def render_template(template, lang_data):
    """Replaces placeholders in an HTML template using a regular expression."""
    def replace_match(match):
        key = match.group(1)
        # Return the value from lang_data if the key exists,
        # otherwise return the original placeholder to make errors visible.
        return str(lang_data.get(key, match.group(0)))

    # Use re.sub() to find all placeholders like __{{key}}__ and replace them.
    return re.sub(r'__\{\{([a-zA-Z0-9_]+)\}\}__', replace_match, template)

def main():
    """Main entry point for the script, dispatching actions based on command-line arguments."""
    if len(sys.argv) < 2: sys.exit("Usage: controller.py <action> [args...]")
    action = sys.argv[1]
    
    # A dispatch table maps the 'action' parameter from the URL to a specific function.
    actions = {
        "get_stations": lambda: print(open(STATIONS_FILE).read()),
        "get_camera_fovs": lambda: print(json.dumps(get_camera_fovs())),
        "get_kp_data": lambda: print(get_kp_data()),
        "get_lightning_data": lambda: print(json.dumps(get_lightning_data(sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().strftime('%Y-%m-%d')))),
        "get_meteor_data": lambda: print(json.dumps(get_meteor_data())),
        "fetch_grid": lambda: print(json.dumps(fetch_grid_file(sys.argv[2], sys.argv[3], sys.argv[4]))),
        "download": lambda: main_download_coordinator(sys.argv[2], sys.argv[3], sys.argv[4]),
        "_internal_download_station": lambda: download_for_single_station(sys.argv[2], sys.argv[3], sys.argv[4]),
      
        "_internal_start_stream": lambda: start_stream_relay(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[7], sys.argv[6].lower() == 'true'),
        "stop_stream": lambda: stop_stream_relay(sys.argv[2]),
        # Internal action for the non-blocking overlay blending subprocess.
        "_internal_blend_overlay": lambda: sys.exit(0 if apply_ffmpeg_overlay(sys.argv[2], sys.argv[3], sys.argv[4]) else 1)
    }

    if action in actions:
        actions[action]()
    elif action == "get_page":
        # Serves the main HTML page for the web interface.
        lang_data = json.loads(sys.argv[2])
        print(render_template(HTML_TEMPLATE, lang_data))
    elif action in ["cancel", "cleanup"]:
        master_task_id = sys.argv[2]
        # Clean up lock files and temporary download files associated with a task.
        for f in os.listdir(LOCK_DIR):
            if f.startswith(master_task_id.split('_')[0]) or f.startswith('task_'):
                try: os.remove(os.path.join(LOCK_DIR, f))
                except OSError: pass
        for f in os.listdir(DOWNLOAD_DIR):
            if f.endswith(".part") or ".tmp" in f:
                try: os.remove(os.path.join(DOWNLOAD_DIR, f))
                except OSError: pass

if __name__ == "__main__":
    main()
