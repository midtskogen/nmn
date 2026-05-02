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

# Import from our new shared utility library
# Imports utility functions shared across multiple backend scripts.
from shared_utils import atomic_json_rw, update_status, uniqid

# --- Configuration (specific to streaming) ---
# Establishes base paths for all necessary directories and configuration files.
def _find_base_dir():
    candidate = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(candidate, 'download')):
        return candidate
    for d in [os.path.join(os.path.dirname(os.path.dirname(candidate)), 'data'),
              '/var/www/html/data', '/data/httpd/norskmeteornettverk.no/data']:
        if os.path.isdir(os.path.join(d, 'download')):
            return d
    return candidate
BASE_DIR = _find_base_dir()
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
        with atomic_json_rw(STREAM_TIME_TRACKER_FILE) as tracker:
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
    
    # Kills the SSH tunnel and ffmpeg processes by their PIDs.
    for pid_name, pid in data.get("pids", {}).items():
        try:
            os.kill(pid, signal.SIGKILL)
            logging.info(f"Killed {pid_name} (PID: {pid}) for task {task_id}.")
        except OSError:
            logging.info(f"{pid_name} (PID: {pid}) for task {task_id} was already gone.")
            
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
                        try:
                            os.kill(pid, signal.SIGKILL)
                            logging.info(f"{log_prefix} Killed stale {pid_name} (PID: {pid}).")
         
                        except OSError: pass
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
        playlist_ready = {"ts": None, "error": None}
        def _playlist_waiter():
            try:
                _wait_for_playlist(ffmpeg_process, playlist_path, log_prefix, timeout_seconds=10)
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
            # 1. Check for Hot-Swap Command
            should_switch = False
            try:
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
