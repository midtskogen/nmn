#!/usr/bin/env python3

import os
import json
import subprocess
import logging
import time
import shutil
import signal
import socket
from datetime import datetime, timezone

# Import from our new shared utility library
# Imports utility functions shared across multiple backend scripts.
from shared_utils import atomic_json_rw, update_status, uniqid

# --- Configuration (specific to streaming) ---
# Establishes base paths for all necessary directories and configuration files.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_DIR = os.path.join(BASE_DIR, 'locks')
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'download')
STREAM_DIR = os.path.join(BASE_DIR, 'streams')
STATIONS_FILE = os.path.join(BASE_DIR, 'stations.json')
STREAM_TIME_TRACKER_FILE = os.path.join(BASE_DIR, 'stream_time_tracker.json')

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
        # Securely copies the grid.png file from the remote station.
        local_filename = f"grid_{station_id}_cam{camera_num}_{uniqid()}.png"
        local_filepath = os.path.join(DOWNLOAD_DIR, local_filename)
        command = ["scp", "-B", "-o", "ConnectTimeout=10", f"{station_id}:/meteor/cam{camera_num}/grid.png", local_filepath]
        subprocess.run(command, check=True, timeout=40, capture_output=True)
        logging.info(f"{log_prefix} Fetched grid to {local_filepath}")

        # Updates the stream's status file with the path to the downloaded grid.
        with atomic_json_rw(status_file, stream_task_id) as data:
            data['grid_local_path'] = local_filepath
        
        return {"success": True, "grid_url": f"download/{local_filename}"}

    except subprocess.TimeoutExpired:
        logging.error(f"{log_prefix} SCP timed out.")
        return {"success": False, "error": "error_grid_fetch_timeout"}
    except subprocess.CalledProcessError as e:
        logging.error(f"{log_prefix} SCP failed. Stderr: {e.stderr.decode()}")
        return {"success": False, "error": "error_grid_not_found"}
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
            
    # Deletes the temporary grid file.
    if grid_path := data.get("grid_local_path"):
        if os.path.exists(grid_path):
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
    time.sleep(4) # Wait briefly to see if the process exits immediately.
    # If the SSH process has already terminated, the tunnel failed.
    if process.poll() is not None:
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore').strip()
        error_message = f"error_ssh_tunnel_failed_with_msg|error={stderr_output}" if stderr_output else "error_ssh_process_terminated"
        logging.error(f"{log_prefix} FAILED. {error_message}")
        raise RuntimeError(error_message)

    logging.info(f"{log_prefix} Tunnel appears stable (PID: {process.pid}) on port {local_port}.")
    return process, local_port


def _start_ffmpeg_relay(local_port, resolution, stream_dir, hevc_supported, log_prefix):
    """
    Starts an ffmpeg process that connects to the local end of the SSH tunnel,
    and re-streams the video into an HLS (HTTP Live Streaming) format.
    """
    stream_index = '1' if resolution == 'lowres' else '0' # Camera provides different stream indexes for hi/low res.
    rtsp_url = f"rtsp://127.0.0.1:{local_port}/user=admin&password=&channel=1&stream={stream_index}.sdp"
    playlist_path = os.path.join(stream_dir, 'playlist.m3u8')

    # Verifies that the local port is open, confirming the SSH tunnel is active.
    try:
        with socket.create_connection(("127.0.0.1", local_port), timeout=2):
            logging.info(f"{log_prefix} Port {local_port} is open. SSH tunnel is confirmed active.")
    except (socket.timeout, ConnectionRefusedError) as e:
        raise RuntimeError(f"error_local_tunnel_inactive|port={local_port}")

    # Probes the video stream to detect its codec (HEVC or H.264).
    try:
        ffprobe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", "-rtsp_transport", "tcp", rtsp_url]
        probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=20, check=True)
        codec_name = probe_result.stdout.strip()
        logging.info(f"{log_prefix} Detected video codec: {codec_name}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logging.error(f"{log_prefix} ffprobe failed to determine codec, defaulting to copy. Error: {e.stderr if hasattr(e, 'stderr') else e}")
        codec_name = 'h264'

  
    video_opts = ["-c:v"]
    # If the source is HEVC but the user's browser doesn't support it, transcode to H.264.
    if codec_name.lower() == 'hevc' and not hevc_supported:
        logging.info(f"{log_prefix} Transcoding HEVC to H.264 for browser compatibility.")
        video_opts.extend(["libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-force_key_frames", "expr:gte(t,n_forced*1)"])
    else:
        # Otherwise, copy the video stream without re-encoding to save CPU.
        logging.info(f"{log_prefix} Codec is '{codec_name}'. Using copy.")
        video_opts.append("copy")
        
    # The ffmpeg command to create the HLS stream (playlist.m3u8 and .ts segment files).
    ffmpeg_command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", rtsp_url, *video_opts, "-an", "-f", "hls", "-hls_time", "1", "-hls_list_size", "2", "-hls_flags", "delete_segments", playlist_path]
    
    logging.info(f"{log_prefix} Starting ffmpeg with command: {' '.join(ffmpeg_command)}")
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Waits for ffmpeg to create the HLS playlist file, confirming it's working.
    for _ in range(30):
        if process.poll() is not None: raise RuntimeError("FFmpeg process failed to start or connect.")
        if os.path.exists(playlist_path) and os.path.getsize(playlist_path) > 0: return process, playlist_path
        time.sleep(1)
    raise RuntimeError("FFmpeg is running but failed to create a valid playlist file.")


def start_stream_relay(task_id, station_id, camera_num, resolution, user_ip, hevc_supported=False):
    """
    Main function to orchestrate the entire live stream setup process.
    This function is intended to be long-running and will manage the stream's lifecycle.
    """
    stream_start_time = time.time()
    status_file = os.path.join(LOCK_DIR, f"{task_id}.json")
    log_prefix = f"StreamRelay {task_id} -"
    logging.info(f"{log_prefix} Request for {station_id}/{camera_num}/{resolution}. HEVC support: {hevc_supported}")
    
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
   
        ssh_process, local_port = _start_ssh_tunnel(station_id, camera_num)

        update_status(status_file, "connecting_camera", {"message": "status_connecting_camera"})
        ffmpeg_process, _ = _start_ffmpeg_relay(local_port, resolution, stream_dir, hevc_supported, log_prefix)
        
        timeout_seconds = _get_timeout_for_station(station_id, resolution, stations_data)
        update_data = {
            "message": "status_stream_ready", "pids": {"ssh_pid": ssh_process.pid, "ffmpeg_pid": ffmpeg_process.pid},
            "stream_dir": stream_dir, "station_id": station_id, "timeout_seconds": timeout_seconds, "resolution": resolution
        }
        update_status(status_file, "ready", update_data)
        
        # This loop keeps the script alive, monitoring the stream processes until the timeout is reached
        # or a process dies or is stopped externally.
        end_time = time.time() + timeout_seconds
        while time.time() < end_time:
            if not os.path.exists(status_file) or ssh_process.poll() is not None or ffmpeg_process.poll() is not None:
                logging.warning(f"A stream process for task {task_id} terminated or was stopped.")
                break
            time.sleep(2)

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
