#!/usr/bin/python3
"""
AS7 Health Check Tool

This script performs a comprehensive diagnostic audit of an AllSky7 meteor detection
station. It checks for common configuration errors, operational issues, and system
health problems to ensure the station is functioning correctly.

The script is designed to be run as the 'root' user for a complete diagnosis, but it
can also be run as the 'ams' user or another user with limited checks.
"""

import os
import pwd
import subprocess
import json
import importlib
import socket
import sys
import time
import re
import glob
from collections import Counter
from shutil import which

class ErrorCatalog:
    """
    A centralized catalog of all possible diagnostic errors.

    Each error includes a type (failure/warning), a description, the reason it's
    a problem, and a recommended fix. This allows for consistent and detailed
    reporting in the final summary.
    """

    _catalog = {
        # User & Permissions
        "NO_AMS_USER": {"type": "failure", "description": "User 'ams' does not exist on the system.", "reason": "The entire software suite is designed to run under the 'ams' user for correct file permissions and ownership.", "fix": "The user 'ams' needs to be created. You may need to re-run the initial setup script."},
        "WRONG_USER": {"type": "warning", "description": "Running as a non-standard user ('{current_user}').", "reason": "The script is designed to be run as 'root' or 'ams'. Running as another user may lead to permission errors on some checks.", "fix": "For a complete diagnosis, run the script as 'root' using 'sudo' or as the 'ams' user."},
        "PERMISSION_DENIED": {"type": "warning", "description": "Could not perform a check due to insufficient permissions.", "reason": "This check requires root (sudo) privileges to read protected system files or other users' data.", "fix": "Re-run the script using 'sudo ./as7-health-check.py' for a complete diagnosis."},
        # Directories & Mounts
        "NOT_MOUNTED": {"type": "failure", "description": "'/mnt/ams2' is NOT a mounted filesystem.", "reason": "The system expects a large, separate drive mounted at /mnt/ams2 to store video data. If it's just a directory on the main drive, the OS drive will fill up quickly.", "fix": "Ensure your data disk is properly formatted, added to '/etc/fstab' to mount on boot, and then mounted with 'sudo mount -a'."},
        "DIR_MISSING": {"type": "failure", "description": "A required directory is missing.", "reason": "The software requires a specific directory structure to save videos, logs, calibration data, and processed meteors.", "fix": "Re-run the setup script section for creating directories, or create it manually ('sudo mkdir {path}') and ensure ownership is correct ('sudo chown ams:ams {path}')."},
        "BAD_OWNER": {"type": "warning", "description": "A directory has an incorrect owner (should be 'ams').", "reason": "The 'ams' user needs to be able to write files to this directory. Incorrect ownership will cause permission errors during operation.", "fix": "Run 'sudo chown -R ams:ams /mnt/ams2' and 'sudo chown -R ams:ams /home/ams/amscams' to correct ownership."},
        # Cloud Archive (Wasabi)
        "S3FS_MISSING": {"type": "failure", "description": "The 's3fs' command is not installed.", "reason": "The 's3fs-fuse' package is required to mount the Wasabi cloud storage.", "fix": "Follow the installation instructions to install 's3fs-fuse' from source."},
        "WASABI_CREDS_MISSING": {"type": "failure", "description": "Wasabi credentials file is missing.", "reason": "The credentials file ('/home/ams/amscams/conf/wasabi.txt') contains the key needed to access your cloud storage.", "fix": "Create the 'wasabi.txt' file in the correct location and add your Wasabi access key and secret key."},
        "WASABI_CREDS_INSECURE": {"type": "failure", "description": "Wasabi credentials file has insecure permissions ({perms}).", "reason": "The credentials file should only be readable by its owner (permissions '600') to protect your secret key.", "fix": "Run 'sudo chmod 600 /home/ams/amscams/conf/wasabi.txt'."},
        "ARCHIVE_NOT_MOUNTED": {"type": "warning", "description": "The cloud archive '/mnt/archive.allsky.tv' is not mounted.", "reason": "The connection to the Wasabi cloud storage is not active. Data is not being backed up, and multi-station features will fail.", "fix": "Run the mount command manually ('sudo -u ams /home/ams/amscams/pythonv2/wasabi.py mnt') and ensure it's added to a startup script."},
        "ARCHIVE_HUNG": {"type": "failure", "description": "The cloud archive mount is hung (stale).", "reason": "The archive appears mounted but is unresponsive. The connection to Wasabi has likely failed.", "fix": "This usually requires a forced unmount and remount. Run 'sudo umount -l /mnt/archive.allsky.tv', then remount it."},
        "ARCHIVE_NOT_WRITABLE": {"type": "failure", "description": "The cloud archive subdirectory for this station ('{path}') is not writable by the 'ams' user.", "reason": "The archive is mounted, but the system cannot write files to the station's subdirectory. This is usually a permissions issue.", "fix": "Verify the subdirectory '{path}' exists and is owned by 'ams'. If correct, check the mount options in 'wasabi.py' for the correct 'uid' and 'gid'."},
        # Scan & Stack
        "MASK_FILE_MISSING": {"type": "warning", "description": "A mask file is missing for one or more cameras.", "reason": "'scan_stack.py' uses mask files to ignore static light sources, preventing many false detections.", "fix": "Create a default mask file for the affected camera in '/home/ams/amscams/conf/'."},
        "PROC_DIR_NOT_WRITABLE": {"type": "failure", "description": "The processing directory '/mnt/ams2/SD/proc2' is not writable by the 'ams' user.", "reason": "The processing pipeline needs to create files in this directory. If it can't write, all processing will halt.", "fix": "Check permissions and run 'sudo chown -R ams:ams /mnt/ams2/SD/proc2' to fix."},
        # Config Files
        "CONFIG_MISSING": {"type": "failure", "description": "A critical config file is missing.", "reason": "Configuration files contain all critical settings for your station. The software cannot run without them.", "fix": "Restore the file from a backup or re-run the setup script to generate a new one."},
        "JSON_PARSE_ERROR": {"type": "failure", "description": "A JSON config file is corrupted and cannot be parsed.", "reason": "This is often caused by a manual editing error (like a missing comma or bracket).", "fix": "Restore from a backup or use an online JSON validator to find the syntax error."},
        "SETUP_JSON_NO_STATION_ID": {"type": "failure", "description": "'station_id' is missing or empty in setup.json.", "reason": "The station ID is the primary identifier for your system on the AllSky7 network.", "fix": "Edit '/home/ams/amscams/conf/setup.json' and add your station ID."},
        "AS6_JSON_SITE_KEY_MISSING": {"type": "failure", "description": "An essential key is missing from the 'site' section of as6.json.", "reason": "Essential station metadata, like latitude and longitude, is required for scientific analysis.", "fix": "Edit '/home/ams/amscams/conf/as6.json' and add the missing information under the 'site' section."},
        "AS6_JSON_LAT_LNG_INVALID": {"type": "failure", "description": "'device_lat' or 'device_lng' in as6.json is not a valid number.", "reason": "The latitude and longitude must be valid decimal numbers for calibration and trajectory analysis.", "fix": "Correct the values in '/home/ams/amscams/conf/as6.json'."},
        "AS6_JSON_LAT_LNG_OUT_OF_RANGE": {"type": "failure", "description": "Latitude or Longitude in as6.json is geographically invalid.", "reason": "Latitude must be between -90 and 90. Longitude must be between -180 and 180.", "fix": "Correct the 'device_lat' and 'device_lng' values in '/home/ams/amscams/conf/as6.json'."},
        "AS6_JSON_NO_CAMS": {"type": "failure", "description": "The 'cameras' section is missing or empty in as6.json.", "reason": "The system needs at least one camera defined to be able to capture video.", "fix": "Run the setup script to configure your cameras, or manually edit the file to add camera information."},
        "AS6_JSON_BAD_IP": {"type": "failure", "description": "A camera has an invalid IP address format in as6.json.", "reason": "The IP address for each camera must be in a valid format (e.g., 192.168.76.71) for the system to connect to it.", "fix": "Correct the IP address for the specified camera in '/home/ams/amscams/conf/as6.json'."},
        "DEFAULTS_PY_NOT_SET": {"type": "failure", "description": "Default station ID 'AMSXXX' found in DEFAULTS.py.", "reason": "This default placeholder value needs to be replaced with your actual station ID during setup.", "fix": "Re-run the configuration part of the setup script."},
        # Config Consistency
        "ID_MISMATCH": {"type": "failure", "description": "Station ID mismatch between setup.json ('{setup_id}') and as6.json ('{as6_id}').", "reason": "Inconsistent station IDs between configuration files can cause processing errors and identification issues.", "fix": "Decide on the correct station ID and update the incorrect file to match."},
        "HOSTNAME_MISMATCH": {"type": "failure", "description": "System hostname ('{hostname}') does not match station ID ('{as6_id}').", "reason": "For network identification and clarity, the machine's hostname should match its station ID.", "fix": "Run 'sudo hostnamectl set-hostname {as6_id}' to set the hostname permanently."},
        # System Health
        "CPU_TEMP_CRITICAL": {"type": "failure", "description": "CPU temperature is critical: {temp}°C.", "reason": "Sustained high CPU temperatures (above 85°C) can cause performance throttling and permanent hardware damage.", "fix": "Immediately check that the computer's fans are working and that air vents are not blocked. Clean any dust from the heatsinks."},
        "SENSORS_MISSING": {"type": "warning", "description": "The 'sensors' command is not installed.", "reason": "The 'lm-sensors' package is required to monitor hardware health like CPU temperature.", "fix": "Install it by running 'sudo apt-get install lm-sensors'."},
        "RAM_CRITICAL": {"type": "failure", "description": "System memory (RAM) usage is critical ({percent:.1f}% used).", "reason": "Extremely high RAM usage will cause system instability and slow performance, likely halting video processing.", "fix": "Close unnecessary applications. If the issue persists, the system may require a RAM upgrade."},
        "SWAP_ACTIVE": {"type": "warning", "description": "Swap usage is high ({percent:.1f}%) and available RAM is low.", "reason": "The system is under active memory pressure, using slow disk swap because physical RAM is exhausted. This will severely degrade performance.", "fix": "Identify the processes consuming memory. This indicates the system has insufficient RAM for its workload and may need an upgrade."},
        "SWAP_IN_USE_OK_RAM": {"type": "warning", "description": "Swap usage is high ({percent:.1f}%), but sufficient RAM is available.", "reason": "The system experienced memory pressure in the past. The data in swap may not be actively used, so performance impact is likely minimal at present.", "fix": "This is a low-priority warning. Monitor for signs of recurring memory pressure. No immediate action is required."},
        "HIGH_LOAD_AVG": {"type": "warning", "description": "System 5-minute load average ({load:.2f}) is high for {cores} CPU cores.", "reason": "The CPU is consistently struggling to keep up with the workload. This can lead to a growing backlog of videos to process.", "fix": "Use 'htop' to identify processes causing high load. If it's the capture/processing scripts, the hardware may be underpowered."},
        "IO_ERROR_DETECTED": {"type": "failure", "description": "I/O errors detected in the kernel log.", "reason": "The operating system is reporting errors when reading from or writing to a storage device. This is a strong indicator of a failing disk.", "fix": "This is a critical hardware warning. Back up your data immediately. Use tools like 'smartctl' to diagnose the disk and prepare to replace it."},
        # Disk Space
        "DISK_CRITICAL": {"type": "failure", "description": "Disk space on {path} is critically low ({percent_free:.2f}% free).", "reason": "If the disk fills up completely, video recording will stop, and the system may become unstable.", "fix": "Delete old, unneeded files from '/mnt/ams2/SD/' and '/mnt/ams2/HD/'. Consider archiving old meteor data."},
        "DISK_LOW": {"type": "warning", "description": "Disk space on {path} is low ({percent_free:.2f}% free).", "reason": "This is an early warning that the video storage drive is filling up.", "fix": "Proactively clean up old video files to prevent the disk from becoming full."},
        # NTP
        "NTP_NO_SYNC": {"type": "failure", "description": "System clock is not synchronized with a time server (NTP).", "reason": "Accurate timing is absolutely critical for meteor observation. If the clock is wrong, the scientific data is useless.", "fix": "Ensure the system is connected to the internet. Run 'sudo timedatectl set-ntp true' and check 'timedatectl status'."},
        # Dependencies
        "PYTHON_PKG_MISSING": {"type": "failure", "description": "A required Python package is missing.", "reason": "The software relies on external Python libraries to function. A missing library will cause scripts to crash.", "fix": "Install the missing package using pip: 'sudo python3 -m pip install {pkg}'."},
        "SYS_PKG_MISSING": {"type": "failure", "description": "A required system command is missing.", "reason": "The software relies on external command-line tools. A missing tool will cause parts of the system to fail.", "fix": "Install the missing package using your system's package manager (e.g., 'sudo apt-get install {pkg_name}')."},
        "SCRIPT_MISSING": {"type": "failure", "description": "A required helper script is missing or not executable.", "reason": "The main scripts call other scripts to perform specific tasks. If one is missing, that functionality will be broken.", "fix": "Restore the script from the repository or a backup. Ensure it is executable ('chmod +x {path}')."},
        # Network
        "NO_INTERNET": {"type": "failure", "description": "No internet connectivity.", "reason": "An internet connection is needed for time sync (NTP), software updates, and data uploads.", "fix": "Check your network cables, router, and firewall configuration."},
        # Camera
        "CAM_UNREACHABLE": {"type": "failure", "description": "A camera is unreachable on the network (ping failed).", "reason": "A camera is not responding. It might be powered off, disconnected, or have a faulty network cable.", "fix": "Check the camera's power supply and network cable. Ensure it's connected to the correct network port on the computer."},
        "CAM_STREAM_DOWN": {"type": "failure", "description": "The video stream for a camera is down.", "reason": "The camera is online (ping successful), but it is not broadcasting a valid video stream.", "fix": "Reboot the camera. If the problem persists, check the camera's web interface for errors and verify the RTSP stream URL."},
        "CAM_BAD_SUBNET": {"type": "warning", "description": "A camera's IP address does not match any local network.", "reason": "The camera may be connected to the wrong network port, or the computer's network interface may be misconfigured.", "fix": "Ensure the camera is plugged into the dedicated camera network port. Verify the IP settings for that port on the computer match the camera's network."},
        # Calibration
        "NO_CAL_FILE": {"type": "warning", "description": "No calibration files were found for one or more cameras.", "reason": "Calibration files are essential for converting pixel positions to astronomical coordinates. Without them, trajectory analysis is impossible.", "fix": "See the detailed sub-checks for more specific reasons and solutions."},
        "SOLVE_FIELD_MISSING": {"type": "failure", "description": "Astrometry solver ('solve-field') is not installed or not in PATH.", "reason": "This is a critical dependency for auto-calibration. The system cannot determine the star field without it.", "fix": "Ensure the 'astrometry.net' package is installed correctly ('sudo apt-get install astrometry.net')."},
        "FAILED_CAL_LOGS": {"type": "warning", "description": "Recent failed calibration logs were found.", "reason": "The calibration process is running but failing to solve the star field, often due to clouds, poor focus, or insufficient stars.", "fix": "Check the latest images in '/mnt/ams2/cal/' and the failed logs in that directory for specific error details."},
        "NO_CAL_SOURCE_IMAGES": {"type": "warning", "description": "No recent source images for calibration were found.", "reason": "The system is not capturing the special 'sense-up' images required for calibration. This could be a cron job issue.", "fix": "Verify that the 'IMX291.py' cron job is present and running correctly."},
        # Processes
        "FFMPEG_DOWN": {"type": "failure", "description": "The FFMPEG capture process is not running for one or more cameras.", "reason": "'ffmpeg' is the program responsible for capturing the video stream. If it's not running, no video is being saved.", "fix": "This is often a symptom of another problem (like an unreachable camera). The 'watch-dog.py' script should try to restart it automatically."},
        "STALE_PROCESS_FOUND": {"type": "warning", "description": "Multiple instances of a key background process ({proc}) are running.", "reason": "This usually indicates a problem with cron job management or a stale lockfile. Conflicting processes can corrupt data.", "fix": "Manually stop all instances of the process ('sudo pkill -f {proc}') and let the watchdog restart it correctly."},
        # Logs & Status
        "WD_LOG_STALE": {"type": "failure", "description": "Watch-dog log ('/tmp/wd.txt') has not been updated in over 6 minutes.", "reason": "The 'watch-dog.py' cron job should run every minute. If its log is old, the cron job is likely failing or cron is not running.", "fix": "Check that the 'cron' service is running ('systemctl status cron'). Manually run the watchdog script to check for errors."},
        "WD_LOG_ERROR": {"type": "failure", "description": "An error was found in the recent watch-dog log entries.", "reason": "The log file itself contains error messages, indicating active problems like camera streams being down.", "fix": "Read the log file ('cat /tmp/wd.txt') to see the specific errors and address them directly."},
        # Captures
        "NO_RECENT_CAPTURES": {"type": "failure", "description": "No recent video captures were found for one or more cameras.", "reason": "The system is not saving new video files. This is a primary indicator that the video capture has failed.", "fix": "This is a critical error. Check camera connectivity and the FFMPEG running process for the affected cameras."},
        "FPS_TOO_LOW": {"type": "warning", "description": "A recent video file has an FPS below 10 ({fps:.1f} FPS).", "reason": "Low FPS indicates a problem with the capture stream, camera settings, or system performance, leading to choppy video.", "fix": "Check camera's stream settings. Ensure the 'ffmpeg' command specifies a stable FPS and that the system is not overloaded."},
        "FPS_INCONSISTENT": {"type": "warning", "description": "Recent {stream_type} video files show high FPS variation (min: {min_fps:.1f}, max: {max_fps:.1f}).", "reason": "Inconsistent frame rates suggest an unstable capture process, which can corrupt video files or cause issues with analysis.", "fix": "This is often linked to an unstable camera connection or an overloaded CPU. Check 'ffmpeg' processes and system load."},
        # Corrupt Files
        "CORRUPT_FILE_FOUND": {"type": "failure", "description": "A likely corrupt (very small) video file was found.", "reason": "The system is creating empty or tiny video files, pointing to a problem with the 'ffmpeg' stream capture.", "fix": "This is often linked to an unstable camera connection or failing 'ffmpeg' process. Check network cables and power."},
        # Processing Queue
        "STALLED_QUEUE": {"type": "failure", "description": "Processing queue is stalled. Oldest file is {age:.1f} hours old.", "reason": "The analysis script ('Process.py') is not processing files. It is likely stuck on a corrupt file or has crashed.", "fix": "Manually run the processing script ('sudo -u ams /home/ams/amscams/pipeline/Process.py ssp') to see the error. Consider moving the oldest file in '/mnt/ams2/SD/proc2/' to unblock the queue."},
        "HIGH_BACKLOG": {"type": "warning", "description": "High processing backlog: {count} new files in the last hour.", "reason": "The system is capturing videos faster than it can process them, possibly due to a meteor shower or insufficient CPU power.", "fix": "This is usually temporary. If it persists for days and system load is high, the hardware may be underpowered."},
        # Cron
        "NO_CRONTAB": {"type": "failure", "description": "No crontab was found for the 'ams' user.", "reason": "The scheduled tasks that automate everything are missing entirely.", "fix": "You must re-run the setup script to install the cron jobs."},
        "CRON_JOB_MISSING": {"type": "failure", "description": "One or more expected cron jobs are missing.", "reason": "A specific, essential scheduled task is not in the crontab, which will cause a part of the system to not function.", "fix": "Re-run the setup script or manually add the missing line to the 'ams' user's crontab ('sudo -u ams crontab -e')."}
    }

    def get_error(self, code):
        """Fetches error details from the catalog by its code."""
        return self._catalog.get(code)

class AS7Diagnostic:
    """
    A diagnostic tool to verify the setup and operational health of the AS7
    Meteor Detection Software.
    """

    def __init__(self, is_root=False):
        """Initializes the diagnostic tool."""
        self.is_root = is_root
        self.GREEN, self.YELLOW, self.RED, self.RESET = '\033[92m', '\033[93m', '\033[91m', '\033[0m'
        self.results = {'success': [], 'warning': [], 'failure': []}
        self.config_data = {}
        self.error_catalog = ErrorCatalog()
        self.logged_errors = []

    def print_header(self):
        """Prints a welcome header for the diagnostic tool."""
        print("="*60)
        print(" AS7 Meteor Detection Software - Comprehensive Health Check")
        print("="*60)
        print("This tool will check if your system is configured and operating correctly.\n")

    def log_success(self, message, indent=0):
        """Formats and prints a success message."""
        prefix = '  -> ' * indent
        print(f"[{self.GREEN}  OK  {self.RESET}] {prefix}{message}")
        self.results['success'].append(message)

    def log_issue(self, error_code, context={}, indent=0):
        """Looks up an error code, formats the message, prints it, and stores it."""
        error_info = self.error_catalog.get_error(error_code)
        if not error_info:
            print(f"[{self.RED}ERROR{self.RESET}] Unknown error code: {error_code}")
            return

        specific_message = self._create_specific_message(error_code, context)
        prefix = '  -> ' * indent
        color = self.RED if error_info['type'] == 'failure' else self.YELLOW
        label = " FAIL " if error_info['type'] == 'failure' else " WARN "
        
        print(f"[{color}{label}{self.RESET}] {prefix}{specific_message}")
        self.results[error_info['type']].append(specific_message)
        self.logged_errors.append({'code': error_code, 'context': context})

    def _create_specific_message(self, error_code, context):
        """Generates a concise, one-line message for an issue using context."""
        if error_code == "PERMISSION_DENIED": return f"Could not perform check: {context.get('check')}"
        if error_code == "DIR_MISSING": return f"Directory is missing: {context.get('path')}"
        if error_code == "BAD_OWNER": return f"Directory owner is not 'ams': {context.get('path')}"
        if error_code == "CONFIG_MISSING": return f"Config file is missing: {context.get('path')}"
        if error_code == "JSON_PARSE_ERROR": return f"Could not parse JSON file: {context.get('path')}"
        if error_code == "AS6_JSON_SITE_KEY_MISSING": return f"Site key '{context.get('key')}' is missing"
        if error_code == "AS6_JSON_BAD_IP": return f"Camera '{context.get('cam_key')}' has an invalid IP"
        if error_code == "AS6_JSON_LAT_LNG_OUT_OF_RANGE": return f"Lat/Lng for station is out of range ({context.get('lat')}, {context.get('lng')})"
        if error_code == "PYTHON_PKG_MISSING": return f"Python package is missing: {context.get('pkg')}"
        if error_code == "SYS_PKG_MISSING": return f"System command '{context.get('pkg')}' is not installed"
        if error_code == "SCRIPT_MISSING": return f"Script is missing or not executable: {context.get('path')}"
        if error_code == "CAM_UNREACHABLE": return f"Camera '{context.get('cam_key')}' ({context.get('ip')}) is unreachable (ping failed)"
        if error_code == "CAM_STREAM_DOWN": return f"Camera '{context.get('cam_key')}' ({context.get('ip')}) {context.get('stream_type')} stream is down (ffprobe failed)"
        if error_code == "CAM_BAD_SUBNET": return f"Camera '{context.get('cam_key')}' ({context.get('ip')}) is not on a known local subnet"
        if error_code == "NO_CAL_FILE": return f"No calibration files found for camera '{context.get('cam_key')}' ({context.get('cams_id')})"
        if error_code == "FFMPEG_DOWN": return f"FFMPEG process for camera '{context.get('cam_key')}' ({context.get('stream')}) is NOT running"
        if error_code == "STALE_PROCESS_FOUND": return f"Found multiple ({context.get('count')}) running instances of '{context.get('proc')}'"
        if error_code == "NO_RECENT_CAPTURES": return f"No recent {context.get('stream_type')} captures for camera '{context.get('cam_key')}'"
        if error_code == "FPS_TOO_LOW": return f"Recent {context.get('stream_type')} file has low FPS: {context.get('fps'):.1f} FPS for {os.path.basename(context.get('file',''))}"
        if error_code == "FPS_INCONSISTENT": return f"High FPS variation for {context.get('stream_type')} captures (Cam: {context.get('cam_key')}). Min: {context.get('min_fps'):.1f}, Max: {context.get('max_fps'):.1f}"
        if error_code == "CORRUPT_FILE_FOUND": return f"Corrupt video file found: {os.path.basename(context.get('file',''))}"
        if error_code == "CRON_JOB_MISSING": return f"Expected cron job is missing: {context.get('job')}"
        if error_code == "WASABI_CREDS_INSECURE": return f"Wasabi credentials file has insecure permissions ({context.get('perms')})"
        if error_code == "MASK_FILE_MISSING": return f"Mask file is missing for camera '{context.get('cam_key')}' ({context.get('cams_id')})"
        
        info = self.error_catalog.get_error(error_code)
        return info['description'].format(**context)

    def run_all_checks(self):
        """Runs all diagnostic checks in a logical order and prints a summary."""
        self.print_header()
        self.check_user()
        self.check_directories_and_mounts()
        self.check_config_files()
        self.check_cloud_archive()
        self.check_config_consistency()
        self.check_system_health()
        self.check_system_resources()
        self.check_filesystem_health()
        self.check_disk_space()
        self.check_ntp_sync()
        self.check_dependencies()
        self.check_scan_stack_dependencies()
        self.check_network_config()
        self.check_network_sanity()
        self.check_api_connectivity()
        self.check_camera_health()
        self.check_calibration_files()
        self.check_running_processes()
        self.check_log_files_and_status()
        self.check_recent_captures()
        self.check_video_fps()
        self.check_for_corrupt_files()
        self.check_processing_health()
        self.check_cron_jobs()
        self.print_summary()

    def _get_top_swap_processes(self):
        """Identifies the top 5 processes using swap memory."""
        swap_users = []
        try:
            psutil = importlib.import_module("psutil")
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    with open(f'/proc/{proc.info["pid"]}/status') as f:
                        for line in f:
                            if line.startswith('VmSwap:'):
                                swap_kb = int(line.split()[1])
                                if swap_kb > 0:
                                    swap_users.append({'name': proc.info['name'], 'swap_mb': swap_kb / 1024})
                                break
                except (FileNotFoundError, PermissionError, ValueError):
                    continue
            return sorted(swap_users, key=lambda x: x['swap_mb'], reverse=True)[:5]
        except (ImportError, Exception):
            return []

    def check_user(self):
        """Verifies the current user and the existence of the 'ams' user."""
        print("\n--- Checking User & Permissions ---")
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            if current_user in ['ams', 'root']:
                self.log_success(f"Running as '{current_user}'.")
            else:
                self.log_issue('WRONG_USER', {'current_user': current_user})
        except Exception: pass
        try:
            pwd.getpwnam('ams')
            self.log_success("User 'ams' exists on the system.")
        except KeyError:
            self.log_issue("NO_AMS_USER")

    def check_directories_and_mounts(self):
        """Checks for the existence, ownership, and mount status of critical directories."""
        print("\n--- Checking Directories & Mounts ---")
        if os.path.ismount("/mnt/ams2"):
            self.log_success("'/mnt/ams2' is correctly mounted.")
        else:
            self.log_issue("NOT_MOUNTED")

        dirs_to_check = [
            "/mnt/ams2", "/mnt/ams2/logs", "/mnt/ams2/backup", "/mnt/ams2/SD", "/mnt/ams2/HD",
            "/mnt/ams2/trash", "/mnt/ams2/temp", "/mnt/ams2/CAMS", "/mnt/ams2/CAMS/queue",
            "/mnt/ams2/CACHE", "/mnt/ams2/meteor_archive", "/mnt/ams2/cal",
            "/mnt/ams2/meteors", "/mnt/ams2/latest", "/home/ams/tmpvids", "/mnt/ams2/SD/proc2"
        ]
        for d in dirs_to_check:
            if os.path.isdir(d):
                self.log_success(f"Directory exists: {d}")
                try:
                    if pwd.getpwuid(os.stat(d).st_uid).pw_name != 'ams':
                        self.log_issue("BAD_OWNER", {'path': d})
                except PermissionError:
                    self.log_issue("PERMISSION_DENIED", {'check': f"ownership of {d}"})
                except KeyError: pass
            else:
                self.log_issue("DIR_MISSING", {'path': d})

    def check_cloud_archive(self):
        """Verifies the Wasabi cloud archive setup, mount status, and writability."""
        print("\n--- Checking Cloud Archive (Wasabi) ---")
        if not which('s3fs'):
            self.log_issue("S3FS_MISSING")
            return

        self.log_success("s3fs command is installed.")
        
        creds_file = "/home/ams/amscams/conf/wasabi.txt"
        if not os.path.isfile(creds_file):
            self.log_issue("WASABI_CREDS_MISSING")
            return
            
        self.log_success("Wasabi credentials file exists.")
        
        try:
            perms = oct(os.stat(creds_file).st_mode)[-3:]
            if perms != '600':
                self.log_issue("WASABI_CREDS_INSECURE", {'perms': perms})
            else:
                self.log_success("Credentials file permissions are secure (600).")
        except PermissionError:
             self.log_issue("PERMISSION_DENIED", {'check': f"reading permissions for {creds_file}"})

        archive_path = "/mnt/archive.allsky.tv"
        try:
            cmd = ['df']
            current_user = pwd.getpwuid(os.getuid()).pw_name
            if self.is_root:
                cmd = ['sudo', '-u', 'ams', 'df']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if archive_path not in result.stdout:
                if not self.is_root and current_user != 'ams':
                    self.log_issue("PERMISSION_DENIED", {'check': f"verifying mount status of '{archive_path}'. Run as root or ams."})
                    return
                else:
                    self.log_issue("ARCHIVE_NOT_MOUNTED")
                    return
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
             self.log_issue("PERMISSION_DENIED", {'check': "running 'df' command, possibly as 'ams' user"})
             return

        try:
            cmd = ['ls', archive_path]
            if self.is_root:
                cmd = ['sudo', '-u', 'ams', 'ls', archive_path]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
            self.log_success(f"Cloud archive '{archive_path}' is mounted and responsive.")
        except subprocess.TimeoutExpired:
            self.log_issue("ARCHIVE_HUNG")
            return
        except subprocess.CalledProcessError as e:
            if "Input/output error" in e.stderr:
                self.log_issue("ARCHIVE_HUNG")
            else:
                self.log_issue("PERMISSION_DENIED", {'check': f"listing contents of {archive_path}. Run as root or ams."})
            return
        except Exception:
            self.log_issue("ARCHIVE_HUNG")
            return

        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json")
        station_id = as6_data.get("site", {}).get("ams_id") if as6_data else None

        if not station_id:
            self.log_issue("PERMISSION_DENIED", {'check': f"archive writability. Station ID could not be found in as6.json."})
            return

        station_archive_path = os.path.join(archive_path, station_id)
        
        dir_exists_cmd = ['test', '-d', station_archive_path]
        if self.is_root:
            dir_exists_check = ['sudo', '-u', 'ams'] + dir_exists_cmd
        else:
            dir_exists_check = dir_exists_cmd

        try:
            subprocess.run(dir_exists_check, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_issue("DIR_MISSING", {'path': station_archive_path})
            return

        test_file = os.path.join(station_archive_path, ".diag_write_test")
        write_check_attempted = False
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            if not self.is_root and current_user != 'ams':
                self.log_issue("PERMISSION_DENIED", {'check': f"write access to '{station_archive_path}'. Run as root or ams."})
            else:
                write_check_attempted = True
                cmd_touch = ['touch', test_file]
                if self.is_root:
                    cmd_touch = ['sudo', '-u', 'ams', 'touch', test_file]
                subprocess.run(cmd_touch, check=True, capture_output=True, text=True)
                self.log_success(f"Cloud archive subdirectory '{station_archive_path}' is writable by user 'ams'.")
        except (subprocess.CalledProcessError, Exception):
            self.log_issue("ARCHIVE_NOT_WRITABLE", {'path': station_archive_path})
        finally:
            if write_check_attempted:
                cmd_rm = ['rm', '-f', test_file]
                if self.is_root:
                    cmd_rm = ['sudo', '-u', 'ams', 'rm', '-f', test_file]
                subprocess.run(cmd_rm, capture_output=True)

    def check_config_files(self):
        """Validates the existence, syntax, and content of critical JSON config files."""
        print("\n--- Checking Configuration Files ---")
        config_files_to_check = {
            "/home/ams/amscams/conf/setup.json": self.validate_setup_json,
            "/home/ams/amscams/conf/as6.json": self.validate_as6_json
        }
        for f, validation_func in config_files_to_check.items():
            if os.path.isfile(f):
                self.log_success(f"Config file exists: {f}")
                try:
                    with open(f, 'r') as json_file:
                        data = json.load(json_file)
                        self.config_data[f] = data
                        validation_func(data)
                except json.JSONDecodeError:
                    self.log_issue("JSON_PARSE_ERROR", {'path': f})
                except PermissionError:
                    self.log_issue("PERMISSION_DENIED", {'check': f"reading {f}"})
                except Exception: pass
            else:
                self.log_issue("CONFIG_MISSING", {'path': f})
        
        defaults_file = "/home/ams/amscams/pipeline/lib/DEFAULTS.py"
        if os.path.exists(defaults_file):
            self.log_success(f"Config file exists: {defaults_file}")
            try:
                with open(defaults_file, 'r') as f: content = f.read()
                if "AMSXXX" in content: self.log_issue("DEFAULTS_PY_NOT_SET")
                else: self.log_success("Station ID appears to be set in DEFAULTS.py.")
            except PermissionError:
                self.log_issue("PERMISSION_DENIED", {'check': f"reading {defaults_file}"})
        else:
            self.log_issue("CONFIG_MISSING", {'path': defaults_file})

    def validate_setup_json(self, data):
        """Specific validation rules for setup.json."""
        if "station_id" not in data or not data["station_id"]:
            self.log_issue("SETUP_JSON_NO_STATION_ID", indent=1)
        else:
            self.log_success("'station_id' is present in setup.json.", indent=1)

    def validate_as6_json(self, data):
        """Specific validation rules for as6.json."""
        site = data.get('site', {})
        if not site: self.log_issue("AS6_JSON_SITE_KEY_MISSING", {'key': 'site'}, indent=1)
        else:
            self.log_success("'site' section is present.", indent=1)
            for key in ['ams_id', 'device_lat', 'device_lng', 'sd_video_dir', 'hd_video_dir']:
                if key not in site or not site[key]: self.log_issue("AS6_JSON_SITE_KEY_MISSING", {'key': key}, indent=2)
                else: self.log_success(f"Site key '{key}' is present and not empty.", indent=2)
            try:
                lat = float(site.get('device_lat', 'x'))
                lng = float(site.get('device_lng', 'x'))
                self.log_success("'device_lat' and 'device_lng' are valid numbers.", indent=2)
                if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                    self.log_issue("AS6_JSON_LAT_LNG_OUT_OF_RANGE", {'lat': lat, 'lng': lng}, indent=3)
                else:
                    self.log_success("Latitude and Longitude are within valid geographical range.", indent=3)
            except (ValueError, TypeError): self.log_issue("AS6_JSON_LAT_LNG_INVALID", indent=2)

        cams = data.get('cameras', {})
        if not cams: self.log_issue("AS6_JSON_NO_CAMS", indent=1)
        else:
            self.log_success("'cameras' section is present and not empty.", indent=1)
            for key, info in cams.items():
                try:
                    socket.inet_aton(info.get('ip', ''))
                    self.log_success(f"Camera '{key}' has a valid IP format.", indent=2)
                except (socket.error, TypeError): self.log_issue("AS6_JSON_BAD_IP", {'cam_key': key}, indent=2)

    def check_config_consistency(self):
        """Checks for mismatches between config files and the system hostname."""
        print("\n--- Checking Configuration Consistency ---")
        setup_data = self.config_data.get("/home/ams/amscams/conf/setup.json")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json")
        if not setup_data or not as6_data: return

        setup_id = setup_data.get("station_id")
        as6_id = as6_data.get("site", {}).get("ams_id")

        if setup_id != as6_id: self.log_issue("ID_MISMATCH", {'setup_id': setup_id, 'as6_id': as6_id})
        else: self.log_success(f"Station ID is consistent between config files ('{setup_id}').")

        try:
            with open('/etc/hostname', 'r') as f: hostname = f.read().strip()
            if as6_id and hostname != as6_id: self.log_issue("HOSTNAME_MISMATCH", {'hostname': hostname, 'as6_id': as6_id})
            else: self.log_success(f"System hostname ('{hostname}') matches station ID.")
        except Exception: pass

    def check_system_health(self):
        """Monitors system hardware, primarily CPU temperature."""
        print("\n--- Checking System Health ---")
        try:
            result = subprocess.run(['sensors'], capture_output=True, text=True, check=False)
            if result.returncode != 0 and 'not found' not in result.stderr.lower():
                if not self.is_root:
                    self.log_issue("PERMISSION_DENIED", {'check': "reading CPU temperature"})
                    return
            
            if result.returncode == 0:
                max_temp = 0
                for line in result.stdout.split('\n'):
                    if '°C' in line and ('Package' in line or 'Core' in line):
                        match = re.search(r'\+(\d{1,3}\.\d)', line)
                        if match: max_temp = max(max_temp, float(match.group(1)))
                if max_temp > 85.0: self.log_issue("CPU_TEMP_CRITICAL", {'temp': max_temp})
                else: self.log_success(f"CPU temperature is normal: {max_temp}°C")
        except FileNotFoundError:
            self.log_issue("SENSORS_MISSING")

    def check_system_resources(self):
        """Checks for high memory usage, swap activity, and system load."""
        print("\n--- Checking System Resources ---")
        try:
            psutil = importlib.import_module("psutil")
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            if mem.percent > 95.0:
                self.log_issue("RAM_CRITICAL", {'percent': mem.percent})
            else:
                self.log_success(f"RAM usage is normal ({mem.percent:.1f}% used).")
            
            if swap.percent > 80.0:
                available_ram_percent = mem.available * 100 / mem.total
                if available_ram_percent < 10.0:
                    top_swappers = self._get_top_swap_processes()
                    self.log_issue("SWAP_ACTIVE", {'percent': swap.percent, 'top_swappers': top_swappers})
                else:
                    self.log_issue("SWAP_IN_USE_OK_RAM", {'percent': swap.percent})
            else:
                self.log_success(f"Swap usage is normal ({swap.percent:.1f}% used).")

            cpu_cores = os.cpu_count()
            load_avg = psutil.getloadavg()
            if load_avg[1] > cpu_cores:
                self.log_issue("HIGH_LOAD_AVG", {'load': load_avg[1], 'cores': cpu_cores})
            else:
                self.log_success(f"System load average is normal ({load_avg[1]:.2f} on {cpu_cores} cores).")
        except ImportError:
            self.log_issue("PYTHON_PKG_MISSING", {'pkg': 'psutil'})
        except Exception: pass

    def check_filesystem_health(self):
        """Scans the kernel log for I/O errors indicating disk failure."""
        print("\n--- Checking Filesystem Health (dmesg) ---")
        if not self.is_root:
            self.log_issue("PERMISSION_DENIED", {'check': "reading kernel log (dmesg). Run as root."})
            return
        try:
            result = subprocess.run(['dmesg'], capture_output=True, text=True)
            io_errors = re.findall(r'.*(I/O error|blk_update_request|ata.*error).*', result.stdout)
            if io_errors:
                error_counts = Counter(err.strip() for err in io_errors)
                self.log_issue("IO_ERROR_DETECTED", {'errors': error_counts})
            else:
                self.log_success("No critical I/O errors found in kernel log.")
        except Exception: pass

    def check_disk_space(self):
        """Checks the available disk space on the primary video storage mount."""
        print("\n--- Checking Disk Space ---")
        path = "/mnt/ams2"
        try:
            stat = os.statvfs(path)
            bytes_available = stat.f_bavail * stat.f_frsize
            bytes_total = stat.f_blocks * stat.f_frsize
            percent_free = (bytes_available / bytes_total) * 100
            if percent_free < 2: self.log_issue("DISK_CRITICAL", {'path': path, 'percent_free': percent_free})
            elif percent_free < 5: self.log_issue("DISK_LOW", {'path': path, 'percent_free': percent_free})
            else: self.log_success(f"Disk space on {path} is sufficient: {bytes_available/1e9:.2f}GB free ({percent_free:.2f}%).")
        except Exception: pass

    def check_ntp_sync(self):
        """Verifies that the system clock is synchronized via NTP."""
        print("\n--- Checking Time Synchronization ---")
        try:
            result = subprocess.run(['timedatectl', 'status'], capture_output=True, text=True)
            if "System clock synchronized: yes" in result.stdout: self.log_success("System clock is synchronized with NTP.")
            else: self.log_issue("NTP_NO_SYNC")
        except Exception: pass

    def check_dependencies(self):
        """Checks for all Python packages, system packages, and external scripts."""
        print("\n--- Checking Dependencies ---")
        
        pkgs = ["netifaces", "requests", "tabulate", "consolemenu", "psutil", "ephem", "timezonefinder"]
        for pkg in pkgs:
            try:
                importlib.import_module(pkg)
                self.log_success(f"Python package installed: {pkg}", indent=1)
            except ImportError: self.log_issue("PYTHON_PKG_MISSING", {'pkg': pkg}, indent=1)

        sys_pkgs = {'curl': 'curl', 'convert': 'imagemagick', 'git': 'git'}
        for pkg, pkg_name in sys_pkgs.items():
            if which(pkg):
                self.log_success(f"System command installed: {pkg}", indent=1)
            else:
                self.log_issue("SYS_PKG_MISSING", {'pkg': pkg, 'pkg_name': pkg_name}, indent=1)
        
        scripts = ["/home/ams/amscams/python/get_latest.py", "/home/ams/amscams/pipeline/system_health.py"]
        for script_path in scripts:
            if os.path.isfile(script_path) and os.access(script_path, os.X_OK):
                self.log_success(f"Script is ready: {os.path.basename(script_path)}", indent=1)
            else:
                self.log_issue("SCRIPT_MISSING", {'path': script_path}, indent=1)

    def check_scan_stack_dependencies(self):
        """Checks dependencies for the video processing pipeline."""
        print("\n--- Checking Scan & Stack Dependencies ---")
        pkgs = ["pandas", "sklearn", "cv2"]
        for pkg in pkgs:
            try:
                importlib.import_module(pkg if pkg != 'cv2' else 'cv2')
                self.log_success(f"Python package for scan_stack installed: {pkg}")
            except ImportError: self.log_issue("PYTHON_PKG_MISSING", {'pkg': 'opencv-python' if pkg == 'cv2' else pkg})

        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            cams_id = cam_info.get('cams_id')
            if cams_id:
                mask_file = f"/home/ams/amscams/conf/mask_{cams_id}.png"
                if not os.path.isfile(mask_file):
                    self.log_issue("MASK_FILE_MISSING", {'cam_key': cam_key, 'cams_id': cams_id})
                else:
                    self.log_success(f"Mask file found for camera '{cam_key}'.")

        proc_dir = "/mnt/ams2/SD/proc2"
        if os.path.isdir(proc_dir):
            test_file = os.path.join(proc_dir, ".diag_write_test")
            write_check_attempted = False
            try:
                current_user = pwd.getpwuid(os.getuid()).pw_name
                if not self.is_root and current_user != 'ams':
                    self.log_issue("PERMISSION_DENIED", {'check': f"write access to '{proc_dir}'. Run as root or ams."})
                else:
                    write_check_attempted = True
                    cmd_touch = ['touch', test_file]
                    if self.is_root:
                        cmd_touch = ['sudo', '-u', 'ams', 'touch', test_file]
                    subprocess.run(cmd_touch, check=True, capture_output=True, text=True)
                    self.log_success(f"Processing directory '{proc_dir}' is writable by user 'ams'.")
            except (subprocess.CalledProcessError, Exception):
                self.log_issue("PROC_DIR_NOT_WRITABLE")
            finally:
                if write_check_attempted:
                    cmd_rm = ['rm', '-f', test_file]
                    if self.is_root:
                        cmd_rm = ['sudo', '-u', 'ams', 'rm', '-f', test_file]
                    subprocess.run(cmd_rm, capture_output=True)

    def check_network_config(self):
        """Performs a basic internet connectivity test."""
        print("\n--- Checking Network ---")
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            self.log_success("Internet connectivity to 8.8.8.8 is working.")
        except OSError: self.log_issue("NO_INTERNET")

    def check_network_sanity(self):
        """Checks if camera IPs fall within the subnets of local network interfaces."""
        print("\n--- Checking Network Sanity ---")
        try:
            netifaces = importlib.import_module("netifaces")
            ipaddress = importlib.import_module("ipaddress")
            
            local_networks = []
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr_info in addrs[netifaces.AF_INET]:
                        if 'addr' in addr_info and 'netmask' in addr_info:
                            try:
                                network = ipaddress.IPv4Network(f"{addr_info['addr']}/{addr_info['netmask']}", strict=False)
                                local_networks.append(network)
                            except ValueError: continue
            
            as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
            for cam_key, cam_info in as6_data.get('cameras', {}).items():
                ip_str = cam_info.get('ip')
                if ip_str:
                    try:
                        cam_ip = ipaddress.IPv4Address(ip_str)
                        if not any(cam_ip in net for net in local_networks):
                            self.log_issue("CAM_BAD_SUBNET", {'cam_key': cam_key, 'ip': ip_str})
                        else:
                            self.log_success(f"Camera '{cam_key}' ({ip_str}) is on a known local subnet.")
                    except ipaddress.AddressValueError: pass
        except ImportError: pass
        except Exception: pass

    def check_api_connectivity(self):
        """Checks if the AllSky7 API is reachable."""
        print("\n--- Checking API Connectivity ---")
        try:
            import requests
            requests.get("https://kyvegys798.execute-api.us-east-1.amazonaws.com/api/allskyapi", timeout=10)
            self.log_success("Successfully connected to the AllSky API.")
        except Exception: pass

    def check_camera_health(self):
        """Pings cameras and checks their live video streams using ffprobe."""
        print("\n--- Checking Camera Health ---")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        ffprobe_exists = which('ffprobe')

        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            ip = cam_info.get('ip')
            if not ip: continue

            res = subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True)
            if res.returncode != 0:
                self.log_issue("CAM_UNREACHABLE", {'cam_key': cam_key, 'ip': ip})
                continue
            
            self.log_success(f"Camera '{cam_key}' ({ip}) is reachable (ping OK).", indent=1)

            if not ffprobe_exists: continue
            
            port = cam_info.get('rtsp_port', '554')
            user = cam_info.get('user', 'admin')
            password = cam_info.get('password', '')
            channel = cam_info.get('channel', '1')

            for stream_type, stream_id in [("HD", "0"), ("SD", "1")]:
                uri = f"user={user}&password={password}&channel={channel}&stream={stream_id}.sdp"
                rtsp_url = f"rtsp://{ip}:{port}/{uri}"
                cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-rtsp_transport', 'tcp', '-timeout', '5000000', rtsp_url]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    self.log_success(f"Camera '{cam_key}' {stream_type} stream is active.", indent=2)
                except subprocess.CalledProcessError:
                    self.log_issue("CAM_STREAM_DOWN", {'cam_key': cam_key, 'ip': ip, 'stream_type': stream_type}, indent=2)

    def check_calibration_files(self):
        """Checks for the presence of essential astrometric calibration files."""
        print("\n--- Checking Calibration Files ---")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        station_id = as6_data.get("site", {}).get("ams_id", "AMSXXX")
        cal_dirs = ["/mnt/ams2/cal/", f"/mnt/ams2/meteor_archive/{station_id}/CAL/"]
        
        any_missing = False
        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            cams_id = cam_info.get('cams_id')
            if cams_id:
                try:
                    found = any(glob.glob(os.path.join(d, '**', f"*{cams_id}*calparams.json"), recursive=True) for d in cal_dirs if os.path.isdir(d))
                    if found: self.log_success(f"Calibration file(s) found for camera '{cam_key}' ({cams_id}).")
                    else: 
                        self.log_issue("NO_CAL_FILE", {'cam_key': cam_key, 'cams_id': cams_id})
                        any_missing = True
                except PermissionError:
                    self.log_issue("PERMISSION_DENIED", {'check': f"reading calibration files for {cam_key}"})
                    any_missing = True
        
        if any_missing:
            self.diagnose_calibration_failures(indent=1)

    def diagnose_calibration_failures(self, indent=0):
        """Sub-diagnostic for when calibration files are missing."""
        if not which('solve-field'):
            self.log_issue("SOLVE_FIELD_MISSING", indent=indent)
        else:
            self.log_success("Astrometry solver 'solve-field' is installed.", indent=indent)
        
        try:
            cmd = ['find', '/mnt/ams2/cal/', '-type', 'f', '-name', '*failed*', '-mmin', '-1440']
            res = subprocess.run(cmd, capture_output=True)
            if res.stdout: self.log_issue("FAILED_CAL_LOGS", indent=indent)
            else: self.log_success("No recent failed calibration logs found.", indent=indent)
        except Exception: pass

        try:
            cmd = ['find', '/mnt/ams2/cal/', '-type', 'f', '-name', '*-stacked.png', '-mmin', '-1440']
            res = subprocess.run(cmd, capture_output=True)
            if not res.stdout: self.log_issue("NO_CAL_SOURCE_IMAGES", indent=indent)
            else: self.log_success("Recent source images for calibration were found.", indent=indent)
        except Exception: pass

    def check_running_processes(self):
        """Checks for running capture processes and for stale/duplicate script instances."""
        print("\n--- Checking Running Processes ---")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        try:
            ps_output = subprocess.check_output(['ps', '-aux'], text=True)
            for cam_key, cam_info in as6_data.get('cameras', {}).items():
                ip = cam_info.get('ip')
                if ip:
                    for stream in ["SD", "HD"]:
                        found = any(f'ffmpeg' in l and ip in l and stream in l and 'grep' not in l for l in ps_output.splitlines())
                        if not found: self.log_issue("FFMPEG_DOWN", {'cam_key': cam_key, 'stream': stream})
                        else: self.log_success(f"FFMPEG process for '{cam_key}' ({stream}) is running.")
        except Exception: pass
        try:
            psutil = importlib.import_module("psutil")
            scripts_to_check = ["Process.py"]
            for script_name in scripts_to_check:
                count = 0
                for proc in psutil.process_iter(['cmdline', 'name']):
                    if not proc.info['cmdline']:
                        continue
                    cmdline_str = ' '.join(proc.info['cmdline'])
                    proc_name = proc.info['name']
                    if script_name in cmdline_str and proc_name in ('python', 'python3'):
                        count += 1
                if count > 1:
                    self.log_issue("STALE_PROCESS_FOUND", {'proc': script_name, 'count': count})
                else:
                    self.log_success(f"No stale instances of '{script_name}' found.")
        except ImportError: pass
        except Exception: pass

    def check_log_files_and_status(self):
        """Checks the freshness of and for errors in the watchdog log."""
        print("\n--- Checking Logs & Status Files ---")
        wd_log = "/tmp/wd.txt"
        if os.path.isfile(wd_log):
            age = time.time() - os.path.getmtime(wd_log)
            if age > 360:
                self.log_issue("WD_LOG_STALE")
            else: self.log_success(f"Watch-dog log was recently updated.")
            
            try:
                with open(wd_log, 'r') as f: last_lines = f.readlines()[-20:]
                if any("error" in l.lower() for l in last_lines): self.log_issue("WD_LOG_ERROR")
                else: self.log_success("No errors found in recent watch-dog log entries.", indent=1)
            except Exception: pass

    def check_recent_captures(self):
        """Verifies that new video files are being created for each camera."""
        print("\n--- Checking Recent Captures ---")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            cams_id = cam_info.get('cams_id')
            if cams_id:
                for stream_type, path in [("SD", "/mnt/ams2/SD"), ("HD", "/mnt/ams2/HD")]:
                    try:
                        cmd = ['find', path, '-name', f"*{cams_id}*.mp4", '-mmin', '-10']
                        res = subprocess.run(cmd, capture_output=True)
                        if not res.stdout: self.log_issue("NO_RECENT_CAPTURES", {'stream_type': stream_type, 'cam_key': cam_key})
                        else: self.log_success(f"Recent {stream_type} captures found for camera '{cam_key}' ({cams_id}).")
                    except Exception: pass

    def check_video_fps(self):
        """Checks the FPS of recent video files for consistency and minimums."""
        print("\n--- Checking Video File FPS ---")
        ffprobe_path = which('ffprobe')
        if not ffprobe_path:
            # check_camera_health or check_dependencies will already flag this
            self.log_success("'ffprobe' not found, skipping FPS check.")
            return

        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            cams_id = cam_info.get('cams_id')
            if not cams_id:
                continue
            
            self.log_success(f"Checking FPS for camera '{cam_key}' ({cams_id})...", indent=1)

            for stream_type, path in [("SD", "/mnt/ams2/SD"), ("HD", "/mnt/ams2/HD")]:
                if not os.path.isdir(path):
                    continue
                
                # 1. Find the 10 most recent files for this cam/stream
                # We search only in the last day for performance.
                cmd = [
                    'find', path, 
                    '-name', f"*{cams_id}*.mp4", 
                    '-mmin', '-1440',  # Only look at files from the last 24h
                    '-type', 'f',
                    '-printf', '%T@ %p\n'
                ]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    if not res.stdout:
                        self.log_success(f"No recent {stream_type} files found to check for '{cam_key}'.", indent=2)
                        continue
                    
                    # Sort by timestamp (first column) descending, take top 10
                    sorted_files = sorted(
                        res.stdout.strip().split('\n'), 
                        key=lambda x: float(x.split(' ', 1)[0]), 
                        reverse=True
                    )
                    files_to_check = [line.split(' ', 1)[1] for line in sorted_files[:10]]

                except subprocess.CalledProcessError:
                    self.log_success(f"No recent {stream_type} files found to check for '{cam_key}'.", indent=2)
                    continue
                except Exception as e:
                    self.log_issue("PERMISSION_DENIED", {'check': f"finding video files in {path}: {e}"}, indent=2)
                    continue

                # 2. Get FPS for each file
                fps_list = []
                low_fps_errors_found = False
                for f in files_to_check:
                    try:
                        cmd_ffprobe = [
                            ffprobe_path, '-v', 'error', 
                            '-select_streams', 'v:0', 
                            '-show_entries', 'stream=r_frame_rate', 
                            '-of', 'default=noprint_wrappers=1:nokey=1',
                            f
                        ]
                        res = subprocess.run(cmd_ffprobe, capture_output=True, text=True, check=True, timeout=5)
                        fps_str = res.stdout.strip()
                        if '/' in fps_str:
                            num, den = map(float, fps_str.split('/'))
                            if den == 0: continue
                            fps = num / den
                            fps_list.append(fps)
                            
                            # 3a. Check for low FPS
                            if fps < 10.0:
                                self.log_issue("FPS_TOO_LOW", {'stream_type': stream_type, 'fps': fps, 'file': f}, indent=3)
                                low_fps_errors_found = True
                        elif fps_str:
                             # Not a fraction, might be a single number
                            fps = float(fps_str)
                            fps_list.append(fps)
                            if fps < 10.0:
                                self.log_issue("FPS_TOO_LOW", {'stream_type': stream_type, 'fps': fps, 'file': f}, indent=3)
                                low_fps_errors_found = True

                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                        # File might be corrupt or unreadable, skip it.
                        pass 
                    except Exception:
                        pass # General catch-all

                if not fps_list:
                    self.log_success(f"Could not determine FPS for any recent {stream_type} files for '{cam_key}'.", indent=2)
                    continue
                
                if not low_fps_errors_found:
                     self.log_success(f"No files with FPS < 10.0 found for {stream_type}.", indent=3)

                # 3b. Check for variation
                min_fps = min(fps_list)
                max_fps = max(fps_list)
                
                if min_fps == 0: # Avoid division by zero
                    continue

                variation_percent = (max_fps - min_fps) / min_fps
                
                if variation_percent > 0.10: # More than 10% variation
                    self.log_issue("FPS_INCONSISTENT", {
                        'stream_type': stream_type,
                        'cam_key': cam_key,
                        'min_fps': min_fps,
                        'max_fps': max_fps
                    }, indent=3)
                else:
                    self.log_success(f"FPS for {stream_type} is stable (Min: {min_fps:.1f}, Max: {max_fps:.1f}).", indent=3)

    def check_for_corrupt_files(self):
        """Searches for recently created video files that are unusually small."""
        print("\n--- Checking for Corrupt Video Files ---")
        try:
            cmd = ['find', "/mnt/ams2/SD", '-type', 'f', '-name', '*.mp4', '-mmin', '+2', '-mmin', '-15', '-size', '-2k', '-printf', '%p %s\\n']
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.stdout:
                for line in res.stdout.strip().split('\n'):
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        file_path, size = parts
                        self.log_issue("CORRUPT_FILE_FOUND", {'file': file_path, 'size': size})
            else:
                self.log_success("No suspiciously small video files found.")
        except Exception: pass

    def check_processing_health(self):
        """Checks the video processing queue for stalls or a large backlog."""
        print("\n--- Checking Processing Queue ---")
        proc_dir = "/mnt/ams2/SD/proc2"
        if not os.path.isdir(proc_dir): return
        
        try:
            cmd_oldest = f"find {proc_dir} -maxdepth 1 -type f -printf '%T@ %p\\n' | sort -n | head -1"
            res = subprocess.run(cmd_oldest, shell=True, capture_output=True, text=True)
            if res.stdout:
                oldest_ts = float(res.stdout.split()[0])
                age_hours = (time.time() - oldest_ts) / 3600
                if age_hours > 3:
                    self.log_issue("STALLED_QUEUE", {'age': age_hours})
                    return 
                else:
                    self.log_success(f"Processing queue is not stalled (oldest file is {age_hours:.1f} hours old).")
            
            cmd_recent = f"find {proc_dir} -maxdepth 1 -type f -mmin -60 | wc -l"
            res = subprocess.run(cmd_recent, shell=True, capture_output=True, text=True)
            count = int(res.stdout.strip())
            if count > 1500:
                self.log_issue("HIGH_BACKLOG", {'count': count})
            else:
                self.log_success(f"Recent file queue size is normal ({count} files in last hour).")

        except Exception: pass

    def check_cron_jobs(self):
        """Verifies that the crontab for the 'ams' user exists and contains all expected jobs."""
        print("\n--- Checking Cron Jobs ---")
        expected_jobs = [
            "as7-latest.py", "watch-dog.py", "NOAA.py", "scan_stack.py", 
            "day_stack.py", "doDay.py", "IMX291.py", "Process.py run_jobs",
            "gitpull.py", "monitor.sh", "downloader.py", "logrotate"
        ]
        try:
            res = subprocess.run(['crontab', '-l', '-u', 'ams'], capture_output=True, text=True, check=True)
            content = res.stdout
            self.log_success("Crontab found for user 'ams'. Verifying jobs...")
            for job in expected_jobs:
                if job not in content:
                    self.log_issue("CRON_JOB_MISSING", {'job': job}, indent=1)
                else:
                    self.log_success(f"Found expected cron job: {job}", indent=1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if not self.is_root:
                self.log_issue("PERMISSION_DENIED", {'check': "reading 'ams' user's crontab"})
            else:
                self.log_issue("NO_CRONTAB")

    def print_summary(self):
        """Prints a final summary of all successes, warnings, and failures."""
        print("\n" + "="*60)
        print(" Health Check Summary")
        print("="*60)
        
        total = sum(len(v) for v in self.results.values())
        print(f"Total checks: {total}")
        print(f"{self.GREEN}Successful: {len(self.results['success'])}{self.RESET}")
        print(f"{self.YELLOW}Warnings:   {len(self.results['warning'])}{self.RESET}")
        print(f"{self.RED}Failures:   {len(self.results['failure'])}{self.RESET}")

        if self.logged_errors:
            print("\n--- Detailed Report ---")
            grouped_errors = {}
            for error in self.logged_errors:
                code = error['code']
                if code not in grouped_errors: grouped_errors[code] = []
                grouped_errors[code].append(error['context'])

            issue_number = 1
            for code, contexts in grouped_errors.items():
                info = self.error_catalog.get_error(code)
                if not info: continue
                
                color = self.RED if info['type'] == 'failure' else self.YELLOW
                header = "Issue" if info['type'] == 'failure' else "Warning"
                
                print(f"\n{color}{issue_number}. {header}: {info['description'].format(**contexts[0])}{self.RESET}")
                
                if code == "IO_ERROR_DETECTED":
                    print("   - Detected Errors:")
                    error_counts = contexts[0].get('errors', {})
                    for err_line, count in error_counts.items():
                        print(f"     - {err_line} (x{count})")
                elif code == "SWAP_ACTIVE":
                    print("   - Top Swap Consumers:")
                    top_swappers = contexts[0].get('top_swappers', [])
                    if not top_swappers:
                        print("     - Could not determine top swap processes.")
                    for proc in top_swappers:
                        print(f"     - {proc['name']}: {proc['swap_mb']:.2f} MB")
                elif len(contexts) == 1:
                     ctx = contexts[0]
                     details = ", ".join(f"{k}: '{v}'" for k, v in ctx.items() if v)
                     if code == "CORRUPT_FILE_FOUND":
                         details = f"file: '{ctx.get('file')}', size: {ctx.get('size')} bytes"
                     elif code == "PERMISSION_DENIED":
                         details = f"{ctx.get('check')}"
                     elif code == "FPS_TOO_LOW":
                         details = f"file: '{ctx.get('file')}', fps: {ctx.get('fps'):.1f}"
                     print(f"   - Affected Item: {details}")
                else:
                    print("   - Affected Items:")
                    for ctx in contexts:
                        if code == "CORRUPT_FILE_FOUND":
                            details = f"file: '{ctx.get('file')}', size: {ctx.get('size')} bytes"
                        elif code == "PERMISSION_DENIED":
                            details = f"{ctx.get('check')}"
                        elif code == "FPS_TOO_LOW":
                            details = f"file: '{ctx.get('file')}', fps: {ctx.get('fps'):.1f}"
                        else:
                            details = ", ".join(f"{k}: '{v}'" for k, v in ctx.items() if v)
                        print(f"     - {details}")
                
                print(f"   - Why it's a problem: {info['reason']}")
                print(f"   - How to fix: {info['fix'].format(**contexts[0])}")
                issue_number += 1
                
        elif not self.results['failure'] and not self.results['warning']:
             print(f"\n{self.GREEN}Everything looks good! Your system appears to be healthy.{self.RESET}")

if __name__ == '__main__':
    is_root = (os.geteuid() == 0)
    diagnostic = AS7Diagnostic(is_root=is_root)

    if not is_root:
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            if current_user != 'ams':
                print(f"{diagnostic.YELLOW}WARNING: Running as non-standard user '{current_user}'.{diagnostic.RESET}")
                print("Some checks for system files and other users' data may be skipped or fail.")
                print("For a complete diagnosis, please run with 'sudo' or as the 'ams' user.\n")
        except KeyError:
            pass
    
    diagnostic.run_all_checks()
