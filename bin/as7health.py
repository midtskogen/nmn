#!/usr/bin/python3
"""
AS7 Health Check Tool

This script performs a comprehensive diagnostic audit of an AllSky7 meteor detection
station. It checks for common configuration errors, operational issues, and system
health problems to ensure the station is functioning correctly.

The script is designed to be run as the 'root' user for a complete diagnosis, but it
can also be run as the 'ams' user or another user with limited checks.

Can also perform checks for the NMN system if run with the --nmn flag.
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
import configparser
import tempfile
import shutil
from collections import Counter
from shutil import which
from datetime import datetime, timedelta

class ErrorCatalog:
    """
    A centralized catalog of all possible diagnostic errors.

    Each error includes a type (failure/warning/info), a description, the reason it's
    a problem, and a recommended fix. This allows for consistent and detailed
    reporting in the final summary.
    """

    _catalog = {
        # User & Permissions
        "NO_AMS_USER": {"type": "failure", "description": "User 'ams' does not exist on the system.", "reason": "The entire software suite is designed to run under the 'ams' user for correct file permissions and ownership.", "fix": "The user 'ams' needs to be created. You may need to re-run the initial setup script."},
        "WRONG_USER": {"type": "info", "description": "Running as a non-standard user ('{current_user}').", "reason": "The script is designed to be run as 'root' or 'ams'. Running as another user may lead to permission errors on some checks.", "fix": "For a complete diagnosis, run the script as 'root' using 'sudo' or as the 'ams' user."},
        "PERMISSION_DENIED": {"type": "info", "description": "Could not perform a check due to insufficient permissions.", "reason": "This check requires root (sudo) privileges to read protected system files or other users' data.", "fix": "Re-run the script using 'sudo ./as7-health-check.py' for a complete diagnosis."},
        # Directories & Mounts
        "NOT_MOUNTED": {"type": "failure", "description": "'/mnt/ams2' is NOT a mounted filesystem.", "reason": "The system expects a large, separate drive mounted at /mnt/ams2 to store video data. If it's just a directory on the main drive, the OS drive will fill up quickly.", "fix": "Ensure your data disk is properly formatted, added to '/etc/fstab' to mount on boot, and then mounted with 'sudo mount -a'."},
        "DIR_MISSING": {"type": "failure", "description": "A required directory is missing.", "reason": "The software requires a specific directory structure to save videos, logs, calibration data, and processed meteors.", "fix": "Re-run the setup script section for creating directories, or create it manually ('sudo mkdir {path}') and ensure ownership is correct ('sudo chown ams:ams {path}')."},
        "BAD_OWNER": {"type": "warning", "description": "A directory has an incorrect owner (should be 'ams').", "reason": "The 'ams' user needs to be able to write files to this directory. Incorrect ownership will cause permission errors during operation.", "fix": "Run 'sudo chown -R ams:ams /mnt/ams2' and 'sudo chown -R ams:ams /home/ams/amscams' to correct ownership."},
        # Cloud Archive (Wasabi)
        "S3FS_MISSING": {"type": "warning", "description": "The 's3fs' command is not installed.", "reason": "The 's3fs-fuse' package is required to mount the Wasabi cloud storage.", "fix": "Follow the installation instructions to install 's3fs-fuse' from source."},
        "WASABI_CREDS_MISSING": {"type": "warning", "description": "Wasabi credentials file is missing.", "reason": "The credentials file ('/home/ams/amscams/conf/wasabi.txt') contains the key needed to access your cloud storage.", "fix": "Create the 'wasabi.txt' file in the correct location and add your Wasabi access key and secret key."},
        "WASABI_CREDS_INSECURE": {"type": "warning", "description": "Wasabi credentials file has insecure permissions ({perms}).", "reason": "The credentials file should only be readable by its owner (permissions '600') to protect your secret key.", "fix": "Run 'sudo chmod 600 /home/ams/amscams/conf/wasabi.txt'."},
        "ARCHIVE_NOT_MOUNTED": {"type": "info", "description": "The cloud archive '/mnt/archive.allsky.tv' is not mounted.", "reason": "The connection to the Wasabi cloud storage is not active. Data is not being backed up, and multi-station features will fail.", "fix": "Run the mount command manually ('sudo -u ams /home/ams/amscams/pythonv2/wasabi.py mnt') and ensure it's added to a startup script."},
        "ARCHIVE_HUNG": {"type": "warning", "description": "The cloud archive mount is hung (stale).", "reason": "The archive appears mounted but is unresponsive. The connection to Wasabi has likely failed.", "fix": "This usually requires a forced unmount and remount. Run 'sudo umount -l /mnt/archive.allsky.tv', then remount it."},
        "ARCHIVE_NOT_WRITABLE": {"type": "warning", "description": "The cloud archive subdirectory for this station ('{path}') is not writable by the 'ams' user.", "reason": "The archive is mounted, but the system cannot write files to the station's subdirectory. This is usually a permissions issue.", "fix": "Verify the subdirectory '{path}' exists and is owned by 'ams'. If correct, check the mount options in 'wasabi.py' for the correct 'uid' and 'gid'."},
        # Scan & Stack
        "MASK_FILE_MISSING": {"type": "info", "description": "A mask file is missing for one or more cameras.", "reason": "'scan_stack.py' uses mask files to ignore static light sources, preventing many false detections.", "fix": "Create a default mask file for the affected camera in '/home/ams/amscams/conf/'."},
        "PROC_DIR_NOT_WRITABLE": {"type": "failure", "description": "The processing directory '/mnt/ams2/SD/proc2' is not writable by the 'ams' user.", "reason": "The processing pipeline needs to create files in this directory. If it can't write, all processing will halt.", "fix": "Check permissions and run 'sudo chown -R ams:ams /mnt/ams2/SD/proc2' to fix."},
        # Config Files
        "CONFIG_MISSING": {"type": "failure", "description": "A critical config file is missing.", "reason": "Configuration files contain all critical settings for your station. The software cannot run without them.", "fix": "Restore the file from a backup or re-run the setup script to generate a new one."},
        "JSON_PARSE_ERROR": {"type": "failure", "description": "A JSON config file is corrupted and cannot be parsed.", "reason": "This is often caused by a manual editing error (like a missing comma or bracket).", "fix": "Restore from a backup or use an online JSON validator to find the syntax error."},
        "SETUP_JSON_NO_STATION_ID": {"type": "failure", "description": "'station_id' is missing or empty in setup.json.", "reason": "The station ID is the primary identifier for your system on the AllSky7 network.", "fix": "Edit '/home/ams/amscams/conf/setup.json' and add your station ID."},
        "AS6_JSON_SITE_KEY_MISSING": {"type": "failure", "description": "An essential key is missing from the 'site' section of as6.json.", "reason": "Essential station metadata, like latitude and longitude, is required for scientific analysis.", "fix": "Edit '/home/ams/amscams/conf/as6.json' and add the missing information under the 'site' section."},
        "AS6_JSON_LAT_LNG_INVALID": {"type": "warning", "description": "'device_lat' or 'device_lng' in as6.json is not a valid number.", "reason": "The latitude and longitude must be valid decimal numbers for calibration and trajectory analysis.", "fix": "Correct the values in '/home/ams/amscams/conf/as6.json'."},
        "AS6_JSON_LAT_LNG_OUT_OF_RANGE": {"type": "warning", "description": "Latitude or Longitude in as6.json is geographically invalid.", "reason": "Latitude must be between -90 and 90. Longitude must be between -180 and 180.", "fix": "Correct the 'device_lat' and 'device_lng' values in '/home/ams/amscams/conf/as6.json'."},
        "AS6_JSON_NO_CAMS": {"type": "failure", "description": "The 'cameras' section is missing or empty in as6.json.", "reason": "The system needs at least one camera defined to be able to capture video.", "fix": "Run the setup script to configure your cameras, or manually edit the file to add camera information."},
        "AS6_JSON_BAD_IP": {"type": "failure", "description": "A camera has an invalid IP address format in as6.json.", "reason": "The IP address for each camera must be in a valid format (e.g., 192.168.76.71) for the system to connect to it.", "fix": "Correct the IP address for the specified camera in '/home/ams/amscams/conf/as6.json'."},
        "DEFAULTS_PY_NOT_SET": {"type": "failure", "description": "Default station ID 'AMSXXX' found in DEFAULTS.py.", "reason": "This default placeholder value needs to be replaced with your actual station ID during setup.", "fix": "Re-run the configuration part of the setup script."},
        # Config Consistency
        "ID_MISMATCH": {"type": "failure", "description": "Station ID mismatch between setup.json ('{setup_id}') and as6.json ('{as6_id}').", "reason": "Inconsistent station IDs between configuration files can cause processing errors and identification issues.", "fix": "Decide on the correct station ID and update the incorrect file to match."},
        "HOSTNAME_MISMATCH": {"type": "failure", "description": "System hostname ('{hostname}') does not match station ID ('{as6_id}').", "reason": "For network identification and clarity, the machine's hostname should match its station ID.", "fix": "Run 'sudo hostnamectl set-hostname {as6_id}' to set the hostname permanently."},
        # System Health
        "CPU_TEMP_CRITICAL": {"type": "failure", "description": "CPU temperature is critical: {temp}°C.", "reason": "Sustained high CPU temperatures (above 85°C) can cause performance throttling and permanent hardware damage.", "fix": "Immediately check that the computer's fans are working and that air vents are not blocked. Clean any dust from the heatsinks."},
        "SENSORS_MISSING": {"type": "info", "description": "The 'sensors' command is not installed.", "reason": "The 'lm-sensors' package is required to monitor hardware health like CPU temperature.", "fix": "Install it by running 'sudo apt-get install lm-sensors'."},
        "RAM_CRITICAL": {"type": "failure", "description": "System memory (RAM) usage is critical ({percent:.1f}% used).", "reason": "Extremely high RAM usage will cause system instability and slow performance, likely halting video processing.", "fix": "Close unnecessary applications. If the issue persists, the system may require a RAM upgrade."},
        "SWAP_ACTIVE": {"type": "info", "description": "Swap usage is high ({percent:.1f}%) and available RAM is low.", "reason": "The system is under active memory pressure, using slow disk swap because physical RAM is exhausted. This will severely degrade performance.", "fix": "Identify the processes consuming memory. This indicates the system has insufficient RAM for its workload and may need an upgrade."},
        "SWAP_IN_USE_OK_RAM": {"type": "info", "description": "Swap usage is high ({percent:.1f}%), but sufficient RAM is available.", "reason": "The system experienced memory pressure in the past. The data in swap may not be actively used, so performance impact is likely minimal at present.", "fix": "This is a low-priority warning. Monitor for signs of recurring memory pressure. No immediate action is required."},
        "HIGH_LOAD_AVG": {"type": "info", "description": "System 5-minute load average ({load:.2f}) is high for {cores} CPU cores (load > {cores}).", "reason": "The CPU is somewhat busy. This is not an immediate problem but indicates a consistent workload.", "fix": "Monitor the load. If it continues to rise, use 'htop' to identify processes."},
        "HIGH_LOAD_AVG_WARN": {"type": "warning", "description": "System 5-minute load average ({load:.2f}) is very high for {cores} CPU cores (load > {cores_x2}).", "reason": "The CPU is consistently struggling to keep up with the workload. This can lead to a growing backlog of videos to process.", "fix": "Use 'htop' to identify processes causing high load. If it's the capture/processing scripts, the hardware may be underpowered."},
        "HIGH_LOAD_AVG_FAIL": {"type": "failure", "description": "System 5-minute load average ({load:.2f}) is critically high for {cores} CPU cores (load > {cores_x4}).", "reason": "The CPU is severely overloaded and cannot keep up with the workload. Video processing will likely halt or fall significantly behind.", "fix": "Use 'htop' to identify processes causing high load. This hardware is likely underpowered for the workload."},
        "IO_ERROR_DETECTED": {"type": "warning", "description": "I/O errors detected in the kernel log.", "reason": "The operating system is reporting errors when reading from or writing to a storage device. This is a strong indicator of a failing disk.", "fix": "This is a critical hardware warning. Back up your data immediately. Use tools like 'smartctl' to diagnose the disk and prepare to replace it."},
        "FS_READ_ONLY": {"type": "failure", "description": "A filesystem was remounted as read-only.", "reason": "This is a critical kernel protection measure, usually triggered by severe filesystem corruption or disk failure. The system cannot write any new data.", "fix": "Check 'dmesg' for details. Reboot the system and run 'fsck' on the affected partition. Prepare to replace the disk."},
        "OOM_KILLER_ACTIVE": {"type": "warning", "description": "The Out-of-Memory (OOM) killer was activated.", "reason": "The system ran out of available RAM and was forced to kill processes to survive. This can silently stop video capture or processing.", "fix": "Use 'htop' or 'free -m' to check memory usage. Identify the memory-hungry process. The system may need a RAM upgrade or swap adjustment."},
        "SEGFAULT_DETECTED": {"type": "warning", "description": "A segmentation fault was detected in the system log.", "reason": "A key program (like 'ffmpeg', 'python', or 's3fs') crashed due to a critical memory error. This indicates software instability or hardware issues.", "fix": "Note the program that segfaulted from the log line. Ensure your software is up to date. If it persists, it could be a sign of faulty RAM."},
        "NTP_LOG_ERRORS": {"type": "info", "description": "NTP time synchronization errors found in the log.", "reason": "The log shows a history of failures to contact time servers, even if the system is currently synced. This indicates an unstable network connection or bad NTP config.", "fix": "Verify network connectivity. Check '/etc/systemd/timesyncd.conf' or 'ntp.conf' to ensure valid time servers are listed."},
        "USB_ERROR_DETECTED": {"type": "info", "description": "USB device errors detected in the log.", "reason": "The kernel is reporting errors communicating with a USB device (e.g., disconnects, failed enumeration). This can indicate faulty hardware, bad cables, or insufficient power.", "fix": "Check all USB connections and cables. If a USB hub is used, ensure it has its own power supply. Check 'dmesg' for more details."},
        # Disk Space
        "DISK_CRITICAL": {"type": "failure", "description": "Disk space on {path} is critically low ({percent_free:.2f}% < 1% free).", "reason": "If the disk fills up completely, video recording will stop, and the system may become unstable.", "fix": "Delete old, unneeded files from '/mnt/ams2/SD/' and '/mnt/ams2/HD/'. Consider archiving old meteor data."},
        "DISK_LOW": {"type": "warning", "description": "Disk space on {path} is low ({percent_free:.2f}% < 3% free).", "reason": "The video storage drive is nearing capacity. Action should be taken soon.", "fix": "Proactively clean up old video files to prevent the disk from becoming full."},
        "DISK_INFO": {"type": "info", "description": "Disk space on {path} is getting low ({percent_free:.2f}% < 5% free).", "reason": "This is an early warning that the video storage drive is filling up.", "fix": "Monitor disk space and plan for cleanup or archiving."},
        # NTP
        "NTP_NO_SYNC": {"type": "warning", "description": "System clock is not synchronized with a time server (NTP).", "reason": "Accurate timing is absolutely critical for meteor observation. If the clock is wrong, the scientific data is useless.", "fix": "Ensure the system is connected to the internet. Run 'sudo timedatectl set-ntp true' and check 'timedatectl status'."},
        # Dependencies
        "PYTHON_PKG_MISSING": {"type": "failure", "description": "A required Python package is missing.", "reason": "The software relies on external Python libraries to function. A missing library will cause scripts to crash.", "fix": "Install the missing package using pip: 'sudo python3 -m pip install {pkg}'."},
        "PYTHON_PKG_MISSING_WARN": {"type": "warning", "description": "A recommended Python package is missing.", "reason": "This package provides non-critical functionality. The system will run, but some features might be limited.", "fix": "Install the missing package using pip: 'sudo python3 -m pip install {pkg}'."},
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
        "STALE_PROCESS_FOUND": {"type": "info", "description": "Multiple instances of a key background process ({proc}) are running.", "reason": "This usually indicates a problem with cron job management or a stale lockfile. Conflicting processes can corrupt data.", "fix": "Manually stop all instances of the process ('sudo pkill -f {proc}') and let the watchdog restart it correctly."},
        # Logs & Status
        "WD_LOG_STALE": {"type": "failure", "description": "Watch-dog log ('/tmp/wd.txt') has not been updated in over 6 minutes.", "reason": "The 'watch-dog.py' cron job should run every minute. If its log is old, the cron job is likely failing or cron is not running.", "fix": "Check that the 'cron' service is running ('systemctl status cron'). Manually run the watchdog script to check for errors."},
        "WD_LOG_ERROR": {"type": "failure", "description": "An error was found in the recent watch-dog log entries.", "reason": "The log file itself contains error messages, indicating active problems like camera streams being down.", "fix": "Read the log file ('cat /tmp/wd.txt') to see the specific errors and address them directly."},
        # Captures
        "NO_RECENT_CAPTURES": {"type": "failure", "description": "No recent video captures were found for one or more cameras.", "reason": "The system is not saving new video files. This is a primary indicator that the video capture has failed.", "fix": "This is a critical error. Check camera connectivity and the FFMPEG running process for the affected cameras."},
        "FPS_TOO_LOW": {"type": "warning", "description": "A recent video file has an FPS below 12.5 ({fps:.1f} FPS).", "reason": "Low FPS indicates a problem with the capture stream, camera settings, or system performance, leading to choppy video.", "fix": "Check camera's stream settings. Ensure the 'ffmpeg' command specifies a stable FPS and that the system is not overloaded."},
        "FPS_INCONSISTENT": {"type": "warning", "description": "Recent {stream_type} video files show high FPS variation (min: {min_fps:.1f}, max: {max_fps:.1f}).", "reason": "Inconsistent frame rates suggest an unstable capture process, which can corrupt video files or cause issues with analysis.", "fix": "This is often linked to an unstable camera connection or an overloaded CPU. Check 'ffmpeg' processes and system load."},
        # Corrupt Files
        "CORRUPT_FILE_FOUND": {"type": "warning", "description": "A likely corrupt (very small) video file was found.", "reason": "The system is creating empty or tiny video files, pointing to a problem with the 'ffmpeg' stream capture.", "fix": "This is often linked to an unstable camera connection or failing 'ffmpeg' process. Check network cables and power."},
        # Processing Queue
        "STALLED_QUEUE": {"type": "failure", "description": "Processing queue is stalled. Oldest file is {age:.1f} hours old.", "reason": "The analysis script ('Process.py') is not processing files. It is likely stuck on a corrupt file or has crashed.", "fix": "Manually run the processing script ('sudo -u ams /home/ams/amscams/pipeline/Process.py ssp') to see the error. Consider moving the oldest file in '/mnt/ams2/SD/proc2/' to unblock the queue."},
        "HIGH_BACKLOG": {"type": "warning", "description": "High processing backlog: {count} new files in the last hour.", "reason": "The system is capturing videos faster than it can process them, possibly due to a meteor shower or insufficient CPU power.", "fix": "This is usually temporary. If it persists for days and system load is high, the hardware may be underpowered."},
        # Cron
        "NO_CRONTAB": {"type": "failure", "description": "No crontab was found for the 'ams' user.", "reason": "The scheduled tasks that automate everything are missing entirely.", "fix": "You must re-run the setup script to install the cron jobs."},
        "CRON_JOB_MISSING": {"type": "failure", "description": "One or more expected cron jobs are missing.", "reason": "A specific, essential scheduled task is not in the crontab, which will cause a part of the system to not function.", "fix": "Re-run the setup script or manually add the missing line to the 'ams' user's crontab ('sudo -u ams crontab -e')."},
        
        # --- NMN Checks ---
        "NMN_NO_METEOR_USER": {"type": "failure", "description": "User 'meteor' does not exist on the system.", "reason": "The NMN software suite is designed to run under the 'meteor' user.", "fix": "The user 'meteor' needs to be created."},
        "NMN_HOME_MISSING": {"type": "failure", "description": "Directory '/home/meteor' is missing.", "reason": "The 'meteor' user's home directory is required for NMN software.", "fix": "Create the directory and set ownership: 'sudo mkdir /home/meteor; sudo chown meteor:meteor /home/meteor'"},
        "NMN_HOME_BAD_OWNER": {"type": "failure", "description": "Directory '/home/meteor' is not owned by 'meteor'.", "reason": "Incorrect ownership will cause permission errors for the NMN software.", "fix": "Run 'sudo chown meteor:meteor /home/meteor'"},
        "NMN_NMN_DIR_MISSING": {"type": "failure", "description": "Directory '/home/meteor/nmn' is missing.", "reason": "This directory contains the NMN software.", "fix": "Restore the NMN software, likely from a git clone."},
        "NMN_NMN_DIR_NOT_WRITABLE": {"type": "failure", "description": "Directory '/home/meteor/nmn' is not writable by 'meteor'.", "reason": "The 'meteor' user must be able to write to this directory.", "fix": "Check permissions and run 'sudo chown -R meteor:meteor /home/meteor/nmn'"},
        "NMN_BIN_NOT_SYMLINK": {"type": "failure", "description": "'/home/meteor/bin' is not a symlink.", "reason": "The 'bin' directory should be a symlink to '/home/meteor/nmn/bin'.", "fix": "Run 'sudo -u meteor ln -s /home/meteor/nmn/bin /home/meteor/bin' (after removing the existing file/dir)."},
        "NMN_BIN_WRONG_SYMLINK": {"type": "failure", "description": "'/home/meteor/bin' symlink points to the wrong target ('{target}').", "reason": "The symlink must point to '/home/meteor/nmn/bin' or 'nmn/bin'.", "fix": "Recreate the symlink: 'sudo -u meteor ln -sfn nmn/bin /home/meteor/bin'"},
        "NMN_PY_IMPORT_MISSING": {"type": "failure", "description": "A required Python package for NMN is missing.", "reason": "A script in '/home/meteor/nmn/bin/' imports a package that is not installed for the 'meteor' user.", "fix": "Install the missing package using pip: 'sudo python3 -m pip install {pkg}' or potentially 'sudo -u meteor python3 -m pip install {pkg}'."},
        "NMN_CONFIG_MISSING": {"type": "failure", "description": "NMN config file '/etc/meteor.cfg' is missing.", "reason": "This file contains critical station configuration.", "fix": "Restore the file from a backup or create it with the required [astronomy] and [station] sections."},
        "NMN_CONFIG_NOT_READABLE": {"type": "failure", "description": "NMN config '/etc/meteor.cfg' is not readable by 'meteor'.", "reason": "The 'meteor' user must be able to read this file.", "fix": "Run 'sudo chmod 644 /etc/meteor.cfg' and verify ownership."},
        "NMN_CONFIG_PARSE_ERROR": {"type": "failure", "description": "NMN config '/etc/meteor.cfg' cannot be parsed.", "reason": "The file is corrupted or not a valid INI file.", "fix": "Check the file for syntax errors."},
        "NMN_CONFIG_SECTION_MISSING": {"type": "failure", "description": "NMN config is missing required section '[{section}]'.", "reason": "The config file must contain both [astronomy] and [station] sections.", "fix": "Edit '/etc/meteor.cfg' and add the missing section."},
        "NMN_CONFIG_KEY_MISSING": {"type": "failure", "description": "NMN config is missing key '{key}' in section '[{section}]'.", "reason": "The config file is missing a required configuration key.", "fix": "Edit '/etc/meteor.cfg' and add the missing key."},
        "NMN_CONFIG_LAT_INVALID": {"type": "failure", "description": "NMN config latitude '{lat}' is invalid or out of range.", "reason": "Latitude must be a number between -90 and 90.", "fix": "Correct the 'latitude' value in '/etc/meteor.cfg'."},
        "NMN_CONFIG_LNG_INVALID": {"type": "failure", "description": "NMN config longitude '{lng}' is invalid or out of range.", "reason": "Longitude must be a number between -180 and 180.", "fix": "Correct the 'longitude' value in '/etc/meteor.cfg'."},
        "NMN_CONFIG_ELEV_INVALID": {"type": "failure", "description": "NMN config elevation '{elev}' is invalid or out of range.", "reason": "Elevation must be a number between -500 and 9000.", "fix": "Correct the 'elevation' value in '/etc/meteor.cfg'."},
        "NMN_CAM_DIR_MISSING": {"type": "failure", "description": "Camera directory is missing: {path}", "reason": "The directory to store camera data is required.", "fix": "Create the directory: 'sudo mkdir {path}; sudo chown meteor:meteor {path}'"},
        "NMN_CAM_DIR_BAD_OWNER": {"type": "failure", "description": "Camera directory has incorrect owner: {path}", "reason": "The 'meteor' user must own this directory to write data.", "fix": "Run 'sudo chown meteor:meteor {path}'"},
        "NMN_CAM_DIR_NOT_WRITABLE": {"type": "failure", "description": "Camera directory is not writable by 'meteor': {path}", "reason": "The 'meteor' user must be able to write to this directory.", "fix": "Check permissions and run 'sudo chown meteor:meteor {path}'"},
        "NMN_CAM_FILE_MISSING": {"type": "failure", "description": "Required camera file is missing: {path}", "reason": "Files like grid.png and lens.pto are required for calibration.", "fix": "Restore the missing file to {path}."},
        "NMN_CAM_FILE_NOT_READABLE": {"type": "failure", "description": "Required camera file is not readable by all users: {path}", "reason": "These files must be world-readable.", "fix": "Run 'sudo chmod 644 {path}' or ensure link target is readable."},
        "NMN_CAM_FILE_ZERO_SIZE": {"type": "failure", "description": "Required camera file is empty (zero bytes): {path}", "reason": "Calibration files cannot be empty.", "fix": "Restore a valid, non-empty version of the file."},
        "NMN_SNAPSHOT_MISSING": {"type": "failure", "description": "Snapshot file is missing: {path}", "reason": "The snapshot.jpg file is required to check camera status.", "fix": "Ensure the camera capture script is running and creating snapshots."},
        "NMN_SNAPSHOT_NOT_READABLE": {"type": "failure", "description": "Snapshot file is not readable by all users: {path}", "reason": "The snapshot file must be world-readable.", "fix": "Run 'sudo chmod 644 {path}'."},
        "NMN_SNAPSHOT_ZERO_SIZE": {"type": "failure", "description": "Snapshot file is empty (zero bytes): {path}", "reason": "An empty snapshot indicates a capture failure.", "fix": "Check the camera and capture script."},
        "NMN_SNAPSHOT_STALE": {"type": "failure", "description": "Snapshot file is stale (older than 2 minutes): {path}", "reason": "The capture script appears to be stalled or not running.", "fix": "Restart the camera capture script or service."},
        "NMN_MIRROR_PROC_DOWN": {"type": "failure", "description": "The 'mirror.py' process is not running.", "reason": "This process is responsible for data mirroring.", "fix": "Check the 'mirror' service status: 'systemctl status mirror'"},
        "NMN_MIRROR_SVC_NOT_ACTIVE": {"type": "failure", "description": "The 'mirror' systemd service is not active.", "reason": "The service is not running or has failed.", "fix": "Run 'sudo systemctl start mirror' and check logs: 'journalctl -u mirror'"},
        "NMN_MIRROR_SVC_NOT_ENABLED": {"type": "failure", "description": "The 'mirror' systemd service is not enabled.", "reason": "The service will not start automatically on boot.", "fix": "Run 'sudo systemctl enable mirror'"},
        "NMN_AUTOSSH_PROC_DOWN": {"type": "failure", "description": "The 'autossh' process for 'meteor@norskmeteornettverk.no' is not running.", "reason": "The reverse SSH tunnel to the NMN server is down.", "fix": "Check the 'autossh-tunnel' service: 'systemctl status autossh-tunnel'"},
        "NMN_AUTOSSH_SVC_NOT_ACTIVE": {"type": "failure", "description": "The 'autossh-tunnel' systemd service is not active.", "reason": "The service is not running or has failed.", "fix": "Run 'sudo systemctl start autossh-tunnel' and check logs: 'journalctl -u autossh-tunnel'"},
        "NMN_AUTOSSH_SVC_NOT_ENABLED": {"type": "failure", "description": "The 'autossh-tunnel' systemd service is not enabled.", "reason": "The service will not start automatically on boot.", "fix": "Run 'sudo systemctl enable autossh-tunnel'"},
        "NMN_SSH_PROC_DOWN": {"type": "failure", "description": "No active 'ssh' child process for 'meteor@norskmeteornettverk.no' was found.", "reason": "The 'autossh' process is running but has failed to establish an active SSH connection.", "fix": "Check 'systemctl status autossh-tunnel' and network connectivity."},
        "NMN_MIRROR_LOG_MISSING": {"type": "warning", "description": "Mirror log file '/home/meteor/mirror.log' is missing.", "reason": "The log file is missing, cannot check for errors.", "fix": "This may be normal if the service has never run. If it should be running, check service status."},
        "NMN_CAM_NO_DATE_DIRS": {"type": "warning", "description": "No date-formatted subdirectories (YYYYMMDD) found in {path}.", "reason": "The capture script may not be saving daily archives.", "fix": "Ensure the capture script is running and has permissions to create directories in {path}."},
        "NMN_CAM_STALE_DATE_DIRS": {"type": "warning", "description": "The most recent date directory in {path} is '{latest_dir}'.", "reason": "The capture script has not created a directory for today or yesterday. It may be stalled.", "fix": "Check the capture script and ensure it is running and processing new data."},
        "NMN_METEOR_DISK_CRITICAL": {"type": "failure", "description": "Disk space on {path} is critically low ({percent_free:.2f}% < 1% free).", "reason": "If the disk fills up completely, video recording will stop.", "fix": "Delete old, unneeded files from '/meteor/'."},
        "NMN_METEOR_DISK_LOW": {"type": "warning", "description": "Disk space on {path} is low ({percent_free:.2f}% < 3% free).", "reason": "The video storage drive is nearing capacity. Action should be taken soon.", "fix": "Proactively clean up old video files to prevent the disk from becoming full."},
        "NMN_METEOR_DISK_INFO": {"type": "info", "description": "Disk space on {path} is getting low ({percent_free:.2f}% < 5% free).", "reason": "This is an early warning that the video storage drive is filling up.", "fix": "Monitor disk space and plan for cleanup or archiving."},
        "NMN_HOST_UNREACHABLE_80": {"type": "failure", "description": "Cannot connect to norskmeteornettverk.no on port 80 (HTTP).", "reason": "Connectivity to the main NMN web server is required for some services or indicates a general network issue.", "fix": "Check this station's internet connection, firewall rules, and DNS settings. Verify the host is online."},
        "NMN_HOST_UNREACHABLE_22": {"type": "failure", "description": "Cannot connect to norskmeteornettverk.no on port 22 (SSH).", "reason": "Connectivity to the NMN SSH server is critical for the autossh reverse tunnel to function.", "fix": "Check this station's internet connection, firewall rules (especially outbound port 22), and DNS settings."}
    }

    def get_error(self, code):
        """Fetches error details from the catalog by its code."""
        return self._catalog.get(code)

class AS7Diagnostic:
    """
    A diagnostic tool to verify the setup and operational health of the AS7
    Meteor Detection Software.
    """

    def __init__(self, is_root=False, do_nmn_checks=False):
        """Initializes the diagnostic tool."""
        self.is_root = is_root
        self.do_nmn_checks = do_nmn_checks # Renamed variable
        self.GREEN, self.YELLOW, self.RED, self.CYAN, self.RESET = '\033[92m', '\033[93m', '\033[91m', '\033[96m', '\033[0m'
        self.results = {'success': [], 'info': [], 'warning': [], 'failure': []}
        self.config_data = {}
        self.error_catalog = ErrorCatalog()
        self.logged_errors = []
        
        self.meteor_pwd = None
        try:
            # Try getting meteor user info early if NMN checks are requested
            if self.do_nmn_checks:
                self.meteor_pwd = pwd.getpwnam('meteor')
        except KeyError:
            pass # Will be handled in the NMN user check function
            
        try:
            self.psutil = importlib.import_module("psutil")
        except ImportError:
            self.psutil = None

    def print_header(self):
        """Prints a welcome header for the diagnostic tool."""
        print("="*60)
        print(" AS7 Meteor Detection Software - Comprehensive Health Check")
        if self.do_nmn_checks:
            print("             (NMN System Checks ENABLED)")
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
        
        if error_info['type'] == 'failure':
            color = self.RED
            label = " FAIL "
        elif error_info['type'] == 'warning':
            color = self.YELLOW
            label = " WARN "
        else: # info
            color = self.CYAN
            label = " INFO "
        
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
        
        if error_code in ["FS_READ_ONLY", "OOM_KILLER_ACTIVE", "SEGFAULT_DETECTED", "NTP_LOG_ERRORS", "USB_ERROR_DETECTED"]:
             return f"Found {context.get('count', 0)} log entries. Last: \"{context.get('last_error', '')}\""
        
        if error_code == "PYTHON_PKG_MISSING": return f"Python package is missing: {context.get('pkg')}"
        if error_code == "PYTHON_PKG_MISSING_WARN": return f"Recommended Python package is missing: {context.get('pkg')}"
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

        # NMN Messages
        if error_code == "NMN_BIN_WRONG_SYMLINK": return f"'/home/meteor/bin' symlink points to '{context.get('target')}', not '/home/meteor/nmn/bin' or 'nmn/bin'"
        if error_code == "NMN_PY_IMPORT_MISSING": return f"NMN script import failed for user 'meteor': '{context.get('pkg')}' is not installed"
        if error_code == "NMN_CONFIG_SECTION_MISSING": return f"NMN config '/etc/meteor.cfg' is missing section '[{context.get('section')}]'"
        if error_code == "NMN_CONFIG_KEY_MISSING": return f"NMN config '/etc/meteor.cfg' is missing key '{context.get('key')}' in section '[{context.get('section')}]'"
        if error_code in ["NMN_CAM_DIR_MISSING", "NMN_CAM_DIR_BAD_OWNER", "NMN_CAM_DIR_NOT_WRITABLE",
                           "NMN_CAM_FILE_MISSING", "NMN_CAM_FILE_NOT_READABLE", "NMN_CAM_FILE_ZERO_SIZE",
                           "NMN_SNAPSHOT_MISSING", "NMN_SNAPSHOT_NOT_READABLE", "NMN_SNAPSHOT_ZERO_SIZE",
                           "NMN_SNAPSHOT_STALE", "NMN_CAM_NO_DATE_DIRS"]:
            return f"{self.error_catalog.get_error(error_code)['description'].format(**context)}"
        if error_code == "NMN_CAM_STALE_DATE_DIRS": return f"Most recent date dir in {context.get('path')} is '{context.get('latest_dir')}', which is not current."

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
        self.check_syslog()
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
        
        if self.do_nmn_checks:
            self.run_nmn_checks() # Call the method
            
        self.print_summary()

    def _get_top_swap_processes(self):
        """Identifies the top 5 processes using swap memory."""
        swap_users = []
        if not self.psutil:
            return []
        try:
            for proc in self.psutil.process_iter(['pid', 'name']):
                try:
                    with open(f'/proc/{proc.info["pid"]}/status') as f:
                        for line in f:
                            if line.startswith('VmSwap:'):
                                swap_kb = int(line.split()[1])
                                if swap_kb > 0:
                                    swap_users.append({'name': proc.info['name'], 'swap_mb': swap_kb / 1024})
                                break
                except (FileNotFoundError, PermissionError, ValueError, self.psutil.NoSuchProcess):
                    continue
            return sorted(swap_users, key=lambda x: x['swap_mb'], reverse=True)[:5]
        except Exception:
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
                    dir_owner_uid = os.stat(d).st_uid
                    dir_owner_name = pwd.getpwuid(dir_owner_uid).pw_name
                    if dir_owner_name != 'ams':
                        self.log_issue("BAD_OWNER", {'path': d})
                except PermissionError:
                    self.log_issue("PERMISSION_DENIED", {'check': f"ownership of {d}"})
                except KeyError: pass # User ID might not exist, ignore
                except Exception as e:
                     self.log_issue("PERMISSION_DENIED", {'check': f"ownership of {d} due to {e}"})
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
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"reading permissions for {creds_file}: {e}"})

        archive_path = "/mnt/archive.allsky.tv"
        try:
            cmd = ['df']
            current_user = pwd.getpwuid(os.getuid()).pw_name
            can_run_as_ams = self.is_root or (current_user == 'ams')

            if can_run_as_ams:
                if self.is_root: # If root, explicitly run as ams
                     cmd = ['sudo', '-u', 'ams', 'df']
                # else: run as current user who is ams
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if archive_path not in result.stdout:
                     self.log_issue("ARCHIVE_NOT_MOUNTED")
                     return
            else: # Cannot run as ams, just check if mounted for current user
                 result = subprocess.run(cmd, capture_output=True, text=True)
                 if archive_path not in result.stdout:
                     self.log_issue("ARCHIVE_NOT_MOUNTED")
                     self.log_issue("PERMISSION_DENIED", {'check': f"full check of {archive_path}. Run as root or ams."})
                     return
                 else:
                     self.log_success(f"'{archive_path}' appears mounted (basic check).")
                     self.log_issue("PERMISSION_DENIED", {'check': f"responsiveness and writability of {archive_path}. Run as root or ams."})
                     return # Cannot perform further checks reliably

        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
             self.log_issue("PERMISSION_DENIED", {'check': "running 'df' command, possibly as 'ams' user"})
             return
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"checking mount status of {archive_path}: {e}"})
             return

        # --- From here assume we can run commands as 'ams' ---
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
                self.log_issue("PERMISSION_DENIED", {'check': f"listing contents of {archive_path} as ams."})
            return
        except Exception as e:
            self.log_issue("ARCHIVE_HUNG", {'details': str(e)}) # Add details if available
            return

        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json")
        station_id = as6_data.get("site", {}).get("ams_id") if as6_data else None

        if not station_id:
            # This case might be hit if as6.json read failed earlier
            self.log_issue("PERMISSION_DENIED", {'check': f"archive writability. Station ID could not be determined."})
            return

        station_archive_path = os.path.join(archive_path, station_id)
        
        dir_exists_cmd = ['test', '-d', station_archive_path]
        if self.is_root:
            dir_exists_check = ['sudo', '-u', 'ams'] + dir_exists_cmd
        else: # running as ams
            dir_exists_check = dir_exists_cmd

        try:
            subprocess.run(dir_exists_check, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_issue("DIR_MISSING", {'path': station_archive_path})
            return

        self._check_dir_writable_by(station_archive_path, 'ams', "ARCHIVE_NOT_WRITABLE", indent=1)

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
                except Exception as e:
                    self.log_issue("PERMISSION_DENIED", {'check': f"processing {f}: {e}"})
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
            except Exception as e:
                 self.log_issue("PERMISSION_DENIED", {'check': f"reading {defaults_file}: {e}"})
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
                    ip_addr = info.get('ip')
                    if ip_addr: # Only validate if IP exists
                        socket.inet_aton(ip_addr)
                        self.log_success(f"Camera '{key}' has a valid IP format.", indent=2)
                    else:
                        self.log_issue("AS6_JSON_BAD_IP", {'cam_key': key}, indent=2) # Treat missing IP as invalid
                except (socket.error, TypeError):
                     self.log_issue("AS6_JSON_BAD_IP", {'cam_key': key}, indent=2)

    def check_config_consistency(self):
        """Checks for mismatches between config files and the system hostname."""
        print("\n--- Checking Configuration Consistency ---")
        setup_data = self.config_data.get("/home/ams/amscams/conf/setup.json")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json")
        if not setup_data or "station_id" not in setup_data:
             self.log_issue("PERMISSION_DENIED", {'check': "config consistency (setup.json not loaded)"})
             return
        if not as6_data or "site" not in as6_data or "ams_id" not in as6_data["site"]:
             self.log_issue("PERMISSION_DENIED", {'check': "config consistency (as6.json not loaded)"})
             return


        setup_id = setup_data.get("station_id")
        as6_id = as6_data.get("site", {}).get("ams_id")

        if setup_id != as6_id: self.log_issue("ID_MISMATCH", {'setup_id': setup_id, 'as6_id': as6_id})
        else: self.log_success(f"Station ID is consistent between config files ('{setup_id}').")

        try:
            with open('/etc/hostname', 'r') as f: hostname = f.read().strip()
            if as6_id and hostname != as6_id: self.log_issue("HOSTNAME_MISMATCH", {'hostname': hostname, 'as6_id': as6_id})
            else: self.log_success(f"System hostname ('{hostname}') matches station ID.")
        except FileNotFoundError:
             self.log_issue("CONFIG_MISSING", {'path': '/etc/hostname'})
        except PermissionError:
             self.log_issue("PERMISSION_DENIED", {'check': "reading /etc/hostname"})
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"reading /etc/hostname: {e}"})

    def check_system_health(self):
        """Monitors system hardware, primarily CPU temperature."""
        print("\n--- Checking System Health ---")
        try:
            result = subprocess.run(['sensors'], capture_output=True, text=True, check=False)
            # Sensors command might require root or specific group membership
            if result.returncode != 0 and 'not found' not in result.stderr.lower():
                self.log_issue("PERMISSION_DENIED", {'check': "reading CPU temperature (sensors command failed)"})
                return
            
            if result.returncode == 0:
                max_temp = 0
                for line in result.stdout.split('\n'):
                    # More robust regex to handle different outputs
                    match = re.search(r'(?:Package id \d+|Core \d+):\s*\+?(\d{1,3}\.\d)°C', line)
                    if match:
                        try:
                           max_temp = max(max_temp, float(match.group(1)))
                        except ValueError: pass # Ignore if number format is weird
                if max_temp == 0: # If no lines matched expected format
                     self.log_issue("PERMISSION_DENIED", {'check': "parsing CPU temperature from 'sensors' output"})
                elif max_temp > 85.0: self.log_issue("CPU_TEMP_CRITICAL", {'temp': max_temp})
                else: self.log_success(f"CPU temperature is normal: {max_temp}°C")
        except FileNotFoundError:
            self.log_issue("SENSORS_MISSING")
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"running 'sensors': {e}"})


    def check_system_resources(self):
        """Checks for high memory usage, swap activity, and system load."""
        print("\n--- Checking System Resources ---")
        if not self.psutil:
            self.log_issue("PYTHON_PKG_MISSING", {'pkg': 'psutil'})
            return
            
        try:
            mem = self.psutil.virtual_memory()
            swap = self.psutil.swap_memory()

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

            cpu_cores = os.cpu_count() or 1 # Default to 1 core if detection fails
            load_avg = self.psutil.getloadavg()
            load = load_avg[1] # Use 5-minute load average
            if load > (cpu_cores * 4):
                self.log_issue("HIGH_LOAD_AVG_FAIL", {'load': load, 'cores': cpu_cores, 'cores_x4': cpu_cores*4})
            elif load > (cpu_cores * 2):
                self.log_issue("HIGH_LOAD_AVG_WARN", {'load': load, 'cores': cpu_cores, 'cores_x2': cpu_cores*2})
            elif load > cpu_cores:
                self.log_issue("HIGH_LOAD_AVG", {'load': load, 'cores': cpu_cores})
            else:
                self.log_success(f"System load average is normal ({load:.2f} on {cpu_cores} cores).")
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"getting system resources: {e}"})


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
        except FileNotFoundError:
             self.log_issue("SYS_PKG_MISSING", {'pkg': 'dmesg', 'pkg_name': 'util-linux'})
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"running dmesg: {e}"})


    def check_syslog(self):
        """Scans /var/log/syslog for critical system-level errors."""
        print("\n--- Checking System Log (/var/log/syslog) ---")
        if not self.is_root:
            # Check if current user can read the log file directly
            can_read = os.access("/var/log/syslog", os.R_OK) or os.access("/var/log/syslog.1", os.R_OK)
            if not can_read:
                self.log_issue("PERMISSION_DENIED", {'check': "reading system logs. Run as root or add user to 'adm' group."})
                return
            else:
                self.log_success("Reading system logs as non-root (may be incomplete).")

        log_files = ["/var/log/syslog", "/var/log/syslog.1"]
        
        checks = {
            "FS_READ_ONLY": r'Remounting filesystem read-only',
            "OOM_KILLER_ACTIVE": r'(Out of memory: Kill process|oom-killer)',
            "SEGFAULT_DETECTED": r'segfault at',
            "NTP_LOG_ERRORS": r'(ntpd|systemd-timesyncd|chrony).*(fail|timeout|error)',
            "USB_ERROR_DETECTED": r'usb .*: (error|disconnect|failed enumeration)',
        }
        
        found_issues = {code: [] for code in checks}
        scanned_files = False

        for log_path in log_files:
            if not os.path.exists(log_path):
                continue
            # Double-check readability here in case is_root was false but permissions allow it
            if not os.access(log_path, os.R_OK):
                continue
            
            scanned_files = True
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    # Read only last N lines for performance? Might miss older errors.
                    # For now, read the whole file accessible.
                    for line in f:
                        for code, pattern in checks.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                # Limit stored line length
                                found_issues[code].append(line.strip()[:200]) 
                                break # Move to next line
            except Exception as e:
                self.log_issue("PERMISSION_DENIED", {'check': f"reading {log_path}: {e}"})

        if not scanned_files:
            self.log_issue("PERMISSION_DENIED", {'check': "reading system logs (files not found or accessible)."})
            return

        if not any(found_issues.values()):
            self.log_success("No critical errors found in recent system logs.")
        else:
            for code, lines in found_issues.items():
                if lines:
                    count = len(lines)
                    last_error = lines[-1]
                    self.log_issue(code, {'count': count, 'last_error': last_error})

    def _check_disk_usage(self, path, err_crit, err_warn, err_info):
        """Helper function to check disk space on a given path."""
        if not os.path.exists(path):
            self.log_issue("DIR_MISSING", {'path': path})
            return
            
        try:
            stat = os.statvfs(path)
            bytes_available = stat.f_bavail * stat.f_frsize
            bytes_total = stat.f_blocks * stat.f_frsize
            if bytes_total == 0:
                # This could happen for special mounts or network filesystems not fully ready
                self.log_success(f"Cannot determine disk space for {path} (total size is 0). Skipping check.")
                return
                
            percent_free = (bytes_available / bytes_total) * 100
            
            # Use thresholds: FAIL < 1%, WARN < 3%, INFO < 5%
            if percent_free < 1:
                self.log_issue(err_crit, {'path': path, 'percent_free': percent_free})
            elif percent_free < 3:
                self.log_issue(err_warn, {'path': path, 'percent_free': percent_free})
            elif percent_free < 5:
                self.log_issue(err_info, {'path': path, 'percent_free': percent_free})
            else:
                self.log_success(f"Disk space on {path} is sufficient: {bytes_available/1e9:.2f}GB free ({percent_free:.2f}%).")
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"checking disk space for {path}: {e}"})

    def check_disk_space(self):
        """Checks the available disk space on the primary video storage mount."""
        print("\n--- Checking Disk Space ---")
        self._check_disk_usage("/mnt/ams2", "DISK_CRITICAL", "DISK_LOW", "DISK_INFO")

    def check_ntp_sync(self):
        """Verifies that the system clock is synchronized via NTP."""
        print("\n--- Checking Time Synchronization ---")
        try:
            result = subprocess.run(['timedatectl', 'status'], capture_output=True, text=True)
            if "System clock synchronized: yes" in result.stdout: self.log_success("System clock is synchronized with NTP.")
            else: self.log_issue("NTP_NO_SYNC")
        except FileNotFoundError:
             self.log_issue("SYS_PKG_MISSING", {'pkg': 'timedatectl', 'pkg_name': 'systemd'})
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"running timedatectl: {e}"})


    def check_dependencies(self):
        """Checks for all Python packages, system packages, and external scripts."""
        print("\n--- Checking Dependencies ---")
        
        # Critical packages
        pkgs = ["netifaces", "requests", "tabulate", "consolemenu", "ephem"]
        if not self.psutil:
             pkgs.append("psutil") # Add to list to show it's missing
             
        for pkg in pkgs:
            try:
                # Special check for psutil as it might have failed during init
                if pkg == "psutil" and not self.psutil:
                    raise ImportError("psutil not found during init")
                elif pkg != "psutil":
                    importlib.import_module(pkg)
                self.log_success(f"Python package installed: {pkg}", indent=1)
            except ImportError: self.log_issue("PYTHON_PKG_MISSING", {'pkg': pkg}, indent=1)

        # Recommended package (Warning if missing)
        pkg = "timezonefinder"
        try:
            importlib.import_module(pkg)
            self.log_success(f"Python package installed: {pkg}", indent=1)
        except ImportError: self.log_issue("PYTHON_PKG_MISSING_WARN", {'pkg': pkg}, indent=1)

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

    def _check_dir_writable_by(self, path, user_name, error_code, indent=0):
        """Helper to check if a directory is writable by a specific user."""
        if not os.path.isdir(path):
            # This should have been checked before calling
            self.log_issue("DIR_MISSING", {'path': path}, indent=indent)
            return False

        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            current_user = str(os.getuid()) # Fallback to UID if user name not found

        # If running as the target user, use os.access
        if not self.is_root and current_user == user_name:
            if os.access(path, os.W_OK):
                self.log_success(f"Directory '{path}' is writable by current user '{user_name}'.", indent=indent)
                return True
            else:
                self.log_issue(error_code, {'path': path}, indent=indent)
                return False
        # If running as root, use sudo
        elif self.is_root:
            test_file = os.path.join(path, ".diag_write_test")
            cmd_touch = ['sudo', '-u', user_name, 'touch', test_file]
            cmd_rm = ['sudo', '-u', user_name, 'rm', '-f', test_file]
            
            try:
                subprocess.run(cmd_touch, check=True, capture_output=True, text=True, timeout=10)
                self.log_success(f"Directory '{path}' is writable by user '{user_name}'.", indent=indent)
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                self.log_issue(error_code, {'path': path}, indent=indent)
                return False
            finally:
                subprocess.run(cmd_rm, capture_output=True) # Try to clean up even if touch failed
        # If running as non-root, non-target user
        else:
            self.log_issue("PERMISSION_DENIED", {'check': f"writable status of {path} by user {user_name}. Run as root or {user_name}."}, indent=indent)
            return False


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
            self._check_dir_writable_by(proc_dir, 'ams', "PROC_DIR_NOT_WRITABLE")

    def _check_port_open(self, host, port, timeout=5):
        """Helper to check if a TCP port is open on a host."""
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            return True
        except (socket.timeout, socket.error, OSError):
            return False

    def check_network_config(self):
        """Performs a basic internet connectivity test."""
        print("\n--- Checking Network ---")
        if self._check_port_open("8.8.8.8", 53):
            self.log_success("Internet connectivity to 8.8.8.8:53 (Google DNS) is working.")
        else:
            self.log_issue("NO_INTERNET")

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
                                if network.is_private:
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
        except ImportError: pass # netifaces might be missing, already flagged
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"network sanity check: {e}"})


    def check_api_connectivity(self):
        """Checks if the AllSky7 API is reachable."""
        print("\n--- Checking API Connectivity ---")
        try:
            import requests
            requests.get("https://kyvegys798.execute-api.us-east-1.amazonaws.com/api/allskyapi", timeout=10)
            self.log_success("Successfully connected to the AllSky API.")
        except ImportError:
            self.log_issue("PYTHON_PKG_MISSING", {'pkg': 'requests'})
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"API connectivity: {e}"}) # Could be network issue


    def check_camera_health(self):
        """Pings cameras and checks their live video streams using ffprobe."""
        print("\n--- Checking Camera Health ---")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        ffprobe_exists = which('ffprobe')

        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            ip = cam_info.get('ip')
            if not ip: continue

            try:
                # Use timeout with ping
                res = subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True, timeout=2)
                if res.returncode != 0:
                    self.log_issue("CAM_UNREACHABLE", {'cam_key': cam_key, 'ip': ip})
                    continue
            except subprocess.TimeoutExpired:
                 self.log_issue("CAM_UNREACHABLE", {'cam_key': cam_key, 'ip': ip, 'reason': 'ping timeout'})
                 continue
            except Exception as e:
                 self.log_issue("PERMISSION_DENIED", {'check': f"pinging {ip}: {e}"})
                 continue

            
            self.log_success(f"Camera '{cam_key}' ({ip}) is reachable (ping OK).", indent=1)

            if not ffprobe_exists:
                # Log only once per run
                ffprobe_missing_logged = any(e['code'] == 'SYS_PKG_MISSING' and e['context'].get('pkg') == 'ffprobe' for e in self.logged_errors)
                if not ffprobe_missing_logged:
                     self.log_issue("SYS_PKG_MISSING", {'pkg': 'ffprobe', 'pkg_name': 'ffmpeg'}, indent=1)
                continue # Skip stream check if ffprobe is missing
            
            port = cam_info.get('rtsp_port', '554')
            user = cam_info.get('user', 'admin')
            password = cam_info.get('password', '')
            channel = cam_info.get('channel', '1')

            for stream_type, stream_id in [("HD", "0"), ("SD", "1")]:
                uri = f"user={user}&password={password}&channel={channel}&stream={stream_id}.sdp"
                rtsp_url = f"rtsp://{ip}:{port}/{uri}"
                # Increased timeout for ffprobe, especially on initial connection
                cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-rtsp_transport', 'tcp', '-timeout', '7000000', rtsp_url] # 7 seconds
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10) # Overall timeout
                    self.log_success(f"Camera '{cam_key}' {stream_type} stream is active.", indent=2)
                except subprocess.CalledProcessError:
                    self.log_issue("CAM_STREAM_DOWN", {'cam_key': cam_key, 'ip': ip, 'stream_type': stream_type}, indent=2)
                except subprocess.TimeoutExpired:
                     self.log_issue("CAM_STREAM_DOWN", {'cam_key': cam_key, 'ip': ip, 'stream_type': stream_type, 'reason': 'ffprobe timeout'}, indent=2)
                except Exception as e:
                     self.log_issue("PERMISSION_DENIED", {'check': f"running ffprobe for {cam_key} {stream_type}: {e}"}, indent=2)


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
                    found = False
                    for d in cal_dirs:
                        if os.path.isdir(d):
                            # Use glob with recursive=True, requires Python 3.5+
                            if sys.version_info >= (3, 5):
                                if glob.glob(os.path.join(d, '**', f"*{cams_id}*calparams.json"), recursive=True):
                                    found = True
                                    break
                            else: # Fallback for older Python without recursive glob
                                for root, _, files in os.walk(d):
                                    if any(f"*{cams_id}*calparams.json" in f for f in files):
                                        found = True
                                        break
                                if found: break
                                
                    if found: self.log_success(f"Calibration file(s) found for camera '{cam_key}' ({cams_id}).")
                    else: 
                        self.log_issue("NO_CAL_FILE", {'cam_key': cam_key, 'cams_id': cams_id})
                        any_missing = True
                except PermissionError:
                    self.log_issue("PERMISSION_DENIED", {'check': f"reading calibration files for {cam_key}"})
                    any_missing = True
                except Exception as e:
                     self.log_issue("PERMISSION_DENIED", {'check': f"searching calibration files for {cam_key}: {e}"})
                     any_missing = True # Assume missing if we can't search
        
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
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"finding failed cal logs: {e}"})


        try:
            cmd = ['find', '/mnt/ams2/cal/', '-type', 'f', '-name', '*-stacked.png', '-mmin', '-1440']
            res = subprocess.run(cmd, capture_output=True)
            if not res.stdout: self.log_issue("NO_CAL_SOURCE_IMAGES", indent=indent)
            else: self.log_success("Recent source images for calibration were found.", indent=indent)
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"finding cal source images: {e}"})


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
            try:
                age = time.time() - os.path.getmtime(wd_log)
                if age > 360: # 6 minutes
                    self.log_issue("WD_LOG_STALE")
                else:
                    self.log_success(f"Watch-dog log was recently updated ({age:.0f}s ago).")
            except Exception as e:
                 self.log_issue("PERMISSION_DENIED", {'check': f"getting mtime of {wd_log}: {e}"})
            
            try:
                with open(wd_log, 'r') as f: last_lines = f.readlines()[-20:] # Read last 20 lines
                if any("error" in l.lower() for l in last_lines): self.log_issue("WD_LOG_ERROR")
                else: self.log_success("No errors found in recent watch-dog log entries.", indent=1)
            except PermissionError:
                self.log_issue("PERMISSION_DENIED", {'check': f"reading {wd_log}"})
            except Exception as e:
                 self.log_issue("PERMISSION_DENIED", {'check': f"reading {wd_log}: {e}"})
        else:
             self.log_success(f"Watch-dog log ({wd_log}) not found.") # Not necessarily an error


    def check_recent_captures(self):
        """Verifies that new video files are being created for each camera."""
        print("\n--- Checking Recent Captures ---")
        as6_data = self.config_data.get("/home/ams/amscams/conf/as6.json", {})
        for cam_key, cam_info in as6_data.get('cameras', {}).items():
            cams_id = cam_info.get('cams_id')
            if cams_id:
                for stream_type, path in [("SD", "/mnt/ams2/SD"), ("HD", "/mnt/ams2/HD")]:
                    if not os.path.isdir(path):
                        self.log_issue("DIR_MISSING", {'path': path})
                        continue
                    try:
                        # Use find command as it's generally available and handles permissions
                        cmd = ['find', path, '-name', f"*{cams_id}*.mp4", '-mmin', '-10', '-print', '-quit'] # -quit stops after first find
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                        if not res.stdout.strip(): # No output means no files found
                            self.log_issue("NO_RECENT_CAPTURES", {'stream_type': stream_type, 'cam_key': cam_key})
                        else:
                            self.log_success(f"Recent {stream_type} captures found for camera '{cam_key}' ({cams_id}).")
                    except subprocess.TimeoutExpired:
                         self.log_issue("PERMISSION_DENIED", {'check': f"finding recent captures in {path} (timeout)"})
                    except Exception as e:
                         self.log_issue("PERMISSION_DENIED", {'check': f"finding recent captures in {path}: {e}"})


    def check_video_fps(self):
        """Checks the FPS of recent video files for consistency and minimums."""
        print("\n--- Checking Video File FPS ---")
        ffprobe_path = which('ffprobe')
        if not ffprobe_path:
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
                
                # Use find to get recent files, sorted by time, take top 10
                cmd = [
                    'find', path, 
                    '-name', f"*{cams_id}*.mp4", 
                    '-mmin', '-1440', # Only files from last 24h
                    '-type', 'f',
                    '-printf', '%T@ %p\\n'
                ]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
                    if not res.stdout:
                        self.log_success(f"No recent {stream_type} files found to check for '{cam_key}'.", indent=2)
                        continue
                    
                    # Sort by timestamp (first column) descending, get filenames
                    sorted_lines = sorted(
                        res.stdout.strip().split('\n'), 
                        key=lambda x: float(x.split(' ', 1)[0]), 
                        reverse=True
                    )
                    files_to_check = [line.split(' ', 1)[1] for line in sorted_lines[:10]]

                except subprocess.CalledProcessError: # find might return error if dir disappears etc.
                    self.log_success(f"No recent {stream_type} files found to check for '{cam_key}'.", indent=2)
                    continue
                except subprocess.TimeoutExpired:
                     self.log_issue("PERMISSION_DENIED", {'check': f"finding video files in {path} (timeout)"}, indent=2)
                     continue
                except Exception as e:
                    self.log_issue("PERMISSION_DENIED", {'check': f"finding video files in {path}: {e}"}, indent=2)
                    continue

                fps_list = []
                low_fps_errors_found = False
                files_processed = 0
                for f in files_to_check:
                    # Check if file still exists and is readable before running ffprobe
                    if not os.path.isfile(f) or not os.access(f, os.R_OK):
                        continue
                    files_processed += 1
                    try:
                        cmd_ffprobe = [
                            ffprobe_path, '-v', 'error', 
                            '-select_streams', 'v:0', 
                            '-show_entries', 'stream=r_frame_rate', 
                            '-of', 'default=noprint_wrappers=1:nokey=1',
                            f
                        ]
                        # Short timeout per file
                        res = subprocess.run(cmd_ffprobe, capture_output=True, text=True, check=True, timeout=5)
                        fps_str = res.stdout.strip()
                        if '/' in fps_str:
                            num, den = map(float, fps_str.split('/'))
                            if den == 0: continue
                            fps = num / den
                            fps_list.append(fps)
                            
                            if fps < 12.5:
                                self.log_issue("FPS_TOO_LOW", {'stream_type': stream_type, 'fps': fps, 'file': f}, indent=3)
                                low_fps_errors_found = True
                        elif fps_str: # Handle integer FPS value if reported
                            try:
                                fps = float(fps_str)
                                fps_list.append(fps)
                                if fps < 12.5:
                                    self.log_issue("FPS_TOO_LOW", {'stream_type': stream_type, 'fps': fps, 'file': f}, indent=3)
                                    low_fps_errors_found = True
                            except ValueError: pass # Ignore non-numeric output

                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                        # File might be corrupt, unreadable by ffprobe, or timeout
                        pass 
                    except Exception:
                        pass # General catch-all for unexpected errors on one file

                if files_processed == 0:
                     self.log_success(f"Could not access any recent {stream_type} files for '{cam_key}' to check FPS.", indent=2)
                     continue
                elif not fps_list:
                    self.log_success(f"Could not determine FPS for any accessible recent {stream_type} files for '{cam_key}'.", indent=2)
                    continue
                
                if not low_fps_errors_found:
                     self.log_success(f"No files with FPS < 12.5 found for {stream_type}.", indent=3)

                # Check for variation only if we have multiple valid FPS values
                if len(fps_list) > 1:
                    min_fps = min(fps_list)
                    max_fps = max(fps_list)
                    
                    if min_fps == 0: # Avoid division by zero, indicates likely error
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
                elif len(fps_list) == 1:
                     self.log_success(f"Only one {stream_type} file FPS ({fps_list[0]:.1f}) obtained, cannot check variation.", indent=3)


    def check_for_corrupt_files(self):
        """Searches for recently created video files that are unusually small."""
        print("\n--- Checking for Corrupt Video Files ---")
        try:
            # Check files modified between 2 and 15 mins ago, smaller than 2k
            cmd = ['find', "/mnt/ams2/SD", '-type', 'f', '-name', '*.mp4', '-mmin', '+1', '-mmin', '-15', '-size', '-2k', '-printf', '%p %s\\n']
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.stdout:
                for line in res.stdout.strip().split('\n'):
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        file_path, size = parts
                        self.log_issue("CORRUPT_FILE_FOUND", {'file': file_path, 'size': size})
            else:
                self.log_success("No suspiciously small video files found in the last ~15 minutes.")
        except subprocess.TimeoutExpired:
             self.log_issue("PERMISSION_DENIED", {'check': "finding corrupt files (timeout)"})
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"finding corrupt files: {e}"})


    def check_processing_health(self):
        """Checks the video processing queue for stalls or a large backlog."""
        print("\n--- Checking Processing Queue ---")
        proc_dir = "/mnt/ams2/SD/proc2"
        if not os.path.isdir(proc_dir): return
        
        try:
            # Find the oldest file in the directory
            cmd_oldest = f"find {proc_dir} -maxdepth 1 -type f -printf '%T@ %p\\n' | sort -n | head -1"
            res = subprocess.run(cmd_oldest, shell=True, capture_output=True, text=True, timeout=15)
            if res.stdout:
                try:
                    oldest_ts_str, oldest_file = res.stdout.strip().split(' ', 1)
                    oldest_ts = float(oldest_ts_str)
                    age_hours = (time.time() - oldest_ts) / 3600
                    if age_hours > 3:
                        self.log_issue("STALLED_QUEUE", {'age': age_hours, 'file': os.path.basename(oldest_file)})
                        return # Don't check backlog if stalled
                    else:
                        self.log_success(f"Processing queue is not stalled (oldest file is {age_hours:.1f} hours old).")
                except ValueError:
                     self.log_issue("PERMISSION_DENIED", {'check': f"parsing output of find command for oldest file: {res.stdout.strip()}"})
            else:
                 self.log_success("Processing queue is empty or no files found.") # Not an error if empty

            # Count files created in the last hour
            cmd_recent = f"find {proc_dir} -maxdepth 1 -type f -mmin -60 | wc -l"
            res = subprocess.run(cmd_recent, shell=True, capture_output=True, text=True, timeout=15)
            count = int(res.stdout.strip())
            if count > 1500: # Threshold for high backlog
                self.log_issue("HIGH_BACKLOG", {'count': count})
            else:
                self.log_success(f"Recent file queue size is normal ({count} files in last hour).")

        except subprocess.TimeoutExpired:
            self.log_issue("PERMISSION_DENIED", {'check': "checking processing queue (timeout)"})
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"checking processing queue: {e}"})


    def check_cron_jobs(self):
        """Verifies that the crontab for the 'ams' user exists and contains all expected jobs."""
        print("\n--- Checking Cron Jobs ---")
        expected_jobs = [
            "as7-latest.py", "watch-dog.py", "NOAA.py", "scan_stack.py", 
            "day_stack.py", "doDay.py", "IMX291.py", "Process.py run_jobs",
            "gitpull.py", "monitor.sh", "downloader.py", "logrotate"
        ]
        
        # Check if crontab command exists
        if not which('crontab'):
             self.log_issue("SYS_PKG_MISSING", {'pkg': 'crontab', 'pkg_name': 'cron'})
             return

        try:
            # Run crontab -l -u ams. This requires root or being the ams user.
            cmd = ['crontab', '-l', '-u', 'ams']
            # If not root, we cannot specify '-u ams' unless we are ams
            if not self.is_root:
                 current_user = pwd.getpwuid(os.getuid()).pw_name
                 if current_user == 'ams':
                     cmd = ['crontab', '-l'] # Check current user's (ams) crontab
                 else:
                     self.log_issue("PERMISSION_DENIED", {'check': "reading 'ams' user's crontab. Run as root or ams."})
                     return

            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            content = res.stdout
            self.log_success("Crontab found for user 'ams'. Verifying jobs...")
            missing_jobs = []
            for job in expected_jobs:
                if job not in content:
                    self.log_issue("CRON_JOB_MISSING", {'job': job}, indent=1)
                    missing_jobs.append(job)
                else:
                    self.log_success(f"Found expected cron job fragment: {job}", indent=1)
            if not missing_jobs:
                 self.log_success("All expected cron job fragments found.")

        except subprocess.CalledProcessError as e:
            # crontab -l returns non-zero if the crontab is empty or doesn't exist
            if "no crontab for ams" in e.stderr.lower():
                 self.log_issue("NO_CRONTAB")
            else: # Other error (e.g., permission denied if not root/ams)
                 self.log_issue("PERMISSION_DENIED", {'check': f"reading 'ams' user's crontab: {e.stderr.strip()}"})
        except FileNotFoundError:
             self.log_issue("SYS_PKG_MISSING", {'pkg': 'crontab', 'pkg_name': 'cron'})
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"reading 'ams' user's crontab: {e}"})


    # =========================================================================
    # NMN CHECKS
    # =========================================================================

    def run_nmn_checks(self):
        """Runs all NMN-specific checks."""
        print("\n" + "="*60)
        print(" NMN (Norsk Meteornettverk) System Health Check")
        print("="*60)
        # No root check here, functions handle permissions individually

        self.check_nmn_user_and_dirs()
        self.check_nmn_config()
        self.check_nmn_connectivity()
        self.check_nmn_cam_dirs()
        self.check_nmn_python_imports()
        self.check_nmn_processes_and_services()
        self.check_nmn_logs()
        self.check_nmn_disk_space()

    def check_nmn_user_and_dirs(self):
        """Checks for 'meteor' user, home, nmn, and bin symlink."""
        print("\n--- Checking NMN User & Directories ---")
        try:
            self.meteor_pwd = pwd.getpwnam('meteor')
            self.log_success("User 'meteor' exists on the system.")
        except KeyError:
            self.log_issue("NMN_NO_METEOR_USER")
            return # Stop here, other checks will fail

        home_path = "/home/meteor"
        if not os.path.isdir(home_path):
            self.log_issue("NMN_HOME_MISSING")
            return # Stop here
        self.log_success(f"Directory exists: {home_path}")
        
        try:
            stat_info = os.stat(home_path)
            if stat_info.st_uid != self.meteor_pwd.pw_uid:
                self.log_issue("NMN_HOME_BAD_OWNER")
            else:
                self.log_success(f"Directory '{home_path}' is owned by 'meteor'.", indent=1)
        except PermissionError:
             self.log_issue("PERMISSION_DENIED", {'check': f"checking owner of {home_path}"})
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"checking owner of {home_path}: {e}"})


        nmn_path = "/home/meteor/nmn"
        if not os.path.isdir(nmn_path):
            self.log_issue("NMN_NMN_DIR_MISSING")
        else:
            self.log_success(f"Directory exists: {nmn_path}")
            # Writability check requires root or running as meteor
            self._check_dir_writable_by(nmn_path, 'meteor', "NMN_NMN_DIR_NOT_WRITABLE", indent=1)

        bin_path = "/home/meteor/bin"
        if not os.path.lexists(bin_path): # Use lexists to check link itself
             self.log_issue("NMN_BIN_NOT_SYMLINK", {'reason': 'does not exist'})
             return
             
        if not os.path.islink(bin_path):
            self.log_issue("NMN_BIN_NOT_SYMLINK", {'reason': 'is not a symlink'})
        else:
            try:
                target = os.readlink(bin_path)
                # Allow relative or absolute path to the correct dir
                if target == "/home/meteor/nmn/bin" or target == "nmn/bin":
                    self.log_success(f"'bin' symlink is correct (target: '{target}').", indent=1)
                else:
                    self.log_issue("NMN_BIN_WRONG_SYMLINK", {'target': target})
            except PermissionError:
                self.log_issue("PERMISSION_DENIED", {'check': f"reading symlink {bin_path}"})
            except Exception as e:
                self.log_issue("PERMISSION_DENIED", {'check': f"reading symlink {bin_path}: {e}"})

    def _check_file_readable_by(self, path, user_name, error_code, indent=0):
        """Helper to check if a file is readable by a specific user."""
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            current_user = str(os.getuid())

        # If running as the target user, use os.access
        if not self.is_root and current_user == user_name:
            if os.access(path, os.R_OK):
                 self.log_success(f"File '{path}' is readable by current user '{user_name}'.", indent=indent)
                 return True
            else:
                 self.log_issue(error_code, {'path': path}, indent=indent)
                 return False
        # If running as root, use sudo
        elif self.is_root:
            cmd = ['sudo', '-u', user_name, 'test', '-r', path]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
                self.log_success(f"File '{path}' is readable by user '{user_name}'.", indent=indent)
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                self.log_issue(error_code, {'path': path}, indent=indent)
                return False
        # If running as non-root, non-target user
        else:
            # Try reading directly first
            if os.access(path, os.R_OK):
                 self.log_success(f"File '{path}' is readable by current user '{current_user}'.", indent=indent)
                 return True
            else:
                self.log_issue("PERMISSION_DENIED", {'check': f"readable status of {path} by user {user_name}. Run as root or {user_name}."}, indent=indent)
                return False


    def check_nmn_config(self):
        """Checks /etc/meteor.cfg for existence, readability, and content."""
        print("\n--- Checking NMN Configuration ---")
        cfg_path = "/etc/meteor.cfg"
        if not os.path.isfile(cfg_path):
            self.log_issue("NMN_CONFIG_MISSING")
            return
            
        # Check readability by meteor user (best effort if not root)
        if not self._check_file_readable_by(cfg_path, 'meteor', "NMN_CONFIG_NOT_READABLE"):
             # If root failed, definitely an issue. If non-root failed, maybe ok if meteor can read.
             if self.is_root: return
             # If not root, log info and try reading anyway
             self.log_issue("PERMISSION_DENIED", {'check': f"verifying readability of {cfg_path} by meteor. Attempting read as current user."}, indent=1)

        config = configparser.ConfigParser()
        try:
            # Try reading as current user
            config.read(cfg_path)
            self.log_success(f"Successfully parsed config: {cfg_path}")
        except configparser.MissingSectionHeaderError:
             self.log_issue("NMN_CONFIG_PARSE_ERROR", {'reason': 'Missing section header'})
             return
        except PermissionError:
             # This confirms the earlier PERMISSION_DENIED if not root
             self.log_issue("NMN_CONFIG_NOT_READABLE", {'path': cfg_path})
             return
        except Exception as e:
            self.log_issue("NMN_CONFIG_PARSE_ERROR", {'reason': str(e)})
            return

        # Check sections
        required_sections = ['astronomy', 'station']
        for section in required_sections:
            if section not in config:
                self.log_issue("NMN_CONFIG_SECTION_MISSING", {'section': section}, indent=1)
                return
        self.log_success("All required config sections [astronomy, station] are present.", indent=1)
        
        # Check keys
        required_keys = {
            'astronomy': ['latitude', 'longitude', 'elevation', 'temperature', 'pressure'],
            'station': ['name', 'code']
        }
        has_all_keys = True
        for section, keys in required_keys.items():
             if section in config: # Check section exists before checking keys
                for key in keys:
                    if key not in config[section]:
                        self.log_issue("NMN_CONFIG_KEY_MISSING", {'section': section, 'key': key}, indent=2)
                        has_all_keys = False
        
        if not has_all_keys:
             return # Stop validation if keys are missing

        # Validate values only if all keys are present
        try:
            lat = config.getfloat('astronomy', 'latitude')
            if not (-90 <= lat <= 90):
                self.log_issue("NMN_CONFIG_LAT_INVALID", {'lat': lat}, indent=2)
            else:
                self.log_success(f"Latitude is valid: {lat}", indent=2)
        except (ValueError, configparser.NoOptionError):
            self.log_issue("NMN_CONFIG_LAT_INVALID", {'lat': config.get('astronomy','latitude','?')}, indent=2)

        try:
            lng = config.getfloat('astronomy', 'longitude')
            if not (-180 <= lng <= 180):
                self.log_issue("NMN_CONFIG_LNG_INVALID", {'lng': lng}, indent=2)
            else:
                self.log_success(f"Longitude is valid: {lng}", indent=2)
        except (ValueError, configparser.NoOptionError):
            self.log_issue("NMN_CONFIG_LNG_INVALID", {'lng': config.get('astronomy','longitude','?')}, indent=2)

        try:
            elev = config.getfloat('astronomy', 'elevation')
            if not (-500 <= elev <= 9000):
                self.log_issue("NMN_CONFIG_ELEV_INVALID", {'elev': elev}, indent=2)
            else:
                self.log_success(f"Elevation is valid: {elev}", indent=2)
        except (ValueError, configparser.NoOptionError):
            self.log_issue("NMN_CONFIG_ELEV_INVALID", {'elev': config.get('astronomy','elevation','?')}, indent=2)

    def check_nmn_connectivity(self):
        """Checks connectivity to NMN servers on required ports."""
        print("\n--- Checking NMN Network Connectivity ---")
        host = "norskmeteornettverk.no"
        
        if self._check_port_open(host, 80):
            self.log_success(f"Successfully connected to {host} on port 80 (HTTP).")
        else:
            self.log_issue("NMN_HOST_UNREACHABLE_80")

        if self._check_port_open(host, 22):
            self.log_success(f"Successfully connected to {host} on port 22 (SSH).")
        else:
            self.log_issue("NMN_HOST_UNREACHABLE_22")

    def _check_file_link(self, path, err_miss, err_read, err_zero, indent=0):
        """Helper to check a file or symlink for readability and size."""
        if not os.path.lexists(path):
            self.log_issue(err_miss, {'path': path}, indent=indent)
            return None
        
        try:
            stat_res = os.stat(path) # Follows symlinks by default
            
            # Check for world-readable permission using os.access (more reliable than mode parsing)
            is_readable = os.access(path, os.R_OK)

            if not is_readable:
                # Double-check mode just in case os.access failed for other reasons
                try:
                     mode = stat_res.st_mode
                     if not (mode & 0o004): # Check other-readable bit specifically
                          self.log_issue(err_read, {'path': path}, indent=indent)
                          return None
                     else: # Mode says readable, but os.access failed - log info
                          self.log_issue("PERMISSION_DENIED", {'check': f"readability check consistency for {path}"}, indent=indent)
                          # Proceed to check size anyway, as it might be readable by some process
                except Exception:
                     self.log_issue(err_read, {'path': path}, indent=indent)
                     return None # Can't check mode either


            if stat_res.st_size == 0:
                self.log_issue(err_zero, {'path': path}, indent=indent)
                return None
                
            self.log_success(f"File '{os.path.basename(path)}' is valid (readable, non-zero).", indent=indent)
            return stat_res
            
        except FileNotFoundError: # Target of symlink missing
            self.log_issue(err_miss, {'path': f"{path} (symlink target missing?)"}, indent=indent)
            return None
        except PermissionError:
             self.log_issue("PERMISSION_DENIED", {'check': f"stat file {path}"}, indent=indent)
             return None
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"stat file {path}: {e}"}, indent=indent)
            return None

    def _check_cam_date_dirs(self, cam_path, indent=0):
        """Helper to check for YYYYMMDD subdirectories."""
        date_dirs = []
        try:
            entries = os.listdir(cam_path)
        except PermissionError:
             self.log_issue("PERMISSION_DENIED", {'check': f"listing dirs in {cam_path}"}, indent=indent)
             return
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"listing dirs in {cam_path}: {e}"}, indent=indent)
            return

        for entry in entries:
            full_path = os.path.join(cam_path, entry)
            # Check if it matches pattern AND is a directory
            if re.match(r'^(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$', entry) and os.path.isdir(full_path):
                    try:
                        # Validate it's a real date
                        datetime.strptime(entry, '%Y%m%d')
                        date_dirs.append(entry)
                    except ValueError:
                        continue # e.g., 20230231
        
        if not date_dirs:
            self.log_issue("NMN_CAM_NO_DATE_DIRS", {'path': cam_path}, indent=indent)
            return
            
        self.log_success(f"Found {len(date_dirs)} date-formatted directories.", indent=indent)
        
        latest_dir = sorted(date_dirs)[-1]
        try:
            latest_date = datetime.strptime(latest_dir, '%Y%m%d').date()
            now = datetime.now()
            today = now.date()
            
            # Allow for "today" or "yesterday".
            # If it's just after midnight (e.g., 00:01), the last dir might still be yesterday's.
            # We give it a 2-minute grace period.
            if now.hour == 0 and now.minute < 2:
                two_mins_ago = (now - timedelta(minutes=2)).date()
                valid_dates = [today, yesterday, two_mins_ago] # Can be today, yesterday, or day before yesterday
            else:
                valid_dates = [today, yesterday]

            if latest_date not in valid_dates:
                self.log_issue("NMN_CAM_STALE_DATE_DIRS", {'path': cam_path, 'latest_dir': latest_dir}, indent=indent)
            else:
                self.log_success(f"Most recent data directory '{latest_dir}' is current.", indent=indent)
        except Exception:
            pass # Should not happen given regex

    def check_nmn_cam_dirs(self):
        """Checks all /meteor/camX/ directories and their contents."""
        print("\n--- Checking NMN Camera Directories ---")
        if not self.meteor_pwd: # Fetched during init or user check
            self.log_issue("PERMISSION_DENIED", {'check': "NMN cam dirs, 'meteor' user info not available."})
            return
            
        for i in range(1, 8):
            path = f"/meteor/cam{i}/"
            self.log_success(f"Checking: {path}")
            if not os.path.isdir(path):
                self.log_issue("NMN_CAM_DIR_MISSING", {'path': path}, indent=1)
                continue
            
            try:
                stat_info = os.stat(path)
                if stat_info.st_uid != self.meteor_pwd.pw_uid:
                    self.log_issue("NMN_CAM_DIR_BAD_OWNER", {'path': path}, indent=1)
                else:
                    self.log_success("Directory owner is 'meteor'.", indent=1)
            except PermissionError:
                 self.log_issue("PERMISSION_DENIED", {'check': f"checking owner of {path}"})
            except Exception as e:
                 self.log_issue("PERMISSION_DENIED", {'check': f"checking owner of {path}: {e}"})

            
            if not self._check_dir_writable_by(path, 'meteor', "NMN_CAM_DIR_NOT_WRITABLE", indent=1):
                # If check couldn't run due to permissions (non-root), still check contents
                if not self.is_root:
                    self.log_issue("PERMISSION_DENIED", {'check': f"writable status of {path} by meteor. Checking contents anyway."}, indent=1)
                else:
                     continue # If root failed writability, stop checking contents

            # Check special files (grid.png, lens.pto, snapshot.jpg)
            for f in ['grid.png', 'lens.pto']:
                file_path = os.path.join(path, f)
                self._check_file_link(file_path, 'NMN_CAM_FILE_MISSING', 'NMN_CAM_FILE_NOT_READABLE', 'NMN_CAM_FILE_ZERO_SIZE', indent=2)
            
            snap_path = os.path.join(path, 'snapshot.jpg')
            stat_res = self._check_file_link(snap_path, 'NMN_SNAPSHOT_MISSING', 'NMN_SNAPSHOT_NOT_READABLE', 'NMN_SNAPSHOT_ZERO_SIZE', indent=2)
            
            if stat_res:
                try:
                    age = time.time() - stat_res.st_mtime
                    if age > 120: # 2 minutes
                        self.log_issue('NMN_SNAPSHOT_STALE', {'path': snap_path, 'age_sec': age}, indent=2)
                    else:
                        self.log_success(f"Snapshot file is current ({age:.0f}s old).", indent=2)
                except Exception as e:
                     self.log_issue("PERMISSION_DENIED", {'check': f"getting mtime for {snap_path}: {e}"})


            # Check for date directories
            self._check_cam_date_dirs(path, indent=1)

    def check_nmn_python_imports(self):
        """Scans NMN bin directory for imports using AST and checks them as the 'meteor' user."""
        print("\n--- Checking NMN Python Dependencies (as user 'meteor') ---")
        if not self.is_root:
            self.log_issue("PERMISSION_DENIED", {'check': "NMN Python imports as 'meteor'. Must be run as root."})
            return
        if not self.meteor_pwd:
            try:
                self.meteor_pwd = pwd.getpwnam('meteor')
            except KeyError:
                self.log_issue("NMN_NO_METEOR_USER")
                self.log_issue("PERMISSION_DENIED", {'check': "NMN Python imports, 'meteor' user not found."})
                return

        bin_path = "/home/meteor/nmn/bin/"
        if not os.path.isdir(bin_path):
            self.log_issue("DIR_MISSING", {'path': bin_path})
            return

        try:
            # Ensure the directory itself is readable by meteor
            cmd_read_dir = ['sudo', '-u', 'meteor', 'test', '-r', bin_path]
            subprocess.run(cmd_read_dir, check=True, capture_output=True, timeout=5)
            # List files as root
            py_files = glob.glob(os.path.join(bin_path, "*.py"))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
             self.log_issue("PERMISSION_DENIED", {'check': f"reading directory {bin_path} as meteor"}, indent=1)
             return
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"listing python files in {bin_path}: {e}"})
            return

        all_imports = set()
        local_modules = set(os.path.splitext(os.path.basename(f))[0] for f in py_files)
        
        # Import ast module here
        try:
            import ast
        except ImportError:
            self.log_issue("PYTHON_PKG_MISSING", {'pkg': 'ast (standard library)'}) # Should not happen unless python install is broken
            return


        for f in py_files:
            try:
                # Check readability as meteor first
                cmd_read = ['sudo', '-u', 'meteor', 'test', '-r', f]
                subprocess.run(cmd_read, check=True, capture_output=True, timeout=5)

                # Read file content as root
                with open(f, 'r', errors='ignore') as file:
                    content = file.read()
                    
                # Parse the code using AST
                try:
                    tree = ast.parse(content, filename=f)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                top_level_imp = alias.name.split('.')[0]
                                all_imports.add(top_level_imp)
                        elif isinstance(node, ast.ImportFrom):
                            # Handle 'from . import ...' or 'from .. import ...' (relative imports)
                            if node.level > 0: # Relative import, ignore for checking external deps
                                continue
                            if node.module: # Regular 'from module import ...'
                                top_level_imp = node.module.split('.')[0]
                                all_imports.add(top_level_imp)
                except SyntaxError as e:
                    self.log_issue("JSON_PARSE_ERROR", {'path': f, 'reason': f"SyntaxError: {e}"}, indent=1) # Re-use JSON error code for simplicity
                except Exception as e:
                     self.log_issue("PERMISSION_DENIED", {'check': f"parsing python file {f} with ast: {e}"}, indent=1)


            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                self.log_issue("PERMISSION_DENIED", {'check': f"reading python file {f} as meteor"}, indent=1)
            except Exception as e:
                self.log_issue("PERMISSION_DENIED", {'check': f"processing file {f}: {e}"})

        # Remove known local modules from the set of imports to check via system python
        imports_to_check = all_imports - local_modules
        if not imports_to_check:
            self.log_success(f"All found non-stdlib imports ({len(all_imports)}) appear to be local modules within {bin_path}.")
            return

        self.log_success(f"Found {len(imports_to_check)} unique potential non-local, non-stdlib imports to check...")
        failed_imports = 0
        skipped_stdlib = 0
        # Expanded list of standard libraries (might need further refinement)
        stdlib_list = [
            '_ast', '_bisect', '_blake2', '_codecs', '_collections', '_compat_pickle', '_compression', '_contextvars',
            '_csv', '_datetime', '_decimal', '_elementtree', '_functools', '_hashlib', '_heapq', '_imp', '_io',
            '_json', '_locale', '_lsprof', '_lzma', '_markupbase', '_md5', '_multibytecodec', '_multiprocessing',
            '_opcode', '_operator', '_osx_support', '_pickle', '_posixsubprocess', '_py_abc', '_pydecimal',
            '_pyio', '_queue', '_random', '_sha1', '_sha256', '_sha3', '_sha512', '_signal', '_sitebuiltins',
            '_socket', '_sqlite3', '_sre', '_ssl', '_stat', '_statistics', '_string', '_strptime', '_struct',
            '_symtable', '_thread', '_tracemalloc', '_uuid', '_warnings', '_weakref', '_weakrefset', '_zoneinfo',
            'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore', 'atexit', 'audioop',
            'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
            'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
            'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv',
            'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils',
            'doctest', 'email', 'encodings', 'ensurepip', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp',
            'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass',
            'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib',
            'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
            'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal',
            'math', 'mimetypes', 'mmap', 'modulefinder', 'msilib', 'msvcrt', 'multiprocessing', 'netrc',
            'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'parser', 'pathlib',
            'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
            'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'pydoc_data',
            'pyexpat', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter',
            'runpy', 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
            'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'sqlite3', 'sre_compile',
            'sre_constants', 'sre_parse', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
            'subprocess', 'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
            'telnetlib', 'tempfile', 'termios', 'textwrap', 'this', 'threading', 'time', 'timeit', 'tkinter',
            'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types',
            'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave',
            'weakref', 'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp',
            'zipfile', 'zipimport', 'zlib', 'zoneinfo'
        ]


        for pkg in sorted(list(imports_to_check)):
            if pkg in stdlib_list:
                skipped_stdlib += 1
                continue # Skip common stdlibs

            try:
                # Construct the command to add the bin_path to sys.path before importing
                safe_bin_path = bin_path.replace("'", "'\\''") # Basic escaping
                import_command = f"import sys; sys.path.insert(0, '{safe_bin_path}'); import {pkg}"
                cmd_import = ['sudo', '-u', 'meteor', 'python3', '-c', import_command]
                #print(f"DEBUG: Running command: {' '.join(cmd_import)}") # Keep for debugging
                subprocess.run(cmd_import, check=True, capture_output=True, text=True, timeout=10)
                self.log_success(f"Import OK: {pkg}", indent=1)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                stderr_output = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else "No stderr"
                #print(f"DEBUG: Import failed for {pkg}. Stderr: {stderr_output}") # Keep for debugging
                self.log_issue('NMN_PY_IMPORT_MISSING', {'pkg': pkg, 'stderr': stderr_output}, indent=1)
                failed_imports += 1
            except FileNotFoundError:
                 self.log_issue("SYS_PKG_MISSING", {'pkg': 'python3', 'pkg_name': 'python3'}, indent=1)
                 return # Cannot continue if python3 is missing
            except Exception as e:
                self.log_issue("PERMISSION_DENIED", {'check': f"running import test for {pkg} as meteor: {e}"}, indent=1)
                failed_imports += 1

        if skipped_stdlib > 0:
             self.log_success(f"Skipped {skipped_stdlib} standard library imports.")
        if failed_imports == 0:
            self.log_success("All checked non-standard Python dependencies seem importable by user 'meteor'.")

    def _check_systemctl_status(self, service, indent=0):
        """Helper to check systemd service status."""
        if not self.is_root:
            self.log_issue("PERMISSION_DENIED", {'check': f"status of service {service}. Run as root."}, indent=indent)
            return
        
        err_active = f"NMN_{service.upper().replace('-', '_')}_SVC_NOT_ACTIVE"
        err_enabled = f"NMN_{service.upper().replace('-', '_')}_SVC_NOT_ENABLED"
        
        try:
            res_active = subprocess.run(['systemctl', 'is-active', service], capture_output=True, text=True)
            if res_active.returncode != 0: # 0 means active
                self.log_issue(err_active, {'service': service}, indent=indent)
            else:
                self.log_success(f"Service '{service}' is active.", indent=indent)
                
            res_enabled = subprocess.run(['systemctl', 'is-enabled', service], capture_output=True, text=True)
            # is-enabled returns 0 for enabled, 1 for disabled, error > 1
            # We treat 'disabled' as an issue here. Check return code OR output text.
            if res_enabled.returncode != 0 or 'disabled' in res_enabled.stdout.strip().lower():
                self.log_issue(err_enabled, {'service': service}, indent=indent)
            else:
                self.log_success(f"Service '{service}' is enabled.", indent=indent)
        except FileNotFoundError:
            self.log_issue("SYS_PKG_MISSING", {'pkg': 'systemctl', 'pkg_name': 'systemd'}, indent=indent)
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"checking systemctl status for {service}: {e}"}, indent=indent)


    def check_nmn_processes_and_services(self):
        """Checks for running NMN processes and systemd services."""
        print("\n--- Checking NMN Processes & Services ---")
        if not self.psutil:
            self.log_issue("PERMISSION_DENIED", {'check': "NMN processes, psutil not loaded."})
            return

        found_mirror = False
        found_autossh = False
        found_ssh = False
        
        try:
            current_uid = os.getuid()
            # If not root, can we reliably check processes for 'meteor' user? Maybe not.
            # Let's check if *we* are the meteor user first.
            is_meteor_user = False
            if self.meteor_pwd and current_uid == self.meteor_pwd.pw_uid:
                is_meteor_user = True
            
            # Iterate processes - this might be limited if not root
            for proc in self.psutil.process_iter(['cmdline', 'uids']):
                try:
                    # Skip if cmdline is empty or None
                    if not proc.info['cmdline']:
                        continue
                        
                    # Check if process belongs to meteor user if we have the info
                    proc_uid = proc.info['uids'].real if proc.info['uids'] else None
                    is_proc_meteor = self.meteor_pwd and (proc_uid == self.meteor_pwd.pw_uid)

                    # Only check if root OR if we are the meteor user running the check
                    if self.is_root or is_proc_meteor:
                        cmdline_str = ' '.join(proc.info['cmdline'])
                        
                        if '/home/meteor/bin/mirror.py' in cmdline_str or 'nmn/bin/mirror.py' in cmdline_str:
                            found_mirror = True
                        if ('autossh' in proc.info['cmdline'][0]) and ('meteor@norskmeteornettverk.no' in cmdline_str):
                            found_autossh = True
                        if ('ssh' in proc.info['cmdline'][0]) and ('meteor@norskmeteornettverk.no' in cmdline_str):
                            # Basic check for likely tunnel process
                            if '-o ForwardAgent yes' in cmdline_str or '-N' in cmdline_str: 
                                found_ssh = True
                                
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    continue # Process might have died, or we don't have permission
                except Exception: pass # Ignore other potential errors for a single process
        except Exception as e:
             self.log_issue("PERMISSION_DENIED", {'check': f"iterating processes: {e}"})

        # Report process status
        if not found_mirror:
            # If not root and not meteor, we might not have seen it
            if not self.is_root and not is_meteor_user:
                 self.log_issue("PERMISSION_DENIED", {'check': "mirror.py process status. Run as root or meteor."})
            else:
                 self.log_issue("NMN_MIRROR_PROC_DOWN")
        else:
            self.log_success("Process 'mirror.py' is running.")

        if not found_autossh:
            if not self.is_root and not is_meteor_user:
                 self.log_issue("PERMISSION_DENIED", {'check': "autossh process status. Run as root or meteor."})
            else:
                 self.log_issue("NMN_AUTOSSH_PROC_DOWN")
        else:
            self.log_success("Process 'autossh' for NMN is running.")
            
        if not found_ssh:
             if not self.is_root and not is_meteor_user:
                  self.log_issue("PERMISSION_DENIED", {'check': "active ssh tunnel process status. Run as root or meteor."})
             else:
                  self.log_issue("NMN_SSH_PROC_DOWN")
        else:
            self.log_success("Active 'ssh' tunnel process for NMN is running.")

        # Check services (only possible as root)
        self._check_systemctl_status('mirror', indent=1)
        self._check_systemctl_status('autossh-tunnel', indent=1)


    def check_nmn_logs(self):
        """Checks the NMN mirror log."""
        print("\n--- Checking NMN Logs ---")
        log_path = "/home/meteor/mirror.log"
        if not os.path.isfile(log_path):
            self.log_issue("NMN_MIRROR_LOG_MISSING")
            return
            
        try:
            with open(log_path, 'r', errors='ignore') as f:
                content = f.read()
            error_count = content.lower().count('error')
            self.log_success(f"Found {error_count} case-insensitive 'error' lines in {log_path}.")
        except PermissionError:
             self.log_issue("PERMISSION_DENIED", {'check': f"reading {log_path}"})
        except Exception as e:
            self.log_issue("PERMISSION_DENIED", {'check': f"reading {log_path}: {e}"})

    def check_nmn_disk_space(self):
        """Checks the available disk space on the /meteor/ mount."""
        print("\n--- Checking NMN Disk Space ---")
        self._check_disk_usage("/meteor/", "NMN_METEOR_DISK_CRITICAL", "NMN_METEOR_DISK_LOW", "NMN_METEOR_DISK_INFO")

    def print_summary(self):
        """Prints a final summary of all successes, warnings, and failures."""
        print("\n" + "="*60)
        print(" Health Check Summary")
        print("="*60)
        
        total = sum(len(v) for v in self.results.values())
        print(f"Total checks: {total}")
        print(f"{self.GREEN}Successful: {len(self.results['success'])}{self.RESET}")
        print(f"{self.CYAN}Info:       {len(self.results['info'])}{self.RESET}")
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
                
                if info['type'] == 'failure':
                    color = self.RED
                    header = "Issue"
                elif info['type'] == 'warning':
                    color = self.YELLOW
                    header = "Warning"
                else: # info
                    color = self.CYAN
                    header = "Info"
                
                # Use context from the *first* occurrence for the main description
                try:
                    main_desc = info['description'].format(**contexts[0])
                except KeyError:
                    main_desc = info['description'] # Fallback
                    
                print(f"\n{color}{issue_number}. {header}: {main_desc}{self.RESET}")
                
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
                     if code not in ["IO_ERROR_DETECTED", "SWAP_ACTIVE"]:
                         # Don't re-print context if it was in the main description
                         try:
                             formatted_desc = info['description'].format(**ctx)
                             if formatted_desc == main_desc:
                                 pass # Already shown
                             else:
                                 # This logic is for when the main desc is generic but context is specific
                                 details = self._create_specific_message(code, ctx)
                                 if details != main_desc:
                                    print(f"   - Affected Item: {details}")
                         except KeyError:
                             pass # Error formatting, just skip
                else:
                    print("   - Affected Items:")
                    unique_details = set()
                    for ctx in contexts:
                        details = self._create_specific_message(code, ctx)
                        # Avoid re-printing the *exact* same message as the header
                        # and avoid printing duplicate detail lines
                        if details != main_desc and details not in unique_details:
                            print(f"     - {details}")
                            unique_details.add(details)

                print(f"   - Why it's a problem: {info['reason']}")
                try:
                    print(f"   - How to fix: {info['fix'].format(**contexts[0])}")
                except KeyError:
                    print(f"   - How to fix: {info['fix']}") # Fallback
                issue_number += 1
                
        elif not self.results['failure'] and not self.results['warning'] and not self.results['info']:
             print(f"\n{self.GREEN}Everything looks good! Your system appears to be healthy.{self.RESET}")


class TeeLogger(object):
    """A helper class to log stdout/stderr to both a file and the console."""
    def __init__(self, filename, original_stream):
        self.terminal = original_stream
        try:
            # Ensure the directory exists if possible (e.g., /var/log might not exist initially)
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception: pass # Ignore if cannot create dir
            self.log = open(filename, 'w')
        except Exception as e:
            self.terminal.write(f"FATAL: Could not open log file {filename} for writing.\n")
            self.terminal.write(f"Reason: {e}\n")
            self.log = None

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            try:
                self.log.write(message)
            except Exception: # Ignore errors writing to log if it fails mid-run
                pass

    def flush(self):
        # this flush method is needed for python 3
        self.terminal.flush()
        if self.log:
            try:
                self.log.flush()
            except Exception:
                pass

    def close(self):
        if self.log:
            try:
                self.log.close()
            except Exception:
                pass
            self.log = None # Prevent further writes after close


if __name__ == '__main__':
    log_file_path = "/var/log/as7health.log"
    temp_log = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    temp_log_name = None
    diagnostic_tool = None # To access colors if logging fails

    try:
        # Create a temporary file in a secure way
        temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='as7health_', suffix='.log', dir='/tmp')
        temp_log_name = temp_log.name
        
        # Create Tee objects for stdout and stderr
        # Must close the file handle returned by NamedTemporaryFile before TeeLogger opens it
        temp_log.close() 
        sys.stdout = TeeLogger(temp_log_name, original_stdout)
        sys.stderr = TeeLogger(temp_log_name, original_stderr)

        # --- Start of original main execution ---
        is_root = (os.geteuid() == 0)
        do_nmn_checks = "--nmn" in sys.argv
        
        diagnostic_tool = AS7Diagnostic(is_root=is_root, do_nmn_checks=do_nmn_checks)

        if not is_root:
            try:
                current_user = pwd.getpwuid(os.getuid()).pw_name
                if current_user != 'ams':
                    print(f"{diagnostic_tool.YELLOW}WARNING: Running as non-standard user '{current_user}'.{diagnostic_tool.RESET}")
                    print("Some checks for system files and other users' data may be skipped or fail.")
                    print("For a complete diagnosis, please run with 'sudo' or as the 'ams' user.\n")
            except KeyError:
                pass # Ignore if user name cannot be found for current UID
        
        diagnostic_tool.run_all_checks()
        # --- End of original main execution ---

    except Exception as e:
        # If the script itself crashes, log the crash
        print(f"\nCRITICAL SCRIPT ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        # Restore stdout/stderr and close files
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        sys.stdout = original_stdout
        if isinstance(sys.stderr, TeeLogger):
            sys.stderr.close()
        sys.stderr = original_stderr

        # Move the temporary log file to its final destination
        if temp_log_name and os.path.exists(temp_log_name):
            try:
                # Ensure the target directory exists
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                     try:
                          os.makedirs(log_dir, exist_ok=True)
                          # Try to set reasonable permissions if we created it
                          shutil.chown(log_dir, user='root', group='adm') # Or another suitable group
                          os.chmod(log_dir, 0o775)
                     except Exception: pass # Ignore errors creating/chowning dir

                shutil.move(temp_log_name, log_file_path)
                # Try setting permissions on the log file itself
                try:
                     shutil.chown(log_file_path, user='root', group='adm') # Or another suitable group
                     os.chmod(log_file_path, 0o664) # Readable by group
                except Exception: pass # Ignore permission setting errors

                print(f"\nLog file saved to {log_file_path}")
            except PermissionError as e:
                # Use colors if the diagnostic tool was initialized, otherwise plain text
                yellow, reset = ('', '')
                if diagnostic_tool:
                    yellow, reset = diagnostic_tool.YELLOW, diagnostic_tool.RESET
                print(f"\n{yellow}WARNING:{reset} Could not move log file to {log_file_path}.", file=original_stderr)
                print(f"         Reason: {e}", file=original_stderr)
                print(f"         Log is available at: {temp_log_name}", file=original_stderr)
            except Exception as e:
                red, reset = ('', '')
                if diagnostic_tool:
                    red, reset = diagnostic_tool.RED, diagnostic_tool.RESET
                print(f"\n{red}ERROR:{reset} Could not move log file to {log_file_path}.", file=original_stderr)
                print(f"       Reason: {e}", file=original_stderr)
                print(f"       Log is available at: {temp_log_name}", file=original_stderr)
        elif temp_log: # temp_log was created but name is missing or file vanished
             red, reset = ('', '')
             if diagnostic_tool: red, reset = diagnostic_tool.RED, diagnostic_tool.RESET
             print(f"\n{red}ERROR:{reset} Temporary log file {temp_log_name} was not found after script execution.", file=original_stderr)
