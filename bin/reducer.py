#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tkinter as tk
import amscalib2lens
import json
import copy
import argparse
import ffmpeg
import tempfile
import glob
import os
import configparser
import ephem
import math
import sys
import re
import subprocess
import numpy as np
import contextlib
import signal
import atexit
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw
from brightstar import brightstar
from recalibrate import recalibrate
from timestamp import get_timestamp
import pto_mapper


from pathlib import Path

# Ensure local project modules are importable even when this script is executed via symlink
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_DIR = None
for _cand in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_cand / 'bin').is_dir() and (_cand / 'server').is_dir():
        _PROJECT_DIR = _cand
        break
if _PROJECT_DIR is not None:
    _BIN_DIR = _PROJECT_DIR / 'bin'
    _SRC_DIR = _PROJECT_DIR / 'src'
    for _p in (_BIN_DIR, _SRC_DIR, _PROJECT_DIR):
        if _p.exists():
            _ps = str(_p)
            if _ps not in sys.path:
                sys.path.insert(0, _ps)

LAUNCHER_DEFAULTS = {
    'station_display': '',
    'cam_num': '',
    'date': None,
    'hour': '00',
    'minute': '00',
    'duration': '1',
}

# --- Globals for temp file cleanup ---
temp_files_to_clean = []
temp_dirs = []

def cleanup_temp_resources():
    """Cleans up all temporary directories and files created by the script."""
    print("\nPerforming cleanup...")
    for d in temp_dirs:
        try:
            d.cleanup()
        except Exception as e:
            print(f"    - Error cleaning up directory: {e}", file=sys.stderr)
    for f in temp_files_to_clean:
        if os.path.exists(f):
            try:
                print(f"  - Deleting temporary file: {f}")
                os.remove(f)
            except OSError as e:
                print(f"    - Error removing file {f}: {e}", file=sys.stderr)
    print("Cleanup complete.")

def signal_handler(sig, frame):
    print(f"\nSignal {sig} received, exiting gracefully.")
    sys.exit(0)
# --- End Globals ---


class LauncherDialog:
    """
    A dialog window shown when the script is run without arguments.
    Prompts the user for station, camera, and time to fetch files.
    """
    def __init__(self, toplevel_root, main_tk_root):
        self.root = toplevel_root       # This is the Toplevel window
        self.main_tk = main_tk_root     # This is the hidden main Tk window
        self.root.title("Reducer Launcher")
        
        self.station_map = {
            "ams119 (Gaustatoppen)": "ams119",
            "ams123 (Kristiansand)": "ams123",
            "ams135 (Ørsta)": "ams135",
            "ams136 (Västerås)": "ams136",
            "ams171 (Trondheim)": "ams171",
            "ams172 (Sørreisa)": "ams172",
            "ams173 (Oslo)": "ams173",
            "ams174 (Harestua)": "ams174",
            "ams175 (Moss)": "ams175",
            "ams176 (Larvik)": "ams176",
            "ams177 (Skibotn)": "ams177",
            "ams178 (Løten)": "ams178",
            "ams179 (Hågår)": "ams179",
            "ams180 (Finnskogen)": "ams180",
        }
        
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Station
        ttk.Label(frame, text="Station:").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.station_var = tk.StringVar()
        station_combo = ttk.Combobox(frame, textvariable=self.station_var, width=25)
        station_combo['values'] = sorted(list(self.station_map.keys()))
        station_combo.grid(column=1, row=0, sticky=(tk.W, tk.E))
        if LAUNCHER_DEFAULTS.get('station_display'):
            self.station_var.set(LAUNCHER_DEFAULTS['station_display'])
        
        # Camera
        ttk.Label(frame, text="Camera:").grid(column=0, row=1, sticky=tk.W, pady=5)
        self.cam_var = tk.StringVar()
        cam_combo = ttk.Combobox(frame, textvariable=self.cam_var, width=5)
        cam_combo['values'] = ['1', '2', '3', '4', '5', '6', '7']
        cam_combo.grid(column=1, row=1, sticky=tk.W)
        if LAUNCHER_DEFAULTS.get('cam_num'):
            self.cam_var.set(LAUNCHER_DEFAULTS['cam_num'])
        
        # --- Date/Time with Stepper Buttons ---
        now = datetime.now()
        
        # Date
        ttk.Label(frame, text="Date (YYYYMMDD):").grid(column=0, row=2, sticky=tk.W, pady=5)
        date_frame = ttk.Frame(frame)
        date_frame.grid(column=1, row=2, sticky=tk.W)
        ttk.Button(date_frame, text="<", width=2, command=lambda: self._adjust_date(-1)).pack(side=tk.LEFT)
        default_date = LAUNCHER_DEFAULTS.get('date') or now.strftime('%Y%m%d')
        self.date_var = tk.StringVar(value=default_date)
        ttk.Entry(date_frame, textvariable=self.date_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(date_frame, text=">", width=2, command=lambda: self._adjust_date(1)).pack(side=tk.LEFT)
        
        # Hour
        ttk.Label(frame, text="Hour (hh):").grid(column=0, row=3, sticky=tk.W, pady=5)
        hour_frame = ttk.Frame(frame)
        hour_frame.grid(column=1, row=3, sticky=tk.W)
        ttk.Button(hour_frame, text="<", width=2, command=lambda: self._adjust_hour(-1)).pack(side=tk.LEFT)
        self.hour_var = tk.StringVar(value=LAUNCHER_DEFAULTS.get('hour', '00'))
        ttk.Entry(hour_frame, textvariable=self.hour_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(hour_frame, text=">", width=2, command=lambda: self._adjust_hour(1)).pack(side=tk.LEFT)

        # Minute
        ttk.Label(frame, text="Minute (mm):").grid(column=0, row=4, sticky=tk.W, pady=5)
        min_frame = ttk.Frame(frame)
        min_frame.grid(column=1, row=4, sticky=tk.W)
        ttk.Button(min_frame, text="<", width=2, command=lambda: self._adjust_minute(-1)).pack(side=tk.LEFT)
        self.min_var = tk.StringVar(value=LAUNCHER_DEFAULTS.get('minute', '00'))
        ttk.Entry(min_frame, textvariable=self.min_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(min_frame, text=">", width=2, command=lambda: self._adjust_minute(1)).pack(side=tk.LEFT)

        # Duration
        ttk.Label(frame, text="Duration (min):").grid(column=0, row=5, sticky=tk.W, pady=5)
        self.duration_var = tk.StringVar(value=LAUNCHER_DEFAULTS.get('duration', '1'))
        dur_spin = ttk.Spinbox(frame, from_=1, to=1440, textvariable=self.duration_var, width=5)
        dur_spin.grid(column=1, row=5, sticky=tk.W)

        # Status Label
        self.status_label = ttk.Label(frame, text="")
        self.status_label.grid(column=0, row=6, columnspan=2, sticky=tk.W, pady=10)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(column=0, row=7, columnspan=2, sticky=tk.E, pady=10)
        
        self.fetch_button = ttk.Button(btn_frame, text="Fetch", command=self.fetch_files)
        self.fetch_button.grid(column=1, row=0, padx=5)
        
        cancel_button = ttk.Button(btn_frame, text="Cancel", command=self.cancel)
        cancel_button.grid(column=0, row=0, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def _adjust_date(self, amount):
        try:
            current_date = datetime.strptime(self.date_var.get(), '%Y%m%d')
            new_date = current_date + timedelta(days=amount)
            self.date_var.set(new_date.strftime('%Y%m%d'))
        except ValueError:
            self.date_var.set(datetime.now().strftime('%Y%m%d'))

    def _adjust_hour(self, amount):
        try:
            current_hour = int(self.hour_var.get())
            new_hour = (current_hour + amount) % 24
            self.hour_var.set(f"{new_hour:02d}")
        except ValueError:
            self.hour_var.set("00")

    def _adjust_minute(self, amount):
        try:
            current_minute = int(self.min_var.get())
            new_minute = (current_minute + amount) % 60
            self.min_var.set(f"{new_minute:02d}")
        except ValueError:
            self.min_var.set("00")

    def cancel(self):
        self.main_tk.destroy()
        sys.exit(0)

    def fetch_files(self):
        station_display = self.station_var.get()
        cam_num = self.cam_var.get()
        date_str = self.date_var.get()
        hour_str = self.hour_var.get()
        min_str = self.min_var.get()

        try:
            duration = int(self.duration_var.get())
            if duration < 1: duration = 1
        except ValueError:
            duration = 1

        if not station_display or not cam_num:
            messagebox.showerror("Error", "Please select a station and camera.")
            return

        if not re.match(r'^\d{8}$', date_str):
            messagebox.showerror("Error", "Invalid Date format. Must be YYYYMMDD.")
            return

        hostname = self.station_map[station_display]
        LAUNCHER_DEFAULTS['station_display'] = station_display
        LAUNCHER_DEFAULTS['cam_num'] = cam_num
        LAUNCHER_DEFAULTS['date'] = date_str
        LAUNCHER_DEFAULTS['hour'] = hour_str
        LAUNCHER_DEFAULTS['minute'] = min_str
        LAUNCHER_DEFAULTS['duration'] = str(duration)
        self.fetch_button.config(state=tk.DISABLED)

        try:
            start_dt = datetime.strptime(f"{date_str}{hour_str}{min_str}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            messagebox.showerror("Error", "Invalid Date/Time.")
            self.fetch_button.config(state=tk.NORMAL)
            return

        remote_cfg_path = "/etc/meteor.cfg"
        remote_pto_path = f"/meteor/cam{cam_num}/lens.pto"
        self.upload_target = f"{hostname}:/meteor/cam{cam_num}"

        try:
            self.status_label.config(text=f"Fetching config from {hostname}...")
            self.root.update_idletasks()
            local_cfg = scp_file(hostname, remote_cfg_path)
            if local_cfg:
                temp_files_to_clean.append(local_cfg)
            else:
                raise Exception(f"Failed to fetch config file: {remote_cfg_path}")

            self.status_label.config(text=f"Fetching PTO from {hostname}...")
            self.root.update_idletasks()
            local_pto = scp_file(hostname, remote_pto_path)
            if local_pto:
                temp_files_to_clean.append(local_pto)
            else:
                raise Exception(f"Failed to fetch PTO file: {remote_pto_path}")

            local_videos = []
            for i in range(duration):
                current_dt = start_dt + timedelta(minutes=i)
                c_date = current_dt.strftime("%Y%m%d")
                c_hour = current_dt.strftime("%H")
                c_min = current_dt.strftime("%M")
                
                remote_video_path = f"/meteor/cam{cam_num}/{c_date}/{c_hour}/full_{c_min}.mp4"
                
                self.status_label.config(text=f"Fetching video {i+1}/{duration} ({c_hour}:{c_min})...")
                self.root.update_idletasks()
                
                local_vid = scp_file(hostname, remote_video_path)
                if local_vid:
                    temp_files_to_clean.append(local_vid)
                    local_videos.append(local_vid)
                else:
                    print(f"Warning: Could not fetch {remote_video_path}", file=sys.stderr)

            if not local_videos:
                raise Exception("Failed to fetch any video files.")

            self.status_label.config(text="Success! Launching...")
            self.fetched_files = {
                'config': local_cfg,
                'pto': local_pto,
                'video': local_videos,
                'selected_datetime': start_dt
            }
            self.main_tk.destroy()

        except Exception as e:
            messagebox.showerror("Fetch Failed", str(e))
            self.status_label.config(text="Fetch failed. Please try again.")
            self.fetch_button.config(state=tk.NORMAL)


class RecalibrateDialog:
    def __init__(self, parent, image, zoom_instance):
        self.parent = parent
        self.image = image
        self.zoom = zoom_instance
        self.img_data = self.zoom.img_data
        self.pto_data = self.zoom.pto_data
        self.parent.title("Recalibrate")

        self.original_params = self.img_data.copy()

        self.radius_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.IntVar(value=50)
        self.sigma_var = tk.IntVar(value=20)
        self.lens_optimize_var = tk.BooleanVar(value=True)

        self.create_slider("Radius", 0.1, 5.0, self.radius_var, row=0)
        self.create_slider("Blur", 1, 100, self.blur_var, row=1)
        self.create_slider("Sigma", 1, 100, self.sigma_var, row=2)

        lens_optimize_btn = tk.Checkbutton(self.parent, text="Optimize lens parameters", variable=self.lens_optimize_var)
        lens_optimize_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky='w')

        solve_btn = tk.Button(self.parent, text="Solve", command=self.recal)
        solve_btn.grid(row=4, column=0, pady=10, sticky='w')

        reset_btn = tk.Button(self.parent, text="Reset", command=self.reset)
        reset_btn.grid(row=4, column=1, pady=10, sticky='e')

        self.rms_label = tk.Label(self.parent, text="", justify=tk.LEFT)
        self.rms_label.grid(row=5, column=0, columnspan=2, pady=5, sticky='w')

        close_btn = tk.Button(self.parent, text="Close", command=self.parent.destroy)
        close_btn.grid(row=6, column=0, columnspan=2, pady=10, sticky='w')

        help_label = tk.Label(self.parent, font=("Helvetica", 14, "bold"), text="Fine-tuning the calibration")
        help_label.grid(row=0, column=3, sticky='w', padx=(10, 0))

        help_text = "The sliders controls the following values:\n- Radius: search mask size.\n- Blur: mask blurring.\n- Sigma: noise assumption.\n- Optimize lens: full recalibration."
        help_text_label = tk.Label(self.parent, text=help_text, justify=tk.LEFT)
        help_text_label.grid(row=1, column=3, rowspan=6, sticky='w', padx=(10, 0))

        par_text = "Original parameters:\n  pitch=%.2f°\n  yaw=%.2f°\n  roll=%.2f°\n  hfov=%.2f°" % (
            self.original_params.get('p', 0), self.original_params.get('y', 0), self.original_params.get('r', 0), self.original_params.get('v', 0))
        self.par_text_label = tk.Label(self.parent, text=par_text, justify=tk.LEFT)
        self.par_text_label.grid(row=7, column=0, columnspan=4, sticky='w', padx=(10, 0))

    def create_slider(self, label_text, min_val, max_val, variable, row):
        label = tk.Label(self.parent, text=label_text)
        label.grid(row=row, column=0, sticky='w')
        slider_frame = ttk.Frame(self.parent)
        slider_frame.grid(row=row, column=1, sticky='w')
        slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=lambda val: self.update_slider_label(val))
        slider.pack(pady=5)
        value_label = tk.Label(slider_frame, text=f"{variable.get():.2f}")
        value_label.pack()
        slider.value_label = value_label

    def update_slider_label(self, value):
        for child in self.parent.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ttk.Scale) and subchild.cget('variable') == self.radius_var:
                         subchild.value_label.config(text=f"{float(value):.2f}")

    def reset(self):
        self.zoom.img_data.clear()
        self.zoom.img_data.update(self.original_params)
        self.zoom.show_image()

    def recal(self):
        self.rms_label.config(text="Solving...")
        self.parent.update_idletasks()
        
        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".png") as img_f:
            self.image.save(img_f, format='PNG')
            img_temp_filename = img_f.name
        
        img_idx = self.zoom.image_index
        vars_to_optimize = [f'v{img_idx}', f'r{img_idx}', f'p{img_idx}', f'y{img_idx}']
        if self.lens_optimize_var.get():
            vars_to_optimize.extend([f'a{img_idx}', f'b{img_idx}', f'c{img_idx}', f'd{img_idx}', f'e{img_idx}'])

        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".pto") as pto_f:
            pto_mapper.write_pto_file(self.pto_data, pto_f.name, optimize_vars=vars_to_optimize)
            old_lens_filename = pto_f.name

        new_lens_filename = tempfile.NamedTemporaryFile(delete=True, dir="/tmp", suffix=".pto").name
        log_file_path = tempfile.mktemp(suffix=".log", dir="/tmp")

        try:
            starttime = None
            try:
                if getattr(self.zoom, 'timestamps', None):
                    starttime = float(self.zoom.timestamps[0])
            except Exception:
                starttime = None
            if starttime is None:
                starttime = datetime.now(timezone.utc).timestamp()

            with open(log_file_path, 'w') as log_f:
                with contextlib.redirect_stdout(log_f):
                    recalibrate(
                        starttime, old_lens_filename, img_temp_filename, new_lens_filename, pos,
                        image=self.zoom.image_index, radius=self.radius_var.get(),
                        lensopt=self.lens_optimize_var.get(), faintest=4, brightest=-5,
                        objects=500, blur=self.blur_var.get(), verbose=True, sigma=self.sigma_var.get()
                    )

            with open(log_file_path, 'r') as log_f: output = log_f.read()
            rms_values = re.findall(r"after \d+ iteration\(s\):\s*([\d.]+)\s*units", str(output))
            if rms_values:
                self.rms_label.config(text=f"Final RMS: {float(rms_values[-1]):.2f}")
            else:
                self.rms_label.config(text="RMS: Not found")

            _, new_images_data = pto_mapper.parse_pto_file(new_lens_filename)
            new_img_data = new_images_data[self.zoom.image_index]
            for param in ['p', 'y', 'r', 'v', 'a', 'b', 'c', 'd', 'e']:
                if param in new_img_data: self.img_data[param] = new_img_data[param]

            self.zoom.show_image()
            self.zoom.pto_dirty = True
        except Exception as e:
            self.rms_label.config(text=f"Error: {e}")
        finally:
            if os.path.exists(img_temp_filename): os.remove(img_temp_filename)
            if os.path.exists(old_lens_filename): os.remove(old_lens_filename)
            if os.path.exists(new_lens_filename): os.remove(new_lens_filename)
            if os.path.exists(log_file_path): os.remove(log_file_path)

def read_frames(filename, directory, skip_seconds=0, total_seconds=None):
    try:
        probe = ffmpeg.probe(filename)
        total_frames = int(next(s['nb_frames'] for s in probe['streams'] if s['codec_type'] == 'video'))
    except (ffmpeg.Error, StopIteration, KeyError, ValueError):
        total_frames = None

    print(f"Decoding video {filename}...")
    input_kwargs = {}
    if skip_seconds > 0: input_kwargs['ss'] = skip_seconds
    if total_seconds is not None: input_kwargs['t'] = total_seconds

    stream = ffmpeg.input(filename, **input_kwargs).output(f'{directory}/%04d.tif', format='image2', vsync=0).overwrite_output()
    args = stream.get_args()
    
    final_args = ['ffmpeg']
    if total_frames: final_args.extend(['-progress', 'pipe:1'])
    final_args.extend(args)

    proc = subprocess.Popen(final_args, stdout=subprocess.PIPE if total_frames else None, stderr=subprocess.PIPE, text=True, errors='ignore')

    if not total_frames:
        _, err = proc.communicate()
        if proc.returncode != 0: raise ffmpeg.Error('ffmpeg', None, err)
        return

    bar_length = 40
    for line in iter(proc.stdout.readline, ''):
        match = re.search(r'frame=\s*(\d+)', line)
        if match:
            current_frame = int(match.group(1))
            progress = current_frame / total_frames
            block = int(round(bar_length * progress))
            sys.stdout.write(f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {current_frame}/{total_frames} ({progress*100:.0f}%)")
            sys.stdout.flush()

    sys.stdout.write(f"\rProgress: [{'#' * bar_length}] {total_frames}/{total_frames} (100%)\n\n")
    if proc.wait() != 0: raise ffmpeg.Error('ffmpeg', None, proc.stderr.read())


def read_first_frame(filename, directory, skip_seconds=0):
    print(f"Decoding first frame of {filename}...")
    input_kwargs = {}
    if skip_seconds > 0:
        input_kwargs['ss'] = skip_seconds
    stream = ffmpeg.input(filename, **input_kwargs).output(f'{directory}/%04d.tif', vframes=1, format='image2', vsync=0).overwrite_output()
    args = ['ffmpeg'] + stream.get_args()
    proc = subprocess.run(args, capture_output=True, text=True, errors='ignore')
    if proc.returncode != 0:
        raise ffmpeg.Error('ffmpeg', None, proc.stderr)


def parse_start_time_arg(s):
    if not s:
        return None
    st = str(s).strip()
    if st.upper().endswith('UTC'):
        st = st[:-3].strip()
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(st, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        iso = st.replace('Z', '+00:00')
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None

def timestamp(timestamps, files, i):
    if i < len(timestamps) and timestamps[i] is not None: return timestamps[i]
    try:
        datetime_obj = get_timestamp(files[i])
    except FileNotFoundError:
        return None
    if not datetime_obj: datetime_obj = get_timestamp(files[i], robust=True)
    
    if datetime_obj:
        ts = datetime_obj.timestamp()
        timestamps[i] = ts
        return ts
    else:
        return None

def interpolate_timestamps(timestamps):
    if not timestamps: return []
    interpolated_stamps = list(timestamps)
    fractions = []
    prev = interpolated_stamps[0]
    i = 1
    previ = -1
    while i < len(interpolated_stamps):
        if interpolated_stamps[i] == prev:
            while i < len(interpolated_stamps) and interpolated_stamps[i] == prev: i += 1
            fractions.append(i - previ)
            previ = i
            if i < len(interpolated_stamps): prev = interpolated_stamps[i]
        else:
            previ += 1
            while i < len(interpolated_stamps) and interpolated_stamps[i] != prev: i -= 1
            fractions.append(i + 2 - previ)
            previ = i + 1
            if i + 1 < len(interpolated_stamps): prev = interpolated_stamps[i + 1]
            else: break
        if not fractions or fractions[-1] == 0:
            i+=1
            continue
        i += fractions[-1]

    fractions.append(len(interpolated_stamps) - previ)
    if len(fractions) > 1: fractions[-1] = fractions[-2]
    prev = interpolated_stamps[0]
    f = 0.0
    if len(fractions) >= 2 and fractions[1] != 0: f = 1 - fractions[0] / fractions[1]
    fi = 0
    if fractions and fractions[0] != 0: fractions[0] = fractions[1]
    
    for i in range(1, len(interpolated_stamps)):
        if interpolated_stamps[i] is None: interpolated_stamps[i] = interpolated_stamps[i - 1]
    
    for i in range(len(interpolated_stamps)):
        if interpolated_stamps[i] == prev:
            if fi < len(fractions) and fractions[fi] > 0: f = f + 1.0 / fractions[fi]
            interpolated_stamps[i] += round(f, 2)
        else:
            prev = interpolated_stamps[i]
            f = 0.0
            fi += 1
    return interpolated_stamps

def midpoint(az1, alt1, az2, alt2):
    x1, y1 = math.radians(az1), math.radians(alt1)
    x2, y2 = math.radians(az2), math.radians(alt2)
    Bx = math.cos(y2) * math.cos(x2 - x1)
    By = math.cos(y2) * math.sin(x2 - x1)
    y3 = math.atan2(math.sin(y1) + math.sin(y2), math.sqrt((math.cos(y1) + Bx)**2 + By**2))
    x3 = x1 + math.atan2(By, math.cos(y1) + Bx)
    a = math.sin((y2 - y1) / 2)**2 + math.cos(y1) * math.cos(y2) * math.sin((x2 - x1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return math.degrees(x3), math.degrees(y3), math.degrees(c)

class Zoom_Advanced(ttk.Frame):
    def __init__(self, mainframe, files, timestamps, pto_data, image_index, **kwargs):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Click Coords')
        self.frames_ready = kwargs.pop('frames_ready', True)
        self.overlay = []
        self.canvas = tk.Canvas(self.master, highlightthickness=0, cursor="draft_small", bg="black")
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.mousepos = "\n"
        self.files = files
        self.timestamps = timestamps
        self.pto_data = pto_data
        self.global_options, self.images_data = pto_data
        self.image_index = image_index
        self.img_data = self.images_data[self.image_index]
        self.num = 0
        self.image = Image.open(files[0])
        self.width, self.height = self.image.size
        self.x = 0.0
        self.y = 0.0
        self.imscale = 1.0
        self.delta = 1.05
        self.contrast = 1
        self.brightness = 1
        self.color = 1
        self.sharpness = 1
        self.show_text = 0
        self.show_info = 0
        self.show_graph = 1
        self.boost = 1
        self.star_objects = 128
        self.restart = False
        self.load_cancel_event = threading.Event()
        self.load_thread = None
        self.offsetx = 0
        self.offsety = 0
        self.undo_stack = []
        self.redo_stack = []
        self.dragged_point_index = None
        self.canvas.bind('<ButtonRelease-1>', self.drag_release)
        self.background_removal_active = False
        self.background_image = None
        self._create_background_image()
        self.positions = []
        self.centroid = [None] * len(files)
        self.curve_coeffs = None
        self.curve_orientation = None
        self.predicted_point = None
        self.last_click_to_highlight = None
        self.last_orientation_backup = None 
        
        self.is_calibrate_mode = kwargs.get('is_calibrate_mode', False)
        self.hostname = kwargs.get('hostname')
        self.cam_num = kwargs.get('cam_num')
        self.date_str = kwargs.get('date_str')
        self.upload_hostname = kwargs.get('upload_hostname')
        self.upload_dir = kwargs.get('upload_dir')
        self.pto_dirty = False

        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)

        self.canvas.bind('<Configure>', self.show_image)
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>', self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<Button-5>', self.wheel)
        self.canvas.bind('<Button-4>', self.wheel)
        self.canvas.bind('<Left>', self.left_key)
        self.canvas.bind('<Right>', self.right_key)
        self.canvas.bind('<Prior>', self.page_up)
        self.canvas.bind('<Next>', self.page_down)
        self.canvas.bind('<Key>', self.key)
        self.canvas.bind('<Button-3>', self.click)
        self.canvas.bind("<Motion>", self.moved)
        self.canvas.bind('<ButtonRelease-1>', self.drag_release)
        self.canvas.focus_set()

        self.show_image()

    def cancel_background_load(self, wait=False, timeout=2.0):
        try:
            self.load_cancel_event.set()
        except Exception:
            pass
        if wait and self.load_thread and self.load_thread.is_alive():
            try:
                self.load_thread.join(timeout=timeout)
            except Exception:
                pass

    def set_sequence(self, files, timestamps):
        self.files = files
        self.timestamps = timestamps
        if self.num >= len(self.files):
            self.num = 0
        if len(self.centroid) != len(self.files):
            self.centroid = [None] * len(self.files)
        self.frames_ready = True
        try:
            self.image = Image.open(self.files[self.num])
        except Exception:
            pass
        self._create_background_image()
        self._update_highlight_state()
        self.show_image()
        
    def _image_coords_to_celestial(self, x, y):
        pano_coords = pto_mapper.map_image_to_pano(self.pto_data, self.image_index, x, y)
        if pano_coords:
            pano_x, pano_y = pano_coords
            pano_w = self.global_options.get('w')
            pano_h = self.global_options.get('h')
            if self.global_options.get('f', 2) == 2:
                az_rad = (pano_x / pano_w) * 2 * math.pi
                alt_rad = (0.5 - pano_y / pano_h) * math.pi
                return math.degrees(az_rad), math.degrees(alt_rad)
            else: return -998, -998
        else: return -999, -999

    def _create_background_image(self):
        if not self.files: return
        try:
            print("Creating background image for subtraction...")
            img = Image.open(self.files[0])
            img_arr = np.array(img)
            pad_width = ((32, 32), (32, 32), (0, 0)) 
            padded_arr = np.pad(img_arr, pad_width, mode='edge')
            padded_img = Image.fromarray(padded_arr)
            blurred_padded = padded_img.filter(ImageFilter.GaussianBlur(radius=32))
            self.background_image = blurred_padded.crop((32, 32, 32 + img.width, 32 + img.height))
        except Exception as e:
            print(f"Failed to create background image: {e}", file=sys.stderr)
            self.background_image = None

    def _save_state_for_undo(self):
        self.redo_stack.clear()
        state = {'positions': copy.deepcopy(self.positions), 'centroid': copy.deepcopy(self.centroid)}
        self.undo_stack.append(state)
        if len(self.undo_stack) > 50: self.undo_stack.pop(0)

    def undo(self, event=None):
        if not self.undo_stack: return
        current_state = {'positions': copy.deepcopy(self.positions), 'centroid': copy.deepcopy(self.centroid)}
        self.redo_stack.append(current_state)
        last_state = self.undo_stack.pop()
        self.positions = last_state['positions']
        self.centroid = last_state['centroid']
        self.update_curve_fit()
        self.update_prediction()
        self._update_highlight_state()
        self.show_image()

    def redo(self, event=None):
        if not self.redo_stack: return
        current_state = {'positions': copy.deepcopy(self.positions), 'centroid': copy.deepcopy(self.centroid)}
        self.undo_stack.append(current_state)
        next_state = self.redo_stack.pop()
        self.positions = next_state['positions']
        self.centroid = next_state['centroid']
        self.update_curve_fit()
        self.update_prediction()
        self._update_highlight_state()
        self.show_image()
        
    def _auto_recalibrate(self):
        print("Calibrate mode: Automatically running initial plate solve...")
        dialog_root = tk.Toplevel(self.master)
        dialog = RecalibrateDialog(dialog_root, self.image, self)
        dialog.lens_optimize_var.set(False)
        print("  - 'Optimize lens parameters' is unchecked for the initial solve.")
        dialog.recal()

    def moved(self, event):
        if not hasattr(self, 'x'):
            return
        x, y = event.x / self.imscale + self.x, event.y / self.imscale + self.y
        x += self.offsetx
        y += self.offsety
        
        pano_coords = pto_mapper.map_image_to_pano(self.pto_data, self.image_index, x, y)
        if pano_coords:
            pano_x, pano_y = pano_coords
            pano_w = self.global_options.get('w')
            pano_h = self.global_options.get('h')
            if self.global_options.get('f', 2) == 2:
                az_rad = (pano_x / pano_w) * 2 * math.pi
                alt_rad = (0.5 - pano_y / pano_h) * math.pi
                az_deg = math.degrees(az_rad)
                alt_deg = math.degrees(alt_rad)
                ra, dec = pos.radec_of(str(az_deg % 360), str(alt_deg))
                self.mousepos="\n  cursor pos = %.2f° az, %.2f° alt / %s ra %.2f dec" % (az_deg % 360, alt_deg, str(ra), math.degrees(float(repr(dec))))
            else: self.mousepos = "\n  cursor pos = (non-equirectangular)"
        else: self.mousepos = "\n  cursor pos = outside panorama"
        self.show_image()
    
    def _save_and_deploy_calibration(self, local_pto_path, log_callback=None): 
        def log_message(msg, file=None):
            if file == sys.stderr: msg = f"ERROR: {msg}"
            if log_callback: log_callback(msg)
            else: print(msg, file=file if file else sys.stdout)
        
        log_message("Deploying new calibration to remote host...")
        try:
            remote_pto_dated = f"/meteor/cam{self.cam_num}/lens-{self.date_str}.pto"
            remote_grid_dated = f"/meteor/cam{self.cam_num}/grid-{self.date_str}.png"
            remote_pto_link = f"/meteor/cam{self.cam_num}/lens.pto"
            remote_grid_link = f"/meteor/cam{self.cam_num}/grid.png"
            drawgrid_script = "/home/meteor/bin/drawgrid.py"

            remote_full_path = f"{self.hostname}:{remote_pto_dated}"
            log_message(f"  - Copying {local_pto_path} to {remote_full_path}...")
            proc = subprocess.run(['scp', local_pto_path, remote_full_path], check=True, capture_output=True, text=True, errors='ignore')
            if proc.stdout: log_message(f"    - SCP STDOUT: {proc.stdout.strip()}")

            remote_command = (f"{drawgrid_script} {remote_pto_dated} {remote_grid_dated} && "
                              f"rm -f {remote_pto_link} {remote_grid_link} && "
                              f"ln -s {remote_pto_dated} {remote_pto_link} && "
                              f"ln -s {remote_grid_dated} {remote_grid_link}")
            log_message("  - Running remote commands to update grid and links...")
            proc = subprocess.run(['ssh', self.hostname, remote_command], check=True, capture_output=True, text=True, errors='ignore')
            log_message("  - Remote deployment successful.")

        except Exception as e:
            error_msg = f"An unexpected error occurred during deployment: {e}"
            log_message(error_msg, file=sys.stderr)
            if not log_callback: messagebox.showerror("Deployment Error", error_msg)
    
    def show_quit_menu(self):
        top = tk.Toplevel(self.master)
        top.title("Exit Options")
        top.geometry("320x120")
        try:
            x = self.master.winfo_x() + (self.master.winfo_width() // 2) - 160
            y = self.master.winfo_y() + (self.master.winfo_height() // 2) - 60
            top.geometry(f"+{x}+{y}")
        except: pass

        tk.Label(top, text="What would you like to do?", font=("Helvetica", 11)).pack(pady=15)
        btn_frame = tk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=5)
        btn_frame.pack_configure(anchor=tk.CENTER)

        def quit_app():
            self.cancel_background_load(wait=True)
            top.destroy(); sys.exit(0)
        def new_event():
            self.cancel_background_load(wait=True)
            self.restart = True; top.destroy(); self.master.destroy()
        def cancel(): top.destroy()

        tk.Button(btn_frame, text="Quit App", command=quit_app, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="New Event", command=new_event, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=cancel, width=10).pack(side=tk.LEFT, padx=10)
        top.transient(self.master)
        top.grab_set()
        self.master.wait_window(top)

    def key(self, event):
        is_control_pressed = (event.state & 0x4) != 0
        if is_control_pressed and event.keysym.lower() == 'z': self.undo(); return
        if is_control_pressed and event.keysym.lower() == 'y': self.redo(); return

        if event.keysym == 'Insert':
            self.star_objects = min(500, self.star_objects + 10)
            self.show_image()
            return
        if event.keysym == 'Delete':
            self.star_objects = max(10, self.star_objects - 10)
            self.show_image()
            return

        key_char = event.char
        is_upper = key_char.isupper()
        
        actions = {'p': ('p', 0.01), 'y': ('y', 0.01), 'r': ('r', 0.01), 'z': ('v', 0.01),
                   'a': ('a', 0.001), 'b': ('b', 0.001), 'c': ('c', 0.001), 'd': ('d', -0.5), 'e': ('e', 0.5)}

        if key_char.lower() in actions:
            param, change = actions[key_char.lower()]
            if is_upper: change *= -1
            self.img_data[param] = self.img_data.get(param, 0) + change * self.boost
            self.pto_dirty = True 

        elif key_char == '?':
            root = tk.Toplevel(self.master)
            RecalibrateDialog(root, self.image, self)
            root.mainloop()
        elif key_char == 'x': 
            self._save_state_for_undo()
            if self.curve_coeffs is not None:
                for p in self.positions:
                    snapped_pos = self._project_point_on_curve(p['current'])
                    p['current'] = snapped_pos
                    self._create_or_update_centroid(p['frame'], snapped_pos)
                self.update_curve_fit()
                self.update_prediction()
        elif key_char == 'X' or key_char == 'T': 
            self._save_state_for_undo()
            for p in self.positions:
                p['current'] = p['original']
                self._update_centroid_entry(p['frame'], p['original'])
            self.update_curve_fit()
            self.update_prediction()
        elif key_char == 't': self._temporally_respace_points()
        elif key_char == '*':
            _, fresh_images_data = pto_mapper.parse_pto_file(args.ptofile)
            self.images_data[self.image_index] = fresh_images_data[self.image_index]
            self.img_data = self.images_data[self.image_index]
            self.pto_dirty = True
        elif key_char == 'l':
            try:
                pto_mapper.write_pto_file(self.pto_data, args.ptofile)
                self.pto_dirty = False 
                if self.is_calibrate_mode: self._save_and_deploy_calibration(args.ptofile)
                else: print(f"Saved PTO file to {args.ptofile}")
            except Exception as e: print(f"Error saving/deploying PTO file: {e}", file=sys.stderr)
        elif key_char == 'o':
            self.last_orientation_backup = {'p': self.img_data.get('p', 0), 'y': self.img_data.get('y', 0),
                                            'r': self.img_data.get('r', 0), 'v': self.img_data.get('v', 0)}
            threading.Thread(target=self._optimize_orientation, daemon=True).start()
            return
        elif key_char == 'O':
            if self.last_orientation_backup:
                self.img_data.update(self.last_orientation_backup)
                self.last_orientation_backup = None
                self.pto_dirty = True
        elif key_char == 'S': self.save_centroid_txt()
        elif key_char == 's': self.save_event_txt()
        elif key_char == 'h': self.show_text ^= 1
        elif key_char == 'i': self.show_info ^= 1
        elif key_char == 'g': self.show_graph ^= 1
        elif key_char == '!': self.boost = 100 if self.boost == 1 else 1
        elif key_char == 'q':
            self.show_quit_menu()
            if self.restart: return
        elif key_char == '1': self.contrast -= 0.1
        elif key_char == '2': self.contrast += 0.1
        elif key_char == '3': self.brightness -= 0.1
        elif key_char == '4': self.brightness += 0.1
        elif key_char == '5': self.color -= 0.1
        elif key_char == '6': self.color += 0.1
        elif key_char == '7': self.sharpness -= 0.2
        elif key_char == '8': self.sharpness += 0.2
        elif key_char == '0': self.contrast = self.brightness = self.color = self.sharpness = 1
        elif key_char == '-':
            self.background_removal_active = True
            self.show_image()
        elif key_char == '+':
            self.background_removal_active = False
            self.show_image()
        elif key_char == 'u': self.upload_data()
        self.show_image()

    def _optimize_orientation(self):
        def _update_ui_start(): self.canvas.config(cursor="watch")
        self.master.after_idle(_update_ui_start)
        
        img_temp_filename, old_lens_filename, new_lens_filename, log_file_path = None, None, None, None
        try:
            starttime = None
            try:
                if getattr(self, 'timestamps', None):
                    starttime = float(self.timestamps[0])
            except Exception:
                starttime = None
            if starttime is None:
                starttime = datetime.now(timezone.utc).timestamp()

            with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".png") as img_f:
                self.image.save(img_f, format='PNG')
                img_temp_filename = img_f.name
            img_idx = self.image_index
            vars_to_optimize = [f'v{img_idx}', f'r{img_idx}', f'p{img_idx}', f'y{img_idx}']
            with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".pto") as pto_f:
                pto_mapper.write_pto_file(self.pto_data, pto_f.name, optimize_vars=vars_to_optimize)
                old_lens_filename = pto_f.name
            new_lens_filename = tempfile.NamedTemporaryFile(delete=True, dir="/tmp", suffix=".pto").name
            log_file_path = tempfile.mktemp(suffix=".log", dir="/tmp")
            with open(log_file_path, 'w') as log_f:
                with contextlib.redirect_stdout(log_f):
                    recalibrate(starttime, old_lens_filename, img_temp_filename, new_lens_filename, pos,
                        image=self.image_index, radius=1.0, lensopt=False, faintest=4, brightest=-5,
                        objects=500, blur=50, verbose=True, sigma=20)
            if not os.path.exists(new_lens_filename) or os.path.getsize(new_lens_filename) == 0:
                raise RuntimeError("Optimization failed to produce a new calibration file.")
            _, new_images_data = pto_mapper.parse_pto_file(new_lens_filename)
            new_img_data = new_images_data[self.image_index]
            def _update_gui_success():
                for param in ['p', 'y', 'r', 'v']:
                    if param in new_img_data: self.img_data[param] = new_img_data[param]
                self.pto_dirty = True
                self.show_image() 
            self.master.after_idle(_update_gui_success)
        except Exception as e:
            err_msg = str(e)
            def _update_gui_error(err_msg=err_msg):
                messagebox.showerror("Optimization Error", err_msg)
                if self.last_orientation_backup:
                    self.img_data.update(self.last_orientation_backup)
                    self.last_orientation_backup = None
                    self.pto_dirty = True
                self.show_image() 
            self.master.after_idle(_update_gui_error)
        finally:
            def _update_gui_finish():
                if img_temp_filename and os.path.exists(img_temp_filename): os.remove(img_temp_filename)
                if old_lens_filename and os.path.exists(old_lens_filename): os.remove(old_lens_filename)
                if new_lens_filename and os.path.exists(new_lens_filename): os.remove(new_lens_filename)
                if log_file_path and os.path.exists(log_file_path): os.remove(log_file_path)
                self.canvas.config(cursor="draft_small")
                self.show_image()
            self.master.after_idle(_update_gui_finish)

    def _update_centroid_entry(self, frame_num, xy_coords):
        if self.centroid[frame_num] is None: return
        x, y = xy_coords
        az_deg, alt_deg = self._image_coords_to_celestial(x, y)
        parts = self.centroid[frame_num].split(' ')
        parts[2] = f"{alt_deg:.2f}"
        parts[3] = f"{az_deg % 360:.2f}"
        self.centroid[frame_num] = ' '.join(parts)
        
    def _estimate_photometry(self, image, center_xy, aperture_r=5, annulus_r_inner=7, annulus_r_outer=10):
        try:
            img_gray = image.convert('L')
            img_array = np.array(img_gray)
            y_coords, x_coords = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
            center_x, center_y = center_xy
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            aperture_mask = dist_from_center <= aperture_r
            annulus_mask = (dist_from_center > annulus_r_inner) & (dist_from_center <= annulus_r_outer)
            background_pixels = img_array[annulus_mask]
            if background_pixels.size == 0: return 0.0
            mean_background = np.mean(background_pixels)
            aperture_pixels = img_array[aperture_mask]
            total_aperture_brightness = np.sum(aperture_pixels)
            background_contribution = mean_background * aperture_pixels.size
            integrated_brightness = total_aperture_brightness - background_contribution
            return max(0, round(integrated_brightness / 100.0, 2))
        except Exception as e:
            print(f"Photometry brightness estimation failed: {e}", file=sys.stderr)
            return 0.0

    def _estimate_saturation_brightness(self, image, center_xy, tolerance_percent=0.05):
        try:
            img_gray = image.convert('L')
            img_array = np.array(img_gray)
            height, width = img_array.shape
            center_x, center_y = int(round(center_xy[0])), int(round(center_xy[1]))
            if not (0 <= center_x < width and 0 <= center_y < height): return 0.0
            start_value = float(img_array[center_y, center_x])
            tolerance_value = 255 * tolerance_percent
            min_val = start_value - tolerance_value
            max_val = start_value + tolerance_value
            q = deque([(center_x, center_y)])
            visited = set([(center_x, center_y)])
            area = 0
            while q:
                x, y = q.popleft()
                area += 1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        if (0 <= nx < width and 0 <= ny < height and min_val <= img_array[ny, nx] <= max_val):
                            q.append((nx, ny))
            return float(area)
        except Exception: return 0.0

    def _create_or_update_centroid(self, frame_num, xy_coords):
        image_to_analyze = Image.open(self.files[frame_num])
        img_gray = image_to_analyze.convert('L')
        img_array = np.array(img_gray)
        center_x, center_y = int(round(xy_coords[0])), int(round(xy_coords[1]))
        brightness = 0.0
        
        if not (0 <= center_y < img_array.shape[0] and 0 <= center_x < img_array.shape[1]): brightness = 0.0
        else:
            start_pixel_value = img_array[center_y, center_x]
            saturation_threshold = 255 * 0.95
            brightness = self._estimate_photometry(image_to_analyze, xy_coords)
            if start_pixel_value >= saturation_threshold:
                brightness += self._estimate_saturation_brightness(image_to_analyze, xy_coords)

        x, y = xy_coords
        az_deg, alt_deg = self._image_coords_to_celestial(x, y)
        diff = self.timestamps[frame_num] - self.timestamps[0]
        ts = datetime.fromtimestamp(self.timestamps[frame_num], timezone.utc)
        self.centroid[frame_num] = f'{frame_num} {diff:.2f} {alt_deg:.2f} {az_deg % 360:.2f} {brightness:.2f} {args.name} {ts.strftime("%Y-%m-%d %H:%M:%S.%f UTC")}'

    def _draw_brightness_graph(self, canvas_bbox):
        if len(self.positions) < 2: return
        graph_data = []
        current_frame_data = None
        for p in self.positions:
            frame = p['frame']
            if self.centroid and frame < len(self.centroid) and self.centroid[frame]:
                try:
                    time = self.timestamps[frame]
                    brightness = float(self.centroid[frame].split(' ')[4])
                    point = (time, brightness)
                    graph_data.append(point)
                    if frame == self.num: current_frame_data = point
                except (IndexError, ValueError): continue

        if len(graph_data) < 2: return
        padding = 10
        graph_width = 200
        graph_height = 100
        graph_x = canvas_bbox[2] - graph_width - padding
        graph_y = canvas_bbox[1] + padding

        self.overlay.append(self.canvas.create_rectangle(
            graph_x, graph_y, graph_x + graph_width, graph_y + graph_height,
            fill="gray10", outline="gray50", stipple="gray50"))

        times, brightnesses = zip(*graph_data)
        min_time, max_time = min(times), max(times)
        min_bright, max_bright = min(brightnesses), max(brightnesses)
        time_range = max_time - min_time if max_time > min_time else 1
        bright_range = max_bright - min_bright if max_bright > min_bright else 1

        scaled_points = []
        for t, b in graph_data:
            px = graph_x + ((t - min_time) / time_range) * graph_width
            py = graph_y + graph_height - ((b - min_bright) / bright_range) * graph_height
            scaled_points.extend([px, py])

        self.overlay.append(self.canvas.create_line(scaled_points, fill="cyan", width=1))
        self.overlay.append(self.canvas.create_text(graph_x + 5, graph_y + 5, text=f"{max_bright:.1f}", anchor="nw", fill="yellow", font=("helvetica", 8)))
        self.overlay.append(self.canvas.create_text(graph_x + 5, graph_y + graph_height - 5, text=f"{min_bright:.1f}", anchor="sw", fill="yellow", font=("helvetica", 8)))

        if current_frame_data:
            t, b = current_frame_data
            px = graph_x + ((t - min_time) / time_range) * graph_width
            py = graph_y + graph_height - ((b - min_bright) / bright_range) * graph_height
            self.overlay.append(self.canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill="red", outline=""))

    def save_event_txt(self, filepath="event.txt"): 
        valid_positions = [p for p in self.positions if self.centroid[p['frame']] is not None]
        centroid2 = [l for l in self.centroid if l is not None]
        if not centroid2: return
        
        with open(filepath, "w") as f:
            print("[trail]", file=f)
            print(f"frames = {len(centroid2)}", file=f)
            duration_val = round(float(centroid2[-1].split(' ')[1]) - float(centroid2[0].split(' ')[1]), 2)
            print(f"duration = {duration_val}", file=f)
            pos_str = " ".join([f"{p['current'][0]:.2f},{p['current'][1]:.2f}" for p in valid_positions])
            print(f"positions = {pos_str}", file=f)
            coord_str_list = [f"{i.split(' ')[3]},{i.split(' ')[2]}" for i in centroid2]
            print(f"coordinates = {' '.join(coord_str_list)}", file=f)
            ts_str = " ".join([str(self.timestamps[int(i.split(' ')[0])]) for i in centroid2])
            print(f"timestamps = {ts_str}", file=f)
            midaz, midalt, arc = midpoint(float(centroid2[0].split(" ")[3]), float(centroid2[0].split(" ")[2]),
                                          float(centroid2[-1].split(" ")[3]), float(centroid2[-1].split(" ")[2]))
            print(f"midpoint = {midaz},{midalt}", file=f)
            print(f"arc = {arc}", file=f)
            brightness_values = [c.split(' ')[4] for c in centroid2]
            print(f"brightness = {' '.join(brightness_values)}", file=f)
            print("frame_brightness = " + " ".join(["0"] * len(centroid2)), file=f)
            print("size = " + " ".join(["0"] * len(centroid2)), file=f)
            print("manual = 1\n", file=f)
            print("[video]", file=f)
            start_info = " ".join(centroid2[0].split(" ")[6:])
            start_ts = self.timestamps[int(centroid2[0].split(" ")[0])]
            print(f"start = {start_info} ({start_ts})", file=f)
            end_info = " ".join(centroid2[-1].split(" ")[6:])
            end_ts = self.timestamps[int(centroid2[-1].split(" ")[0])]
            print(f"end = {end_info} ({end_ts})", file=f)
            print(f"width = {self.width}\nheight = {self.height}\n", file=f)
            print("[summary]", file=f)
            print(f"timestamp = {start_info} ({start_ts})", file=f)
            print(f"startpos = {coord_str_list[0].replace(',', ' ')}", file=f)
            print(f"endpos = {coord_str_list[-1].replace(',', ' ')}", file=f)
            print(f"duration = {duration_val}", file=f)
        print(f"Saved event data to {filepath}") 

    def save_centroid_txt(self, filepath="centroid.txt"):
        try:
            with open(filepath, "w") as f:
                for l in self.centroid:
                    if l is not None: f.write(l + "\n")
            print(f"Saved centroid data to {filepath}")
        except Exception as e: print(f"Error saving centroid file '{filepath}': {e}", file=sys.stderr)

    def upload_data(self, event=None):
        if not self.upload_hostname or not self.upload_dir:
            print("Upload target not configured. Use --upload-target HOST:DIR.")
            return

        if not self.positions:
            messagebox.showwarning("Upload Error", "No points selected. Cannot determine upload timestamp.")
            return
            
        first_point_frame = self.positions[0]['frame']
        first_timestamp = self.timestamps[first_point_frame]

        temp_event = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix="_event.txt", dir="/tmp")
        temp_event.close()
        self.save_event_txt(filepath=temp_event.name)

        temp_centroid = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix="_centroid.txt", dir="/tmp")
        temp_centroid.close()
        self.save_centroid_txt(filepath=temp_centroid.name)

        temp_lens_path = ""
        if self.pto_dirty:
            temp_lens = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix="_lens.pto", dir="/tmp")
            temp_lens.close()
            pto_mapper.write_pto_file(self.pto_data, temp_lens.name)
            temp_lens_path = temp_lens.name

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--upload-worker",
            "--upload-hostname", self.upload_hostname,
            "--upload-dir", self.upload_dir,
            "--upload-timestamp", str(first_timestamp),
            "--upload-event-file", temp_event.name,
            "--upload-centroid-file", temp_centroid.name,
        ]

        if temp_lens_path:
            cmd.extend(["--upload-lens-file", temp_lens_path])

        proc = subprocess.Popen([c for c in cmd if c != ""], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)

        def _check_upload_worker_started():
            rc = proc.poll()
            if rc is None:
                return

            messagebox.showerror(
                "Upload Error",
                f"Upload worker exited immediately (exit code {rc})."
            )

        try:
            self.master.after(500, _check_upload_worker_started)
            self.master.after(2000, _check_upload_worker_started)
        except Exception:
            pass
        self.pto_dirty = False

    def load_event_file(self, filepath):
        try:
            config = configparser.ConfigParser()
            config.read(filepath)
            positions_str = config.get('trail', 'positions')
            timestamps_str = config.get('trail', 'timestamps')
            brightness_str = config.get('trail', 'brightness', fallback="0")
            event_brightnesses = brightness_str.split(' ')

            event_positions_xy = []
            for part in positions_str.split(' '):
                if ',' in part:
                    x_str, y_str = part.split(',')
                    event_positions_xy.append((float(x_str), float(y_str)))
            event_timestamps = [float(t) for t in timestamps_str.split(' ')]
            frame_indices = []

            tolerance = 1.0 / args.fps
            app_ts_map = {round(ts, 2): i for i, ts in enumerate(self.timestamps)}

            for t_event in event_timestamps:
                if round(t_event, 2) in app_ts_map:
                    frame_indices.append(app_ts_map[round(t_event, 2)])
                    continue
                min_diff = float('inf')
                best_match_idx = -1
                for i, app_ts in enumerate(self.timestamps):
                    diff = abs(t_event - app_ts)
                    if diff < min_diff:
                        min_diff = diff
                        best_match_idx = i
                if best_match_idx != -1 and min_diff < tolerance:
                    frame_indices.append(best_match_idx)
                else:
                    print(f"Warning: Timestamp {t_event:.4f} not found.", file=sys.stderr)
                    self.positions.clear()
                    self.centroid = [None] * len(self.files)
                    return

            print(f"Successfully loaded {len(frame_indices)} points from {filepath}")
            self.positions = []
            for i, frame_idx in enumerate(frame_indices):
                pos_xy = event_positions_xy[i]
                point_data = {'frame': frame_idx, 'original': pos_xy, 'current': pos_xy}
                self.positions.append(point_data)
                x, y = pos_xy
                az_deg, alt_deg = self._image_coords_to_celestial(x, y)
                diff = self.timestamps[frame_idx] - self.timestamps[0]
                ts_obj = datetime.fromtimestamp(self.timestamps[frame_idx], timezone.utc)
                brightness_val = event_brightnesses[i] if i < len(event_brightnesses) else "0.0"
                self.centroid[frame_idx] = f'{frame_idx} {diff:.2f} {alt_deg:.2f} {az_deg % 360:.2f} {brightness_val} {args.name} {ts_obj.strftime("%Y-%m-%d %H:%M:%S.%f UTC")}'

            if self.positions:
                self.update_curve_fit()
                self.update_prediction()
                self.change_image(self.positions[0]['frame'])

        except Exception as e:
            print(f"Error loading event file '{filepath}': {e}", file=sys.stderr)

    def _update_highlight_state(self):
        self.last_click_to_highlight = None
        if self.positions:
            last_recorded_frame = self.positions[-1]['frame']
            if self.num <= last_recorded_frame:
                for pos_data in reversed(self.positions):
                    if pos_data['frame'] <= self.num:
                        self.last_click_to_highlight = pos_data['current']
                        break

    def change_image(self, new_num):
        if not self.frames_ready:
            return
        if 0 <= new_num < len(self.files):
            self.num = new_num
            self.image = Image.open(self.files[self.num])
            self._update_highlight_state()
            self.show_image()

    def left_key(self, event): self.change_image(self.num - 1)
    def right_key(self, event): self.change_image(self.num + 1)
    def page_up(self, event): self.change_image(self.num - 10)
    def page_down(self, event): self.change_image(self.num + 10)

    def move_from(self, event):
        self.dragged_point_index = None
        for i, pos_data in reversed(list(enumerate(self.positions))):
            px, py = pos_data['current']
            point_widget_x = (px - self.x - self.offsetx) * self.imscale
            point_widget_y = (py - self.y - self.offsety) * self.imscale
            distance = math.sqrt((event.x - point_widget_x)**2 + (event.y - point_widget_y)**2)
            if distance <= 5: self.dragged_point_index = i; return
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        if self.dragged_point_index is not None:
            new_img_x = (event.x / self.imscale) + self.x + self.offsetx
            new_img_y = (event.y / self.imscale) + self.y + self.offsety
            self.positions[self.dragged_point_index]['current'] = (new_img_x, new_img_y)
        else: self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()

    def drag_release(self, event):
        if self.dragged_point_index is not None:
            self._save_state_for_undo()
            dragged_point_data = self.positions[self.dragged_point_index]
            frame_num = dragged_point_data['frame']
            final_coords = dragged_point_data['current']
            self._create_or_update_centroid(frame_num, final_coords)
            self.update_curve_fit()
            self.update_prediction()
            self.dragged_point_index = None
            self.show_image()

    def wheel(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)
        if not (bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]): return
        scale = self.delta if (event.num == 4 or event.delta > 0) else 1/self.delta
        if self.imscale * scale > max(self.width/10, self.height/10) or self.imscale * scale < 0.1 : return
        self.imscale *= scale
        self.canvas.scale('all', x, y, scale, scale)
        self.show_image()

    def converted_image(self, img_a, contrast, brightness, color, sharpness):
        if contrast != 1: img_a = ImageEnhance.Contrast(img_a).enhance(contrast)
        if brightness != 1: img_a = ImageEnhance.Brightness(img_a).enhance(brightness)
        if color != 1: img_a = ImageEnhance.Color(img_a).enhance(color)
        if sharpness != 1: img_a = ImageEnhance.Sharpness(img_a).enhance(sharpness)
        return img_a

    def show_image(self, event=None):
        if not hasattr(self, 'container'):
            return
        try:
            if not self.canvas.winfo_exists():
                return
        except Exception:
            return
        try:
            bbox1 = self.canvas.bbox(self.container)
        except tk.TclError:
            return
        if not bbox1:
            try:
                self.canvas.coords(self.container, 0, 0, self.width * self.imscale, self.height * self.imscale)
            except Exception:
                pass
            try:
                bbox1 = self.canvas.bbox(self.container)
            except tk.TclError:
                return
        if not bbox1:
            bbox1 = (0, 0, self.width * self.imscale, self.height * self.imscale)
        bbox2 = (self.canvas.canvasx(0), self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()), self.canvas.canvasy(self.canvas.winfo_height()))
        scroll_bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        self.canvas.configure(scrollregion=scroll_bbox)
        x1 = max(bbox2[0] - bbox1[0], 0); y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]; y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        self.offsetx = (bbox2[0] - bbox1[0]) / self.imscale if bbox2[0] < bbox1[0] else 0
        self.offsety = (bbox2[1] - bbox1[1]) / self.imscale if bbox2[1] < bbox1[1] else 0
        if int(x2 - x1) <= 0 or int(y2 - y1) <= 0: return

        pos.date = datetime.fromtimestamp(self.timestamps[self.num], timezone.utc)
        point_info_text = ""
        for p in self.positions:
            if p['frame'] == self.num:
                px, py = p['current']
                brightness, altitude, azimuth = 0.0, 0.0, 0.0
                if self.centroid and self.num < len(self.centroid) and self.centroid[self.num]:
                    try:
                        parts = self.centroid[self.num].split(' ')
                        altitude = float(parts[2]); azimuth = float(parts[3]); brightness = float(parts[4])
                    except (IndexError, ValueError): pass
                point_info_text = f"\n  marked pos = {azimuth:.2f}, {altitude:.2f}, brightness = {brightness:.2f}"
                break
        
        all_stars = brightstar(self.pto_data, pos, 6.5, -30, int(self.star_objects), map_to_source_image=True, include_img_idx=True)
        stars = [s[1:] for s in all_stars if s[0] == self.image_index]

        self.x = x1 / self.imscale; self.y = y1 / self.imscale
        crop_x2 = min(x2 / self.imscale, self.width); crop_y2 = min(y2 / self.imscale, self.height)
        image = self.image.crop((self.x, self.y, crop_x2, crop_y2))

        if self.background_removal_active and self.background_image:
            bg_crop = self.background_image.crop((self.x, self.y, crop_x2, crop_y2))
            img_arr = np.array(image.convert('L')).astype(np.float32)
            bg_arr = np.array(bg_crop.convert('L')).astype(np.float32)
            subtracted_arr = np.clip(img_arr - bg_arr, 0, 255)
            max_val = subtracted_arr.max()
            if max_val > 0: subtracted_arr = (subtracted_arr / max_val) * 255.0
            image = Image.fromarray(subtracted_arr.astype(np.uint8)).convert('RGB')

        image = self.converted_image(image, self.contrast, self.brightness, self.color, self.sharpness)
        image = image.resize((int(x2 - x1), int(y2 - y1)))

        if self.curve_coeffs is not None and len(self.positions) >= 3:
            p = np.poly1d(self.curve_coeffs)
            p_first = self.positions[0]['current']; p_last = self.positions[-1]['current']
            if self.curve_orientation == 'y_is_f_of_x':
                x_coords = [p['current'][0] for p in self.positions]
                min_x, max_x = min(x_coords), max(x_coords)
                span = max_x - min_x
                extension = span * 0.2 if span > 0 else 50
                if p_last[0] >= p_first[0]: start_point, end_point = min_x, max_x + extension
                else: start_point, end_point = min_x - extension, max_x
                fit_x = np.linspace(start_point, end_point, 100)
                fit_y = p(fit_x)
            else:
                y_coords = [p['current'][1] for p in self.positions]
                min_y, max_y = min(y_coords), max(y_coords)
                span = max_y - min_y
                extension = span * 0.2 if span > 0 else 50
                if p_last[1] >= p_first[1]: start_point, end_point = min_y, max_y + extension
                else: start_point, end_point = min_y - extension, max_y
                fit_y = np.linspace(start_point, end_point, 100)
                fit_x = p(fit_y)

            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            pil_points = []
            for i in range(len(fit_x)):
                px = (fit_x[i] - self.x) * self.imscale
                py = (fit_y[i] - self.y) * self.imscale
                pil_points.append((px, py))
            if len(pil_points) > 1: draw.line(pil_points, fill=(0, 255, 255, 64), width=2, joint='curve')
            if image.mode != 'RGBA': image = image.convert('RGBA')
            image = Image.alpha_composite(image, overlay)
            image = image.convert('RGB')

        imagetk = ImageTk.PhotoImage(image)
        for c in self.overlay: self.canvas.delete(c)
        self.overlay.clear()
        canvas_origin_x = max(bbox2[0], bbox1[0])
        canvas_origin_y = max(bbox2[1], bbox1[1])
        imageid = self.canvas.create_image(canvas_origin_x, canvas_origin_y, anchor='nw', image=imagetk)
        self.canvas.lower(imageid)
        self.canvas.imagetk = imagetk

        for pos_data in self.positions:
            ox, oy = pos_data['original']
            canvas_ox = (ox - self.x) * self.imscale + canvas_origin_x
            canvas_oy = (oy - self.y) * self.imscale + canvas_origin_y
            size = 2
            if bbox2[0] < canvas_ox < bbox2[2] and bbox2[1] < canvas_oy < bbox2[3]:
                self.overlay.append(self.canvas.create_oval(canvas_ox - size, canvas_oy - size, canvas_ox + size, canvas_oy + size, fill="grey50", outline=""))
            px, py = pos_data['current']
            canvas_x = (px - self.x) * self.imscale + canvas_origin_x
            canvas_y = (py - self.y) * self.imscale + canvas_origin_y
            size = 5
            if bbox2[0] < canvas_x < bbox2[2] and bbox2[1] < canvas_y < bbox2[3]:
                self.overlay.append(self.canvas.create_line(canvas_x - size, canvas_y - size, canvas_x + size, canvas_y + size, fill="dark green", width=2))
                self.overlay.append(self.canvas.create_line(canvas_x - size, canvas_y + size, canvas_x + size, canvas_y - size, fill="dark green", width=2))
        
        if self.last_click_to_highlight and any(p['frame'] == self.num for p in self.positions):
            px, py = self.last_click_to_highlight
            canvas_x = (px - self.x) * self.imscale + canvas_origin_x
            canvas_y = (py - self.y) * self.imscale + canvas_origin_y
            radius = 12
            if bbox2[0] < canvas_x < bbox2[2] and bbox2[1] < canvas_y < bbox2[3]:
                self.overlay.append(self.canvas.create_oval(canvas_x - radius, canvas_y - radius, canvas_x + radius, canvas_y + radius, outline="red", width=2)) 
        if self.predicted_point:
            canvas_x = (self.predicted_point[0] - self.x) * self.imscale + canvas_origin_x
            canvas_y = (self.predicted_point[1] - self.y) * self.imscale + canvas_origin_y
            size = 10
            if bbox2[0] < canvas_x < bbox2[2] and bbox2[1] < canvas_y < bbox2[3]:
                self.overlay.append(self.canvas.create_line(canvas_x - size, canvas_y, canvas_x + size, canvas_y, fill="red", width=1))
                self.overlay.append(self.canvas.create_line(canvas_x, canvas_y - size, canvas_x, canvas_y + size, fill="red", width=1))

        if self.show_graph: self._draw_brightness_graph(bbox2)

        ts = datetime.fromtimestamp(self.timestamps[self.num], timezone.utc)
        info_text = (f"  time = {ts.strftime('%Y-%m-%d %H:%M:%S.%f UTC')} ({self.timestamps[self.num]:.2f})\n"
                     f"  pitch={self.img_data.get('p', 0):.2f}° yaw={self.img_data.get('y', 0):.2f}° roll={self.img_data.get('r', 0):.2f}° hfov={self.img_data.get('v', 0):.2f}°\n"
                     f"  radial=({self.img_data.get('a', 0):.3f}, {self.img_data.get('b', 0):.3f}, {self.img_data.get('c', 0):.3f}) radial shift=({-self.img_data.get('d', 0):.1f}, {self.img_data.get('e', 0):.1f})"
                     f"{self.mousepos}{point_info_text}\n  h = toggle help text")
        self.overlay.append(self.canvas.create_text(bbox2[0]+5, bbox2[1]+5, anchor="nw", text=info_text, fill="yellow", font=("helvetica", 10)))

        if self.show_text:
            help_text_content = ("\n" * 6 +
                "  NAVIGATION: arrows=move frame, pgup/dn=move 10 frames, mouse wheel=zoom, LMB=drag, RMB=mark & next\n" +
                "  MAIN: q=quit, h=toggle help, i=toggle star info, g=toggle brightness graph\n" +
                "  STARS: red dots=stars, i=toggle star labels, Ins/Del=more/less stars (10..500, brightest)\n" +
                "  EDITING: Ctrl+Z=undo, Ctrl+Y=redo, x=snap to line, X=undo snap, t=temp. space, T=undo temp.\n" +
                "  CALIBRATION: ?=open dialog, o=optimise orientation, O=undo optimisation, *=reset orientation\n" +
                "  FINE-TUNE: p/P=pitch, y/Y=yaw, r/R=roll, z/Z=hfov, a/A,b/B,c/C=radial, d/D,e/E=radial shift\n" +
                "  IMAGE: -/+=bg removal, 1/2=contrast, 3/4=brightness, 5/6=color, 7/8=sharpness, 0=reset\n" +
                "  SAVE/UPLOAD: l=save pto, s=save event.txt, S=save centroid.txt" + (f", u=upload to {self.upload_hostname}\n" if self.upload_hostname else "\n") +
                "  MODIFIERS: !=toggle boost (100x)")
            self.overlay.append(self.canvas.create_text(bbox2[0]+5, bbox2[1]+5, anchor="nw", text=help_text_content, fill="yellow", font=("helvetica", 10)))

        for s in stars:
            sx, sy, _, _, name, mag = s
            canvas_x = (sx - self.x) * self.imscale + canvas_origin_x
            canvas_y = (sy - self.y) * self.imscale + canvas_origin_y
            if bbox2[0] < canvas_x < bbox2[2] and bbox2[1] < canvas_y < bbox2[3]:
                self.overlay.append(self.canvas.create_oval(canvas_x-1, canvas_y-1, canvas_x+1, canvas_y+1, width=(max(0.1, 5-mag)*2 if self.show_info else 1), outline="red"))
                if self.show_info: self.overlay.append(self.canvas.create_text(canvas_x+3, canvas_y+3, text=f"{name} ({mag})", anchor="nw", fill="green", font=("helvetica", 10)))

    def _distance(self, p1, p2): return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _project_point_on_curve(self, point_to_project):
        if self.curve_coeffs is None: return point_to_project
        p_curve = np.poly1d(self.curve_coeffs)
        search_span_x = (self.width / 10) / self.imscale
        search_span_y = (self.height / 10) / self.imscale
        if self.curve_orientation == 'y_is_f_of_x':
            search_x = np.linspace(point_to_project[0] - search_span_x, point_to_project[0] + search_span_x, 500)
            search_y = p_curve(search_x)
        else:
            search_y = np.linspace(point_to_project[1] - search_span_y, point_to_project[1] + search_span_y, 500)
            search_x = p_curve(search_y)
        min_dist_sq = float('inf')
        closest_point = point_to_project
        for i in range(len(search_x)):
            dist_sq = (search_x[i] - point_to_project[0])**2 + (search_y[i] - point_to_project[1])**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; closest_point = (search_x[i], search_y[i])
        return closest_point

    def _temporally_respace_points(self):
        if self.curve_coeffs is None or len(self.positions) < 3: return
        print("Applying local temporal respacing...")
        N_POINTS = 2000
        p_curve = np.poly1d(self.curve_coeffs)
        start_coords, end_coords = self.positions[0]['current'], self.positions[-1]['current']
        if self.curve_orientation == 'y_is_f_of_x':
            domain = np.linspace(start_coords[0], end_coords[0], N_POINTS)
            curve_x, curve_y = domain, p_curve(domain)
        else:
            domain = np.linspace(start_coords[1], end_coords[1], N_POINTS)
            curve_y, curve_x = domain, p_curve(domain)
        arc_lengths = np.zeros(N_POINTS)
        for i in range(1, N_POINTS):
            dist = np.sqrt((curve_x[i] - curve_x[i-1])**2 + (curve_y[i] - curve_y[i-1])**2)
            arc_lengths[i] = arc_lengths[i-1] + dist
        original_arc_lengths = []
        for p in self.positions:
            dists = np.sqrt((curve_x - p['current'][0])**2 + (curve_y - p['current'][1])**2)
            closest_idx = np.argmin(dists)
            original_arc_lengths.append(arc_lengths[closest_idx])
        new_positions = [p['current'] for p in self.positions]
        for i in range(1, len(self.positions) - 1):
            t_prev = self.timestamps[self.positions[i-1]['frame']]
            t_curr = self.timestamps[self.positions[i]['frame']]
            t_next = self.timestamps[self.positions[i+1]['frame']]
            arc_prev, arc_next = original_arc_lengths[i-1], original_arc_lengths[i+1]
            local_duration = t_next - t_prev
            if local_duration <= 0: continue
            time_fraction = (t_curr - t_prev) / local_duration
            target_arc_length = arc_prev + time_fraction * (arc_next - arc_prev)
            new_x = np.interp(target_arc_length, arc_lengths, curve_x)
            new_y = np.interp(target_arc_length, arc_lengths, curve_y)
            new_positions[i] = (new_x, new_y)
        print("Recalculating brightness for all points...")
        for i, p in enumerate(self.positions):
            p['current'] = new_positions[i]
            self._create_or_update_centroid(p['frame'], p['current'])
        self.update_prediction()

    def update_prediction(self):
        if len(self.positions) < 2: self.predicted_point = None; return
        p_n_minus_1 = self.positions[-2]['current']
        p_n = self.positions[-1]['current']
        direction_vector = (p_n[0] - p_n_minus_1[0], p_n[1] - p_n_minus_1[1])
        norm = self._distance((0,0), direction_vector)
        if norm == 0: self.predicted_point = None; return
        normalized_vector = (direction_vector[0] / norm, direction_vector[1] / norm)
        distances = [self._distance(self.positions[i]['current'], self.positions[i+1]['current']) for i in range(len(self.positions) - 1)]
        dist_next = 0
        if len(distances) < 2: dist_next = distances[-1] if distances else 10
        else:
            indices = np.arange(1, len(distances) + 1)
            coeffs = np.polyfit(indices, distances, 1)
            p = np.poly1d(coeffs)
            dist_next = p(len(distances) + 1)
        if dist_next < 0: dist_next = distances[-1] if distances else 10
        p_pred_linear = (p_n[0] + normalized_vector[0] * dist_next, p_n[1] + normalized_vector[1] * dist_next)
        self.predicted_point = self._project_point_on_curve(p_pred_linear)

    def update_curve_fit(self):
        if len(self.positions) < 3: self.curve_coeffs = None; return
        x_coords = np.array([p['current'][0] for p in self.positions])
        y_coords = np.array([p['current'][1] for p in self.positions])
        if np.ptp(x_coords) > np.ptp(y_coords):
            self.curve_coeffs = np.polyfit(x_coords, y_coords, 2)
            self.curve_orientation = 'y_is_f_of_x'
        else:
            self.curve_coeffs = np.polyfit(y_coords, x_coords, 2)
            self.curve_orientation = 'x_is_f_of_y'
            
    def click(self, event):
        if not self.frames_ready:
            return
        self._save_state_for_undo()
        existing_frames = {p['frame'] for p in self.positions}
        if self.num in existing_frames:
            if self.positions:
                last_click_frame = self.positions[-1]['frame']
                if self.num < last_click_frame:
                    truncate_index = -1
                    for i, pos_data in enumerate(self.positions):
                        if pos_data['frame'] >= self.num: truncate_index = i; break
                    if truncate_index != -1:
                        frames_to_clear = {p['frame'] for p in self.positions[truncate_index:]}
                        for frame_idx in frames_to_clear:
                            if frame_idx < len(self.centroid): self.centroid[frame_idx] = None
                        self.positions = self.positions[:truncate_index]
        x, y = event.x / self.imscale + self.x, event.y / self.imscale + self.y
        x += self.offsetx; y += self.offsety
        new_point = {'frame': self.num, 'original': (x,y), 'current': (x,y)}
        insert_index = len(self.positions)
        for i, pos_data in enumerate(self.positions):
            if pos_data['frame'] > self.num: insert_index = i; break
        self.positions.insert(insert_index, new_point)
        if len(self.positions) >= 2: self.update_prediction()
        else: self.predicted_point = None
        if len(self.positions) >= 3: self.update_curve_fit()
        else: self.curve_coeffs = None
        self._update_highlight_state()
        self._create_or_update_centroid(self.num, (x, y))
        self.right_key(event)

def _recursive_extract(timestamps, files, start_idx, end_idx, report_progress_callback, cancel_event=None):
    if cancel_event is not None and cancel_event.is_set():
        return
    if start_idx > end_idx: return
    if start_idx == end_idx:
        if cancel_event is not None and cancel_event.is_set():
            return
        if timestamps[start_idx] is None:
            timestamp(timestamps, files, start_idx)
            if timestamps[start_idx] is not None: report_progress_callback()
        return

    if cancel_event is not None and cancel_event.is_set():
        return
    if timestamps[start_idx] is None:
        timestamp(timestamps, files, start_idx)
        if timestamps[start_idx] is not None: report_progress_callback()
    if timestamps[end_idx] is None:
        timestamp(timestamps, files, end_idx)
        if timestamps[end_idx] is not None: report_progress_callback()
    if timestamps[start_idx] is not None and timestamps[start_idx] == timestamps[end_idx]:
        frames_filled = 0
        for i in range(start_idx + 1, end_idx):
            if cancel_event is not None and cancel_event.is_set():
                return
            if timestamps[i] is None:
                timestamps[i] = timestamps[start_idx]
                frames_filled += 1
        if frames_filled > 0: report_progress_callback(num_frames=frames_filled)
        return
    if end_idx <= start_idx + 1: return
    mid_idx = (start_idx + end_idx) // 2
    _recursive_extract(timestamps, files, start_idx, mid_idx, report_progress_callback, cancel_event=cancel_event)
    _recursive_extract(timestamps, files, mid_idx + 1, end_idx, report_progress_callback, cancel_event=cancel_event)

def extract_timestamps(files, cancel_event=None):
    total_files = len(files)
    if not files: return []
    raw_timestamps = [None] * total_files
    print("Extracting timestamps...")
    progress_state = {'processed_count': 0}
    bar_length = 40
    def _update_progress_bar():
        count = progress_state['processed_count']
        count = min(count, total_files)
        progress = count / total_files
        block = int(round(bar_length * progress))
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {count}/{total_files} ({progress*100:.0f}%)"
        sys.stdout.write(text)
        sys.stdout.flush()
    def _report_progress(num_frames=1):
        progress_state['processed_count'] += num_frames
        _update_progress_bar()
    _recursive_extract(raw_timestamps, files, 0, total_files - 1, _report_progress, cancel_event=cancel_event)
    for i in range(total_files):
        if cancel_event is not None and cancel_event.is_set():
            break
        if raw_timestamps[i] is None:
            timestamp(raw_timestamps, files, i)
            if raw_timestamps[i] is not None: _report_progress()
    sys.stdout.write(f"\rProgress: [{'#' * bar_length}] {total_files}/{total_files} (100%)\n\n")
    return raw_timestamps

def scp_file(hostname, remote_path):
    try:
        temp_f = tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=os.path.basename(remote_path))
        local_path = temp_f.name
        temp_f.close() 
        remote_full_path = f"{hostname}:{remote_path}"
        print(f"Copying {remote_full_path} to {local_path}...")
        proc = subprocess.run(['scp', remote_full_path, local_path], check=True, capture_output=True, text=True, errors='ignore')
        print(f"Copy successful: {remote_full_path}")
        return local_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error copying file: {e}", file=sys.stderr)
        if 'local_path' in locals() and os.path.exists(local_path): os.remove(local_path)
        return None


def run_upload_worker(args):
    upload_root = tk.Tk()
    upload_root.title("Upload Progress")
    upload_root.geometry("700x450")
    text_widget = tk.Text(upload_root, wrap=tk.WORD, height=25, width=90, bg="black", fg="gray90", font=("monospace", 9))
    text_widget.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
    text_widget.tag_config("red", foreground="#ff5555")
    close_button = tk.Button(upload_root, text="Close", command=upload_root.destroy, state=tk.DISABLED)
    close_button.pack(pady=5)

    def log_to_window(message, tag=None):
        msg = str(message) + "\n"

        def _append():
            try:
                if tag:
                    text_widget.insert(tk.END, msg, tag)
                else:
                    text_widget.insert(tk.END, msg)
                text_widget.see(tk.END)
            except Exception:
                pass

        try:
            upload_root.after(0, _append)
        except Exception:
            pass

    def _upload_task():
        try:
            ts = float(args.upload_timestamp)
            dt = datetime.fromtimestamp(ts, timezone.utc)
            date_dir = dt.strftime("%Y%m%d")
            time_dir = dt.strftime("%H%M%S")

            remote_subdir = f"{args.upload_dir.rstrip('/')}/events/{date_dir}/{time_dir}"
            log_to_window(f"Target directory: {args.upload_hostname}:{remote_subdir}")
            log_to_window(f"Running: ssh {args.upload_hostname} mkdir -p {remote_subdir}")
            proc = subprocess.run(['ssh', args.upload_hostname, 'mkdir', '-p', remote_subdir], capture_output=True, text=True, check=True, errors='ignore')
            if proc.stderr:
                log_to_window(f"STDERR:\n{proc.stderr.strip()}")

            files_to_upload = {
                'event.txt': args.upload_event_file,
                'centroid.txt': args.upload_centroid_file,
            }
            if args.upload_lens_file:
                files_to_upload['lens.pto'] = args.upload_lens_file

            log_to_window("Copying files via scp...")
            for dest_name, local_path in files_to_upload.items():
                remote_destination = f"{args.upload_hostname}:{remote_subdir}/{dest_name}"
                log_to_window(f"  - Copying {local_path} to {remote_destination}...")
                proc = subprocess.run(['scp', local_path, remote_destination], capture_output=True, text=True, check=True, errors='ignore')
                if proc.stderr:
                    log_to_window(f"    - SCP STDERR for {dest_name}: {proc.stderr.strip()}")

            log_to_window("  - SCP complete.")

            if args.upload_lens_file:
                match = re.match(r'/meteor/cam(\d+)', args.upload_dir)
                if match:
                    cam_num = match.group(1)
                    log_to_window(f"Deploying new calibration (lens.pto) to cam{cam_num} on {args.upload_hostname}...")
                    local_pto_path = args.upload_lens_file
                    remote_pto_dated = f"/meteor/cam{cam_num}/lens-{date_dir}.pto"
                    remote_grid_dated = f"/meteor/cam{cam_num}/grid-{date_dir}.png"
                    remote_pto_link = f"/meteor/cam{cam_num}/lens.pto"
                    remote_grid_link = f"/meteor/cam{cam_num}/grid.png"
                    drawgrid_script = "/home/meteor/bin/drawgrid.py"
                    remote_full_path = f"{args.upload_hostname}:{remote_pto_dated}"
                    log_to_window(f"  - Copying {local_pto_path} to {remote_full_path}...")
                    subprocess.run(['scp', local_pto_path, remote_full_path], check=True, capture_output=True, text=True, errors='ignore')
                    remote_command = (f"{drawgrid_script} {remote_pto_dated} {remote_grid_dated} && "
                                      f"rm -f {remote_pto_link} {remote_grid_link} && "
                                      f"ln -s {remote_pto_dated} {remote_pto_link} && "
                                      f"ln -s {remote_grid_dated} {remote_grid_link}")
                    log_to_window("  - Running remote commands to update grid and links...")
                    subprocess.run(['ssh', args.upload_hostname, remote_command], check=True, capture_output=True, text=True, errors='ignore')
                    log_to_window("  - Remote deployment successful.")

            log_to_window("Running remote report script...")
            report_script = "/home/meteor/bin/report.py"
            remote_event_file = f"{remote_subdir}/event.txt"
            log_to_window(f"Running: ssh {args.upload_hostname} {report_script} {remote_event_file}")
            proc = subprocess.run(['ssh', args.upload_hostname, report_script, remote_event_file], capture_output=True, text=True, check=False, errors='ignore')
            log_to_window("\n--- Report Output ---")
            if proc.stdout:
                log_to_window(proc.stdout.strip())
            if proc.stderr:
                log_to_window(f"STDERR:\n{proc.stderr.strip()}")
            log_to_window("--- End of Report ---")
            log_to_window("Upload finished", "red")
        except subprocess.CalledProcessError as e:
            log_to_window(f"\n--- COMMAND FAILED ---")
            log_to_window(f"Command: {' '.join(e.cmd)}")
            log_to_window(f"Return Code: {e.returncode}")
            if e.stdout:
                log_to_window(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr:
                log_to_window(f"STDERR:\n{e.stderr.strip()}")
        except Exception as e:
            log_to_window(f"\n--- UNEXPECTED ERROR ---")
            log_to_window(str(e))
        finally:
            log_to_window("Cleaning up temporary local files...")
            for p in [args.upload_event_file, args.upload_centroid_file, args.upload_lens_file]:
                if p:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            def _finish_ui():
                try:
                    close_button.config(state=tk.NORMAL)
                except Exception:
                    pass
                try:
                    upload_root.after(10000, upload_root.destroy)
                except Exception:
                    pass

            try:
                upload_root.after(0, _finish_ui)
            except Exception:
                pass

    threading.Thread(target=_upload_task, daemon=False).start()
    upload_root.mainloop()

if __name__ == '__main__':
    atexit.register(cleanup_temp_resources)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if '--upload-worker' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--upload-worker', action='store_true')
        parser.add_argument('--upload-hostname', required=True)
        parser.add_argument('--upload-dir', required=True)
        parser.add_argument('--upload-timestamp', required=True)
        parser.add_argument('--upload-event-file', required=True)
        parser.add_argument('--upload-centroid-file', required=True)
        parser.add_argument('--upload-lens-file', default='')
        wargs = parser.parse_args()
        run_upload_worker(wargs)
        sys.exit(0)

    while True:
        args = None
        calibrate_info = {
            'is_calibrate_mode': False, 'hostname': None, 'cam_num': None,
            'date_str': None, 'upload_hostname': None, 'upload_dir': None,
            'selected_datetime': None 
        }

        if len(sys.argv) == 1:
            print("No arguments detected, launching GUI selector...")
            root = tk.Tk()
            root.withdraw() 
            dialog_toplevel = tk.Toplevel(root)
            dialog = LauncherDialog(dialog_toplevel, root)
            root.mainloop() 
            if not hasattr(dialog, 'fetched_files'):
                print("Launcher cancelled. Exiting."); sys.exit(0)
            
            parser = argparse.ArgumentParser() 
            args = parser.parse_args([]) 
            args.station = dialog.fetched_files['config']
            args.ptofile = dialog.fetched_files['pto']
            args.imgfiles = dialog.fetched_files['video'] 
            args.upload_target = dialog.upload_target
            args.image = 0
            args.fps = 10
            args.name = "NUL" 
            args.start = None
            args.skip = 0.0
            args.total = None
            args.calibrate = None
            args.event_file = None
            args.latitude = None; args.longitude = None; args.elevation = None
            if 'selected_datetime' in dialog.fetched_files:
                calibrate_info['selected_datetime'] = dialog.fetched_files['selected_datetime']
            print("Launcher success. Proceeding with fetched files.")
        else:
            parser = argparse.ArgumentParser(description='Click on images to find coordinates.')
            parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
            parser.add_argument('-f', '--fps', dest='fps', help='frames per second (default: 10 or extracted from images)', default=10, type=float)
            parser.add_argument('-n', '--name', dest='name', help='station name (default: NUL or extracted from station config)', default="NUL", type=str)
            parser.add_argument('-c', '--config', dest='station', help='station config file', type=str)
            parser.add_argument('-d', '--date', dest='start', help='start time (default: extracted from images))', type=str)
            parser.add_argument('-s', dest='skip', help='Seconds of the initial video to skip (default: 0)', default=0.0, type=float)
            parser.add_argument('-t', dest='total', help='Total seconds of video to load after skipping.', type=float, default=None)
            parser.add_argument('--calibrate', dest='calibrate', help='Fetch files from a remote host.', type=str)
            parser.add_argument('--load-event', dest='event_file', help='load a previously saved event.txt file', type=str)
            parser.add_argument('--upload-target', dest='upload_target', help='Remote upload destination', type=str, default=None)
            parser.add_argument('--latitude', type=float, help='Observer latitude')
            parser.add_argument('--longitude', type=float, help='Observer longitude')
            parser.add_argument('--elevation', type=float, help='Observer elevation')
            parser.add_argument('ptofile', nargs='?', default=None, help='input .pto')
            parser.add_argument('imgfiles', nargs='*', help='input image or video files')
            args = parser.parse_args()

            if args.upload_target is None and args.imgfiles and ':' in args.imgfiles[-1]:
                if not os.path.exists(args.imgfiles[-1]):
                    args.upload_target = args.imgfiles.pop(-1)
                    print(f"Note: Last argument '{args.upload_target}' interpreted as upload target.")

        if args.upload_target:
            try:
                calibrate_info['upload_hostname'], calibrate_info['upload_dir'] = args.upload_target.split(':', 1)
            except ValueError: parser.error("Invalid --upload-target format.")
            if re.search(r'[;&|`$(){}<> \'\"]', calibrate_info['upload_hostname']): parser.error("Invalid hostname.")
            if not calibrate_info['upload_dir'].startswith('/') or re.search(r'[;&|`$(){}<> \'\"]', calibrate_info['upload_dir']): parser.error("Invalid directory.")

        if not args.calibrate and (not args.ptofile or not args.imgfiles) and len(sys.argv) > 1:
            parser.error("ptofile and imgfiles are required when not using --calibrate.")
        
        if args.calibrate:
            parts = args.calibrate.split(':')
            if len(parts) not in [2, 3]: sys.exit(f"Error: Invalid format for --calibrate.")
            hostname = parts[0]; cam_num = parts[1]
            if len(parts) == 3:
                date_time_str = parts[2]
                match = re.match(r'(\d{8})/(\d{2})(\d{2})', date_time_str)
                if not match: sys.exit(f"Error: Invalid date/time format.")
                date_str, hour_str, minute_str = match.groups()
            else:
                now = datetime.now()
                date_str = now.strftime('%Y%m%d'); hour_str = "00"; minute_str = "00"
                print(f"No date provided for --calibrate, defaulting to {date_str}/{hour_str}{minute_str}")
            
            calibrate_info.update({'is_calibrate_mode': True, 'hostname': hostname, 'cam_num': cam_num, 'date_str': date_str})
            if 'selected_datetime' not in calibrate_info or calibrate_info['selected_datetime'] is None:
                 try: calibrate_info['selected_datetime'] = datetime.strptime(f"{date_str}{hour_str}{minute_str}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                 except: pass

            if args.total is None: args.total = 1.0
            remote_cfg_path = "/etc/meteor.cfg"
            remote_pto_path = f"/meteor/cam{cam_num}/lens.pto"
            remote_video_path = f"/meteor/cam{cam_num}/{date_str}/{hour_str}/full_{minute_str}.mp4"
            print("Copying remote files in parallel...")
            files_to_copy = {'station': remote_cfg_path, 'pto': remote_pto_path, 'video': remote_video_path}
            results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_key = {executor.submit(scp_file, hostname, path): key for key, path in files_to_copy.items()}
                for future in future_to_key:
                    key = future_to_key[future]
                    try:
                        local_path = future.result()
                        if not local_path: raise ValueError("scp_file returned None")
                        results[key] = local_path
                        temp_files_to_clean.append(local_path)
                    except Exception as e:
                        print(f"Fatal error during parallel file download: {e}", file=sys.stderr)
                        sys.exit("Aborting due to scp failure.")
            args.station = results['station']
            args.ptofile = results['pto']
            args.imgfiles = [results['video']]

        pos = ephem.Observer()
        if all(arg is not None for arg in [args.latitude, args.longitude, args.elevation]):
            pos.lat, pos.lon, pos.elevation = str(args.latitude), str(args.longitude), args.elevation
            pos.temp, pos.pressure = 15.0, 1010.0 
        elif args.station:
            print(f"Using location from station file: {args.station}")
            try:
                config = configparser.ConfigParser()
                config.read(args.station)
                pos.lat = config.get('astronomy', 'latitude')
                pos.lon = config.get('astronomy', 'longitude')
                pos.elevation = float(config.get('astronomy', 'elevation'))
                pos.temp = float(config.get('astronomy', 'temperature', fallback=15))
                pos.pressure = float(config.get('astronomy', 'pressure', fallback=1010))
                args.name = config.get('station', 'code', fallback=args.name)
            except (configparser.Error, FileNotFoundError) as e: sys.exit(f"Error reading station file: {e}")
        else: sys.exit("Error: Observer location not specified.")

        if args.ptofile.lower().endswith('.json'):
            try:
                with open(args.ptofile) as f: calib_data = json.load(f)
                cal_params_data = calib_data.get('cal_params', calib_data)
                width = cal_params_data.get('imagew', 1920); height = cal_params_data.get('imageh', 1080)
                pto_content = amscalib2lens.generate_pto_from_json(cal_params_data, pos, width, height, match_dist_limit=0.2)
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pto", dir="/tmp") as temp_pto_file:
                    temp_pto_path = temp_pto_file.name
                    temp_files_to_clean.append(temp_pto_path)
                    temp_pto_file.write(pto_content)
                    temp_pto_file.flush() 
                    subprocess.run(['autooptimiser', '-n', temp_pto_path, '-o', temp_pto_path], capture_output=True, text=True)
                args.ptofile = temp_pto_path
            except Exception as e: sys.exit(f"Error during JSON to PTO conversion: {e}")

        is_video_mode = bool(args.imgfiles and args.imgfiles[0].lower().endswith('.mp4'))

        # Determine FPS early for consistent timestamp interpolation.
        fps = args.fps
        if is_video_mode:
            try:
                probe = ffmpeg.probe(args.imgfiles[0])
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream:
                    fps_str = video_stream.get('r_frame_rate')
                    if fps_str:
                        num, den = map(int, fps_str.split('/'))
                        if den > 0:
                            fps = num / den
                            print(f"Extracted FPS from video: {fps:.2f}")
            except Exception as e:
                print(f"Could not extract FPS from metadata, assuming 25. Error: {e}")
                fps = 25.0
        elif fps == 10:
            fps = 25.0
        args.fps = fps

        # --- Quick-start: decode and timestamp only the first frame ---
        images = []
        timestamps = []
        frames_ready = True

        if is_video_mode:
            frames_ready = False
            temp_dir_first = tempfile.TemporaryDirectory(prefix="clickcoords_first_", dir="/tmp")
            temp_dirs.append(temp_dir_first)
            try:
                read_first_frame(args.imgfiles[0], temp_dir_first.name, skip_seconds=args.skip)
                decoded_first = sorted(glob.glob(temp_dir_first.name + "/*.tif"))
                if not decoded_first:
                    raise RuntimeError("No first frame decoded")
                images = [decoded_first[0]]
            except Exception as e:
                sys.exit(f"Failed to decode first frame: {e}")

            # Timestamp just the first frame (fast).
            ts0 = None
            try:
                dt0 = get_timestamp(images[0])
                if not dt0:
                    dt0 = get_timestamp(images[0], robust=True)
                if dt0:
                    ts0 = dt0.timestamp()
            except Exception:
                ts0 = None
            if ts0 is None:
                ts0 = datetime.now(timezone.utc).timestamp()
            if args.start is not None:
                parsed = parse_start_time_arg(args.start)
                if parsed is not None:
                    ts0 = parsed
            timestamps = [ts0]
        else:
            # Image mode: keep existing behavior (already fast enough).
            for f in args.imgfiles:
                images.append(os.path.abspath(f))
            if not images:
                sys.exit("No image files found.")
            starttime = None
            if args.start is not None:
                starttime = parse_start_time_arg(args.start)
            if starttime is None:
                starttime = datetime.now(timezone.utc).timestamp()
            timestamps = [starttime + (i / args.fps) for i in range(len(images))]

        try:
            pto_data = pto_mapper.parse_pto_file(args.ptofile)
        except Exception as e: sys.exit(f"Error parsing PTO file '{args.ptofile}': {e}")

        global_options, images_data = pto_data
        if args.image >= len(images_data): sys.exit(f"Error: Image index out of range.")
        
        img_data = images_data[args.image]
        first_pil = Image.open(images[0])
        img_data['w'], img_data['h'] = first_pil.size

        window = tk.Tk()
        window.geometry("1600x960")
        app = Zoom_Advanced(window, files=images, timestamps=timestamps, pto_data=pto_data, image_index=args.image, frames_ready=frames_ready, **calibrate_info)

        def _on_window_close():
            try:
                app.cancel_background_load(wait=False)
            except Exception:
                pass
            window.destroy()

        try:
            window.protocol("WM_DELETE_WINDOW", _on_window_close)
        except Exception:
            pass

        if is_video_mode:
            def _load_full_sequence():
                full_images = []
                seconds_to_skip = args.skip
                seconds_to_load = args.total
                loaded_duration = 0.0

                for f in args.imgfiles:
                    if app.load_cancel_event.is_set():
                        return
                    if f.lower().endswith('.mp4'):
                        if seconds_to_load is not None and loaded_duration >= seconds_to_load:
                            break
                        try:
                            probe = ffmpeg.probe(f)
                            video_duration = float(next(s['duration'] for s in probe['streams'] if s['codec_type'] == 'video'))
                        except Exception:
                            continue

                        if seconds_to_skip > 0:
                            if seconds_to_skip >= video_duration:
                                seconds_to_skip -= video_duration
                                continue
                            else:
                                current_file_skip = seconds_to_skip
                                seconds_to_skip = 0
                        else:
                            current_file_skip = 0

                        current_file_load_duration = None
                        if seconds_to_load is not None:
                            remaining_to_load = seconds_to_load - loaded_duration
                            available_in_video = video_duration - current_file_skip
                            current_file_load_duration = min(remaining_to_load, available_in_video)
                            if current_file_load_duration <= 0:
                                break

                        temp_dir = tempfile.TemporaryDirectory(prefix="clickcoords_", dir="/tmp")
                        temp_dirs.append(temp_dir)
                        read_frames(f, temp_dir.name, skip_seconds=current_file_skip, total_seconds=current_file_load_duration)
                        decoded_files = sorted(glob.glob(temp_dir.name + "/*.tif"))
                        full_images.extend(decoded_files)
                        loaded_duration += len(decoded_files) / args.fps
                    else:
                        full_images.append(os.path.abspath(f))

                if not full_images:
                    raise RuntimeError("No frames decoded")

                if app.load_cancel_event.is_set():
                    return

                # Full timestamp extraction (slow)
                starttime_local = None
                raw_timestamps = None
                if args.start is None:
                    raw_timestamps = extract_timestamps(full_images, cancel_event=app.load_cancel_event)
                    if app.load_cancel_event.is_set():
                        return
                    valid_timestamps = [t for t in raw_timestamps if t is not None]
                    valid_count = len(valid_timestamps)
                    is_valid = False
                    if valid_count > len(raw_timestamps) * 0.5:
                        unique_vals = set(valid_timestamps)
                        sorted_valid = sorted(valid_timestamps)
                        total_duration = sorted_valid[-1] - sorted_valid[0]
                        if len(valid_timestamps) > 1 and (total_duration <= 0.001 or len(unique_vals) < 2):
                            is_valid = False
                        else:
                            is_valid = True
                            interp_test = interpolate_timestamps(raw_timestamps)
                            expected_duration = len(full_images) / args.fps
                            duration_ocr = interp_test[-1] - interp_test[0]
                            if duration_ocr < expected_duration * 0.1 or duration_ocr > expected_duration * 5.0:
                                is_valid = False

                    if is_valid:
                        full_timestamps = interpolate_timestamps(raw_timestamps)
                        starttime_local = full_timestamps[0]
                    else:
                        full_timestamps = [None] * len(full_images)
                else:
                    starttime_local = parse_start_time_arg(args.start)
                    full_timestamps = [None] * len(full_images)

                if starttime_local is None:
                    first_vid = args.imgfiles[0] if args.imgfiles else ""
                    match = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', os.path.basename(first_vid))
                    if match:
                        y, m, d, H, M, S = map(int, match.groups())
                        try:
                            starttime_local = datetime(y, m, d, H, M, S, tzinfo=timezone.utc).timestamp()
                        except ValueError:
                            starttime_local = None

                if starttime_local is None and calibrate_info.get('selected_datetime'):
                    starttime_local = calibrate_info['selected_datetime'].timestamp()
                if starttime_local is None:
                    starttime_local = datetime.now(timezone.utc).timestamp()

                if not raw_timestamps or full_timestamps[0] is None:
                    full_timestamps = [starttime_local + (i / args.fps) for i in range(len(full_images))]

                if app.load_cancel_event.is_set():
                    return

                def _apply_loaded_sequence():
                    if app.load_cancel_event.is_set():
                        return
                    try:
                        app.set_sequence(full_images, full_timestamps)
                    except Exception:
                        pass

                try:
                    window.after(0, _apply_loaded_sequence)
                except Exception:
                    pass

            app.load_thread = threading.Thread(target=_load_full_sequence, daemon=True)
            app.load_thread.start()

        if args.event_file: app.load_event_file(args.event_file)
        window.mainloop()

        if hasattr(app, 'restart') and app.restart:
            print("\n--- Restarting to select new event ---\n")
            cleanup_temp_resources()
            temp_files_to_clean.clear()
            temp_dirs.clear()
            sys.argv = [sys.argv[0]]
            continue 
        else: break
