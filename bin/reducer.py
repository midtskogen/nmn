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
import tkinter as tk
import contextlib
from collections import deque
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime, timezone
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
from brightstar import brightstar
from recalibrate import recalibrate
from timestamp import get_timestamp
import pto_mapper

class RecalibrateDialog:
    def __init__(self, parent, image, zoom_instance):
        self.parent = parent
        self.image = image # This is a PIL Image
        self.zoom = zoom_instance
        self.img_data = self.zoom.img_data
        self.pto_data = self.zoom.pto_data
        self.parent.title("Recalibrate")

        # Store original parameters for reset
        self.original_params = self.img_data.copy()

        # Initialize variables for sliders
        self.radius_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.IntVar(value=50)
        self.sigma_var = tk.IntVar(value=20)
        self.lens_optimize_var = tk.BooleanVar(value=True)

        # Create sliders
        self.create_slider("Radius", 0.1, 5.0, self.radius_var, row=0)
        self.create_slider("Blur", 1, 100, self.blur_var, row=1)
        self.create_slider("Sigma", 1, 100, self.sigma_var, row=2)

        # Create button for Lens optimization
        lens_optimize_btn = tk.Checkbutton(self.parent, text="Optimize lens parameters", variable=self.lens_optimize_var)
        lens_optimize_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky='w')

        # Create Solve button
        solve_btn = tk.Button(self.parent, text="Solve", command=self.recal)
        solve_btn.grid(row=4, column=0, pady=10, sticky='w')

        # Create Reset button
        reset_btn = tk.Button(self.parent, text="Reset", command=self.reset)
        reset_btn.grid(row=4, column=1, pady=10, sticky='e')

        # RMS Display Label
        self.rms_label = tk.Label(self.parent, text="", justify=tk.LEFT)
        self.rms_label.grid(row=5, column=0, columnspan=2, pady=5, sticky='w')

        # Create Close button
        close_btn = tk.Button(self.parent, text="Close", command=self.parent.destroy)
        close_btn.grid(row=6, column=0, columnspan=2, pady=10, sticky='w')

        # Create labels for help text
        help_label = tk.Label(self.parent, font=("Helvetica", 14, "bold"), text="Fine-tuning the calibration")
        help_label.grid(row=0, column=3, sticky='w', padx=(10, 0))

        help_text = """
        The sliders controls the following values:

        - "Radius" the search mask in degrees to look for the star.
        - "Blur" controls the amount of blurring for the mask.
        - "Sigma" indicates how much noise should be assumed.
        - "Optimise lens parameters" for full recalibration.
        - Click "Solve" to perform the calibration.
        - Click "Reset" to go back to the original parameters.
        - The "Close" button closes the dialog.
        """
        help_text_label = tk.Label(self.parent, text=help_text, justify=tk.LEFT)
        help_text_label.grid(row=1, column=3, rowspan=6, sticky='w', padx=(10, 0))

        par_text = "Original parameters:\n  pitch=%.2f°\n  yaw=%.2f°\n  roll=%.2f°\n  hfov=%.2f°\n  radial=(%.3f, %.3f, %.3f)\n  radial shift=(%.1f, %.1f)" % (
            self.original_params.get('p', 0), self.original_params.get('y', 0), self.original_params.get('r', 0),
            self.original_params.get('v', 0), self.original_params.get('a', 0), self.original_params.get('b', 0), self.original_params.get('c', 0),
            -self.original_params.get('d', 0), self.original_params.get('e', 0)
        )
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
        # Provide immediate feedback in the GUI
        self.rms_label.config(text="Solving...")
        self.parent.update_idletasks()
        
        # Create temporary files safely
        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".png") as img_f:
            self.image.save(img_f, format='PNG')
            img_temp_filename = img_f.name
        
        # Determine which variables to optimize
        img_idx = self.zoom.image_index
        vars_to_optimize = [
            f'v{img_idx}',
            f'r{img_idx}',
            f'p{img_idx}',
            f'y{img_idx}',
        ]
        if self.lens_optimize_var.get():
            vars_to_optimize.extend([
                f'a{img_idx}',
                f'b{img_idx}',
                f'c{img_idx}',
                f'd{img_idx}',
                f'e{img_idx}',
            ])

        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".pto") as pto_f:
            pto_mapper.write_pto_file(self.pto_data, pto_f.name, optimize_vars=vars_to_optimize)
            old_lens_filename = pto_f.name

        new_lens_filename = tempfile.NamedTemporaryFile(delete=True, dir="/tmp", suffix=".pto").name
        # Create a path for a temporary log file
        log_file_path = tempfile.mktemp(suffix=".log", dir="/tmp")

        try:
            # Open the log file for writing
            with open(log_file_path, 'w') as log_f:
                # Redirect stdout to the real file, which has a fileno
                with contextlib.redirect_stdout(log_f):
                    recalibrate(
                        starttime,
                        old_lens_filename,
                        img_temp_filename,
                        new_lens_filename,
                        pos,
                        image=self.zoom.image_index,
                        radius=self.radius_var.get(),
                        lensopt=self.lens_optimize_var.get(),
                        faintest=4,
                        brightest=-5,
                        objects=500,
                        blur=self.blur_var.get(),
                        verbose=True,
                        sigma=self.sigma_var.get()
                    )

            # Read the output from the log file now that the process is complete
            with open(log_file_path, 'r') as log_f:
                output = log_f.read()

            # Parse the captured output for the RMS value
            rms_values = re.findall(r"after \d+ iteration\(s\):\s*([\d.]+)\s*units", str(output))
            if rms_values:
                last_rms = float(rms_values[-1])
                self.rms_label.config(text=f"Final RMS: {last_rms:.2f}")
            else:
                self.rms_label.config(text="RMS: Not found in output")

            _, new_images_data = pto_mapper.parse_pto_file(new_lens_filename)
            new_img_data = new_images_data[self.zoom.image_index]

            params_to_update = ['p', 'y', 'r', 'v', 'a', 'b', 'c', 'd', 'e']
            for param in params_to_update:
                if param in new_img_data:
                    self.img_data[param] = new_img_data[param]

            self.zoom.show_image()
        except Exception as e:
            print(f"Recalibration failed: {e}", file=sys.stderr)
            self.rms_label.config(text=f"Error: {e}")
        finally:
            # Clean up all temporary files
            os.remove(img_temp_filename)
            os.remove(old_lens_filename)
            if os.path.exists(new_lens_filename):
                os.remove(new_lens_filename)
            if os.path.exists(log_file_path):
                os.remove(log_file_path)

def read_frames(filename, directory):
    """Decodes a video file into individual frames, showing a progress bar."""
    try:
        probe = ffmpeg.probe(filename)
        total_frames = int(next(s['nb_frames'] for s in probe['streams'] if s['codec_type'] == 'video'))
    except (ffmpeg.Error, StopIteration, KeyError, ValueError):
        total_frames = None

    print(f"Decoding video {filename}...")

    stream = ffmpeg.input(filename).output(f'{directory}/%04d.tif', format='image2', vsync=0).overwrite_output()
    args = stream.get_args()
    
    final_args = ['ffmpeg']
    if total_frames:
        final_args.extend(['-progress', 'pipe:1'])
    final_args.extend(args)

    proc = subprocess.Popen(final_args, stdout=subprocess.PIPE if total_frames else None, stderr=subprocess.PIPE, text=True, errors='ignore')

    if not total_frames:
        _, err = proc.communicate()
        if proc.returncode != 0:
            raise ffmpeg.Error('ffmpeg', None, err)
        return

    bar_length = 40
    for line in iter(proc.stdout.readline, ''):
        match = re.search(r'frame=\s*(\d+)', line)
        if match:
            current_frame = int(match.group(1))
            progress = current_frame / total_frames
            block = int(round(bar_length * progress))
            text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {current_frame}/{total_frames} ({progress*100:.0f}%)"
            sys.stdout.write(text)
            sys.stdout.flush()

    sys.stdout.write(f"\rProgress: [{'#' * bar_length}] {total_frames}/{total_frames} (100%)\n\n")

    err = proc.stderr.read()
    if proc.wait() != 0:
        raise ffmpeg.Error('ffmpeg', None, err)


def timestamp(timestamps, files, i):
    """
    Gets the timestamp for the file at index i, using a cache.
    It calls get_timestamp, first normally, then in robust mode on failure.
    """
    if i < len(timestamps) and timestamps[i] is not None:
        return timestamps[i]

    datetime_obj = get_timestamp(files[i])
    if not datetime_obj:
        datetime_obj = get_timestamp(files[i], robust=True)

    if datetime_obj:
        ts = datetime_obj.timestamp()
        timestamps[i] = ts
        return ts
    else:
        if i > 0 and timestamps[i-1] is not None:
            timestamps[i] = timestamps[i-1]
            return timestamps[i-1]
        else:
            sys.stderr.write(f"Error: Could not extract timestamp from image: {files[i]}\n")
            if i == 0:
                sys.stderr.write("Please provide a start time using the -d argument.\n")
                sys.exit(1)
            return None

def interpolate_timestamps(timestamps):
    """Performs interpolation on a pre-populated list of timestamps."""
    if not timestamps:
        return []
    
    interpolated_stamps = list(timestamps) # Make a copy

    fractions = []
    prev = interpolated_stamps[0]

    i = 1
    previ = -1
    while i < len(interpolated_stamps):
        if interpolated_stamps[i] == prev:
            while i < len(interpolated_stamps) and interpolated_stamps[i] == prev:
                i += 1
            fractions.append(i - previ)
            previ = i
            if i < len(interpolated_stamps):
                prev = interpolated_stamps[i]
        else:
            previ += 1
            while i < len(interpolated_stamps) and interpolated_stamps[i] != prev:
                i -= 1
            fractions.append(i + 2 - previ)
            previ = i + 1
            if i + 1 < len(interpolated_stamps):
                prev = interpolated_stamps[i + 1]
            else:
                break
        
        if not fractions or fractions[-1] == 0:
            i+=1
            continue
            
        i += fractions[-1]

    fractions.append(len(interpolated_stamps) - previ)
    if len(fractions) > 1:
        fractions[-1] = fractions[-2]

    prev = interpolated_stamps[0]
    f = 0.0
    if len(fractions) >= 2 and fractions[1] != 0:
        f = 1 - fractions[0] / fractions[1]

    fi = 0
    if fractions and fractions[0] != 0:
        fractions[0] = fractions[1]
    
    for i in range(1, len(interpolated_stamps)):
        if interpolated_stamps[i] is None:
            interpolated_stamps[i] = interpolated_stamps[i - 1]
    
    for i in range(len(interpolated_stamps)):
        if interpolated_stamps[i] == prev:
            if fi < len(fractions) and fractions[fi] > 0:
                f = f + 1.0 / fractions[fi]
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
    ''' Advanced zoom of the image '''
    def __init__(self, mainframe, files, timestamps, pto_data, image_index):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Click Coords')
        self.overlay = []
        self.canvas = tk.Canvas(self.master, highlightthickness=0, cursor="draft_small", bg="black")
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        
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
        self.canvas.focus_set()

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
        self.imscale = 1.0
        self.delta = 1.05
        self.contrast = 1
        self.brightness = 1
        self.color = 1
        self.sharpness = 1
        self.show_text = 0
        self.show_info = 0
        self.boost = 1
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
        
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        
        self.canvas.update()
        self.show_image()
        
    def _image_coords_to_celestial(self, x, y):
        """Converts image (x, y) coordinates to celestial (azimuth, altitude)."""
        pano_coords = pto_mapper.map_image_to_pano(self.pto_data, self.image_index, x, y)
        if pano_coords:
            pano_x, pano_y = pano_coords
            pano_w = self.global_options.get('w')
            pano_h = self.global_options.get('h')
            # Check for equirectangular projection (f=2)
            if self.global_options.get('f', 2) == 2:
                az_rad = (pano_x / pano_w) * 2 * math.pi
                alt_rad = (0.5 - pano_y / pano_h) * math.pi
                az_deg = math.degrees(az_rad)
                alt_deg = math.degrees(alt_rad)
                return az_deg, alt_deg
            else:
                # Non-equirectangular projection
                return -998, -998
        else:
            # Outside panorama
            return -999, -999

    def _create_background_image(self):
        """Pads, blurs, and crops the first frame to create a static background image."""
        if not self.files:
            return
        try:
            print("Creating background image for subtraction...")
            img = Image.open(self.files[0])
            
            # Convert the PIL image to a numpy array to use advanced padding
            img_arr = np.array(img)
            
            # Pad the array by extending the edge pixels
            pad_width = ((32, 32), (32, 32), (0, 0)) # Pad height, width, but not color channels
            padded_arr = np.pad(img_arr, pad_width, mode='edge')

            # Convert the padded array back to a PIL image
            padded_img = Image.fromarray(padded_arr)
            
            # Blur the correctly padded image
            blurred_padded = padded_img.filter(ImageFilter.GaussianBlur(radius=32))
            
            # Crop the padding off to get a blurred image of the original size
            self.background_image = blurred_padded.crop((32, 32, 32 + img.width, 32 + img.height))
            print("Background image created.")
        except Exception as e:
            print(f"Failed to create background image: {e}", file=sys.stderr)
            self.background_image = None

    def _save_state_for_undo(self):
        """Saves a deep copy of the current points and centroids to the undo stack."""
        # A new action clears the redo stack.
        self.redo_stack.clear()

        state = {
            'positions': copy.deepcopy(self.positions),
            'centroid': copy.deepcopy(self.centroid)
        }
        self.undo_stack.append(state)

        # To prevent high memory usage, limit the undo history.
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self, event=None):
        """Reverts to the previous state from the undo stack."""
        if not self.undo_stack:
            return

        # Save the current state to the redo stack before reverting.
        current_state = {
            'positions': copy.deepcopy(self.positions),
            'centroid': copy.deepcopy(self.centroid)
        }
        self.redo_stack.append(current_state)

        # Pop the last state and restore it.
        last_state = self.undo_stack.pop()
        self.positions = last_state['positions']
        self.centroid = last_state['centroid']
        
        # Refresh the display to show the reverted state.
        self.update_curve_fit()
        self.update_prediction()
        self._update_highlight_state()
        self.show_image()

    def redo(self, event=None):
        """Re-applies a state from the redo stack."""
        if not self.redo_stack:
            return

        # Save the current state for a potential future undo.
        current_state = {
            'positions': copy.deepcopy(self.positions),
            'centroid': copy.deepcopy(self.centroid)
        }
        self.undo_stack.append(current_state)

        # Pop from the redo stack and restore.
        next_state = self.redo_stack.pop()
        self.positions = next_state['positions']
        self.centroid = next_state['centroid']

        # Refresh the display.
        self.update_curve_fit()
        self.update_prediction()
        self._update_highlight_state()
        self.show_image()
        
    def moved(self, event):
        x, y = event.x / self.imscale + self.x, event.y / self.imscale + self.y
        x += self.offsetx
        y += self.offsety
        
        pano_coords = pto_mapper.map_image_to_pano(self.pto_data, self.image_index, x, y)
        if pano_coords:
            pano_x, pano_y = pano_coords
            pano_w = self.global_options.get('w')
            pano_h = self.global_options.get('h')
            # Assuming equirectangular projection (f=2)
            if self.global_options.get('f', 2) == 2:
                az_rad = (pano_x / pano_w) * 2 * math.pi
                alt_rad = (0.5 - pano_y / pano_h) * math.pi
                az_deg = math.degrees(az_rad)
                alt_deg = math.degrees(alt_rad)
                
                ra, dec = pos.radec_of(str(az_deg % 360), str(alt_deg))
                self.mousepos="\n  cursor pos = %.2f° az, %.2f° alt / %s ra %.2f dec" % (az_deg % 360, alt_deg, str(ra), math.degrees(float(repr(dec))))
            else:
                self.mousepos = "\n  cursor pos = (non-equirectangular)"
        else:
            self.mousepos = "\n  cursor pos = outside panorama"
        self.show_image()

    def key(self, event):
        # Check for Control key modifier (mask 0x4)
        is_control_pressed = (event.state & 0x4) != 0

        if is_control_pressed and event.keysym.lower() == 'z':
            self.undo()
            return # Stop further processing
        if is_control_pressed and event.keysym.lower() == 'y':
            self.redo()
            return # Stop further processing

        key_char = event.char
        is_upper = key_char.isupper()
        
        actions = {
            'p': ('p', 0.01), 'y': ('y', 0.01), 'r': ('r', 0.01),
            'z': ('v', 0.01), 'a': ('a', 0.001), 'b': ('b', 0.001),
            'c': ('c', 0.001), 'd': ('d', -0.5), 'e': ('e', 0.5)
        }

        if key_char.lower() in actions:
            param, change = actions[key_char.lower()]
            if is_upper: change *= -1
            self.img_data[param] = self.img_data.get(param, 0) + change * self.boost

        elif key_char == '?':
            root = tk.Toplevel(self.master)
            RecalibrateDialog(root, self.image, self)
            root.mainloop()
        elif key_char == 'x': # Snap points to fitted line
            self._save_state_for_undo()
            if self.curve_coeffs is not None:
                for p in self.positions:
                    snapped_pos = self._project_point_on_curve(p['current'])
                    p['current'] = snapped_pos
                    #self._update_centroid_entry(p['frame'], snapped_pos)
                    self._create_or_update_centroid(p['frame'], snapped_pos)
                self.update_curve_fit()
                self.update_prediction()
        elif key_char == 'X' or key_char == 'T': # Undo snap or temporal space
            self._save_state_for_undo()
            for p in self.positions:
                p['current'] = p['original']
                self._update_centroid_entry(p['frame'], p['original'])
            self.update_curve_fit()
            self.update_prediction()
        elif key_char == 't': # Temporally re-space points
            self._temporally_respace_points()
        elif key_char == '*':
            _, fresh_images_data = pto_mapper.parse_pto_file(args.ptofile)
            self.images_data[self.image_index] = fresh_images_data[self.image_index]
            self.img_data = self.images_data[self.image_index]
        elif key_char == 'l':
            try:
                pto_mapper.write_pto_file(self.pto_data, args.ptofile)
                print(f"Saved PTO file to {args.ptofile}")
            except Exception as e:
                print(f"Error saving PTO file: {e}", file=sys.stderr)
        elif key_char == 'S': # Note: 'S' is already uppercase
            with open("centroid.txt","w") as f:
                for l in self.centroid:
                    if l is not None: f.write(l + "\n")
            print("Saved centroid data to centroid.txt")
        elif key_char == 's':
            self.save_event_txt()
        elif key_char == 'h': self.show_text ^= 1
        elif key_char == 'i': self.show_info ^= 1
        elif key_char == '!': self.boost = 100 if self.boost == 1 else 1
        elif key_char == 'q':
            if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
                exit(0)
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
        
        self.show_image()

    def _update_centroid_entry(self, frame_num, xy_coords):
        """Recalculates and updates the centroid string for a given frame and xy coordinate."""
        if self.centroid[frame_num] is None:
            return
            
        x, y = xy_coords
        az_deg, alt_deg = self._image_coords_to_celestial(x, y)

        parts = self.centroid[frame_num].split(' ')
        parts[2] = f"{alt_deg:.2f}"
        parts[3] = f"{az_deg % 360:.2f}"
        self.centroid[frame_num] = ' '.join(parts)
        
    def _estimate_photometry(self, image, center_xy, aperture_r=5, annulus_r_inner=7, annulus_r_outer=10):
        """
        Estimates object brightness using aperture photometry.
        """
        try:
            # Work with a grayscale version of the image
            img_gray = image.convert('L')
            img_array = np.array(img_gray)

            y_coords, x_coords = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
            center_x, center_y = center_xy

            # Create masks for the aperture and the background annulus
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

            aperture_mask = dist_from_center <= aperture_r
            annulus_mask = (dist_from_center > annulus_r_inner) & (dist_from_center <= annulus_r_outer)

            # Calculate background level from the annulus
            background_pixels = img_array[annulus_mask]
            if background_pixels.size == 0:
                return 0.0 # Not enough pixels for background estimate

            mean_background = np.mean(background_pixels)

            # Calculate total brightness in the aperture
            aperture_pixels = img_array[aperture_mask]
            total_aperture_brightness = np.sum(aperture_pixels)

            # Subtract the background contribution from the aperture
            background_contribution = mean_background * aperture_pixels.size
            integrated_brightness = total_aperture_brightness - background_contribution

            # Return a non-negative, scaled value
            return max(0, round(integrated_brightness / 100.0, 2))

        except Exception as e:
            print(f"Photometry brightness estimation failed: {e}", file=sys.stderr)
            return 0.0

    def _estimate_saturation_brightness(self, image, center_xy, tolerance_percent=0.05):
        """
        Estimates brightness by measuring the area of contiguous pixels with
        a similar color to a center point, using a flood-fill algorithm.
        """
        try:
            img_gray = image.convert('L')
            img_array = np.array(img_gray)
            height, width = img_array.shape

            center_x, center_y = int(round(center_xy[0])), int(round(center_xy[1]))

            # Ensure the starting point is within the image bounds
            if not (0 <= center_x < width and 0 <= center_y < height):
                return 0.0

            # Get the color value of the starting pixel
            start_value = float(img_array[center_y, center_x])

            # Calculate the allowed color range based on the 5% tolerance
            tolerance_value = 255 * tolerance_percent
            min_val = start_value - tolerance_value
            max_val = start_value + tolerance_value

            q = deque([(center_x, center_y)])
            visited = set([(center_x, center_y)])
            area = 0

            while q:
                x, y = q.popleft()
                area += 1

                # Check the four direct neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy

                    # If the neighbor hasn't been visited yet...
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        # ...and it is a valid pixel within the color tolerance, add it to the queue.
                        if (0 <= nx < width and 0 <= ny < height and
                                min_val <= img_array[ny, nx] <= max_val):
                            q.append((nx, ny))

            return float(area)

        except Exception as e:
            print(f"Saturation brightness estimation failed: {e}", file=sys.stderr)
            return 0.0

    def _create_or_update_centroid(self, frame_num, xy_coords):
        """
        Selects the best brightness algorithm based on pixel saturation
        and then creates/updates the centroid string for a given frame.
        """
        image_to_analyze = Image.open(self.files[frame_num])

        # --- Dispatcher Logic ---
        img_gray = image_to_analyze.convert('L')
        img_array = np.array(img_gray)
        center_x, center_y = int(round(xy_coords[0])), int(round(xy_coords[1]))
        brightness = 0.0
        
        # Check if click is in bounds before accessing pixel data
        if not (0 <= center_y < img_array.shape[0] and 0 <= center_x < img_array.shape[1]):
            brightness = 0.0
        else:
            start_pixel_value = img_array[center_y, center_x]
            # A pixel is "white" if its value is 95% of maximum (255) or more
            saturation_threshold = 255 * 0.95

            brightness = self._estimate_photometry(image_to_analyze, xy_coords)
            if start_pixel_value >= saturation_threshold:
                # For saturated ("white") pixels, use the flood-fill area algorithm
                brightness += self._estimate_saturation_brightness(image_to_analyze, xy_coords)
        # --- End Dispatcher Logic ---

        # Calculate celestial coordinates
        x, y = xy_coords
        az_deg, alt_deg = self._image_coords_to_celestial(x, y)

        # Create the full centroid string with the new brightness value
        diff = self.timestamps[frame_num] - self.timestamps[0]
        ts = datetime.fromtimestamp(self.timestamps[frame_num], timezone.utc)

        self.centroid[frame_num] = f'{frame_num} {diff:.2f} {alt_deg:.2f} {az_deg % 360:.2f} {brightness:.2f} {args.name} {ts.strftime("%Y-%m-%d %H:%M:%S.%f UTC")}'

    def _draw_brightness_graph(self, canvas_bbox):
        """Draws a brightness vs. time graph on the canvas overlay."""
        if len(self.positions) < 2:
            return

        # 1. Extract (time, brightness) data points for the plot
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
                    if frame == self.num:
                        current_frame_data = point
                except (IndexError, ValueError):
                    continue

        if len(graph_data) < 2:
            return

        # 2. Define graph geometry in the upper-right corner
        padding = 10
        graph_width = 200
        graph_height = 100
        graph_x = canvas_bbox[2] - graph_width - padding
        graph_y = canvas_bbox[1] + padding

        # Draw background and border for the graph
        self.overlay.append(self.canvas.create_rectangle(
            graph_x, graph_y, graph_x + graph_width, graph_y + graph_height,
            fill="gray10", outline="gray50", stipple="gray50"
        ))

        # 3. Get data ranges for scaling
        times, brightnesses = zip(*graph_data)
        min_time, max_time = min(times), max(times)
        min_bright, max_bright = min(brightnesses), max(brightnesses)

        time_range = max_time - min_time if max_time > min_time else 1
        bright_range = max_bright - min_bright if max_bright > min_bright else 1

        # 4. Draw the graph line
        scaled_points = []
        for t, b in graph_data:
            px = graph_x + ((t - min_time) / time_range) * graph_width
            py = graph_y + graph_height - ((b - min_bright) / bright_range) * graph_height
            scaled_points.extend([px, py])

        self.overlay.append(self.canvas.create_line(scaled_points, fill="cyan", width=1))

        # Draw max/min brightness labels
        self.overlay.append(self.canvas.create_text(
            graph_x + 5, graph_y + 5, text=f"{max_bright:.1f}", anchor="nw", fill="yellow", font=("helvetica", 8)
        ))
        self.overlay.append(self.canvas.create_text(
            graph_x + 5, graph_y + graph_height - 5, text=f"{min_bright:.1f}", anchor="sw", fill="yellow", font=("helvetica", 8)
        ))

        # 5. Draw a marker for the current frame's position on the graph
        if current_frame_data:
            t, b = current_frame_data
            px = graph_x + ((t - min_time) / time_range) * graph_width
            py = graph_y + graph_height - ((b - min_bright) / bright_range) * graph_height
            radius = 3
            self.overlay.append(self.canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="red", outline=""
            ))

    def save_event_txt(self):
        # Use a filtered list of positions that corresponds to non-None centroid entries
        valid_positions = [p for p in self.positions if self.centroid[p['frame']] is not None]
        centroid2 = [l for l in self.centroid if l is not None]

        if not centroid2:
            print("No centroid data to save to event.txt")
            return
            
        with open("event.txt", "w") as f:
            print("[trail]", file=f)
            print(f"frames = {len(centroid2)}", file=f)
            print(f"duration = {round(float(centroid2[-1].split(' ')[1]) - float(centroid2[0].split(' ')[1]), 2)}", file=f)
            
            pos_str = " ".join([f"{p['current'][0]:.2f},{p['current'][1]:.2f}" for p in valid_positions])
            print(f"positions = {pos_str}", file=f)

            coord_str = " ".join([f"{i.split(' ')[3]},{i.split(' ')[2]}" for i in centroid2])
            print(f"coordinates = {coord_str}", file=f)

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
            
            print("manual = 1", file=f)
            print(file=f)

            print("[video]", file=f)
            start_info = " ".join(centroid2[0].split(" ")[6:])
            start_ts = self.timestamps[int(centroid2[0].split(" ")[0])]
            print(f"start = {start_info} ({start_ts})", file=f)
            
            end_info = " ".join(centroid2[-1].split(" ")[6:])
            end_ts = self.timestamps[int(centroid2[-1].split(" ")[0])]
            print(f"end = {end_info} ({end_ts})", file=f)
            
            print(f"width = {self.width}", file=f)
            print(f"height = {self.height}", file=f)
        
        print("Saved event data to event.txt")

    def load_event_file(self, filepath):
        """
        Loads positions and timestamps from an event.txt file, validates the data,
        and sets up the application state for editing.
        """
        try:
            config = configparser.ConfigParser()
            config.read(filepath)

            positions_str = config.get('trail', 'positions')
            timestamps_str = config.get('trail', 'timestamps')
            brightness_str = config.get('trail', 'brightness', fallback="0")
            event_brightnesses = brightness_str.split(' ')

            # Process the position and timestamp strings into lists
            event_positions_xy = []
            for part in positions_str.split(' '):
                if ',' in part:
                    x_str, y_str = part.split(',')
                    event_positions_xy.append((float(x_str), float(y_str)))
            
            event_timestamps = [float(t) for t in timestamps_str.split(' ')]

            # Create a lookup map of application timestamps for validation
            # We round to 2 decimal places for robust float comparison
            app_ts_map = {round(ts, 2): i for i, ts in enumerate(self.timestamps)}
            frame_indices = []

            for t_event in event_timestamps:
                if round(t_event, 2) in app_ts_map:
                    frame_indices.append(app_ts_map[round(t_event, 2)])
                else:
                    print(f"Warning: Timestamp {t_event:.2f} from event file not found in loaded images.", file=sys.stderr)
                    print("Ignoring event file and starting normally.", file=sys.stderr)
                    return # Abort loading

            print(f"Successfully loaded {len(frame_indices)} points from {filepath}")
            self.positions = []
            for i, frame_idx in enumerate(frame_indices):
                pos_xy = event_positions_xy[i]
                point_data = {'frame': frame_idx, 'original': pos_xy, 'current': pos_xy}
                self.positions.append(point_data)

                # Re-create the corresponding centroid entry from scratch
                x, y = pos_xy
                az_deg, alt_deg = self._image_coords_to_celestial(x, y)

                diff = self.timestamps[frame_idx] - self.timestamps[0]
                ts_obj = datetime.fromtimestamp(self.timestamps[frame_idx], timezone.utc)
                brightness_val = event_brightnesses[i] if i < len(event_brightnesses) else "0.0"
                self.centroid[frame_idx] = f'{frame_idx} {diff:.2f} {alt_deg:.2f} {az_deg % 360:.2f} {brightness_val} {args.name} {ts_obj.strftime("%Y-%m-%d %H:%M:%S.%f UTC")}'
            
            # Update the UI and jump to the first frame of the event
            if self.positions:
                self.update_curve_fit()
                self.update_prediction()
                self.change_image(self.positions[0]['frame'])

        except Exception as e:
            print(f"Error loading event file '{filepath}': {e}", file=sys.stderr)
            print("Starting normally with a clean slate.", file=sys.stderr)

    def _update_highlight_state(self):
        self.last_click_to_highlight = None
        if self.positions:
            last_recorded_frame = self.positions[-1]['frame']
            if self.num < last_recorded_frame:
                for pos_data in reversed(self.positions):
                    if pos_data['frame'] <= self.num:
                        self.last_click_to_highlight = pos_data['current']
                        break

    def change_image(self, new_num):
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
        """
        Check if the click is near a marked point. If so, prepare to drag it.
        Otherwise, prepare to pan the image.
        """
        self.dragged_point_index = None

        # Check for proximity to any existing points
        for i, pos_data in reversed(list(enumerate(self.positions))):
            px, py = pos_data['current']

            # Convert point's image coordinates to widget coordinates
            point_widget_x = (px - self.x - self.offsetx) * self.imscale
            point_widget_y = (py - self.y - self.offsety) * self.imscale

            # Calculate distance in screen pixels
            distance = math.sqrt((event.x - point_widget_x)**2 + (event.y - point_widget_y)**2)

            # If click is within 5 pixels, select this point for dragging
            if distance <= 5:
                self.dragged_point_index = i
                return  # Exit to prevent starting a pan

        # If no point was close enough, start a pan action
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        """
        If a point is being dragged, update its position.
        Otherwise, pan the image.
        """
        if self.dragged_point_index is not None:
            # We are dragging a point. Convert widget coordinates to image coordinates.
            new_img_x = (event.x / self.imscale) + self.x + self.offsetx
            new_img_y = (event.y / self.imscale) + self.y + self.offsety

            # Update the point's 'current' position
            self.positions[self.dragged_point_index]['current'] = (new_img_x, new_img_y)
        else:
            # We are panning the canvas
            self.canvas.scan_dragto(event.x, event.y, gain=1)
        
        self.show_image() # Redraw the canvas

    def drag_release(self, event):
        """
        Finalize the drag-and-drop operation when the left mouse button is released.
        """
        if self.dragged_point_index is not None:
            self._save_state_for_undo()

            # Get data from the point that was just moved
            dragged_point_data = self.positions[self.dragged_point_index]
            frame_num = dragged_point_data['frame']
            final_coords = dragged_point_data['current']

            # Update the corresponding centroid data and recalculate the curve
            self._create_or_update_centroid(frame_num, final_coords)
            self.update_curve_fit()
            self.update_prediction()

            # Reset the drag state
            self.dragged_point_index = None
            self.show_image() # Perform a final redraw

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
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)
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
        
        # Check if a point exists for the current frame to display its info
        point_info_text = ""
        for p in self.positions:
            if p['frame'] == self.num:
                px, py = p['current']
                brightness, altitude, azimuth = 0.0, 0.0, 0.0
                
                # Try to parse data from the corresponding centroid entry
                if self.centroid and self.num < len(self.centroid) and self.centroid[self.num]:
                    try:
                        parts = self.centroid[self.num].split(' ')
                        altitude = float(parts[2])
                        azimuth = float(parts[3])
                        brightness = float(parts[4])
                    except (IndexError, ValueError):
                        pass # Keep default values if parsing fails
                
                point_info_text = f"\n  marked pos = {azimuth:.2f}, {altitude:.2f}, brightness = {brightness:.2f}"
                break # Found the point for the current frame
        
        all_stars = brightstar(self.pto_data, pos, 6.5, -30, int(128*self.imscale), map_to_source_image=True, include_img_idx=True)
        stars = [s[1:] for s in all_stars if s[0] == self.image_index]

        self.x = x1 / self.imscale; self.y = y1 / self.imscale
        crop_x2 = min(x2 / self.imscale, self.width); crop_y2 = min(y2 / self.imscale, self.height)
        
        image = self.image.crop((self.x, self.y, crop_x2, crop_y2))

        if self.background_removal_active and self.background_image:
            bg_crop = self.background_image.crop((self.x, self.y, crop_x2, crop_y2))

            # Convert to numpy arrays for subtraction
            img_arr = np.array(image.convert('L')).astype(np.float32)
            bg_arr = np.array(bg_crop.convert('L')).astype(np.float32)
            
            subtracted_arr = img_arr - bg_arr

            # 1. Clip all negative values to 0. This ensures black stays black.
            subtracted_arr = np.clip(subtracted_arr, 0, 255)
            
            # 2. Renormalize the result to stretch the brightest part back to white.
            max_val = subtracted_arr.max()
            if max_val > 0:
                subtracted_arr = (subtracted_arr / max_val) * 255.0
            
            # Convert back to a displayable image
            image = Image.fromarray(subtracted_arr.astype(np.uint8)).convert('RGB')

        image = self.converted_image(image, self.contrast, self.brightness, self.color, self.sharpness)
        imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))

        for c in self.overlay: self.canvas.delete(c)
        self.overlay.clear()

        canvas_origin_x = max(bbox2[0], bbox1[0])
        canvas_origin_y = max(bbox2[1], bbox1[1])

        imageid = self.canvas.create_image(canvas_origin_x, canvas_origin_y, anchor='nw', image=imagetk)
        self.canvas.lower(imageid)
        self.canvas.imagetk = imagetk

        if self.curve_coeffs is not None and len(self.positions) >= 3:
            p = np.poly1d(self.curve_coeffs)
            p_first = self.positions[0]['current']
            p_last = self.positions[-1]['current']

            if self.curve_orientation == 'y_is_f_of_x':
                x_coords = [p['current'][0] for p in self.positions]
                min_x, max_x = min(x_coords), max(x_coords)
                span = max_x - min_x
                extension = span * 0.2 if span > 0 else 50
                
                if p_last[0] >= p_first[0]:
                    start_point, end_point = min_x, max_x + extension
                else:
                    start_point, end_point = min_x - extension, max_x
                
                fit_x = np.linspace(start_point, end_point, 100)
                fit_y = p(fit_x)
            else:
                y_coords = [p['current'][1] for p in self.positions]
                min_y, max_y = min(y_coords), max(y_coords)
                span = max_y - min_y
                extension = span * 0.2 if span > 0 else 50
                
                if p_last[1] >= p_first[1]:
                    start_point, end_point = min_y, max_y + extension
                else:
                    start_point, end_point = min_y - extension, max_y

                fit_y = np.linspace(start_point, end_point, 100)
                fit_x = p(fit_y)

            canvas_points = []
            for i in range(len(fit_x)):
                canvas_x = (fit_x[i] - self.x) * self.imscale + canvas_origin_x
                canvas_y = (fit_y[i] - self.y) * self.imscale + canvas_origin_y
                canvas_points.extend([canvas_x, canvas_y])
            
            if len(canvas_points) > 2:
                self.overlay.append(self.canvas.create_line(canvas_points, fill="cyan", width=2, smooth=True))

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
        
        # The red circle should only be drawn if a point exists for the current frame.
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

        if self.show_info:
            self.overlay.append(self.canvas.create_text(canvas_x+3, canvas_y+3, text=f"{name} ({mag})", anchor="nw", fill="green", font=("helvetica", 10)))

        self._draw_brightness_graph(bbox2)

        ts = datetime.fromtimestamp(self.timestamps[self.num], timezone.utc)
        info_text = (f"  time = {ts.strftime('%Y-%m-%d %H:%M:%S.%f UTC')} ({self.timestamps[self.num]:.2f})\n"
                     f"  pitch={self.img_data.get('p', 0):.2f}° yaw={self.img_data.get('y', 0):.2f}° roll={self.img_data.get('r', 0):.2f}° "
                     f"hfov={self.img_data.get('v', 0):.2f}°\n"
                     f"  radial=({self.img_data.get('a', 0):.3f}, {self.img_data.get('b', 0):.3f}, {self.img_data.get('c', 0):.3f}) "
                     f"radial shift=({-self.img_data.get('d', 0):.1f}, {self.img_data.get('e', 0):.1f})"
                     f"{self.mousepos}"
                     f"{point_info_text}\n  h = toggle help text")
        
        help_id = self.canvas.create_text(bbox2[0]+5, bbox2[1]+5, anchor="nw", text=info_text, fill="yellow", font=("helvetica", 10))
        self.overlay.append(help_id)

        if self.show_text:
            help_text_content = ("\n" * 6 +
                "  q=quit, -/+=bg removal, 1/2=contrast, 3/4=brightness, 5/6=color, 7/8=sharpness, 0=reset\n" + # This line is changed
                "  p/P=pitch, y/Y=yaw, r/R=roll, z/Z=hfov\n" +
                "  a/A,b/B,c/C = radial distortion params\n" +
                "  d/D,e/E = radial distortion shift\n" +
                "  i=toggle star info, !=toggle boost (100x)\n" +
                "  x=snap points to line, X=undo snap\n" +
                "  t=temporally snap points, T=undo temporal snap\n" +
                "  Ctrl+Z=undo, Ctrl+Y=redo\n" +
                "  *=reset orientation, l=save ptofile\n" +
                "  s=save event.txt, S=save centroid.txt\n" +
                "  arrows=move frame, pgup/dn=move 10 frames\n" +
                "  mouse wheel=zoom, LMB=drag, RMB=mark & next")
            text_id = self.canvas.create_text(bbox2[0]+5, bbox2[1]+5, anchor="nw", text=help_text_content, fill="yellow", font=("helvetica", 10))
            self.overlay.append(text_id)

        for s in stars:
            sx, sy, _, _, name, mag = s
            canvas_x = (sx - self.x) * self.imscale + canvas_origin_x
            canvas_y = (sy - self.y) * self.imscale + canvas_origin_y
            
            if bbox2[0] < canvas_x < bbox2[2] and bbox2[1] < canvas_y < bbox2[3]:
                self.overlay.append(self.canvas.create_oval(canvas_x-1, canvas_y-1, canvas_x+1, canvas_y+1, width=(max(0.1, 5-mag)*2 if self.show_info else 1), outline="red"))
                if self.show_info:
                    self.overlay.append(self.canvas.create_text(canvas_x+3, canvas_y+3, text=f"{name} ({mag})", anchor="nw", fill="green", font=("helvetica", 10)))

    def _distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _project_point_on_curve(self, point_to_project):
        """Finds the closest point on the fitted curve to a given point."""
        if self.curve_coeffs is None:
            return point_to_project

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
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = (search_x[i], search_y[i])
                
        return closest_point

    def _temporally_respace_points(self):
        """
        Adjusts each point's position along the fitted curve to be smoothly
        spaced in time relative to its immediate neighbors.
        """
        if self.curve_coeffs is None or len(self.positions) < 3:
            print("Not enough points to perform local temporal respacing.", file=sys.stderr)
            return

        print("Applying local temporal respacing...")
        
        # 1. Create a single, high-resolution lookup table for the entire curve
        N_POINTS = 2000
        p_curve = np.poly1d(self.curve_coeffs)
        start_coords, end_coords = self.positions[0]['current'], self.positions[-1]['current']

        if self.curve_orientation == 'y_is_f_of_x':
            domain = np.linspace(start_coords[0], end_coords[0], N_POINTS)
            curve_x, curve_y = domain, p_curve(domain)
        else: # x_is_f_of_y
            domain = np.linspace(start_coords[1], end_coords[1], N_POINTS)
            curve_y, curve_x = domain, p_curve(domain)

        arc_lengths = np.zeros(N_POINTS)
        for i in range(1, N_POINTS):
            dist = np.sqrt((curve_x[i] - curve_x[i-1])**2 + (curve_y[i] - curve_y[i-1])**2)
            arc_lengths[i] = arc_lengths[i-1] + dist

        # 2. Find the arc length corresponding to each of the original points
        original_arc_lengths = []
        for p in self.positions:
            # Find the closest point on the discretized curve to the current point
            dists = np.sqrt((curve_x - p['current'][0])**2 + (curve_y - p['current'][1])**2)
            closest_idx = np.argmin(dists)
            original_arc_lengths.append(arc_lengths[closest_idx])
        
        # 3. Iterate through each interior point and calculate its new local position
        new_positions = [p['current'] for p in self.positions] # Start with current positions
        
        for i in range(1, len(self.positions) - 1):
            # Get data for the local window (previous, current, next)
            t_prev = self.timestamps[self.positions[i-1]['frame']]
            t_curr = self.timestamps[self.positions[i]['frame']]
            t_next = self.timestamps[self.positions[i+1]['frame']]
            
            arc_prev, arc_next = original_arc_lengths[i-1], original_arc_lengths[i+1]
            
            # Calculate where the current point should be based on time
            local_duration = t_next - t_prev
            if local_duration <= 0: continue
            
            time_fraction = (t_curr - t_prev) / local_duration
            
            # Determine the target arc length within the local segment
            target_arc_length = arc_prev + time_fraction * (arc_next - arc_prev)

            # Interpolate from the main lookup table to find the new coordinate
            new_x = np.interp(target_arc_length, arc_lengths, curve_x)
            new_y = np.interp(target_arc_length, arc_lengths, curve_y)
            new_positions[i] = (new_x, new_y)

        # 4. Apply the new positions and recalculate all data
        print("Recalculating brightness for all points... (this may take a moment)")
        for i, p in enumerate(self.positions):
            p['current'] = new_positions[i]
            self._create_or_update_centroid(p['frame'], p['current'])

        self.update_prediction()

    def update_prediction(self):
        if len(self.positions) < 2:
            self.predicted_point = None
            return

        p_n_minus_1 = self.positions[-2]['current']
        p_n = self.positions[-1]['current']
        
        direction_vector = (p_n[0] - p_n_minus_1[0], p_n[1] - p_n_minus_1[1])
        norm = self._distance((0,0), direction_vector)
        
        if norm == 0:
            self.predicted_point = None
            return
            
        normalized_vector = (direction_vector[0] / norm, direction_vector[1] / norm)

        distances = [self._distance(self.positions[i]['current'], self.positions[i+1]['current']) for i in range(len(self.positions) - 1)]
        
        dist_next = 0
        if len(distances) < 2:
            dist_next = distances[-1] if distances else 10
        else:
            indices = np.arange(1, len(distances) + 1)
            coeffs = np.polyfit(indices, distances, 1)
            p = np.poly1d(coeffs)
            dist_next = p(len(distances) + 1)

        if dist_next < 0:
            dist_next = distances[-1] if distances else 10

        p_pred_linear = (p_n[0] + normalized_vector[0] * dist_next, p_n[1] + normalized_vector[1] * dist_next)
        
        self.predicted_point = self._project_point_on_curve(p_pred_linear)

    def update_curve_fit(self):
        """Fits a 2nd degree polynomial to the clicked points."""
        if len(self.positions) < 3:
            self.curve_coeffs = None
            return

        x_coords = np.array([p['current'][0] for p in self.positions])
        y_coords = np.array([p['current'][1] for p in self.positions])

        if np.ptp(x_coords) > np.ptp(y_coords):
            self.curve_coeffs = np.polyfit(x_coords, y_coords, 2)
            self.curve_orientation = 'y_is_f_of_x'
        else:
            self.curve_coeffs = np.polyfit(y_coords, x_coords, 2)
            self.curve_orientation = 'x_is_f_of_y'
            
    def click(self, event):
        self._save_state_for_undo()

        # Determine if the current frame already has a point marked on it.
        existing_frames = {p['frame'] for p in self.positions}

        # The deletion of subsequent points will only happen if the user clicks on a
        # frame that *already* has a point.
        if self.num in existing_frames:
            if self.positions:
                last_click_frame = self.positions[-1]['frame']
                if self.num < last_click_frame:
                    truncate_index = -1
                    for i, pos_data in enumerate(self.positions):
                        if pos_data['frame'] >= self.num:
                            truncate_index = i
                            break
                    
                    if truncate_index != -1:
                        frames_to_clear = {p['frame'] for p in self.positions[truncate_index:]}
                        for frame_idx in frames_to_clear:
                            if frame_idx < len(self.centroid):
                                self.centroid[frame_idx] = None
                        self.positions = self.positions[:truncate_index]

        # Get the coordinates for the new point
        x, y = event.x / self.imscale + self.x, event.y / self.imscale + self.y
        x += self.offsetx; y += self.offsety
        
        new_point = {'frame': self.num, 'original': (x,y), 'current': (x,y)}

        # Find the correct insertion index to keep the list sorted by frame number.
        # This correctly handles both appending to the end and inserting in the middle.
        insert_index = len(self.positions)
        for i, pos_data in enumerate(self.positions):
            if pos_data['frame'] > self.num:
                insert_index = i
                break
        self.positions.insert(insert_index, new_point)

        # Update curve fitting, prediction, and other states
        if len(self.positions) >= 3:
            self.update_curve_fit()
        else:
            self.curve_coeffs = None

        if len(self.positions) >= 2:
            self.update_prediction()
        else:
            self.predicted_point = None

        self._update_highlight_state()
        
        # Calculate celestial coordinates and update the centroid data
        self._create_or_update_centroid(self.num, (x, y))
        
        # Advance to the next frame
        self.right_key(event)

def _recursive_extract(timestamps, files, start_idx, end_idx, report_progress_callback):
    """
    Recursively divides the file list to find and fill blocks of identical timestamps.
    This is a helper function for extract_timestamps.
    """
    if start_idx > end_idx:
        return

    if start_idx == end_idx:
        if timestamps[start_idx] is None:
            timestamp(timestamps, files, start_idx)
            if timestamps[start_idx] is not None:
                report_progress_callback()
        return

    if timestamps[start_idx] is None:
        timestamp(timestamps, files, start_idx)
        if timestamps[start_idx] is not None:
            report_progress_callback()
    if timestamps[end_idx] is None:
        timestamp(timestamps, files, end_idx)
        if timestamps[end_idx] is not None:
            report_progress_callback()

    if timestamps[start_idx] is not None and timestamps[start_idx] == timestamps[end_idx]:
        frames_filled = 0
        for i in range(start_idx + 1, end_idx):
            if timestamps[i] is None:
                timestamps[i] = timestamps[start_idx]
                frames_filled += 1
        if frames_filled > 0:
            report_progress_callback(num_frames=frames_filled)
        return
    
    if end_idx <= start_idx + 1:
        return

    mid_idx = (start_idx + end_idx) // 2
    _recursive_extract(timestamps, files, start_idx, mid_idx, report_progress_callback)
    _recursive_extract(timestamps, files, mid_idx + 1, end_idx, report_progress_callback)


def extract_timestamps(files):
    """
    Optimized timestamp extraction using a divide-and-conquer strategy
    to quickly fill blocks of files with the same timestamp.
    """
    total_files = len(files)
    if not files:
        return []
    
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

    _recursive_extract(raw_timestamps, files, 0, total_files - 1, _report_progress)

    for i in range(total_files):
        if raw_timestamps[i] is None:
            timestamp(raw_timestamps, files, i)
            if raw_timestamps[i] is not None:
                _report_progress()
    
    sys.stdout.write(f"\rProgress: [{'#' * bar_length}] {total_files}/{total_files} (100%)\n\n")

    return raw_timestamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Click on images to find coordinates.')
    parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
    parser.add_argument('-f', '--fps', dest='fps', help='frames per second (default: 10 or extracted from images)', default=10, type=int)
    parser.add_argument('-n', '--name', dest='name', help='station name (default: NUL or extracted from station config)', default="NUL", type=str)
    parser.add_argument('-s', '--station', dest='station', help='station config file', type=str)
    parser.add_argument('-d', '--date', dest='start', help='start time (default: extracted from images))', type=str)
    parser.add_argument('--load-event', dest='event_file', help='load a previously saved event.txt file for editing', type=str)
    parser.add_argument('--latitude', type=float, help='Observer latitude in degrees. Required if --station is not used.')
    parser.add_argument('--longitude', type=float, help='Observer longitude in degrees. Required if --station is not used.')
    parser.add_argument('--elevation', type=float, help='Observer elevation in meters. Required if --station is not used.')
    parser.add_argument(action='store', dest='ptofile', help='input .pto or .json calibration file')
    parser.add_argument(action='store', nargs="*", dest='imgfiles', help='input image or video files (.mp4 or image files)')
    args = parser.parse_args()

    # Establish observer location from command-line args or station file
    pos = ephem.Observer()
    if all(arg is not None for arg in [args.latitude, args.longitude, args.elevation]):
        print("Using location from command-line arguments.")
        pos.lat, pos.lon, pos.elevation = str(args.latitude), str(args.longitude), args.elevation
        pos.temp, pos.pressure = 15.0, 1010.0  # Set reasonable defaults
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
        except (configparser.Error, FileNotFoundError) as e:
            sys.exit(f"Error reading station file '{args.station}': {e}")
    else:
        sys.exit("Error: Observer location not specified. Please use the -s/--station option, or --latitude, --longitude, and --elevation.")

    # This list will hold paths to any temporary files that need to be deleted later
    temp_files_to_clean = []

    # If input is a JSON file, convert it to an optimized PTO in memory
    if args.ptofile.lower().endswith('.json'):
        print(f"Detected JSON calibration file: {args.ptofile}. Converting...")
        try:
            with open(args.ptofile) as f:
                calib_data = json.load(f)

            # Use the already configured observer 'pos' to generate PTO data
            width = calib_data.get('imagew', 1920)
            height = calib_data.get('imageh', 1080)
            pto_content = amscalib2lens.generate_pto_from_json(calib_data, pos, width, height, match_dist_limit=0.2)

            # Hugin's optimizer requires a file, so we use a temporary one.
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pto", dir="/tmp") as temp_pto_file:
                temp_pto_path = temp_pto_file.name
                temp_files_to_clean.append(temp_pto_path)
                temp_pto_file.write(pto_content)
                temp_pto_file.flush() # Ensure data is written before optimizer runs

                print(f"Running Hugin's autooptimiser on temporary file {temp_pto_path}...")
                proc = subprocess.run(['autooptimiser', '-n', temp_pto_path, '-o', temp_pto_path], capture_output=True, text=True)
                
                if proc.returncode != 0:
                    print("\nWarning: autooptimiser failed during conversion.", file=sys.stderr)
                    print(proc.stderr, file=sys.stderr)
                else:
                    print("Optimization complete.")
            
            # The rest of the script will now use the newly created and optimized PTO file
            args.ptofile = temp_pto_path

        except (FileNotFoundError, json.JSONDecodeError) as e:
            sys.exit(f"Error reading JSON file: {e}")
        except (IOError, ValueError) as e:
            sys.exit(f"Error during JSON to PTO conversion: {e}")
        except FileNotFoundError:
            sys.exit("\nError: 'autooptimiser' command not found. Please ensure Hugin is installed.")

    # Process video files into image frames
    temp_dirs = []
    images = []
    for f in args.imgfiles:
        if f.lower().endswith('.mp4'):
            temp_dir = tempfile.TemporaryDirectory(prefix="clickcoords_", dir="/tmp")
            temp_dirs.append(temp_dir)
            try:
                read_frames(f, temp_dir.name)
                images.extend(sorted(glob.glob(temp_dir.name + "/*.tif")))
            except Exception as e:
                print(e, file=sys.stderr)
        else:
            images.append(os.path.abspath(f))

    if not images:
        sys.exit("No image files found.")

    # Extract or generate timestamps for all frames
    timestamps = [None] * len(images)
    if args.start is None:
        raw_timestamps = extract_timestamps(images)
        print("Interpolating timestamps...")
        timestamps = interpolate_timestamps(raw_timestamps)
        starttime = timestamps[0] if timestamps and timestamps[0] is not None else datetime.now().timestamp()
    else:
        starttime = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        timestamps = [starttime + (x / args.fps) for x in range(len(images))]

    # Parse the final PTO data (either original or from conversion)
    try:
        pto_data = pto_mapper.parse_pto_file(args.ptofile)
    except Exception as e:
        sys.exit(f"Error parsing PTO file '{args.ptofile}': {e}")

    global_options, images_data = pto_data
    if args.image >= len(images_data):
        sys.exit(f"Error: Image index {args.image} is out of range for PTO file with {len(images_data)} images.")
    
    img_data = images_data[args.image]
    first_pil = Image.open(images[0])
    img_data['w'], img_data['h'] = first_pil.size

    # Launch the GUI
    window = tk.Tk()
    window.geometry("1600x960")
    app = Zoom_Advanced(window, files=images, timestamps=timestamps, pto_data=pto_data, image_index=args.image)

    if args.event_file:
        app.load_event_file(args.event_file)
    window.mainloop()

    # Clean up all temporary directories and files
    for d in temp_dirs:
        d.cleanup()
    for f in temp_files_to_clean:
        if os.path.exists(f):
            print(f"Cleaning up temporary file: {f}")
            os.remove(f)
