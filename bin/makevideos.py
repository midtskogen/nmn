#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Processes meteor observation videos to create stacked images, grid overlays,
and gnomonic projections. This script uses Python libraries and a
ThreadPoolExecutor to handle tasks in parallel, maximizing CPU utilization.
"""

import argparse
import base64
import re
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import math
import pto_mapper


# Assuming user-provided scripts are in the same directory or python path.
try:
    from logos import NMN_LOGO_B64, SBSDNB_LOGO_B64
except ImportError as e:
    print(f"Error: A required local module is missing: {e}", file=sys.stderr)
    print("Please ensure logos.py is accessible.", file=sys.stderr)
    sys.exit(1)

# Image and video processing libraries
try:
    from PIL import Image, ImageDraw, ImageFont
    import ffmpeg
except ImportError as e:
    print(f"Error: A required library is not installed: {e}", file=sys.stderr)
    print("Please install required libraries with: 'pip install Pillow ffmpeg-python'", file=sys.stderr)
    sys.exit(1)


# --- Script Constants ---
OVERLAY_OPACITY = 0.6
BIN_DIR = Path(__file__).parent.resolve()

# --- Helper Functions ---
def refraction(alt):
    """
    Applies atmospheric refraction correction to an altitude in degrees.
    This is a Python implementation of the formula from makevideos.sh.
    """
    # Avoid division by zero errors for edge cases
    if alt + 4.4 == 0: return alt
    
    tan_arg = math.radians(alt + (7.31 / (alt + 4.4)))
    
    # Avoid division by zero if tan(arg) is zero
    if abs(math.tan(tan_arg)) < 1e-9: return alt
    
    # The formula from the shell script
    corrected_alt = alt - 0.006 / math.tan(tan_arg)
    return round(corrected_alt, 2)

# --- Image, Video, & File Manipulation Functions ---

def modify_pto_canvas(input_path, output_path, width, height, verbose=False):
    """Modifies the canvas size (w, h) in a PTO project file."""
    description = f"Modifying PTO for video canvas: {Path(output_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Input: {input_path}, New Canvas: {width}x{height}")

    modified_lines = []
    found_p_line = False
    try:
        with open(input_path, 'r') as f_in:
            for line in f_in:
                if line.strip().startswith('p '):
                    line = re.sub(r'\bw\d+\b', f'w{width}', line)
                    line = re.sub(r'\bh\d+\b', f'h{height}', line)
                    found_p_line = True
                modified_lines.append(line)

        if not found_p_line:
            print(f"Warning: 'p' line not found in {input_path}. File may be invalid.", file=sys.stderr)

        with open(output_path, 'w') as f_out:
            f_out.writelines(modified_lines)

    except IOError as e:
        print(f"Error modifying PTO file: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"   ...Done: {description}.")
    return output_path

def composite_logos(base_image_path, output_path, nmn_logo_path, sbsdnb_logo_path, verbose=False):
    """Composites logos onto a base image using Pillow."""
    description = f"Adding logos to {Path(base_image_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Base: {base_image_path}, NMN: {nmn_logo_path}, SBSDNB: {sbsdnb_logo_path}")

    with Image.open(base_image_path).convert("RGBA") as base_img, \
         Image.open(nmn_logo_path).convert("RGBA") as nmn_logo, \
         Image.open(sbsdnb_logo_path).convert("RGBA") as sbsdnb_logo:

        base_img.paste(nmn_logo, (16, 16), nmn_logo)
        sbsdnb_x = base_img.width - sbsdnb_logo.width - 16
        base_img.paste(sbsdnb_logo, (sbsdnb_x, 16), sbsdnb_logo)
        base_img.convert("RGB").save(output_path)
    print(f"   ...Done: {description}.")
    return output_path

def draw_text_on_image(image_path, text, output_path, verbose=False):
    """Draws a multi-line label on the bottom-left of an image."""
    description = "Adding label to corrected grid"
    print(f"-> {description}...")
    if verbose:
        print(f"   Image: {image_path}, Text: '{text.replace('\n', ' ')}'")

    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        font = None
        for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Verdana.ttf"]:
            try:
                font = ImageFont.truetype(font_name, 18)
                break
            except IOError:
                continue
        if not font:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        position = (16, img.height - text_height - 16)
        draw.text(position, text, font=font, fill="white", stroke_width=1, stroke_fill="black")
        img.save(output_path)
    print(f"   ...Done: {description}.")
    return output_path

def set_image_opacity(input_path, output_path, opacity, verbose=False):
    """Creates a new image with adjusted opacity in its alpha channel."""
    description = f"Setting image opacity to {opacity*100:.0f}%: {Path(output_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Input: {input_path}")
    with Image.open(input_path).convert("RGBA") as img:
        alpha = img.getchannel('A')
        new_alpha = alpha.point(lambda i: int(i * opacity))
        img.putalpha(new_alpha)
        img.save(output_path)
    print(f"   ...Done: {description}.")
    return output_path

def alpha_composite_images(base_path, overlay_path, output_path, verbose=False):
    """Composites a pre-processed overlay onto a base image."""
    description = f"Creating composite image with grid: {Path(output_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Base: {base_path}, Overlay: {overlay_path}")

    with Image.open(base_path).convert("RGBA") as base, \
         Image.open(overlay_path).convert("RGBA") as overlay:
        if base.size != overlay.size:
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        
        composited = Image.alpha_composite(base, overlay)
        composited.convert("RGB").save(output_path)
    print(f"   ...Done: {description}.")
    return output_path

def crop_image(input_path, output_path, crop_box, verbose=False):
    """Crops an image while preserving its mode (e.g., RGBA)."""
    description = f"Cropping grid image: {Path(output_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Input: {input_path}, CropBox: {crop_box}")
    with Image.open(input_path) as img:
        cropped = img.crop(crop_box)
        cropped.save(output_path)
    print(f"   ...Done: {description}.")
    return output_path

def overlay_video_with_image(video_path, overlay_path, output_path, verbose=False):
    """Overlays an image on a video using the ffmpeg-python library."""
    description = f"Creating video with grid overlay: {Path(output_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Video Input: {video_path}, Overlay Input: {overlay_path}")
    
    try:
        input_video = ffmpeg.input(video_path)
        overlay_image = ffmpeg.input(overlay_path)

        (
            ffmpeg
            .overlay(input_video, overlay_image)
            .output(output_path, vcodec='libx264')
            .overwrite_output()
            .run(quiet=(not verbose), capture_stdout=True, capture_stderr=True)
        )
        print(f"   ...Done: {description}.")
    except ffmpeg.Error as e:
        print("\n--- ERROR during ffmpeg-python execution ---", file=sys.stderr)
        print(f"FFmpeg stderr:\n{e.stderr.decode('utf8')}", file=sys.stderr)
        sys.exit(1)
    return output_path


# --- Core Script Logic ---

def get_event_data(event_file):
    """Parses the event.txt file to extract key information."""
    data = {}
    try:
        with open(event_file, 'r') as f:
            for line in f:
                if '=' not in line:
                    continue
                key, value = (part.strip() for part in line.strip().split('=', 1))
                parts = value.split()
                if key == 'start':
                    timestamp = None
                    for part in parts:
                        try:
                            cleaned_part = ''.join(c for c in part if c.isdigit() or c == '.')
                            if float(cleaned_part) > 1_000_000_000:
                                timestamp = int(float(cleaned_part))
                                break
                        except (ValueError, TypeError):
                            continue
                    if timestamp is None:
                        raise ValueError("Could not find a valid Unix timestamp in 'start' line.")
                    data['timestamp'] = timestamp
                    data['clock'] = f"{parts[0]} {parts[1].split('.')[0]}"
                elif key == 'duration':
                    data['duration'] = int(float(value) + 0.5)
                elif key == 'positions':
                    data['begin'], data['end'] = parts[0], parts[-1]
                elif key == 'coordinates':
                    data['start_azalt'], data['end_azalt'] = parts[0], parts[-1]
                elif key in ('sunalt', 'manual'):
                    data[key] = int(float(value))
                elif key in ('latitude', 'longitude'):
                    data[key] = float(value)
    except FileNotFoundError:
        print(f"Error: Event file '{event_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"Error parsing '{event_file}': {e}", file=sys.stderr)
        sys.exit(1)
    return data

def run_command(command, description, verbose=False):
    """Executes a shell command with progress indication and error handling."""
    if not verbose:
        print(f"-> {description}...")
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        if verbose:
            output_file = "N/A"
            if ' -o ' in command: output_file = command.split(' -o ')[-1].split(' ')[0]
            elif '>' in command: output_file = command.split('>')[-1].strip()
            
            log_message = (
                f"\n{'='*70}\n"
                f"Action: {description}\n"
                f"Producing: {output_file}\n"
                f"Command: {command}\n"
            )
            print(log_message, end="")
            if result.stdout.strip():
                print(f"--- stdout ---\n{result.stdout.strip()}")
            if result.stderr.strip():
                sys.stderr.write(f"--- stderr ---\n{result.stderr.strip()}\n")
            print(f"{'='*70}\n")
        else:
            print(f"   ...Done: {description}.")
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR executing external command ---\nCommand failed: {command}", file=sys.stderr)
        if e.stdout: print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        if e.stderr: print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def _run_full_view_sequentially(event_data, filenames, tmpdir, verbose):
    """Runs the full view processing pipeline sequentially."""
    print("\n--- Processing Full View (Sequential) ---")
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    
    grid_labels_path = f"{tmpdir}/grid-labels.png"
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['lens_pto']} {grid_labels_path}"
    run_command(drawgrid_cmd, "Generating full view grid", verbose)

    grid_labels_transparent_path = f"{tmpdir}/grid-labels-transparent.png"
    set_image_opacity(grid_labels_path, grid_labels_transparent_path, OVERLAY_OPACITY, verbose=verbose)
    
    print("-> Creating final full view images and videos...")
    alpha_composite_images(filenames['jpg'], grid_labels_transparent_path, filenames['jpggrid'], verbose=verbose)
    overlay_video_with_image(filenames['full'], grid_labels_transparent_path, filenames['mp4grid'], verbose=verbose)

def _run_full_view_in_parallel(event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg: Future):
    """Runs the full view processing pipeline in parallel."""
    print("\n--- Processing Full View ---")
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    
    grid_labels_path = f"{tmpdir}/grid-labels.png"
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['lens_pto']} {grid_labels_path}"
    future_grid_png = executor.submit(run_command, drawgrid_cmd, "Generating full view grid", verbose)

    future_transparent_grid = executor.submit(lambda: set_image_opacity(future_grid_png.result() and grid_labels_path, f"{tmpdir}/grid-labels-transparent.png", OVERLAY_OPACITY, verbose))
    
    print("-> Submitting final full view image and video tasks...")
    future_jpg_grid = executor.submit(lambda: alpha_composite_images(future_stacked_jpg.result() and filenames['jpg'], future_transparent_grid.result(), filenames['jpggrid'], verbose))
    future_mp4_grid = executor.submit(lambda: overlay_video_with_image(filenames['full'], future_transparent_grid.result(), filenames['mp4grid'], verbose))

    for future in as_completed([future_jpg_grid, future_mp4_grid]):
        future.result()

def _create_gnomonic_grid_and_image(event_data, filenames, tmpdir, verbose):
    """
    Synchronous helper function that creates the gnomonic grid and final composite image.
    This entire function is intended to be run as a single task in a thread pool.
    """
    # 1. Stitch gnomonic JPG and add logos
    tmp_gnomonic_jpg = f"{tmpdir}/{filenames['name']}-gnomonic-tmp.png"
    stitcher_cmd_jpg = (f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 "
                        f"{filenames['gnomonic_pto']} {filenames['jpg']} {tmp_gnomonic_jpg}")
    run_command(stitcher_cmd_jpg, "Stitching gnomonic JPG", verbose)
    
    composite_logos(tmp_gnomonic_jpg, filenames['gnomonic'], 
                    f"{tmpdir}/nmn.png", f"{tmpdir}/sbsdnb.png", verbose=verbose)
    
    # 2. Recalibrate if necessary
    if event_data.get('recalibrate', False):
        recalibrate_cmd = (f"{sys.executable} {BIN_DIR}/recalibrate.py -c meteor.cfg "
                           f"{event_data['timestamp'] + event_data['duration'] // 2} "
                           f"{filenames['gnomonic_grid_pto']} {filenames['gnomonic']} "
                           f"{filenames['gnomonic_corr_grid_pto']}")
        run_command(recalibrate_cmd, "Recalibrating gnomonic view", verbose)
    else:
        shutil.copy(filenames['gnomonic_grid_pto'], filenames['gnomonic_corr_grid_pto'])
    
    # 3. Generate and prepare grid overlay
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['gnomonic_corr_grid_pto']} {filenames['gnomonic_corr_grid_png']}"
    run_command(drawgrid_cmd, "Generating gnomonic grid", verbose)
    
    draw_text_on_image(filenames['gnomonic_corr_grid_png'], event_data['label'], filenames['gnomonic_corr_grid_png'], verbose=verbose)
    
    gnomonic_grid_transparent = f"{tmpdir}/gnomonic_grid_transparent.png"
    set_image_opacity(filenames['gnomonic_corr_grid_png'], gnomonic_grid_transparent, OVERLAY_OPACITY, verbose=verbose)
    
    cropped_gnomonic_grid = f"{tmpdir}/gnomonic_grid_cropped.png"
    crop_image(gnomonic_grid_transparent, cropped_gnomonic_grid, crop_box=(0, 740, 1920, 740 + 1080), verbose=verbose)

    # 4. Create final composite static image
    alpha_composite_images(filenames['gnomonic'], gnomonic_grid_transparent, filenames['gnomonicgrid'], verbose=verbose)

    return {"cropped_grid": cropped_gnomonic_grid}


def _run_gnomonic_view_in_parallel(event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg: Future):
    """Runs the gnomonic processing by splitting it into parallel video and image/grid pipelines."""
    print("\n--- Processing Gnomonic View ---")
    azalt_start, azalt_end = event_data.get('start_azalt'), event_data.get('end_azalt')
    if not azalt_start or not azalt_end:
        print("Skipping gnomonic view: 'coordinates' not found in event.txt.")
        return

    # 1. Reproject for gnomonic view (initial task, required by both sub-pipelines)
    reproject_cmd = (
        f"{sys.executable} {BIN_DIR}/reproject.py -f 45 --width 1920 --height 2560 "
        f"-o {filenames['gnomonic_pto']} -g {filenames['gnomonic_grid_pto']} "
        f"-e {azalt_end} {filenames['lens_pto']} {azalt_start}"
    )
    future_reproject = executor.submit(run_command, reproject_cmd, "Reprojecting for gnomonic view", verbose)
    future_reproject.result()

    # 2. GNOMONIC VIDEO PIPELINE (long-running)
    # Starts the creation of the raw gnomonic video, which runs in parallel with the grid creation.
    future_modified_pto = executor.submit(modify_pto_canvas, filenames['gnomonic_pto'], 
                                          filenames['gnomonic_mp4_pto'], 1920, 1080, verbose)
    stitch_cmd_mp4 = f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 {filenames['gnomonic_mp4_pto']} {filenames['full']} {filenames['gnomonicmp4']}"
    future_gnomonic_mp4 = executor.submit(lambda: run_command(future_modified_pto.result() and stitch_cmd_mp4, "Creating gnomonic video", verbose))

    # 3. GNOMONIC IMAGE/GRID PIPELINE (long-running)
    # Starts the creation of all grid assets. This task runs in parallel with the video pipeline.
    future_stacked_jpg.result() # Must wait for the main stacked JPG before starting.
    future_grid_assets = executor.submit(_create_gnomonic_grid_and_image, event_data, filenames, tmpdir, verbose)

    # 4. FINAL ASSEMBLY
    # Wait for the video and the grid assets, then combine them.
    grid_assets = future_grid_assets.result()
    future_gnomonic_mp4.result()
    
    overlay_video_with_image(filenames['gnomonicmp4'], grid_assets["cropped_grid"], filenames['gnomonicgridmp4'], verbose)

def _run_gnomonic_view_sequentially(event_data, filenames, tmpdir, verbose):
    """Runs the entire gnomonic processing pipeline sequentially."""
    print("\n--- Processing Gnomonic View (Sequential) ---")
    azalt_start, azalt_end = event_data.get('start_azalt'), event_data.get('end_azalt')
    if not azalt_start or not azalt_end: return

    reproject_cmd = (f"{sys.executable} {BIN_DIR}/reproject.py -f 45 --width 1920 --height 2560 "
                     f"-o {filenames['gnomonic_pto']} -g {filenames['gnomonic_grid_pto']} "
                     f"-e {azalt_end} {filenames['lens_pto']} {azalt_start}")
    run_command(reproject_cmd, "Reprojecting for gnomonic view", verbose)

    grid_assets = _create_gnomonic_grid_and_image(event_data, filenames, tmpdir, verbose)

    print("-> Creating final gnomonic video...")
    modify_pto_canvas(filenames['gnomonic_pto'], filenames['gnomonic_mp4_pto'], 1920, 1080, verbose=verbose)
    stitch_cmd_mp4 = f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 {filenames['gnomonic_mp4_pto']} {filenames['full']} {filenames['gnomonicmp4']}"
    run_command(stitch_cmd_mp4, "Creating gnomonic video", verbose)
    
    overlay_video_with_image(filenames['gnomonicmp4'], grid_assets["cropped_grid"], filenames['gnomonicgridmp4'], verbose=verbose)

def main(file_prefix, verbose=False, nothreads=False):
    """Main function to orchestrate the video processing pipeline."""
    print(f"--- Initializing Video Pipeline ---")
    if verbose: print("--- Verbose mode enabled ---")
    if nothreads: print("--- Multithreading disabled ---")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/nmn.png", "wb") as f: f.write(base64.b64decode(NMN_LOGO_B64))
        with open(f"{tmpdir}/sbsdnb.png", "wb") as f: f.write(base64.b64decode(SBSDNB_LOGO_B64))

        event_data = get_event_data('event.txt')
        name = file_prefix
        
        filenames = {
            'name': name, 'full': f"{name}.mp4", 'jpg': f"{name}.jpg", 'jpggrid': f"{name}-grid.jpg",
            'mp4grid': f"{name}-grid.mp4", 'gnomonic': f"{name}-gnomonic.jpg", 'gnomonicgrid': f"{name}-gnomonic-grid.jpg",
            'gnomonicmp4': f"{name}-gnomonic.mp4", 'gnomonicgridmp4': f"{name}-gnomonic-grid.mp4",
            'gnomonic_pto': 'gnomonic.pto', 'gnomonic_grid_pto': 'gnomonic_grid.pto',
            'gnomonic_corr_grid_pto': 'gnomonic_corr_grid.pto', 'gnomonic_corr_grid_png': 'gnomonic_corr_grid.png',
            'gnomonic_mp4_pto': 'gnomonic_mp4.pto', 'lens_pto': 'lens.pto'
        }

        event_data['recalibrate'] = event_data.get('manual', 0) == 0 and event_data.get('sunalt', 0) < -9
        pos_label = f"{event_data.get('latitude', 0):.4f}N {event_data.get('longitude', 0):.4f}E"
        event_data['label'] = f"{event_data.get('clock', '')}\n{pos_label}"
        gnomonic_enabled = 'begin' in event_data and 'start_azalt' in event_data

        if nothreads:
            stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
            run_command(stack_cmd, "Stacking video frames to create JPG", verbose)
            _run_full_view_sequentially(event_data, filenames, tmpdir, verbose)
            if gnomonic_enabled:
                _run_gnomonic_view_sequentially(event_data, filenames, tmpdir, verbose)
        else:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
                future_stacked_jpg = executor.submit(run_command, stack_cmd, "Stacking video frames to create JPG", verbose)

                future_full_view = executor.submit(_run_full_view_in_parallel, event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg)
                
                if gnomonic_enabled:
                    future_gnomonic = executor.submit(_run_gnomonic_view_in_parallel, event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg)
                    future_gnomonic.result() # Wait for gnomonic pipeline to finish
                else:
                    print("\nSkipping gnomonic view: requires 'positions' and 'coordinates' in event.txt.")

                future_full_view.result() # Wait for full view pipeline to finish

        composite_logos(filenames['jpg'], filenames['jpg'], f"{tmpdir}/nmn.png", f"{tmpdir}/sbsdnb.png", verbose=verbose)

        # --- Calculate and store refined coordinates for final printout ---
        if gnomonic_enabled and Path(filenames['gnomonic_corr_grid_pto']).exists():
            try:
                print("\n--- Calculating Refined Endpoints ---")
                # 1. Parse PTO files
                lens_pto_data = pto_mapper.parse_pto_file(filenames['lens_pto'])
                gnomonic_pto_data = pto_mapper.parse_pto_file(filenames['gnomonic_corr_grid_pto'])

                # 2. Get scale from lens.pto (used for az/alt conversion)
                scale = lens_pto_data[0]['w'] / 360.0

                # 3. Get initial pixel coordinates from event.txt
                start_px, start_py = map(float, event_data['begin'].split(','))
                end_px, end_py = map(float, event_data['end'].split(','))

                # 4. Map fisheye pixel coords to equirectangular pano coords using lens.pto
                start_pano = pto_mapper.map_image_to_pano(lens_pto_data, 0, start_px, start_py)
                end_pano = pto_mapper.map_image_to_pano(lens_pto_data, 0, end_px, end_py)

                if start_pano and end_pano:
                    start_az2, start_alt2 = start_pano
                    end_az2, end_alt2 = end_pano

                    # 5. Map equirectangular pano coords to gnomonic image coords
                    start_res = pto_mapper.map_pano_to_image(gnomonic_pto_data, start_az2, start_alt2)
                    end_res = pto_mapper.map_pano_to_image(gnomonic_pto_data, end_az2, end_alt2)

                    if start_res and end_res:
                        _, gnomonic_startx, gnomonic_starty = start_res
                        _, gnomonic_endx, gnomonic_endy = end_res

                        # 6. Refine track using refinetrack.py
                        gnomonic_image_path = filenames['gnomonic']
                        gnomonic_mask_path = f"{filenames['name']}-gnomonic-mask.jpg"
                        if Path(gnomonic_mask_path).exists():
                            gnomonic_image_path = gnomonic_mask_path
                        
                        refine_cmd = [
                            sys.executable, str(BIN_DIR / "refinetrack.py"),
                            gnomonic_image_path,
                            f"{gnomonic_startx},{gnomonic_starty}",
                            f"{gnomonic_endx},{gnomonic_endy}"
                        ]
                        result = subprocess.run(refine_cmd, capture_output=True, text=True, check=True)
                        points_str = result.stdout.strip()
                        
                        # 7. Map refined gnomonic image points back to pano coordinates
                        point_coords = re.split(r'[,\s]+', points_str)
                        refined_start_x, refined_start_y = float(point_coords[0]), float(point_coords[1])
                        refined_end_x, refined_end_y = float(point_coords[2]), float(point_coords[3])

                        start_pano_coords = pto_mapper.map_image_to_pano(gnomonic_pto_data, 0, refined_start_x, refined_start_y)
                        end_pano_coords = pto_mapper.map_image_to_pano(gnomonic_pto_data, 0, refined_end_x, refined_end_y)

                        if start_pano_coords and end_pano_coords:
                            # 8. Convert pano coordinates to az/alt and apply refraction
                            corr_startaz = round(start_pano_coords[0] / scale, 2)
                            corr_startalt = 90 - (start_pano_coords[1] / scale)
                            corr_startalt = refraction(corr_startalt)

                            corr_endaz = round(end_pano_coords[0] / scale, 2)
                            corr_endalt = 90 - (end_pano_coords[1] / scale)
                            corr_endalt = refraction(corr_endalt)

                            # Store for printing after "Pipeline Finished"
                            event_data['refined_coords'] = (corr_startaz, corr_startalt, corr_endaz, corr_endalt)

            except Exception as e:
                print(f"Warning: Could not calculate refined endpoints: {e}", file=sys.stderr)
                if verbose:
                    traceback.print_exc(file=sys.stderr)

    print("\n--- Pipeline Finished ---")

    # Print the refined coordinates if they were calculated successfully
    if event_data.get('refined_coords'):
        s_az, s_alt, e_az, e_alt = event_data['refined_coords']
        print(f"Start: {s_az:.2f} {s_alt:.2f}  End: {e_az:.2f} {e_alt:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes meteor observation videos using Python libraries.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("file_prefix", help="The base name for input/output files (e.g., 'event_20250101_123456').")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed information about commands being run.")
    parser.add_argument("--nothreads", action="store_true", help="Disable multithreading and run all tasks sequentially.")
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    lockfile = Path("processing.lock")
    if lockfile.exists():
        print("Error: Lockfile 'processing.lock' exists. Another instance may be running.", file=sys.stderr)
        sys.exit(1)
    
    try:
        lockfile.touch()
        main(args.file_prefix, verbose=args.verbose, nothreads=args.nothreads)
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        if lockfile.exists():
            lockfile.unlink()
