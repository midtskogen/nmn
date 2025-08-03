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

def process_full_view(event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg: Future):
    """Handles all processing for the full view, using an executor for parallel tasks."""
    print("\n--- Processing Full View ---")
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    
    # --- Submit initial, independent tasks ---
    grid_labels_path = f"{tmpdir}/grid-labels.png"
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['lens_pto']} {grid_labels_path}"
    future_grid_png = executor.submit(run_command, drawgrid_cmd, "Generating full view grid", verbose)

    # --- Process grid transparency (depends on grid generation) ---
    grid_labels_transparent_path = f"{tmpdir}/grid-labels-transparent.png"
    future_grid_png.result() # Wait for grid generation to complete
    future_transparent_grid = executor.submit(set_image_opacity, grid_labels_path, grid_labels_transparent_path, OVERLAY_OPACITY, verbose)
    
    # --- Create final full view images and videos (dependencies on above tasks) ---
    print("-> Submitting final full view image and video tasks...")
    
    # Composite JPG with grid (depends on stacked JPG and transparent grid)
    future_stacked_jpg.result()
    future_transparent_grid.result()
    future_jpg_grid = executor.submit(alpha_composite_images, filenames['jpg'], grid_labels_transparent_path, filenames['jpggrid'], verbose)
    
    # Overlay video with grid (depends on transparent grid)
    future_mp4_grid = executor.submit(overlay_video_with_image, filenames['full'], grid_labels_transparent_path, filenames['mp4grid'], verbose)

    # Wait for all full-view tasks to complete before returning
    for future in as_completed([future_jpg_grid, future_mp4_grid]):
        future.result()

def process_gnomonic_view(event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg: Future):
    """Handles all processing related to the gnomonic projection, using an executor for parallel tasks."""
    print("\n--- Processing Gnomonic View ---")
    azalt_start, azalt_end = event_data.get('start_azalt'), event_data.get('end_azalt')
    if not azalt_start or not azalt_end:
        print("Skipping gnomonic view: 'coordinates' not found in event.txt.")
        return

    # 1. Reproject for gnomonic view (initial task)
    reproject_cmd = (
        f"{sys.executable} {BIN_DIR}/reproject.py -f 45 --width 1920 --height 2560 "
        f"-o {filenames['gnomonic_pto']} -g {filenames['gnomonic_grid_pto']} "
        f"-e {azalt_end} {filenames['lens_pto']} {azalt_start}"
    )
    future_reproject = executor.submit(run_command, reproject_cmd, "Reprojecting for gnomonic view", verbose)
    
    # --- Gnomonic Image Pipeline (depends on reprojection and stacked JPG) ---
    future_reproject.result() # Wait for reprojection to finish
    
    # 2. Stitch JPG (This is the critical step that depends on the external JPG)
    future_stacked_jpg.result() # WAIT for the stacked JPG to be created by the other pipeline.
    tmp_gnomonic_jpg = f"{tmpdir}/{filenames['name']}-gnomonic-tmp.png"
    stitcher_cmd_jpg = (f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 "
                        f"{filenames['gnomonic_pto']} {filenames['jpg']} {tmp_gnomonic_jpg}")
    future_stitched_gnomonic_jpg = executor.submit(run_command, stitcher_cmd_jpg, "Stitching gnomonic JPG", verbose)

    # 3. Modify PTO for MP4 (can run in parallel with JPG stitching)
    future_modified_pto = executor.submit(modify_pto_canvas, filenames['gnomonic_pto'], 
                                          filenames['gnomonic_mp4_pto'], 1920, 1080, verbose)

    # Add logos to stitched JPG (depends on stitching)
    future_stitched_gnomonic_jpg.result()
    future_gnomonic_with_logos = executor.submit(composite_logos, tmp_gnomonic_jpg, filenames['gnomonic'], 
                                                 f"{tmpdir}/nmn.png", f"{tmpdir}/sbsdnb.png", verbose=verbose)
    
    # 4. Recalibrate (depends on logos being added)
    future_gnomonic_with_logos.result()
    if event_data.get('recalibrate', False):
        recalibrate_cmd = (f"{sys.executable} {BIN_DIR}/recalibrate.py -c meteor.cfg "
                           f"{event_data['timestamp'] + event_data['duration'] // 2} "
                           f"{filenames['gnomonic_grid_pto']} {filenames['gnomonic']} {filenames['gnomonic_corr_grid_pto']}")
        future_recalibrated = executor.submit(run_command, recalibrate_cmd, "Recalibrating gnomonic view", verbose)
    else:
        shutil.copy(filenames['gnomonic_grid_pto'], filenames['gnomonic_corr_grid_pto'])
        future_recalibrated = executor.submit(lambda: True) # Dummy future
    
    # --- Gnomonic Grid Overlay Pipeline (depends on recalibration) ---
    future_recalibrated.result()
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['gnomonic_corr_grid_pto']} {filenames['gnomonic_corr_grid_png']}"
    run_command(drawgrid_cmd, "Generating gnomonic grid", verbose) # This chain is sequential
    draw_text_on_image(filenames['gnomonic_corr_grid_png'], event_data['label'], filenames['gnomonic_corr_grid_png'], verbose=verbose)
    gnomonic_grid_transparent = f"{tmpdir}/gnomonic_grid_transparent.png"
    set_image_opacity(filenames['gnomonic_corr_grid_png'], gnomonic_grid_transparent, OVERLAY_OPACITY, verbose=verbose)
    cropped_gnomonic_grid = f"{tmpdir}/gnomonic_grid_cropped.png"
    future_cropped_grid = executor.submit(crop_image, gnomonic_grid_transparent, cropped_gnomonic_grid, 
                                          crop_box=(0, 740, 1920, 740 + 1080), verbose=verbose)

    # --- Gnomonic Video Pipeline (depends on various earlier steps) ---
    future_modified_pto.result()
    stitch_cmd_mp4 = f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 {filenames['gnomonic_mp4_pto']} {filenames['full']} {filenames['gnomonicmp4']}"
    future_gnomonic_mp4 = executor.submit(run_command, stitch_cmd_mp4, "Creating gnomonic video", verbose)

    # --- Final Assembly (Submit final tasks and wait for completion) ---
    print("-> Submitting final gnomonic image and video tasks...")
    
    # Composite final gnomonic image with grid
    future_gnomonic_grid = executor.submit(alpha_composite_images, filenames['gnomonic'], 
                                           gnomonic_grid_transparent, filenames['gnomonicgrid'], verbose)
    
    # Overlay final gnomonic video with cropped grid
    future_gnomonic_mp4.result()
    future_cropped_grid.result()
    future_gnomonic_grid_mp4 = executor.submit(overlay_video_with_image, filenames['gnomonicmp4'], 
                                               cropped_gnomonic_grid, filenames['gnomonicgridmp4'], verbose)
    
    for future in as_completed([future_gnomonic_grid, future_gnomonic_grid_mp4]):
        future.result()

def main(file_prefix, verbose=False, nothreads=False):
    """Main function to orchestrate the video processing pipeline."""
    print(f"--- Initializing Video Pipeline ---")
    if verbose:
        print("--- Verbose mode enabled ---")
    if nothreads:
        print("--- Multithreading disabled ---")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare logos
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
            # --- Sequential (Original) Execution ---
            print("\n--- Processing Full View (Sequential) ---")
            stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
            run_command(stack_cmd, "Stacking video frames to create JPG", verbose)
            ts2 = event_data['timestamp'] + event_data['duration'] // 2
            grid_labels_path = f"{tmpdir}/grid-labels.png"
            drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['lens_pto']} {grid_labels_path}"
            run_command(drawgrid_cmd, "Generating full view grid", verbose)
            grid_labels_transparent_path = f"{tmpdir}/grid-labels-transparent.png"
            set_image_opacity(grid_labels_path, grid_labels_transparent_path, OVERLAY_OPACITY, verbose=verbose)
            print("-> Creating final full view images and videos...")
            alpha_composite_images(filenames['jpg'], grid_labels_transparent_path, filenames['jpggrid'], verbose=verbose)
            overlay_video_with_image(filenames['full'], grid_labels_transparent_path, filenames['mp4grid'], verbose=verbose)
            if gnomonic_enabled:
                process_gnomonic_view_sequential(event_data, filenames, tmpdir, verbose) # Call sequential version
            else:
                print("\nSkipping gnomonic view: requires 'positions' and 'coordinates' in event.txt.")
        else:
            # --- Parallel Execution ---
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Start the one task that has a cross-pipeline dependency
                stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
                future_stacked_jpg = executor.submit(run_command, stack_cmd, "Stacking video frames to create JPG", verbose)

                # Submit the two main pipelines, passing the future for the stacked JPG to both
                future_full_view = executor.submit(process_full_view, event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg)
                
                future_gnomonic_view = None
                if gnomonic_enabled:
                    future_gnomonic_view = executor.submit(process_gnomonic_view, event_data, filenames, tmpdir, verbose, executor, future_stacked_jpg)
                else:
                    print("\nSkipping gnomonic view: requires 'positions' and 'coordinates' in event.txt.")

                # Wait for both pipelines to complete
                future_full_view.result()
                if future_gnomonic_view:
                    future_gnomonic_view.result()

        # --- Final Touches (runs after all processing is complete) ---
        composite_logos(filenames['jpg'], filenames['jpg'], f"{tmpdir}/nmn.png", f"{tmpdir}/sbsdnb.png", verbose=verbose)

    print("\n--- Pipeline Finished ---")

def process_gnomonic_view_sequential(event_data, filenames, tmpdir, verbose):
    """The original sequential version of gnomonic processing for --nothreads mode."""
    print("\n--- Processing Gnomonic View (Sequential) ---")
    azalt_start, azalt_end = event_data.get('start_azalt'), event_data.get('end_azalt')
    
    reproject_cmd = (f"{sys.executable} {BIN_DIR}/reproject.py -f 45 --width 1920 --height 2560 "
                     f"-o {filenames['gnomonic_pto']} -g {filenames['gnomonic_grid_pto']} "
                     f"-e {azalt_end} {filenames['lens_pto']} {azalt_start}")
    run_command(reproject_cmd, "Reprojecting for gnomonic view", verbose)

    tmp_gnomonic_jpg = f"{tmpdir}/{filenames['name']}-gnomonic-tmp.png"
    stitcher_cmd_jpg = (f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 "
                        f"{filenames['gnomonic_pto']} {filenames['jpg']} {tmp_gnomonic_jpg}")
    run_command(stitcher_cmd_jpg, "Stitching gnomonic JPG", verbose)
    composite_logos(tmp_gnomonic_jpg, filenames['gnomonic'], f"{tmpdir}/nmn.png", f"{tmpdir}/sbsdnb.png", verbose=verbose)

    if event_data.get('recalibrate', False):
        recalibrate_cmd = (f"{sys.executable} {BIN_DIR}/recalibrate.py -c meteor.cfg "
                           f"{event_data['timestamp'] + event_data['duration'] // 2} "
                           f"{filenames['gnomonic_grid_pto']} {filenames['gnomonic']} "
                           f"{filenames['gnomonic_corr_grid_pto']}")
        run_command(recalibrate_cmd, "Recalibrating gnomonic view", verbose)
    else:
        shutil.copy(filenames['gnomonic_grid_pto'], filenames['gnomonic_corr_grid_pto'])

    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['gnomonic_corr_grid_pto']} {filenames['gnomonic_corr_grid_png']}"
    run_command(drawgrid_cmd, "Generating gnomonic grid", verbose)
    draw_text_on_image(filenames['gnomonic_corr_grid_png'], event_data['label'], filenames['gnomonic_corr_grid_png'], verbose=verbose)
    
    gnomonic_grid_transparent = f"{tmpdir}/gnomonic_grid_transparent.png"
    set_image_opacity(filenames['gnomonic_corr_grid_png'], gnomonic_grid_transparent, OVERLAY_OPACITY, verbose=verbose)
    
    cropped_gnomonic_grid = f"{tmpdir}/gnomonic_grid_cropped.png"
    crop_image(gnomonic_grid_transparent, cropped_gnomonic_grid, crop_box=(0, 740, 1920, 740 + 1080), verbose=verbose)

    print("-> Creating final gnomonic images and videos...")
    modify_pto_canvas(filenames['gnomonic_pto'], filenames['gnomonic_mp4_pto'], 1920, 1080, verbose=verbose)
    stitch_cmd_mp4 = f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 {filenames['gnomonic_mp4_pto']} {filenames['full']} {filenames['gnomonicmp4']}"
    run_command(stitch_cmd_mp4, "Creating gnomonic video", verbose)
    
    alpha_composite_images(filenames['gnomonic'], gnomonic_grid_transparent, filenames['gnomonicgrid'], verbose=verbose)
    overlay_video_with_image(filenames['gnomonicmp4'], cropped_gnomonic_grid, filenames['gnomonicgridmp4'], verbose=verbose)

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
