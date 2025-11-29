#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Processes meteor observation videos to create stacked images, grid overlays,
and gnomonic projections. This script uses Python libraries and a
ThreadPoolExecutor to handle tasks in parallel, maximizing CPU utilization.

It also includes a '--client' mode, which prepares the initial event video
from raw camera footage and then generates the initial gnomonic projection
and cropped 'fireball.jpg'.
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
import termios
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import math
import pto_mapper
import errno
import datetime


# Assuming user-provided scripts are in the same directory or python path.
try:
    from logos import NMN_LOGO_B64, SBSDNB_LOGO_B64, AS7_LOGO_B64
except ImportError as e:
    print(f"Error: A required local module is missing: {e}", file=sys.stderr)
    print("Please ensure logos.py is accessible and contains NMN, SBSDNB, and AS7 logo data.", file=sys.stderr)
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
OVERLAY_OPACITY = 0.65
BIN_DIR = Path(__file__).parent.resolve()

# --- Helper Functions ---
def refraction(alt):
    """
    Applies atmospheric refraction correction to an altitude in degrees.
    """
    if alt + 4.4 == 0: return alt
    tan_arg = math.radians(alt + (7.31 / (alt + 4.4)))
    if abs(math.tan(tan_arg)) < 1e-9: return alt
    corrected_alt = alt - 0.006 / math.tan(tan_arg)
    return round(corrected_alt, 2)

def get_dynamic_mag_limit(pto_path):
    """
    Parses the PTO file to find the Field of View (v parameter).
    Returns a limiting magnitude based on the FOV:
      - FOV > 115: Mag 3
      - FOV > 100: Mag 4
      - Otherwise: Mag 5 (default)
    """
    default_mag = 5
    try:
        with open(pto_path, 'r') as f:
            for line in f:
                # Look for the image line 'i'
                if line.strip().startswith('i '):
                    # Regex to find 'v' followed by numbers/dots
                    # Matches "v127.111" in "i w1920 ... v127.111 ..."
                    match = re.search(r'\bv([\d\.]+)', line)
                    if match:
                        fov = float(match.group(1))
                        if fov > 115:
                            return 3
                        elif fov > 100:
                            return 4
                        else:
                            return default_mag
    except Exception as e:
        print(f"Warning: Could not determine FOV from {pto_path} (Error: {e}). Using default mag {default_mag}.", file=sys.stderr)
    
    return default_mag

def get_video_resolution(video_path):
    """
    Probes the video file to return its width and height.
    Defaults to 1920x1080 if probing fails.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            return int(video_stream['width']), int(video_stream['height'])
    except Exception as e:
        print(f"Warning: Could not probe resolution of {video_path}: {e}", file=sys.stderr)
    
    return 1920, 1080

def draw_marker_crosses(image_path, pixel_coords, azalt_coords, verbose=False):
    """Draws marker crosses and az/alt labels on an image at specified coordinates."""
    description = f"Drawing marker crosses and labels on {Path(image_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Pixel Coords: {pixel_coords}")
        print(f"   Az/Alt Coords: {azalt_coords}")

    sx, sy, ex, ey = pixel_coords

    if len(azalt_coords) < 4:
        print("Warning: Insufficient az/alt coordinates provided to draw_marker_crosses.", file=sys.stderr)
        return image_path

    start_az, start_alt, end_az, end_alt = azalt_coords

    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # Load font
        font = None
        for font_name in ["Helvetica.ttf", "Arial.ttf", "DejaVuSans.ttf", "Verdana.ttf"]:
            try:
                font = ImageFont.truetype(font_name, 14)
                break
            except IOError:
                continue
        if not font:
            font = ImageFont.load_default()

        start_label = f"({start_az:.2f}, {start_alt:.2f})"
        end_label = f"({end_az:.2f}, {end_alt:.2f})"

        start_text_pos = (sx + 20, sy - 8)
        end_text_pos = (ex + 20, ey - 8)

        # Draw start cross and label
        draw.line((sx-16, sy-16, sx-4, sy-4), fill="white", width=2)
        draw.line((sx+4, sy+4, sx+16, sy+16), fill="white", width=2)
        draw.line((sx+16, sy-16, sx+4, sy-4), fill="white", width=2)
        draw.line((sx-4, sy+4, sx-16, sy+16), fill="white", width=2)
        draw.text(start_text_pos, start_label, font=font, fill="yellow", stroke_width=1, stroke_fill="black")

        # Draw end cross and label
        draw.line((ex-16, ey-16, ex-4, ey-4), fill="white", width=2)
        draw.line((ex+4, ey+4, ex+16, ey+16), fill="white", width=2)
        draw.line((ex+16, ey-16, ex+4, ey-4), fill="white", width=2)
        draw.line((ex-4, ey+4, ex-16, ey+16), fill="white", width=2)
        draw.text(end_text_pos, end_label, font=font, fill="yellow", stroke_width=1, stroke_fill="black")

        img.save(image_path)
    print(f"   ...Done: {description}.")
    return image_path

# --- Refined Endpoint Calculation Helpers ---

def _get_initial_gnomonic_pixels(event_data, lens_pto_data, gnomonic_pto_data):
    """Maps initial fisheye pixel coordinates to the gnomonic image plane."""
    start_px, start_py = map(float, event_data['begin'].split(','))
    end_px, end_py = map(float, event_data['end'].split(','))

    start_pano = pto_mapper.map_image_to_pano(lens_pto_data, 0, start_px, start_py)
    end_pano = pto_mapper.map_image_to_pano(lens_pto_data, 0, end_px, end_py)
    if not (start_pano and end_pano): return None
    
    start_az2, start_alt2 = start_pano
    end_az2, end_alt2 = end_pano

    start_res = pto_mapper.map_pano_to_image(gnomonic_pto_data, start_az2, start_alt2)
    end_res = pto_mapper.map_pano_to_image(gnomonic_pto_data, end_az2, end_alt2)
    if not (start_res and end_res): return None

    _, gnomonic_startx, gnomonic_starty = start_res
    _, gnomonic_endx, gnomonic_endy = end_res
    return gnomonic_startx, gnomonic_starty, gnomonic_endx, gnomonic_endy

def _refine_gnomonic_track(initial_pixels, gnomonic_image_path, frames_count=None):
    """Runs refinetrack.py to get more precise start/end points."""
    gnomonic_startx, gnomonic_starty, gnomonic_endx, gnomonic_endy = initial_pixels
    refine_cmd = [
        sys.executable, str(BIN_DIR / "refinetrack.py"),
        gnomonic_image_path,
        f"{gnomonic_startx},{gnomonic_starty}",
        f"{gnomonic_endx},{gnomonic_endy}"
    ]
    if frames_count and frames_count > 0:
        refine_cmd.append(f"--frames={frames_count}")

    print(f"Executing: {' '.join(refine_cmd)}")
    result = subprocess.run(refine_cmd, capture_output=True, text=True, check=True)
    
    output_parts = result.stdout.strip().split()
    coord_str = " ".join(output_parts[:2])
    point_coords = re.split(r'[,\s]+', coord_str)
    refined_pixels = tuple(map(float, point_coords[:4]))
    brightness_values = output_parts[2:] if len(output_parts) > 2 else []

    return refined_pixels, brightness_values


def _convert_refined_pixels_to_azalt(refined_pixels, gnomonic_pto_data, scale):
    """Maps refined gnomonic pixel coordinates back to az/alt values."""
    refined_start_x, refined_start_y, refined_end_x, refined_end_y = refined_pixels
    
    start_pano_coords = pto_mapper.map_image_to_pano(gnomonic_pto_data, 0, refined_start_x, refined_start_y)
    end_pano_coords = pto_mapper.map_image_to_pano(gnomonic_pto_data, 0, refined_end_x, refined_end_y)
    if not (start_pano_coords and end_pano_coords): return None

    corr_startaz = round(start_pano_coords[0] / scale, 2)
    corr_startalt = refraction(90 - (start_pano_coords[1] / scale))
    corr_endaz = round(end_pano_coords[0] / scale, 2)
    corr_endalt = refraction(90 - (end_pano_coords[1] / scale))
    return corr_startaz, corr_startalt, corr_endaz, corr_endalt

def calculate_refined_endpoints(event_data, filenames, verbose=False):
    """Calculates the refined start/end points of the meteor trail."""
    try:
        print("-> Calculating refined endpoints...")
        lens_pto_data = pto_mapper.parse_pto_file(filenames['lens_pto'])
        gnomonic_pto_data = pto_mapper.parse_pto_file(filenames['gnomonic_corr_grid_pto'])
        scale = lens_pto_data[0]['w'] / 360.0

        initial_gnomonic_pixels = _get_initial_gnomonic_pixels(event_data, lens_pto_data, gnomonic_pto_data)
        if not initial_gnomonic_pixels: return None

        gnomonic_image_path = filenames['gnomonic']
        if Path(f"{filenames['name']}-gnomonic-mask.jpg").exists():
            gnomonic_image_path = f"{filenames['name']}-gnomonic-mask.jpg"

        is_manual = event_data.get('manual', 0) != 0
        sun_is_high = event_data.get('sunalt', -99) > -8
        frames_count = event_data.get('frames')

        should_estimate_brightness = False
        if frames_count and not is_manual:
            existing_brightness = event_data.get('brightness_values')
            if not existing_brightness or len(set(existing_brightness)) == 1:
                should_estimate_brightness = True
        
        frames_to_pass = frames_count if should_estimate_brightness else None
        
        refined_pixels, brightness_values = None, []
        if is_manual or sun_is_high:
            if is_manual: print("   -> Skipping track refinement: 'manual' flag is set.")
            if sun_is_high: print(f"   -> Skipping track refinement: sun altitude ({event_data.get('sunalt')}°) is > -8°.")
            refined_pixels = initial_gnomonic_pixels
        else:
            print("   -> Running track refinement for greater precision.")
            refined_pixels, brightness_values = _refine_gnomonic_track(
                initial_gnomonic_pixels, gnomonic_image_path, frames_to_pass
            )
        
        if not refined_pixels: return None

        azalt = _convert_refined_pixels_to_azalt(refined_pixels, gnomonic_pto_data, scale)
        if not azalt: return None

        result_data = {'gnomonic_pixels': refined_pixels, 'azalt': azalt}

        if frames_to_pass and brightness_values:
            if len(brightness_values) == frames_to_pass:
                result_data['brightness'] = brightness_values
            else:
                print(f"Warning: Brightness update skipped.", file=sys.stderr)

        print("   ...Done: Calculating refined endpoints.")
        return result_data

    except Exception as e:
        print(f"Warning: Could not calculate refined endpoints: {e}", file=sys.stderr)
        if verbose: traceback.print_exc(file=sys.stderr)
        return None

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
            print(f"Warning: 'p' line not found in {input_path}.", file=sys.stderr)

        with open(output_path, 'w') as f_out:
            f_out.writelines(modified_lines)

    except IOError as e:
        print(f"Error modifying PTO file: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"   ...Done: {description}.")
    return output_path

def create_logo_overlay(width, height, logo_paths, output_path=None):
    """
    Creates a transparent image of (width, height) with NMN, SBSDNB, and AS7 logos 
    placed in the Top-Left, Top-Right, and Bottom-Right corners respectively.
    Returns the path to the saved overlay image.
    """
    if output_path is None:
        # Create a temp file if no path specified
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    try:
        # Create transparent base
        base_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        with Image.open(logo_paths['nmn']).convert("RGBA") as nmn_logo, \
             Image.open(logo_paths['sbsdnb']).convert("RGBA") as sbsdnb_logo, \
             Image.open(logo_paths['as7']).convert("RGBA") as as7_logo:

            # 1. NMN Logo (Top Left)
            base_img.paste(nmn_logo, (16, 16), nmn_logo)
            
            # 2. SBSDNB Logo (Top Right)
            sbsdnb_x = width - sbsdnb_logo.width - 16
            base_img.paste(sbsdnb_logo, (sbsdnb_x, 16), sbsdnb_logo)

            # 3. AS7 Logo (Lower Right)
            as7_x = width - as7_logo.width - 16
            as7_y = height - as7_logo.height - 16
            base_img.paste(as7_logo, (as7_x, as7_y), as7_logo)
            
        base_img.save(output_path)
        return output_path

    except Exception as e:
        print(f"Error creating logo overlay: {e}", file=sys.stderr)
        return None

def add_logos(base_image_path, output_path, logo_paths, verbose=False):
    """
    Convenience function to composite logos directly onto an image file.
    Uses create_logo_overlay internally for consistency.
    """
    description = f"Adding logos to {Path(output_path).name}"
    print(f"-> {description}...")
    
    try:
        with Image.open(base_image_path).convert("RGBA") as base:
            # Generate an overlay matching the base image dimensions
            overlay_path = f"{base_image_path}.logos.temp.png"
            create_logo_overlay(base.width, base.height, logo_paths, overlay_path)
            
            with Image.open(overlay_path).convert("RGBA") as overlay:
                 composited = Image.alpha_composite(base, overlay)
                 
                 ext = Path(output_path).suffix.lower()
                 if ext in ['.jpg', '.jpeg']:
                     composited.convert("RGB").save(output_path)
                 else:
                     composited.save(output_path)
            
            if os.path.exists(overlay_path):
                os.remove(overlay_path)

        print(f"   ...Done: {description}.")
    except Exception as e:
        print(f"Error adding logos: {e}", file=sys.stderr)
        if verbose: traceback.print_exc(file=sys.stderr)
    
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
    """
    Composites a pre-processed overlay onto a base image.
    IMPORTANT: Preserves Alpha channel if saving to PNG, enabling semi-transparent video overlays.
    """
    description = f"Creating composite image: {Path(output_path).name}"
    print(f"-> {description}...")
    if verbose:
        print(f"   Base: {base_path}, Overlay: {overlay_path}")

    with Image.open(base_path).convert("RGBA") as base, \
         Image.open(overlay_path).convert("RGBA") as overlay:
        if base.size != overlay.size:
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        
        composited = Image.alpha_composite(base, overlay)
        
        # Check output extension to determine mode
        ext = Path(output_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            # Drop alpha for JPEGs
            composited.convert("RGB").save(output_path)
        else:
            # Keep alpha channel for PNGs (Essential for transparent video overlays)
            composited.save(output_path)
            
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
    description = f"Creating video with overlay: {Path(output_path).name}"
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

def handle_hevc_transcoding(video_path, logo_overlay_path, verbose=False):
    """
    Checks if a video file uses the HEVC codec. If so, renames the original
    to '_hevc.mp4' and creates a new H.264 version with the original name.
    
    OPTIMIZATION: Applies the logo overlay during the transcode step to avoid 
    an extra encoding pass.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Warning: Video file {video_file} not found for HEVC check.", file=sys.stderr)
        return

    try:
        print(f"-> Checking video codec for {video_file.name}...")
        probe = ffmpeg.probe(str(video_file))
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)

        if video_stream and video_stream.get('codec_name') == 'hevc':
            print(f"   -> HEVC codec detected. Transcoding to H.264 (with overlay)...")

            # Define the new name for the original HEVC file
            hevc_file_path = video_file.with_name(f"{video_file.stem}_hevc.mp4")

            # Rename the original file
            print(f"      - Renaming {video_file.name} to {hevc_file_path.name}")
            shutil.move(str(video_file), str(hevc_file_path))

            # Transcode the renamed file back to the original filename with H.264
            # AND apply the overlay in the same pass
            print(f"      - Transcoding and watermarking {hevc_file_path.name} to {video_file.name} (H.264)...")
            try:
                input_video = ffmpeg.input(str(hevc_file_path))
                overlay_img = ffmpeg.input(logo_overlay_path)
                
                (
                    ffmpeg
                    .filter([input_video, overlay_img], 'overlay', 0, 0)
                    .output(str(video_file), vcodec='libx264', crf=23, preset='medium')
                    .overwrite_output()
                    .run(quiet=(not verbose), capture_stdout=True, capture_stderr=True)
                )
                print(f"   ...Done: Transcoding complete.")
            except ffmpeg.Error as e:
                print("\n--- ERROR during HEVC to H.264 transcoding ---", file=sys.stderr)
                print(f"FFmpeg stderr:\n{e.stderr.decode('utf8')}", file=sys.stderr)
                # Attempt to move the file back if transcoding fails
                print(f"      - Transcoding failed. Restoring original file name.", file=sys.stderr)
                shutil.move(str(hevc_file_path), str(video_file))

        else:
            print(f"   ...Done: Codec is not HEVC.")

    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf8') if e.stderr else "No stderr output"
        print(f"Warning: Could not probe video file {video_file.name}. FFmpeg error: {stderr_output}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during HEVC check/transcode: {e}", file=sys.stderr)


# --- Core Script Logic ---

def get_event_data(event_file):
    """Parses the event.txt file to extract key information."""
    data = {}
    in_trail_section = False
    try:
        with open(event_file, 'r') as f:
            for line in f:
                if line.strip() == '[trail]':
                    in_trail_section = True
                    continue
                elif line.strip().startswith('['):
                    in_trail_section = False

                if '=' not in line:
                    continue
                
                key, value = (part.strip() for part in line.strip().split('=', 1))
                parts = value.split()
                
                if in_trail_section:
                    if key == 'duration':
                        data['duration'] = int(float(value) + 0.5)
                    elif key == 'positions':
                        data['begin'], data['end'] = parts[0], parts[-1]
                    elif key == 'coordinates':
                        data['start_azalt'], data['end_azalt'] = parts[0], parts[-1]
                    elif key == 'frames':
                        data['frames'] = int(value)
                    elif key == 'brightness':
                        data['brightness_values'] = [float(b) for b in value.split()]

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

def update_event_file(event_data, filepath="event.txt"):
    """Updates event.txt with refined brightness data."""
    if 'brightness' not in event_data or not event_data['brightness']:
        return

    print("-> Updating event.txt with refined brightness data...")
    try:
        lines = Path(filepath).read_text().splitlines()
        new_lines = []
        in_trail_section = False
        was_updated = False

        for line in lines:
            if line.strip() == '[trail]':
                in_trail_section = True
            elif line.strip().startswith('['):
                in_trail_section = False

            key = line.split('=')[0].strip() if '=' in line else None
            
            if in_trail_section and key == 'brightness':
                brightness_str = " ".join(map(str, event_data['brightness']))
                new_lines.append(f"brightness = {brightness_str}")
                was_updated = True
            else:
                new_lines.append(line)

        if not was_updated:
             print(f"Warning: 'brightness' key not found in [trail] section of {filepath}. File not updated.", file=sys.stderr)
             return

        Path(filepath).write_text("\n".join(new_lines) + "\n")
        print(f"   ...Done: Updated brightness in {filepath}.")

    except FileNotFoundError:
        print(f"Error: Could not find {filepath} to update.", file=sys.stderr)
    except Exception as e:
        print(f"Error updating {filepath}: {e}", file=sys.stderr)

def run_command(command, description, verbose=False):
    """Executes a shell command with progress indication and error handling."""
    if not verbose:
        print(f"-> {description}...")
    
    do_capture = not verbose
    try:
        stdout_dest = subprocess.PIPE if do_capture else None
        stderr_dest = subprocess.PIPE if do_capture else None

        result = subprocess.run(
            command, 
            check=True, 
            shell=True, 
            stdout=stdout_dest,
            stderr=stderr_dest,
            text=True
        )

        if not verbose:
            print(f"   ...Done: {description}.")
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR executing external command ---\nCommand failed: {command}", file=sys.stderr)
        if do_capture:
            if e.stdout: print(f"Stdout:\n{e.stdout}", file=sys.stderr)
            if e.stderr: print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        raise

def _run_full_view_sequentially(event_data, filenames, tmpdir, logo_paths, verbose):
    """Runs the full view processing pipeline sequentially."""
    print("\n--- Processing Full View (Sequential) ---")
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    
    # 1. Generate Grid
    grid_labels_path = f"{tmpdir}/grid-labels.png"
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['lens_pto']} {grid_labels_path}"
    run_command(drawgrid_cmd, "Generating full view grid", verbose)

    # 2. Make grid transparent
    grid_labels_transparent_path = f"{tmpdir}/grid-labels-transparent.png"
    set_image_opacity(grid_labels_path, grid_labels_transparent_path, OVERLAY_OPACITY, verbose=verbose)
    
    # 3. Create Logo Overlay Layer (Dynamic Size based on video)
    # Get actual video resolution
    vw, vh = get_video_resolution(filenames['full'])
    logo_layer_path = f"{tmpdir}/logo_layer_full.png"
    create_logo_overlay(vw, vh, logo_paths, logo_layer_path)

    # 4. Composite Logos onto the Transparent Grid
    # This ensures mp4grid gets both grid and logos
    final_overlay_path = f"{tmpdir}/grid_plus_logos.png"
    alpha_composite_images(grid_labels_transparent_path, logo_layer_path, final_overlay_path, verbose=verbose)

    print("-> Creating final full view images and videos...")
    
    # --- Create [prefix]-orig.mp4 (Original Video + Logos) ---
    # overlay_video_with_image defaults to libx264
    overlay_video_with_image(filenames['full'], logo_layer_path, filenames['orig'], verbose=verbose)

    # 5. Composite Final Overlay (Grid+Logos) onto the clean stacked JPG to create [prefix]-grid.jpg
    alpha_composite_images(filenames['jpg'], final_overlay_path, filenames['jpggrid'], verbose=verbose)
    
    # 6. Overlay Final Overlay (Grid+Logos) onto the video to create [prefix]-grid.mp4
    overlay_video_with_image(filenames['full'], final_overlay_path, filenames['mp4grid'], verbose=verbose)

def _run_full_view_in_parallel(event_data, filenames, tmpdir, logo_paths, verbose, executor, future_stacked_jpg: Future):
    """Runs the full view processing pipeline in parallel."""
    print("\n--- Processing Full View ---")
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    
    # 1. Grid Generation
    grid_labels_path = f"{tmpdir}/grid-labels.png"
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -d {ts2} {filenames['lens_pto']} {grid_labels_path}"
    future_grid_png = executor.submit(run_command, drawgrid_cmd, "Generating full view grid", verbose)

    # 2. Transparency
    future_transparent_grid = executor.submit(lambda: set_image_opacity(future_grid_png.result() and grid_labels_path, f"{tmpdir}/grid-labels-transparent.png", OVERLAY_OPACITY, verbose))
    
    # 3. Create Logo Layer (Dynamic Size)
    # Determine video resolution (Synchronous is fine, it's fast)
    vw, vh = get_video_resolution(filenames['full'])
    logo_layer_path = f"{tmpdir}/logo_layer_full.png"
    future_logo_layer = executor.submit(create_logo_overlay, vw, vh, logo_paths, logo_layer_path)

    # 4. Composite Logos onto Transparent Grid
    final_overlay_path = f"{tmpdir}/grid_plus_logos.png"
    future_final_overlay = executor.submit(lambda: alpha_composite_images(future_transparent_grid.result(), future_logo_layer.result(), final_overlay_path, verbose))

    print("-> Submitting final full view image and video tasks...")
    
    # --- Create [prefix]-orig.mp4 (Original Video + Logos) ---
    # We use the future_logo_layer result directly on the original video
    future_orig_mp4 = executor.submit(lambda: overlay_video_with_image(filenames['full'], future_logo_layer.result(), filenames['orig'], verbose))

    # 5. Create [prefix]-grid.jpg (Clean JPG + GridWithLogos)
    future_jpg_grid = executor.submit(lambda: alpha_composite_images(future_stacked_jpg.result() and filenames['jpg'], future_final_overlay.result(), filenames['jpggrid'], verbose))
    
    # 6. Create [prefix]-grid.mp4 (Full Video + GridWithLogos)
    future_mp4_grid = executor.submit(lambda: overlay_video_with_image(filenames['full'], future_final_overlay.result(), filenames['mp4grid'], verbose))

    # Add future_orig_mp4 to the completion wait list
    for future in as_completed([future_jpg_grid, future_mp4_grid, future_orig_mp4]):
        future.result()

# --- Gnomonic View Processing Helpers ---

def generate_gnomonic_projection(event_data, filenames, tmpdir, verbose, stacked_jpg_path, padding_value=1024):
    """Generates the base gnomonic projection image (clean)."""
    azalt_start = event_data.get('start_azalt')
    azalt_end = event_data.get('end_azalt')
    if not azalt_start or not azalt_end:
        print("Warning: Skipping gnomonic projection generation: 'coordinates' not found in event.txt.", file=sys.stderr)
        return None

    reproject_cmd = (
        f"{sys.executable} {BIN_DIR}/reproject.py -f 45 --width 1920 --height 2560 "
        f"-o {filenames['gnomonic_pto']} -g {filenames['gnomonic_grid_pto']} "
        f"-e {azalt_end} {filenames['lens_pto']} {azalt_start}"
    )
    run_command(reproject_cmd, "Reprojecting for gnomonic view", verbose)

    tmp_gnomonic_jpg = f"{tmpdir}/{filenames['name']}-gnomonic-tmp.png"
    stitcher_cmd_jpg = (f"{sys.executable} {BIN_DIR}/stitcher.py --pad {padding_value} "
                        f"{filenames['gnomonic_pto']} {filenames['jpg']} {tmp_gnomonic_jpg}")
    run_command(stitcher_cmd_jpg, "Stitching gnomonic JPG", verbose)
    
    shutil.move(tmp_gnomonic_jpg, filenames['gnomonic'])
    
    print(f"   ...Done: Generated base gnomonic image '{filenames['gnomonic']}'.")
    return filenames['gnomonic']

def recalibrate_gnomonic_view(event_data, filenames, verbose):
    """Recalibrates the gnomonic view if necessary, with a fallback."""
    if event_data.get('recalibrate', False):
        try:
            recalibrate_cmd = (f"{sys.executable} {BIN_DIR}/recalibrate.py -c meteor.cfg "
                               f"{event_data['timestamp'] + event_data['duration'] // 2} "
                               f"{filenames['gnomonic_grid_pto']} {filenames['gnomonic']} "
                               f"{filenames['gnomonic_corr_grid_pto']}")
            run_command(recalibrate_cmd, "Recalibrating gnomonic view", verbose)
        except subprocess.CalledProcessError as e:
            print(f"Warning: recalibrate.py failed. Using uncorrected grid PTO.", file=sys.stderr)
            if verbose:
                print(f"Stderr from recalibrate.py:\n{e.stderr}", file=sys.stderr)
            shutil.copy(filenames['gnomonic_grid_pto'], filenames['gnomonic_corr_grid_pto'])
    else:
        shutil.copy(filenames['gnomonic_grid_pto'], filenames['gnomonic_corr_grid_pto'])

def _generate_decorated_grid(event_data, filenames, refined_data, verbose):
    """Generates the grid, and draws labels and marker crosses."""
    grid_path = filenames['gnomonic_corr_grid_png']
    ts2 = event_data['timestamp'] + event_data['duration'] // 2
    
    # Calculate Magnitude Limit based on FOV
    mag_limit = get_dynamic_mag_limit(filenames['gnomonic_corr_grid_pto'])
    
    # Generate base grid using dynamic magnitude limit
    drawgrid_cmd = f"{sys.executable} {BIN_DIR}/drawgrid.py -c meteor.cfg -f {mag_limit} -d {ts2} {filenames['gnomonic_corr_grid_pto']} {grid_path}"
    run_command(drawgrid_cmd, "Generating gnomonic grid", verbose)
    
    if refined_data and 'gnomonic_pixels' in refined_data and 'azalt' in refined_data:
        draw_marker_crosses(grid_path, refined_data['gnomonic_pixels'], refined_data['azalt'], verbose=verbose)

    draw_text_on_image(grid_path, event_data['label'], grid_path, verbose=verbose)
    return grid_path

def _create_gnomonic_grid_and_image(event_data, filenames, tmpdir, logo_paths, verbose):
    """
    Creates the gnomonic grid, applies logos, and produces the final composite image.
    Handles the difference between Tall JPG projection and Cropped Video projection.
    """
    recalibrate_gnomonic_view(event_data, filenames, verbose)
    
    refined_data = calculate_refined_endpoints(event_data, filenames, verbose)
    if refined_data:
        event_data['refined_coords'] = refined_data.get('azalt')
        if refined_data.get('brightness'):
            event_data['brightness'] = refined_data.get('brightness')

    # 1. Generate Base Grid (Tall) with dynamic star labels
    grid_path = _generate_decorated_grid(event_data, filenames, refined_data, verbose)
    
    # 2. Make Grid Transparent
    gnomonic_grid_transparent = f"{tmpdir}/gnomonic_grid_transparent.png"
    set_image_opacity(grid_path, gnomonic_grid_transparent, OVERLAY_OPACITY, verbose=verbose)
    
    # --- PROJECTION IMAGE CREATION (TALL 1920x2560) ---
    # Create a tall logo layer for the static JPG
    logo_layer_tall = f"{tmpdir}/logo_layer_tall.png"
    # Using 2560 height to place logos at the very top and very bottom of the strip
    create_logo_overlay(1920, 2560, logo_paths, logo_layer_tall)
    
    # Composite Tall Grid + Tall Logos
    overlay_tall = f"{tmpdir}/overlay_tall.png"
    alpha_composite_images(gnomonic_grid_transparent, logo_layer_tall, overlay_tall, verbose=verbose)
    
    # Create final [prefix]-gnomonic-grid.jpg
    alpha_composite_images(filenames['gnomonic'], overlay_tall, filenames['gnomonicgrid'], verbose=verbose)


    # --- VIDEO GRID OVERLAY PREPARATION (CROPPED 1920x1080) ---
    # Crop the TRANSPARENT GRID (without logos yet) to 1080p
    cropped_grid_clean = f"{tmpdir}/gnomonic_grid_cropped_clean.png"
    crop_image(gnomonic_grid_transparent, cropped_grid_clean, crop_box=(0, 740, 1920, 740 + 1080), verbose=verbose)
    
    # We return the CLEAN cropped grid.
    # Why? Because we will overlay logos onto the BASE gnomonic video separately.
    # If we add logos to the grid here, we'd get double logos (ghosting).
    return {"cropped_grid": cropped_grid_clean}


def _run_gnomonic_view_in_parallel(event_data, filenames, tmpdir, logo_paths, verbose, executor, future_stacked_jpg: Future):
    """Runs the gnomonic processing by splitting it into parallel video and image/grid pipelines."""
    print("\n--- Processing Gnomonic View ---")
    azalt_start, azalt_end = event_data.get('start_azalt'), event_data.get('end_azalt')
    if not azalt_start or not azalt_end:
        print("Skipping gnomonic view: 'coordinates' not found in event.txt.")
        return

    # 1. Generate the base gnomonic projection image.
    stacked_jpg_path = future_stacked_jpg.result() and filenames['jpg']
    future_gnomonic_base_image = executor.submit(generate_gnomonic_projection, 
                                                 event_data, filenames, tmpdir, verbose, stacked_jpg_path)
    future_gnomonic_base_image.result() 

    # 2. GNOMONIC VIDEO PIPELINE
    future_modified_pto = executor.submit(modify_pto_canvas, filenames['gnomonic_pto'], 
                                          filenames['gnomonic_mp4_pto'], 1920, 1080, verbose)
    
    # Create raw stitch
    raw_gnomonic_mp4 = f"{tmpdir}/raw_gnomonic.mp4"
    stitch_cmd_mp4 = f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 {filenames['gnomonic_mp4_pto']} {filenames['full']} {raw_gnomonic_mp4}"
    future_stitch = executor.submit(lambda: run_command(future_modified_pto.result() and stitch_cmd_mp4, "Creating gnomonic video", verbose))

    # Create 1080p Logo Layer
    logo_layer_1080p = f"{tmpdir}/logo_layer_1080p.png"
    # Ensure this exists (might have been created in full view, but parallel safety says create again or check)
    create_logo_overlay(1920, 1080, logo_paths, logo_layer_1080p)

    # 3. GNOMONIC IMAGE/GRID PIPELINE
    future_grid_assets = executor.submit(_create_gnomonic_grid_and_image, event_data, filenames, tmpdir, logo_paths, verbose)

    # 4. FINAL ASSEMBLY
    grid_assets = future_grid_assets.result() # Contains CLEAN cropped grid
    future_stitch.result() # Wait for raw stitching
    
    # A. Watermark the Base Gnomonic Video -> [prefix]-gnomonic.mp4
    overlay_video_with_image(raw_gnomonic_mp4, logo_layer_1080p, filenames['gnomonicmp4'], verbose)
    
    # B. Create Grid Video: Overlay CLEAN grid onto WATERMARKED base video -> [prefix]-gnomonic-grid.mp4
    # This prevents double logos (ghosting) while ensuring correct placement.
    overlay_video_with_image(filenames['gnomonicmp4'], grid_assets["cropped_grid"], filenames['gnomonicgridmp4'], verbose)

def _run_gnomonic_view_sequentially(event_data, filenames, tmpdir, logo_paths, verbose):
    """Runs the entire gnomonic processing pipeline sequentially."""
    print("\n--- Processing Gnomonic View (Sequential) ---")
    azalt_start, azalt_end = event_data.get('start_azalt'), event_data.get('end_azalt')
    if not azalt_start or not azalt_end: return

    # 1. Generate base gnomonic image (clean).
    generate_gnomonic_projection(event_data, filenames, tmpdir, verbose, filenames['jpg'])

    # 2. Create the decorated grid (returns clean cropped grid)
    grid_assets = _create_gnomonic_grid_and_image(event_data, filenames, tmpdir, logo_paths, verbose)

    # 3. Create Gnomonic Video
    print("-> Creating final gnomonic video...")
    modify_pto_canvas(filenames['gnomonic_pto'], filenames['gnomonic_mp4_pto'], 1920, 1080, verbose=verbose)
    
    raw_gnomonic_mp4 = f"{tmpdir}/raw_gnomonic.mp4"
    stitch_cmd_mp4 = f"{sys.executable} {BIN_DIR}/stitcher.py --pad 0 {filenames['gnomonic_mp4_pto']} {filenames['full']} {raw_gnomonic_mp4}"
    run_command(stitch_cmd_mp4, "Creating gnomonic video", verbose)
    
    # Prepare Logo Layer
    logo_layer_1080p = f"{tmpdir}/logo_layer_1080p.png"
    if not Path(logo_layer_1080p).exists():
        create_logo_overlay(1920, 1080, logo_paths, logo_layer_1080p)

    # A. Watermark Base Video
    overlay_video_with_image(raw_gnomonic_mp4, logo_layer_1080p, filenames['gnomonicmp4'], verbose)
    
    # B. Overlay Grid (Clean) on Watermarked Video
    overlay_video_with_image(filenames['gnomonicmp4'], grid_assets["cropped_grid"], filenames['gnomonicgridmp4'], verbose=verbose)

# --- Client Mode Functions ---

def search_for_videos(video_dir, start_unix):
    """Searches for three consecutive one-minute video files."""
    found_files = []
    for i in [-1, 0, 1]:
        found_file = None
        base_time = start_unix + (i * 60)
        for offset in [0, -60, 60]:
            search_time = base_time + offset
            dt_obj = datetime.datetime.fromtimestamp(search_time, tz=datetime.timezone.utc)
            file_path = Path(video_dir) / dt_obj.strftime('%Y%m%d/%H') / f"full_{dt_obj.strftime('%M')}.mp4"
            if file_path.exists():
                found_file = file_path
                break
        found_files.append(found_file)
    return found_files

def run_client_mode(output_name, video_dir, start_unix, length_sec, verbose):
    """Runs the script in client mode."""
    print("--- Running in Client Mode ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory {tmpdir}")

        dest_conf = Path("metdetect.conf")
        if not dest_conf.exists():
            src_conf = Path(video_dir) / "metdetect.conf"
            if src_conf.exists():
                try:
                    shutil.copy(src_conf, dest_conf)
                    print(f"Copied {src_conf} to current directory.")
                except shutil.SameFileError: pass
                except shutil.Error as e: print(f"Error copying metdetect.conf: {e}", file=sys.stderr)
            else:
                 print("Warning: metdetect.conf not found.", file=sys.stderr)

        meteor_cfg_src = Path("/etc/meteor.cfg")
        if meteor_cfg_src.exists(): shutil.copy(meteor_cfg_src, ".")
        else: print("Warning: /etc/meteor.cfg not found.", file=sys.stderr)

        full1, full2, full3 = search_for_videos(video_dir, start_unix)
        video_files = [full1, full2, full3]
        if not all(video_files):
            print("Error: One or more source videos not found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found source videos:\n- {full1}\n- {full2}\n- {full3}")
        
        ts = 0
        try:
            probe = ffmpeg.probe(str(full1))
            rate_str = next((s['r_frame_rate'] for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if rate_str is None: raise ValueError("No video stream found")
            
            creation_time_str = probe.get('format', {}).get('tags', {}).get('creation_time')
            if creation_time_str:
                dt_obj = datetime.datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))
                ts = int(dt_obj.timestamp())
        except (ffmpeg.Error, ValueError) as e:
            print(f"Could not probe {full1} for metadata: {e}", file=sys.stderr)

        if ts == 0:
            print("Warning: Could not get timestamp from video metadata, falling back to filename parsing.")
            match = re.search(r'/(\d{8})/(\d{2})/full_(\d{2})\.mp4', str(full1))
            if match:
                date_str = f"{match.group(1)} {match.group(2)}:{match.group(3)}:00"
                dt_obj = datetime.datetime.strptime(date_str, "%Y%m%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
                ts = int(dt_obj.timestamp())
            else:
                print("Error: Could not determine timestamp. Exiting.", file=sys.stderr)
                sys.exit(1)
        
        skip_sec = start_unix - ts - 10
        len_padded = length_sec + 14
        if skip_sec < 0:
            print(f"Warning: Calculated skip time is negative ({skip_sec}s).", file=sys.stderr)
            skip_sec = max(0, skip_sec)

        full_mp4 = f"{output_name}.mp4"
        ts_clip_start = ts + skip_sec
        ts_clip_end = ts_clip_start + len_padded
        start_hms = datetime.datetime.fromtimestamp(ts_clip_start, tz=datetime.timezone.utc).strftime('%H:%M:%S')
        end_hms = datetime.datetime.fromtimestamp(ts_clip_end, tz=datetime.timezone.utc).strftime('%H:%M:%S')
        
        print(f"Generating {full_mp4} ... {start_hms} - {end_hms} ... ", end="", flush=True)
        
        filelist_path = Path(tmpdir) / "filelist.txt"
        with open(filelist_path, "w") as f:
            for vfile in video_files:
                f.write(f"file '{vfile.resolve()}'\n")

        try:
            (
                ffmpeg
                .input(str(filelist_path), format='concat', safe=0)
                .output(full_mp4, c='copy', ss=skip_sec, t=len_padded)
                .overwrite_output()
                .run(quiet=(not verbose), capture_stdout=True, capture_stderr=True)
            )
            print("done")
        except ffmpeg.Error as e:
            print("\n--- ERROR during ffmpeg-python execution ---", file=sys.stderr)
            print(f"FFmpeg stderr:\n{e.stderr.decode('utf8')}", file=sys.stderr)
            sys.exit(1)
        
        day_str = datetime.datetime.fromtimestamp(start_unix, tz=datetime.timezone.utc).strftime('%Y%m%d')
        
        def find_and_copy_latest_file(pattern, dest_name, event_day_str):
            base_dir = Path(video_dir)
            latest_file_to_copy = base_dir / dest_name
            files = sorted(base_dir.glob(pattern))
            for f in files:
                match = re.search(r'-(\d{8})\.', f.name)
                if match and match.group(1) <= event_day_str:
                    latest_file_to_copy = f
                else: break 
            if latest_file_to_copy.exists():
                shutil.copy(latest_file_to_copy, dest_name)
                return latest_file_to_copy
            return None

        find_and_copy_latest_file("grid-*.png", "grid.png", day_str)
        find_and_copy_latest_file("lens-*.pto", "lens.pto", day_str)
        find_and_copy_latest_file("mask-*.jpg", "mask.jpg", day_str)

        print("\n--- Generating images for cropping ---")
        
        filenames = {
            'name': output_name, 'full': f"{output_name}.mp4", 'jpg': f"{output_name}.jpg",
            'gnomonic': f"{output_name}-gnomonic.jpg", 'gnomonic_pto': 'gnomonic.pto',
            'gnomonic_grid_pto': 'gnomonic_grid.pto', 'lens_pto': 'lens.pto',
            'gnomonic_corr_grid_pto': 'gnomonic_corr_grid.pto'
        }
        
        with open(f"{tmpdir}/nmn.png", "wb") as f: f.write(base64.b64decode(NMN_LOGO_B64))
        with open(f"{tmpdir}/sbsdnb.png", "wb") as f: f.write(base64.b64decode(SBSDNB_LOGO_B64))
        with open(f"{tmpdir}/as7.png", "wb") as f: f.write(base64.b64decode(AS7_LOGO_B64))

        logo_paths = {'nmn': f"{tmpdir}/nmn.png", 'sbsdnb': f"{tmpdir}/sbsdnb.png", 'as7': f"{tmpdir}/as7.png"}

        stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
        run_command(stack_cmd, "Stacking video frames to create JPG", verbose)
        
        event_data = get_event_data("event.txt")
        event_data['recalibrate'] = event_data.get('manual', 0) == 0 and event_data.get('sunalt', 0) < -9

        gnomonic_image_path = generate_gnomonic_projection(
            event_data, filenames, tmpdir, verbose, filenames['jpg'], padding_value=32
        )

        if gnomonic_image_path:
            recalibrate_gnomonic_view(event_data, filenames, verbose)
            
            # Run crop on CLEAN images
            meteorcrop_cmd = f"{sys.executable} {BIN_DIR / 'meteorcrop.py'} ."
            run_command(meteorcrop_cmd, "Cropping meteor track to create fireball.jpg", verbose)
            
            # Watermark the Gnomonic Image
            add_logos(filenames['gnomonic'], filenames['gnomonic'], logo_paths, verbose)

        # Watermark the Stacked Image
        add_logos(filenames['jpg'], filenames['jpg'], logo_paths, verbose)

        try:
            event_data = get_event_data("event.txt")
            start_coords = event_data.get('start_azalt', 'N/A,N/A').replace(',', ' ')
            end_coords = event_data.get('end_azalt', 'N/A,N/A').replace(',', ' ')
            print(f"Start: {start_coords}  End: {end_coords}")
        except SystemExit:
            print("Could not read event.txt to report coordinates.")

def main(args):
    """Main function to orchestrate the video processing pipeline."""
    if args.client:
        run_client_mode(args.file_prefix, args.video_dir, args.start, args.length, args.verbose)
        return
        
    if args.video_dir or args.start is not None or args.length is not None:
        print("Warning: --video-dir, --start, and --length are only used with the --client flag. They will be ignored.", file=sys.stderr)

    print(f"--- Initializing Full Video Pipeline ---")
    if args.verbose: print("--- Verbose mode enabled ---")
    if args.nothreads: print("--- Multithreading disabled ---")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/nmn.png", "wb") as f: f.write(base64.b64decode(NMN_LOGO_B64))
        with open(f"{tmpdir}/sbsdnb.png", "wb") as f: f.write(base64.b64decode(SBSDNB_LOGO_B64))
        with open(f"{tmpdir}/as7.png", "wb") as f: f.write(base64.b64decode(AS7_LOGO_B64))

        logo_paths = {'nmn': f"{tmpdir}/nmn.png", 'sbsdnb': f"{tmpdir}/sbsdnb.png", 'as7': f"{tmpdir}/as7.png"}

        event_data = get_event_data('event.txt')
        name = args.file_prefix
        
        filenames = {
            'name': name, 
            'full': f"{name}.mp4", 
            'orig': f"{name}-orig.mp4",
            'jpg': f"{name}.jpg", 
            'jpggrid': f"{name}-grid.jpg",
            'mp4grid': f"{name}-grid.mp4", 
            'gnomonic': f"{name}-gnomonic.jpg", 
            'gnomonicgrid': f"{name}-gnomonic-grid.jpg",
            'gnomonicmp4': f"{name}-gnomonic.mp4", 
            'gnomonicgridmp4': f"{name}-gnomonic-grid.mp4",
            'gnomonic_pto': 'gnomonic.pto', 
            'gnomonic_grid_pto': 'gnomonic_grid.pto',
            'gnomonic_corr_grid_pto': 'gnomonic_corr_grid.pto', 
            'gnomonic_corr_grid_png': 'gnomonic_corr_grid.png',
            'gnomonic_mp4_pto': 'gnomonic_mp4.pto', 
            'lens_pto': 'lens.pto'
        }

        event_data['recalibrate'] = event_data.get('manual', 0) == 0 and event_data.get('sunalt', 0) < -9
        pos_label = f"{event_data.get('latitude', 0):.4f}N {event_data.get('longitude', 0):.4f}E"
        event_data['label'] = f"{event_data.get('clock', '')}\n{pos_label}"
        gnomonic_enabled = 'begin' in event_data and 'start_azalt' in event_data

        if args.nothreads:
            stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
            run_command(stack_cmd, "Stacking video frames to create JPG", args.verbose)
            _run_full_view_sequentially(event_data, filenames, tmpdir, logo_paths, args.verbose)
            if gnomonic_enabled:
                _run_gnomonic_view_sequentially(event_data, filenames, tmpdir, logo_paths, args.verbose)
        else:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                stack_cmd = f"{sys.executable} {BIN_DIR}/stack.py --output {filenames['jpg']} {filenames['full']}"
                future_stacked_jpg = executor.submit(run_command, stack_cmd, "Stacking video frames to create JPG", args.verbose)

                future_full_view = executor.submit(_run_full_view_in_parallel, event_data, filenames, tmpdir, logo_paths, args.verbose, executor, future_stacked_jpg)
                
                if gnomonic_enabled:
                    future_gnomonic = executor.submit(_run_gnomonic_view_in_parallel, event_data, filenames, tmpdir, logo_paths, args.verbose, executor, future_stacked_jpg)
                    future_gnomonic.result()
                else:
                    print("\nSkipping gnomonic view: requires 'positions' and 'coordinates' in event.txt.")

                future_full_view.result()

        # Watermark the base images (clean stacked and gnomonic) at the end
        add_logos(filenames['jpg'], filenames['jpg'], logo_paths, verbose=args.verbose)
        if gnomonic_enabled and Path(filenames['gnomonic']).exists():
             add_logos(filenames['gnomonic'], filenames['gnomonic'], logo_paths, verbose=args.verbose)


        # --- HEVC Transcoding Step ---
        print("\n--- Finalizing Videos ---")
        
        # We create a 1080p overlay specifically for the HEVC transcode process
        with tempfile.TemporaryDirectory() as hevc_tmp:
            # Determine actual video resolution for the overlay
            vw, vh = get_video_resolution(filenames['full'])
            hevc_overlay_path = f"{hevc_tmp}/hevc_overlay.png"
            create_logo_overlay(vw, vh, logo_paths, hevc_overlay_path)
            handle_hevc_transcoding(filenames['full'], hevc_overlay_path, args.verbose)

    # Update event.txt before finishing
    update_event_file(event_data)

    print("\n--- Pipeline Finished ---")

    if event_data.get('refined_coords'):
        s_az, s_alt, e_az, e_alt = event_data['refined_coords']
        print(f"Start: {s_az:.2f} {s_alt:.2f}  End: {e_az:.2f} {e_alt:.2f}")


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH: return False
        elif err.errno == errno.EPERM: return True
        else: raise
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes meteor observation videos.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("file_prefix", help="The base name for input/output files.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed info.")
    
    client_group = parser.add_argument_group('Client Mode')
    client_group.add_argument("--client", action="store_true", help="Run in client mode.")
    client_group.add_argument("--video-dir", help="[Client mode] Source video directory.")
    client_group.add_argument("--start", type=int, help="[Client mode] Start timestamp.")
    client_group.add_argument("--length", type=int, help="[Client mode] Duration in seconds.")

    full_group = parser.add_argument_group('Full Pipeline Mode')
    full_group.add_argument("--nothreads", action="store_true", help="Disable multithreading.")

    args = parser.parse_args()

    if args.client and (args.video_dir is None or args.start is None or args.length is None):
        parser.error("--client mode requires --video-dir, --start, and --length.")

    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    lockfile = Path("processing.lock")

    if lockfile.exists():
        try:
            pid = int(lockfile.read_text())
            if check_pid(pid):
                print(f"Error: Process {pid} is still running.", file=sys.stderr)
                sys.exit(1)
            else:
                lockfile.unlink()
        except (ValueError, FileNotFoundError):
            lockfile.unlink(missing_ok=True)

    # --- TERMINAL STATE SAVING ---
    original_tty_settings = None
    fd = sys.stdin.fileno()
    if sys.stdin.isatty():
        try:
            original_tty_settings = termios.tcgetattr(fd)
        except termios.error:
            pass

    try:
        lockfile.write_text(str(os.getpid()))
        main(args)
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        if lockfile.exists() and lockfile.read_text() == str(os.getpid()):
            lockfile.unlink()
        
        # --- TERMINAL STATE RESTORATION ---
        if original_tty_settings:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, original_tty_settings)
            except termios.error:
                pass
            # Explicitly verify echo is on (extra safety)
            try:
                subprocess.run(['stty', 'echo'], check=False, stderr=subprocess.DEVNULL)
            except Exception:
                pass
