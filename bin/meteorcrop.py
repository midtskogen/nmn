#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts a normalized image and video of a meteor track from a gnomonic projection.

This script leverages the rotation and cropping features of the stitcher to
extract a meteor track with maximum efficiency.

A single, precise PTO file is generated to command the stitcher to reproject the
source media (image and video) directly into the final rotated and cropped geometry.

A reusable background plate is generated once from the first frame of the source
video. This plate is used for background subtraction in both the image and video
processing steps to ensure consistency and improve performance.
"""

import argparse
import configparser
import ctypes
import math
import sys
import traceback
import subprocess
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Tuple, List, Optional
import re
import shutil

import numpy as np

# Third-party libraries must be installed (e.g., pip install Wand tqdm opencv-python-headless).
# The pto_mapper.py and stitcher.py scripts should be in the same directory or Python path.
try:
    import pto_mapper
    from wand.api import library
    from wand.color import Color
    from wand.image import Image
    from tqdm import tqdm
    import cv2
except ImportError as e:
    print(f"Error: A required library is missing. {e}", file=sys.stderr)
    print("Please install required packages using: pip install Wand tqdm opencv-python-headless", file=sys.stderr)
    sys.exit(1)


# --- Custom Exception Classes for Robust Error Handling ---

class ScriptError(Exception):
    """Base class for custom exceptions in this script."""
    pass

class FileNotFoundError(ScriptError):
    """Raised when a required file (config, pto, image) is not found."""
    pass

class ConfigError(ScriptError):
    """Raised when there's an error reading required data from event.txt."""
    pass

class ProjectionError(ScriptError):
    """Raised when coordinate projection fails."""
    pass


# --- Custom Motion Blur Binding for Wand ---
library.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,  # angle
)

class MotionBlurImage(Image):
    """A Wand Image subclass with a custom motion_blur method."""
    def motion_blur(self, radius: float = 0.0, sigma: float = 0.0, angle: float = 0.0):
        library.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class Settings:
    """Configuration constants for the script."""
    TRACK_WIDTH = 128
    OUTPUT_FILENAME = "fireball.jpg"
    OUTPUT_VIDEO_FILENAME = "fireball.mp4"
    OUTPUT_ORIG_FILENAME = "fireball_orig.jpg"
    OUTPUT_ORIG_VIDEO_FILENAME = "fireball_orig.mp4"


    class VideoTrim:
        """Parameters for detecting the meteor event to trim the video."""
        ENABLED = True
        BASELINE_FRAMES = 30
        NOISE_RANGE_FACTOR = 2
        MIN_EVENT_DURATION_S = 0.2
        PADDING_S = 0.5


def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extracts a meteor track from a gnomonic image and video."
    )
    parser.add_argument(
        "event_dir",
        type=Path,
        help="Path to the directory containing the event.txt file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video", "both"],
        default="image",
        help="Specify the output mode: 'image', 'video', or 'both'. Defaults to 'image'.",
    )
    return parser.parse_args()


# --- Helper functions for progress indicators ---

def get_video_info(video_path: Path) -> Tuple[float, float, int]:
    """Uses ffprobe to get duration (s), frame rate (fps), and total frames."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=r_frame_rate,duration,nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    
    duration_str, fps_str, frames_str = '0', '25/1', '0'

    for line in lines:
        if '/' in line: fps_str = line
        elif '.' in line: duration_str = line
        elif line.isdigit(): frames_str = line

    num, den = map(float, fps_str.split('/'))
    frame_rate = num / den if den != 0 else 0
            
    return float(duration_str), frame_rate, int(frames_str)


def run_command_with_progress(cmd: List[str], desc: str, total: float):
    """
    Runs a command and displays a tqdm progress bar by parsing its stderr.
    Handles both ffmpeg (time=) and stitcher (PROGRESS:) formats.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    is_ffmpeg = 'ffmpeg' in cmd[0]
    unit = 's' if is_ffmpeg else '%'
    bar_format = '{l_bar}{bar}| {n:.2f}/{total:.2f}s' if is_ffmpeg else '{l_bar}{bar}| {n:.1f}/{total:.1f}%'
    time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
    stitcher_pattern = re.compile(r"PROGRESS:(\d+\.?\d*)")

    stderr_lines = []
    with tqdm(total=round(total, 2), desc=desc, unit=unit, bar_format=bar_format) as pbar:
        last_update = 0.0
        for line in iter(process.stderr.readline, ''):
            stderr_lines.append(line) # Store stderr for error reporting
            current_update = None
            if is_ffmpeg:
                match = time_pattern.search(line)
                if match:
                    h, m, s, hs = map(int, match.groups())
                    elapsed = h * 3600 + m * 60 + s + hs / 100
                    current_update = min(elapsed, total)
            else:
                match = stitcher_pattern.search(line)
                if match:
                    current_update = float(match.group(1))
            
            if current_update is not None and current_update > last_update:
                increment = current_update - last_update
                pbar.update(increment)
                last_update = current_update

        if pbar.n < total: pbar.update(total - pbar.n)
    
    stdout, _ = process.communicate() # Get stdout, stderr is already read
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, output=stdout, stderr="".join(stderr_lines)
        )


def get_projection_coords(event_dir: Path, config: configparser.ConfigParser) -> Tuple[List[float], List[float]]:
    """Uses pto_mapper to transform celestial coordinates (az/alt) to pixels."""
    try:
        recalibrated = config.getint('summary', 'recalibrated', fallback=1)
        start_pos_str = config.get('summary', 'startpos')
        end_pos_str = config.get('summary', 'endpos')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ConfigError(f"Missing required data in 'event.txt': {e}") from e

    pto_filename = 'gnomonic_grid.pto' if recalibrated != 0 else 'gnomonic_corr_grid.pto'
    pto_file = event_dir / pto_filename
    if not pto_file.is_file():
        raise FileNotFoundError(f"Projection file not found at '{pto_file}'")

    try:
        pto_data = pto_mapper.parse_pto_file(str(pto_file))
        global_options, _ = pto_data
    except Exception as e:
        raise ProjectionError(f"Error parsing PTO file '{pto_file}': {e}") from e

    pano_w, pano_h = global_options.get('w'), global_options.get('h')
    if not pano_w or not pano_h:
        raise ProjectionError("PTO 'p' line must contain 'w' and 'h' parameters.")

    start_az, start_alt = map(float, start_pos_str.split())
    end_az, end_alt = map(float, end_pos_str.split())
    
    start_pano_x = start_az * pano_w / 360.0
    start_pano_y = (90.0 - start_alt) * pano_h / 180.0
    
    end_pano_x = end_az * pano_w / 360.0
    end_pano_y = (90.0 - end_alt) * pano_h / 180.0

    start_result = pto_mapper.map_pano_to_image(pto_data, start_pano_x, start_pano_y)
    end_result = pto_mapper.map_pano_to_image(pto_data, end_pano_x, end_pano_y)

    if not start_result: raise ProjectionError("Could not map start coordinates to an image.")
    if not end_result: raise ProjectionError("Could not map end coordinates to an image.")

    return [start_result[1], start_result[2]], [end_result[1], end_result[2]]


def create_fireball_pto(base_pto_path: Path, output_pto_path: Path, start_xy: List[float], end_xy: List[float]):
    """Generates a precise, video-compatible PTO file for the stitcher."""
    angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
    angle_deg = math.degrees(angle_rad)
    track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    mid_x, mid_y = (start_xy[0] + end_xy[0]) / 2, (start_xy[1] + end_xy[1]) / 2

    final_h = Settings.TRACK_WIDTH
    # Always calculate a width compatible with video codecs to ensure consistency
    padded_len = track_len + 3 * final_h
    final_w = (int(padded_len) + 31) & -32
    
    pto_data = pto_mapper.parse_pto_file(str(base_pto_path))
    global_options, images = pto_data
    pano_w, pano_h = global_options.get('w'), global_options.get('h')

    cx, cy = pano_w / 2.0, pano_h / 2.0
    tx, ty = mid_x - cx, mid_y - cy
    rot_rad = math.radians(-angle_deg)
    cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)
    rotated_mid_x = (tx * cos_r - ty * sin_r) + cx
    rotated_mid_y = (tx * sin_r + ty * cos_r) + cy

    s_left = rotated_mid_x - final_w / 2
    s_top = rotated_mid_y - final_h / 2
    s_right = s_left + final_w
    s_bottom = s_top + final_h

    global_options['r'] = -angle_deg
    global_options['S'] = f"{int(s_left)},{int(s_top)},{int(s_right)},{int(s_bottom)}"
    pto_mapper.write_pto_file((global_options, images), str(output_pto_path))
    print(f"Generated '{output_pto_path.name}' for direct rotation and cropping.")


def create_background_plate(event_dir: Path, pto_path: Path, source_video_path: Path) -> Path:
    """Creates a single, high-quality background plate for reuse."""
    print("Creating shared background plate...")
    first_frame_tmp = event_dir / "fireball-firstframe.jpg"
    bg_plate_path = event_dir / "fireball-bg-plate.jpg"
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"

    ffmpeg_extract_cmd = ["ffmpeg", "-i", str(source_video_path), "-vframes", "1", "-q:v", "2", "-y", str(first_frame_tmp)]
    subprocess.run(ffmpeg_extract_cmd, check=True, capture_output=True)

    stitcher_cmd_img = [sys.executable, str(stitcher_path), "--pad", "128", str(pto_path), str(first_frame_tmp), str(bg_plate_path)]
    subprocess.run(stitcher_cmd_img, check=True, capture_output=True)

    with Image(filename=str(bg_plate_path)) as bg_img:
        # --- Use a much faster, localized 2D blur method ---
        # Store original size, resize down to 2% of the original, then resize back up.
        # This preserves local gradients while being extremely fast.
        original_width, original_height = bg_img.width, bg_img.height
        
        # Calculate 2% of the original dimensions, ensuring they are at least 1 pixel.
        new_width = max(1, int(original_width * 0.05))
        new_height = max(1, int(original_height * 0.05))
        
        bg_img.resize(new_width, new_height)
        bg_img.resize(original_width, original_height)
        bg_img.save(filename=str(bg_plate_path))

    first_frame_tmp.unlink(missing_ok=True)
    return bg_plate_path


def scale_luminance(image_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates a scaling factor from an image array and applies it to the
    luminance, returning the processed array and the factor.
    """
    print("Applying brightness multiplication directly with NumPy...")

    # Ensure the array is a float type for calculations.
    img_float_array = image_array.astype(np.float32)

    # 1. Calculate the original luminance of each pixel using the Rec. 709 formula.
    luminance = (0.2126 * img_float_array[:, :, 0] +  # Red channel
                 0.7152 * img_float_array[:, :, 1] +  # Green channel
                 0.0722 * img_float_array[:, :, 2])   # Blue channel

    # 2. Find the robust max value from the luminance data.
    max_val = np.percentile(luminance, 99.95)
    
    scaling_factor = 1.0  # Default to no scaling
    final_image_array = image_array

    if max_val > 1:
        # 3. Calculate the scaling factor.
        scaling_factor = 255.0 / max_val
        print(f"Luminance max (99.95th percentile): {max_val:.2f}. Scaling by {scaling_factor:.4f}x.")

        # 4. Calculate the ratio of new luminance to old luminance.
        epsilon = 1e-6 # Avoid division by zero
        luminance_ratio = (luminance * scaling_factor) / (luminance + epsilon)
        
        # 5. Apply this ratio to the original R, G, B channels.
        final_image_array = img_float_array * luminance_ratio[:, :, np.newaxis]

        # 6. Clip the final values and convert back to an 8-bit integer.
        final_image_array = np.clip(final_image_array, 0, 255).astype(np.uint8)

    return final_image_array, scaling_factor


def extract_meteor_track(image_path: Path, pto_path: Path, event_dir: Path, background_plate_path: Path) -> Image:
    """
    Isolates the meteor track using a direct, luminance-based threshold
    comparison between the source image and the background plate.
    """
    stitched_track_path = event_dir / "fireball-stitched.jpg"
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"

    # 1. Stitch the source image.
    stitcher_cmd = [sys.executable, str(stitcher_path), "--pad", "128", str(pto_path), str(image_path), str(stitched_track_path)]
    subprocess.run(stitcher_cmd, check=True, capture_output=True)

    # --- MODIFICATION: Save original stitched image ---
    orig_output_path = event_dir / Settings.OUTPUT_ORIG_FILENAME
    shutil.copy(str(stitched_track_path), str(orig_output_path))
    print(f"Saved original stitched image as '{orig_output_path.name}'")
    # --- END MODIFICATION ---

    print("Applying luminance-gated background removal to image...")
    with Image(filename=str(stitched_track_path)) as main_img, \
         Image(filename=str(background_plate_path)) as bg_img:
        
        # Convert both images to NumPy arrays for pixel-level operations
        main_array = np.array(main_img, dtype=np.float32)
        bg_array = np.array(bg_img, dtype=np.float32)

    # 2. Calculate luminance for both images using the standard formula.
    lum_A = (0.2126 * main_array[:, :, 0] + 0.7152 * main_array[:, :, 1] + 0.0722 * main_array[:, :, 2])
    lum_B = (0.2126 * bg_array[:, :, 0] + 0.7152 * bg_array[:, :, 1] + 0.0722 * bg_array[:, :, 2])
    
    threshold = 32
    
    # 3. Create a boolean mask where the condition lum(A) >= lum(B) + threshold is true.
    mask = lum_A >= (lum_B + threshold)
    
    # 4. Create a black canvas of the same size.
    black_array = np.zeros_like(main_array, dtype=np.float32)
    
    # 5. Use the mask to construct the final image. Where the mask is True, use the
    #    original pixel; where it's False, use the corresponding pixel from the black canvas.
    final_array = np.where(mask[:, :, np.newaxis], main_array, black_array)
    
    # Convert back to an 8-bit image for saving.
    final_array_uint8 = final_array.astype(np.uint8)
    
    with Image.from_array(final_array_uint8) as final_image_obj:
        final_image = final_image_obj.clone()
        
    stitched_track_path.unlink(missing_ok=True)
    return final_image


def detect_meteor_activity(video_path: Path, trim_config: Settings.VideoTrim) -> Optional[Tuple[float, float]]:
    """Analyzes a video to find the start and end time of significant brightness changes."""
    print("Step 2/3: Detecting meteor activity...")
    try:
        _, frame_rate, _ = get_video_info(video_path)
        if frame_rate == 0:
             print("Warning: Could not determine frame rate. Skipping trim.")
             return None

        ffprobe_cmd = [
            "ffprobe", "-v", "error", "-f", "lavfi", f"movie={video_path.as_posix()},signalstats",
            "-show_entries", "frame_tags=lavfi.signalstats.YAVG", "-of", "csv=p=0"
        ]
        result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True, timeout=120)
        brightness = [float(line.strip(',')) for line in result.stdout.strip().split('\n') if line.strip()]

        if not brightness:
            print("Warning: Could not parse brightness values. Skipping trim.")
            return None
        
        num_frames = len(brightness)
        timestamps = [i / frame_rate for i in range(num_frames)]
        
        baseline_frames = min(trim_config.BASELINE_FRAMES, num_frames)
        if baseline_frames < 2: return None
        
        initial_brightness = brightness[:baseline_frames]
        baseline = np.median(initial_brightness)
        noise_range = np.max(initial_brightness) - np.min(initial_brightness)
        threshold = baseline + (trim_config.NOISE_RANGE_FACTOR * noise_range)

        active_indices = [i for i, b in enumerate(brightness) if b > threshold]
        if not active_indices:
            print("No significant activity detected. Skipping trim.")
            return None

        start_time = timestamps[active_indices[0]]
        end_time = timestamps[active_indices[-1]]

        if (end_time - start_time) < trim_config.MIN_EVENT_DURATION_S:
            print(f"Detected event is too short ({end_time - start_time:.2f}s). Skipping trim.")
            return None

        start_padded = max(0, start_time - trim_config.PADDING_S)
        end_padded = min(timestamps[-1], end_time + trim_config.PADDING_S)
        
        print(f"Detected activity from {start_padded:.2f}s to {end_padded:.2f}s.")
        return start_padded, end_padded

    except Exception as e:
        print(f"Warning: An unexpected error occurred during activity detection: {e}", file=sys.stderr)
        return None


def create_fireball_video(event_dir: Path, pto_path: Path, background_plate_path: Path):
    """
    Creates a video by using a highly compatible, three-step masking process
    to isolate the meteor track and work around FFmpeg build limitations.
    """
    print("\nStarting video creation process...")
    source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
    if not source_video_path: raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"

    temp_stitched_video = event_dir / "fireball-stitched.mp4"
    temp_diff_video = event_dir / "fireball-diff.mp4"
    temp_mask_video = event_dir / "fireball-mask.mp4"
    temp_subtracted_video = event_dir / "fireball-subtracted.mp4"
    final_video_path = event_dir / Settings.OUTPUT_VIDEO_FILENAME
    final_orig_video_path = event_dir / Settings.OUTPUT_ORIG_VIDEO_FILENAME

    
    try:
        # Step 1: Stitch the video
        stitcher_cmd_vid = [
            sys.executable, str(stitcher_path), "--pad", "0", str(pto_path), 
            str(source_video_path), str(temp_stitched_video)
        ]
        # Stitcher progress is percentage-based (total=100)
        run_command_with_progress(stitcher_cmd_vid, desc="Step 1: Stitching", total=100)

        duration, _, _ = get_video_info(temp_stitched_video)

        # Step 2A: Create a simple 'difference' video.
        filter_graph_diff = "[0:v][1:v]blend=c0_mode=difference"
        ffmpeg_diff_cmd = [
            "ffmpeg", "-i", str(temp_stitched_video),
            "-loop", "1", "-i", str(background_plate_path),
            "-filter_complex", filter_graph_diff,
            "-t", str(duration), "-y", str(temp_diff_video)
        ]
        run_command_with_progress(ffmpeg_diff_cmd, desc="Step 2A: Creating Difference", total=duration)
        
        # Step 2B: Create a black-and-white mask from the difference video.
        threshold = 32
        filter_graph_mask = f"geq=lum='if(gt(lum(X,Y),{threshold}),255,0)'"
        ffmpeg_mask_cmd = [
            "ffmpeg", "-i", str(temp_diff_video),
            "-vf", filter_graph_mask,
            "-t", str(duration), "-y", str(temp_mask_video)
        ]
        run_command_with_progress(ffmpeg_mask_cmd, desc="Step 2B: Generating Mask", total=duration)

        # Step 2C: Apply the mask to the original stitched video.
        filter_graph_apply_mask = "[0:v][1:v]blend=c0_mode=multiply"
        ffmpeg_apply_mask_cmd = [
            "ffmpeg", "-i", str(temp_stitched_video),
            "-i", str(temp_mask_video),
            "-filter_complex", filter_graph_apply_mask,
            "-t", str(duration), "-y", str(temp_subtracted_video)
        ]
        run_command_with_progress(ffmpeg_apply_mask_cmd, desc="Step 2C: Applying Mask", total=duration)
        
        # Step 3: Trim the final video.
        trim_times = detect_meteor_activity(temp_subtracted_video, Settings.VideoTrim)
        print("Step 3: Finalizing video...")
        final_filter = "format=yuv420p"

        if trim_times:
            start_time, end_time = trim_times
            trim_duration = end_time - start_time
            # Trim the final processed video
            final_cmd = [
                "ffmpeg", "-ss", f"{start_time:.3f}", "-i", str(temp_subtracted_video),
                "-t", f"{trim_duration:.3f}", "-vf", final_filter, "-y", str(final_video_path)
            ]
            run_command_with_progress(final_cmd, desc="Step 3: Trimming Final", total=trim_duration)
            
            # --- MODIFICATION: Trim the original stitched video ---
            final_orig_cmd = [
                "ffmpeg", "-ss", f"{start_time:.3f}", "-i", str(temp_stitched_video),
                "-t", f"{trim_duration:.3f}", "-vf", final_filter, "-c:v", "libx264", "-y", str(final_orig_video_path)
            ]
            run_command_with_progress(final_orig_cmd, desc="Step 3: Trimming Original", total=trim_duration)
            print(f"✅ Saved original trimmed video to '{final_orig_video_path.name}'")
            # --- END MODIFICATION ---

        else:
            final_duration, _, _ = get_video_info(temp_subtracted_video)
            final_cmd = ["ffmpeg", "-i", str(temp_subtracted_video), "-vf", final_filter, "-y", str(final_video_path)]
            run_command_with_progress(final_cmd, desc="Step 3: Finalizing Final", total=final_duration)
            
            # --- MODIFICATION: Save the untrimmed original stitched video ---
            shutil.copy(str(temp_stitched_video), str(final_orig_video_path))
            print(f"✅ Saved original untrimmed video to '{final_orig_video_path.name}'")
            # --- END MODIFICATION ---

        print(f"✅ Success! Created '{final_video_path.name}'")

    except (subprocess.CalledProcessError, ScriptError) as e:
        stdout_output = e.stdout if hasattr(e, 'stdout') else ''
        stderr_output = e.stderr if hasattr(e, 'stderr') else ''
        command = ' '.join(e.cmd) if hasattr(e, 'cmd') else 'N/A'
        print(f"❌ Error during video processing:\nCOMMAND: {command}\nSTDOUT:\n{stdout_output}\nSTDERR:\n{stderr_output}", file=sys.stderr)
        sys.exit(1)
    finally:
        for f in [temp_stitched_video, temp_diff_video, temp_mask_video, temp_subtracted_video]:
            f.unlink(missing_ok=True)


def main():
    """Main execution flow of the script."""
    args = get_args()
    event_dir = args.event_dir
    background_plate_path = None
    pto_path = event_dir / "fireball.pto"
    
    scaling_factor = None

    try:
        if not event_dir.is_dir():
            raise FileNotFoundError(f"The specified event directory does not exist: '{event_dir}'")

        config_path = event_dir / 'event.txt'
        if not config_path.is_file():
            raise FileNotFoundError(f"'event.txt' not found in directory '{event_dir}'")
        
        config = configparser.ConfigParser()
        config.read(config_path)

        start_xy, end_xy = get_projection_coords(event_dir, config)
        
        gnomonic_base_pto_path = event_dir / 'gnomonic.pto'
        if not gnomonic_base_pto_path.is_file():
            raise FileNotFoundError(f"Base projection PTO file not found at '{gnomonic_base_pto_path}'")

        create_fireball_pto(gnomonic_base_pto_path, pto_path, start_xy, end_xy)
        
        source_video_path = None
        if args.mode in ["image", "video", "both"]:
            source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
            if not source_video_path: raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
            background_plate_path = create_background_plate(event_dir, pto_path, source_video_path)

        if args.mode in ["image", "both"]:
            if not source_video_path: raise ScriptError("Source video path needed for image processing not found.")
            source_image_path = source_video_path.with_suffix('.jpg')
            if not source_image_path.is_file():
                raise FileNotFoundError(f"Could not find corresponding source image '{source_image_path.name}'")
            
            print("Processing image to extract meteor track...")
            track_image = extract_meteor_track(source_image_path, pto_path, event_dir, background_plate_path)
            output_path = event_dir / Settings.OUTPUT_FILENAME
            track_image.save(filename=str(output_path))
            track_image.close()
            print(f"✅ Success! Created '{output_path.name}'")
        
        if args.mode in ["video", "both"]:
            create_fireball_video(event_dir, pto_path, background_plate_path)
            
    except ScriptError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("An unexpected error occurred:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if background_plate_path and background_plate_path.exists():
            background_plate_path.unlink()


if __name__ == "__main__":
    main()
