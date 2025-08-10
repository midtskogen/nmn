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

# --- Standard Library Imports ---
import argparse
import configparser
import ctypes
import math
import shutil
import subprocess
import sys
import traceback
import re
from pathlib import Path
from typing import List, Optional, Tuple

# --- Third-Party Imports ---
import cv2
import numpy as np
from tqdm import tqdm
from wand.api import library
from wand.image import Image

# --- Local Application/Library Specific Imports ---
try:
    import pto_mapper
except ImportError as e:
    print(f"Error: A required local script is missing. {e}", file=sys.stderr)
    print("Please ensure 'pto_mapper.py' is in the same directory or Python path.", file=sys.stderr)
    sys.exit(1)


# ==============================================================================
#  CUSTOM EXCEPTION AND CONFIGURATION CLASSES
# ==============================================================================

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
    # General processing settings
    TRACK_WIDTH: int = 128
    BG_SUBTRACT_THRESHOLD: int = 40

    # Final output filenames
    OUTPUT_FILENAME: str = "fireball.jpg"
    OUTPUT_VIDEO_FILENAME: str = "fireball.mp4"
    OUTPUT_ORIG_FILENAME: str = "fireball_orig.jpg"
    OUTPUT_ORIG_VIDEO_FILENAME: str = "fireball_orig.mp4"

    # Intermediate (temporary) filenames
    TEMP_PTO: str = "fireball.pto"
    TEMP_FIRST_FRAME: str = "fireball-firstframe.jpg"
    TEMP_BG_PLATE: str = "fireball-bg-plate.jpg"
    TEMP_STITCHED_IMG: str = "fireball-stitched.jpg"
    TEMP_STITCHED_VID: str = "fireball-stitched.mp4"
    TEMP_SUBTRACTED_VID: str = "fireball-subtracted.mp4"

    # Parameters for trimming the video to the main event
    class VideoTrim:
        ENABLED: bool = True
        BASELINE_FRAMES: int = 30
        NOISE_RANGE_FACTOR: float = 2.0
        MIN_EVENT_DURATION_S: float = 0.2
        PADDING_S: float = 0.5


# ==============================================================================
#  COMMAND-LINE AND HELPER FUNCTIONS
# ==============================================================================

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


def get_video_info(video_path: Path) -> Tuple[float, float, int]:
    """Uses ffprobe to get duration (s), frame rate (fps), and total frames."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=r_frame_rate,duration,nb_read_frames",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = {
            k.strip(): v.strip()
            for k, v in (line.split("=") for line in result.stdout.strip().split("\n") if "=" in line)
        }

        duration_str = info.get("duration", "0")
        fps_str = info.get("r_frame_rate", "0/1")
        frames_str = info.get("nb_read_frames", "0")

        num, den = map(float, fps_str.split("/"))
        frame_rate = num / den if den != 0 else 0

        return float(duration_str), frame_rate, int(frames_str)
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
        print(f"Warning: Could not get video info from '{video_path.name}'. {e}", file=sys.stderr)
        return 0.0, 0.0, 0


def run_command_with_progress(cmd: List[str], desc: str, total: float):
    """
    Runs a command and displays a tqdm progress bar by parsing its stderr.
    Handles both ffmpeg (time=...) and stitcher (PROGRESS:...) formats.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    is_ffmpeg = "ffmpeg" in cmd[0]
    unit = "s" if is_ffmpeg else "%"
    bar_format = "{l_bar}{bar}| {n:.2f}/{total:.2f}s" if is_ffmpeg else "{l_bar}{bar}| {n:.1f}/{total:.1f}%"
    
    # Pre-compile regex for performance in the loop
    time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
    stitcher_pattern = re.compile(r"PROGRESS:(\d+\.?\d*)")

    stderr_lines = []
    with tqdm(total=round(total, 2), desc=desc, unit=unit, bar_format=bar_format) as pbar:
        last_update = 0.0
        for line in iter(process.stderr.readline, ""):
            stderr_lines.append(line)  # Store stderr for error reporting
            current_update = None
            if is_ffmpeg:
                match = time_pattern.search(line)
                if match:
                    h, m, s, hs = map(int, match.groups())
                    elapsed = h * 3600 + m * 60 + s + hs / 100
                    current_update = min(elapsed, total)
            else: # stitcher
                match = stitcher_pattern.search(line)
                if match:
                    current_update = float(match.group(1))

            if current_update is not None and current_update > last_update:
                increment = current_update - last_update
                pbar.update(increment)
                last_update = current_update

        # Ensure the progress bar completes fully
        if pbar.n < total:
            pbar.update(total - pbar.n)

    stdout, _ = process.communicate()  # Get remaining stdout
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, output=stdout, stderr="".join(stderr_lines)
        )


# ==============================================================================
#  COORDINATE AND PTO FILE HANDLING
# ==============================================================================

def get_projection_coords(event_dir: Path, config: configparser.ConfigParser) -> Tuple[List[float], List[float]]:
    """Uses pto_mapper to transform celestial coordinates (az/alt) to pixels."""
    try:
        recalibrated = config.getint("summary", "recalibrated", fallback=1)
        start_pos_str = config.get("summary", "startpos")
        end_pos_str = config.get("summary", "endpos")
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ConfigError(f"Missing required data in 'event.txt': {e}") from e

    # Determine which grid file to use based on calibration status
    pto_filename = "gnomonic_grid.pto" if recalibrated != 0 else "gnomonic_corr_grid.pto"
    pto_file = event_dir / pto_filename
    if not pto_file.is_file():
        raise FileNotFoundError(f"Projection file not found at '{pto_file}'")

    try:
        pto_data = pto_mapper.parse_pto_file(str(pto_file))
        global_options, _ = pto_data
    except Exception as e:
        raise ProjectionError(f"Error parsing PTO file '{pto_file}': {e}") from e

    pano_w, pano_h = global_options.get("w"), global_options.get("h")
    if not pano_w or not pano_h:
        raise ProjectionError("PTO 'p' line must contain 'w' and 'h' parameters.")

    # Convert az/alt to panoramic pixel coordinates
    start_az, start_alt = map(float, start_pos_str.split())
    end_az, end_alt = map(float, end_pos_str.split())
    start_pano_x = start_az * pano_w / 360.0
    start_pano_y = (90.0 - start_alt) * pano_h / 180.0
    end_pano_x = end_az * pano_w / 360.0
    end_pano_y = (90.0 - end_alt) * pano_h / 180.0

    # Map panoramic coordinates to source image coordinates
    start_result = pto_mapper.map_pano_to_image(pto_data, start_pano_x, start_pano_y)
    end_result = pto_mapper.map_pano_to_image(pto_data, end_pano_x, end_pano_y)

    if not start_result:
        raise ProjectionError("Could not map start coordinates to an image.")
    if not end_result:
        raise ProjectionError("Could not map end coordinates to an image.")

    return [start_result[1], start_result[2]], [end_result[1], end_result[2]]


def create_fireball_pto(base_pto_path: Path, output_pto_path: Path, start_xy: List[float], end_xy: List[float]):
    """Generates a precise, video-compatible PTO file for the stitcher."""
    angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
    angle_deg = math.degrees(angle_rad)
    track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    mid_x, mid_y = (start_xy[0] + end_xy[0]) / 2, (start_xy[1] + end_xy[1]) / 2

    final_h = Settings.TRACK_WIDTH
    # Add padding and round width to be divisible by 32 for video codec compatibility
    padded_len = track_len + 3 * final_h
    final_w = (int(padded_len) + 31) & -32

    pto_data = pto_mapper.parse_pto_file(str(base_pto_path))
    global_options, images = pto_data
    pano_w, pano_h = global_options.get("w"), global_options.get("h")

    # Calculate the new center point after rotation
    cx, cy = pano_w / 2.0, pano_h / 2.0
    tx, ty = mid_x - cx, mid_y - cy
    rot_rad = math.radians(-angle_deg)
    cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)
    rotated_mid_x = (tx * cos_r - ty * sin_r) + cx
    rotated_mid_y = (tx * sin_r + ty * cos_r) + cy

    # Define the crop area (S parameter)
    s_left = rotated_mid_x - final_w / 2
    s_top = rotated_mid_y - final_h / 2
    s_right = s_left + final_w
    s_bottom = s_top + final_h

    # Update PTO with rotation and crop parameters
    global_options["r"] = -angle_deg
    global_options["S"] = f"{int(s_left)},{int(s_top)},{int(s_right)},{int(s_bottom)}"
    pto_mapper.write_pto_file((global_options, images), str(output_pto_path))
    print(f"Generated '{output_pto_path.name}' for direct rotation and cropping.")


# ==============================================================================
#  IMAGE AND VIDEO PROCESSING
# ==============================================================================

def create_background_plate(event_dir: Path, pto_path: Path, source_video_path: Path) -> Path:
    """Creates a single, high-quality, blurred background plate for reuse."""
    print("Creating shared background plate...")
    first_frame_tmp = event_dir / Settings.TEMP_FIRST_FRAME
    bg_plate_path = event_dir / Settings.TEMP_BG_PLATE
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"

    # 1. Extract the first frame from the source video.
    ffmpeg_extract_cmd = ["ffmpeg", "-i", str(source_video_path), "-vframes", "1", "-q:v", "2", "-y", str(first_frame_tmp)]
    subprocess.run(ffmpeg_extract_cmd, check=True, capture_output=True)

    # 2. Stitch the first frame using the generated PTO file.
    stitcher_cmd_img = [sys.executable, str(stitcher_path), "--pad", "128", str(pto_path), str(first_frame_tmp), str(bg_plate_path)]
    subprocess.run(stitcher_cmd_img, check=True, capture_output=True)

    # 3. Apply a fast, heavy blur to the stitched frame to create a clean background.
    with Image(filename=str(bg_plate_path)) as bg_img:
        # This method is much faster than a large-radius Gaussian blur.
        # It preserves local color gradients while smoothing out star details.
        original_width, original_height = bg_img.width, bg_img.height
        new_width = max(1, int(original_width * 0.1))
        new_height = max(1, int(original_height * 0.1))

        bg_img.resize(new_width, new_height)
        bg_img.resize(original_width, original_height)
        bg_img.save(filename=str(bg_plate_path))

    first_frame_tmp.unlink(missing_ok=True)
    return bg_plate_path


def extract_meteor_track(image_path: Path, pto_path: Path, event_dir: Path, background_plate_path: Path) -> Image:
    """Isolates the meteor track from a source image using background subtraction."""
    stitched_track_path = event_dir / Settings.TEMP_STITCHED_IMG
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"

    # 1. Stitch the source image using the generated PTO file.
    stitcher_cmd = [sys.executable, str(stitcher_path), "--pad", "128", str(pto_path), str(image_path), str(stitched_track_path)]
    subprocess.run(stitcher_cmd, check=True, capture_output=True)

    # Save a copy of the original stitched image before background removal.
    orig_output_path = event_dir / Settings.OUTPUT_ORIG_FILENAME
    shutil.copy(str(stitched_track_path), str(orig_output_path))
    print(f"Saved original stitched image as '{orig_output_path.name}'")

    print("Applying luminance-gated background removal to image...")
    with Image(filename=str(stitched_track_path)) as main_img, \
         Image(filename=str(background_plate_path)) as bg_img:
        main_array = np.array(main_img, dtype=np.float32)
        bg_array = np.array(bg_img, dtype=np.float32)

    # 2. Calculate luminance (brightness) for both images using Rec. 709 standard.
    lum_A = (0.2126 * main_array[:, :, 2] + 0.7152 * main_array[:, :, 1] + 0.0722 * main_array[:, :, 0])
    lum_B = (0.2126 * bg_array[:, :, 2] + 0.7152 * bg_array[:, :, 1] + 0.0722 * bg_array[:, :, 0])
    
    # 3. Create a mask where the main image is significantly brighter than the background.
    mask = lum_A >= (lum_B + Settings.BG_SUBTRACT_THRESHOLD)
    
    # 4. Use the mask to create the final image, keeping original pixels where the
    #    mask is true and making them black otherwise.
    black_array = np.zeros_like(main_array, dtype=np.float32)
    # np.newaxis adds a dimension to the mask to match the 3-channel image array.
    final_array = np.where(mask[:, :, np.newaxis], main_array, black_array)
    
    final_array_uint8 = final_array.astype(np.uint8)
    
    with Image.from_array(final_array_uint8) as final_image_obj:
        final_image = final_image_obj.clone()
        
    stitched_track_path.unlink(missing_ok=True)
    return final_image


def detect_meteor_activity(video_path: Path, trim_config: Settings.VideoTrim) -> Optional[Tuple[float, float]]:
    """Analyzes video brightness to find the meteor event's start and end times."""
    print("Step 2/3: Detecting meteor activity...")
    try:
        _, frame_rate, num_frames = get_video_info(video_path)
        if frame_rate == 0:
            print("Warning: Could not determine frame rate. Skipping trim.")
            return None

        # Use ffprobe's signalstats filter to get the average brightness (YAVG) for each frame.
        ffprobe_cmd = [
            "ffprobe", "-v", "error", "-f", "lavfi",
            f"movie={video_path.as_posix()},signalstats",
            "-show_entries", "frame_tags=lavfi.signalstats.YAVG",
            "-of", "csv=p=0",
        ]
        result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True, timeout=120)
        brightness = [float(line.strip(',')) for line in result.stdout.strip().split("\n") if line]

        if not brightness:
            print("Warning: Could not parse brightness values. Skipping trim.")
            return None
        
        # Establish a baseline brightness and noise level from the start of the video.
        baseline_frames = min(trim_config.BASELINE_FRAMES, num_frames)
        if baseline_frames < 2:
            return None
        
        initial_brightness = brightness[:baseline_frames]
        baseline = np.median(initial_brightness) + 0.02
        noise_range = np.max(initial_brightness) - np.min(initial_brightness)
        threshold = baseline + (trim_config.NOISE_RANGE_FACTOR * noise_range)

        # Find all frames where brightness exceeds the noise threshold.
        active_indices = [i for i, b in enumerate(brightness) if b > threshold]
        if not active_indices:
            print("No significant activity detected. Skipping trim.")
            return None

        timestamps = [i / frame_rate for i in range(num_frames)]
        start_time = timestamps[active_indices[0]]
        end_time = timestamps[active_indices[-1]]

        if (end_time - start_time) < trim_config.MIN_EVENT_DURATION_S:
            print(f"Detected event is too short ({end_time - start_time:.2f}s). Skipping trim.")
            return None

        # Add padding to the start and end times for a better viewing experience.
        start_padded = max(0, start_time - trim_config.PADDING_S)
        end_padded = min(timestamps[-1], end_time + trim_config.PADDING_S)
        
        print(f"Detected activity from {start_padded:.2f}s to {end_padded:.2f}s.")
        return start_padded, end_padded

    except Exception as e:
        print(f"Warning: An unexpected error occurred during activity detection: {e}", file=sys.stderr)
        return None

def _finalize_videos(event_dir: Path, original_vid: Path, processed_vid: Path, trim_times: Optional[Tuple[float, float]]):
    """Helper function to trim and save the final MP4, WebM, and inverted WebM videos."""
    print("Step 3/3: Finalizing videos...")
    # Define MP4 output paths
    final_video_path = event_dir / Settings.OUTPUT_VIDEO_FILENAME
    final_orig_video_path = event_dir / Settings.OUTPUT_ORIG_VIDEO_FILENAME
    # Ensure yuv420p pixel format for maximum compatibility with players and web.
    final_filter = "format=yuv420p"

    # --- Create MP4 Videos ---
    if trim_times:
        start_time, end_time = trim_times
        trim_duration = end_time - start_time
        
        print(f"Trimming final video ({final_video_path.name})...")
        final_cmd = [
            "ffmpeg", "-ss", f"{start_time:.3f}", "-i", str(processed_vid),
            "-t", f"{trim_duration:.3f}", "-vf", final_filter, "-y", str(final_video_path)
        ]
        subprocess.run(final_cmd, check=True, capture_output=True)
        
        print(f"Trimming original video ({final_orig_video_path.name})...")
        final_orig_cmd = [
            "ffmpeg", "-ss", f"{start_time:.3f}", "-i", str(original_vid),
            "-t", f"{trim_duration:.3f}", "-vf", final_filter, "-c:v", "libx264", "-y", str(final_orig_video_path)
        ]
        subprocess.run(final_orig_cmd, check=True, capture_output=True)
        print(f"✅ Saved '{final_orig_video_path.name}'")
    else: # No trimming needed, save full-length videos
        print(f"Saving final video ({final_video_path.name})...")
        final_cmd = ["ffmpeg", "-i", str(processed_vid), "-vf", final_filter, "-y", str(final_video_path)]
        subprocess.run(final_cmd, check=True, capture_output=True)
        
        print(f"Saving original video ({final_orig_video_path.name})...")
        shutil.copy(str(original_vid), str(final_orig_video_path))
        print(f"✅ Saved '{final_orig_video_path.name}'")
    
    print(f"✅ Success! Created '{final_video_path.name}'")

    # --- Create WebM Videos from the newly created MP4s ---
    final_webm_path = final_video_path.with_suffix(".webm")
    final_orig_webm_path = final_orig_video_path.with_suffix(".webm")

    # Common FFmpeg options for high-quality, web-optimized WebM conversion
    webm_opts = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-an", "-y"]

    print(f"Creating WebM version ({final_webm_path.name})...")
    webm_cmd = ["ffmpeg", "-i", str(final_video_path)] + webm_opts + [str(final_webm_path)]
    subprocess.run(webm_cmd, check=True, capture_output=True)
    print(f"✅ Success! Created '{final_webm_path.name}'")
    
    # --- Create Inverted WebM Video ---
    final_neg_webm_path = final_video_path.with_name("fireball_neg.webm")
    print(f"Creating inverted WebM version ({final_neg_webm_path.name})...")
    # Apply luminance inversion filter to the common options
    neg_webm_cmd = ["ffmpeg", "-i", str(final_video_path), "-vf", "lutyuv=y=negval"] + webm_opts + [str(final_neg_webm_path)]
    subprocess.run(neg_webm_cmd, check=True, capture_output=True)
    print(f"✅ Success! Created '{final_neg_webm_path.name}'")


    print(f"Creating WebM version ({final_orig_webm_path.name})...")
    orig_webm_cmd = ["ffmpeg", "-i", str(final_orig_video_path)] + webm_opts + [str(final_orig_webm_path)]
    subprocess.run(orig_webm_cmd, check=True, capture_output=True)
    print(f"✅ Success! Created '{final_orig_webm_path.name}'")


def create_fireball_video(event_dir: Path, pto_path: Path, background_plate_path: Path):
    """Creates a background-subtracted and trimmed video of the meteor track."""
    print("\nStarting video creation process...")
    source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
    if not source_video_path:
        raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
    
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"
    temp_stitched_video = event_dir / Settings.TEMP_STITCHED_VID
    temp_subtracted_video = event_dir / Settings.TEMP_SUBTRACTED_VID
    
    try:
        # Step 1: Stitch the source video using the generated PTO.
        stitcher_cmd_vid = [
            sys.executable, str(stitcher_path), "--pad", "0", str(pto_path),
            str(source_video_path), str(temp_stitched_video)
        ]
        run_command_with_progress(stitcher_cmd_vid, desc="Step 1/3: Stitching", total=100)

        # Step 2: Perform background subtraction using a single, efficient FFmpeg command.
        # This graph creates a difference matte, thresholds it to a mask, and applies the mask.
        duration, _, _ = get_video_info(temp_stitched_video)
        filter_graph = (
            f"[0:v][1:v]blend=c0_mode=difference[diff];"
            f"[diff]geq=lum='if(gt(lum(X,Y),{Settings.BG_SUBTRACT_THRESHOLD}),255,0)'[mask];"
            f"[0:v][mask]blend=c0_mode=multiply"
        )
        ffmpeg_subtract_cmd = [
            "ffmpeg", "-i", str(temp_stitched_video),
            "-loop", "1", "-i", str(background_plate_path),
            "-filter_complex", filter_graph,
            "-t", str(duration), "-y", str(temp_subtracted_video)
        ]
        run_command_with_progress(ffmpeg_subtract_cmd, desc="Step 2/3: Subtracting Background", total=duration)
        
        # Step 3: Detect meteor activity and finalize both videos.
        trim_times = detect_meteor_activity(temp_subtracted_video, Settings.VideoTrim)
        _finalize_videos(event_dir, temp_stitched_video, temp_subtracted_video, trim_times)

    except (subprocess.CalledProcessError, ScriptError) as e:
        stdout = getattr(e, "stdout", "")
        stderr = getattr(e, "stderr", "")
        command = " ".join(getattr(e, "cmd", []))
        print(f"❌ Error during video processing:\nCOMMAND: {command}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up temporary video files
        for f in [temp_stitched_video, temp_subtracted_video]:
            f.unlink(missing_ok=True)


# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution flow of the script."""
    args = get_args()
    event_dir = args.event_dir.resolve()
    background_plate_path = None
    pto_path = event_dir / Settings.TEMP_PTO

    try:
        if not event_dir.is_dir():
            raise FileNotFoundError(f"The specified event directory does not exist: '{event_dir}'")

        config_path = event_dir / "event.txt"
        if not config_path.is_file():
            raise FileNotFoundError(f"'event.txt' not found in directory '{event_dir}'")
        
        config = configparser.ConfigParser()
        config.read(config_path)

        # --- Setup Phase ---
        start_xy, end_xy = get_projection_coords(event_dir, config)
        
        gnomonic_base_pto_path = event_dir / "gnomonic.pto"
        if not gnomonic_base_pto_path.is_file():
            raise FileNotFoundError(f"Base projection PTO file not found at '{gnomonic_base_pto_path}'")

        create_fireball_pto(gnomonic_base_pto_path, pto_path, start_xy, end_xy)
        
        source_video_path = None
        if args.mode in ["image", "video", "both"]:
            source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
            if not source_video_path:
                raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
            background_plate_path = create_background_plate(event_dir, pto_path, source_video_path)

        # --- Image Processing ---
        if args.mode in ["image", "both"]:
            if not source_video_path:
                raise ScriptError("Source video path needed for image processing not found.")
            source_image_path = source_video_path.with_suffix(".jpg")
            if not source_image_path.is_file():
                raise FileNotFoundError(f"Could not find corresponding source image '{source_image_path.name}'")
            
            print("\nProcessing image to extract meteor track...")
            track_image = extract_meteor_track(source_image_path, pto_path, event_dir, background_plate_path)
            output_path = event_dir / Settings.OUTPUT_FILENAME
            track_image.save(filename=str(output_path))
            track_image.close()
            print(f"✅ Success! Created '{output_path.name}'")
        
        # --- Video Processing ---
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
        # Final cleanup of shared resources
        if background_plate_path and background_plate_path.exists():
            background_plate_path.unlink()
        if pto_path.exists():
             pto_path.unlink()


if __name__ == "__main__":
    main()
