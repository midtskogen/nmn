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

    # Clean the strings to remove byte string formatting (e.g., "b'123.45'")
    start_pos_str = start_pos_str.replace("b'", "").replace("'", "")
    end_pos_str = end_pos_str.replace("b'", "").replace("'", "")

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
    try:
        start_az, start_alt = map(float, start_pos_str.split())
        end_az, end_alt = map(float, end_pos_str.split())
    except ValueError as e:
        raise ConfigError(
            f"Could not convert coordinate strings to floats. "
            f"Problematic values might be startpos='{start_pos_str}' or endpos='{end_pos_str}'. Original error: {e}"
        ) from e


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


def create_fireball_pto(base_pto_path: Path, output_pto_path: Path, start_xy: List[float], end_xy: List[float]) -> Tuple[int, int]:
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
    return final_w, final_h


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
        new_width = max(1, int(original_width * 0.3))
        new_height = max(1, int(original_height * 0.3))

        bg_img.resize(new_width, new_height)
        bg_img.resize(original_width, original_height)
        bg_img.save(filename=str(bg_plate_path))

    first_frame_tmp.unlink(missing_ok=True)
    return bg_plate_path


def stack_video_to_image(video_path: Path) -> np.ndarray:
    """
    Stacks all frames of a video file to create a single max-brightness image.
    This function is inspired by the logic in stack.py.
    Returns a BGR numpy array.
    """
    print(f"Stacking frames from '{video_path.name}' to create a composite image...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ScriptError(f"Could not open video file for stacking: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise ScriptError(f"Video file '{video_path.name}' contains no frames.")

    # Initialize a YUV stack. U and V are set to 128 (neutral grey).
    yuv_stack = np.zeros((height, width, 3), dtype=np.uint8)
    yuv_stack[:, :, 1:] = 128

    with tqdm(total=frame_count, desc="Stacking Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Create a mask where the new frame's luma is brighter than the stack's
            update_mask = yuv_frame[:, :, 0] > yuv_stack[:, :, 0]
            
            # Apply the mask to update all channels (Y, U, and V)
            yuv_stack[update_mask] = yuv_frame[update_mask]
            pbar.update(1)

    cap.release()
    bgr_result = cv2.cvtColor(yuv_stack, cv2.COLOR_YUV2BGR)
    print("✅ Frame stacking complete.")
    return bgr_result


def subtract_background_from_image_array(main_image_bgr: np.ndarray, background_plate_path: Path) -> Image:
    """
    Performs luminance-gated background subtraction on a numpy image array (BGR).
    Returns a wand.Image object.
    """
    print("Applying luminance-gated background removal to stacked image...")
    with Image(filename=str(background_plate_path)) as bg_img:
        # Ensure background matches dimensions of main image array
        if bg_img.width != main_image_bgr.shape[1] or bg_img.height != main_image_bgr.shape[0]:
             bg_img.resize(main_image_bgr.shape[1], main_image_bgr.shape[0])
        # Convert wand's RGB image to a BGR numpy array to match the input
        bg_array_rgb = np.array(bg_img, dtype=np.uint8)
        bg_array_bgr = cv2.cvtColor(bg_array_rgb, cv2.COLOR_RGB2BGR)

    main_array = main_image_bgr.astype(np.float32)
    bg_array = bg_array_bgr.astype(np.float32)

    # Calculate luminance using Rec. 709 standard on BGR arrays
    # Index 2 is Red, 1 is Green, 0 is Blue for a BGR array
    lum_A = (0.2126 * main_array[:, :, 2] + 0.7152 * main_array[:, :, 1] + 0.0722 * main_array[:, :, 0])
    lum_B = (0.2126 * bg_array[:, :, 2] + 0.7152 * bg_array[:, :, 1] + 0.0722 * bg_array[:, :, 0])

    mask = lum_A >= (lum_B + Settings.BG_SUBTRACT_THRESHOLD)
    
    black_array = np.zeros_like(main_array, dtype=np.float32)
    final_array = np.where(mask[:, :, np.newaxis], main_array, black_array)
    final_array_uint8 = final_array.astype(np.uint8)
    
    # wand.Image.from_array expects an RGB image, so convert final BGR back to RGB
    final_array_rgb = cv2.cvtColor(final_array_uint8, cv2.COLOR_BGR2RGB)

    with Image.from_array(final_array_rgb, 'RGB') as final_image_obj:
        return final_image_obj.clone()

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
    # Use the numpy-based subtraction function for consistency.
    # Load the stitched image with OpenCV (as BGR) and process.
    stitched_bgr = cv2.imread(str(stitched_track_path))
    final_image = subtract_background_from_image_array(stitched_bgr, background_plate_path)
    
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


def create_fireball_video(event_dir: Path, pto_path: Path, background_plate_path: Path, final_w: int, final_h: int, delete_stitched_vid: bool = True):
    """
    Creates a background-subtracted and trimmed video of the meteor track.
    If delete_stitched_vid is False, the temporary stitched video is not removed.
    """
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
            sys.executable, str(stitcher_path), "--pad", "128", str(pto_path),
            str(source_video_path), str(temp_stitched_video)
        ]
        run_command_with_progress(stitcher_cmd_vid, desc="Step 1/3: Stitching", total=100)

        # Step 2: Perform background subtraction using a single, efficient FFmpeg command.
        # This graph scales both inputs to ensure they match and are codec-compliant before blending.
        duration, _, _ = get_video_info(temp_stitched_video)
        filter_graph = (
            f"[0:v]scale={final_w}:{final_h},format=yuv420p,split=2[v1][v2];"
            f"[1:v]scale={final_w}:{final_h},format=yuv420p[imgscaled];"
            f"[v1][imgscaled]blend=c0_mode=difference[diff];"
            f"[diff]geq=lum='if(gt(lum(X,Y),{Settings.BG_SUBTRACT_THRESHOLD}),255,0)'[mask];"
            f"[v2][mask]blend=c0_mode=multiply"
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
        if delete_stitched_vid:
            temp_stitched_video.unlink(missing_ok=True)
        temp_subtracted_video.unlink(missing_ok=True)


# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution flow of the script."""
    args = get_args()
    event_dir = args.event_dir.resolve()
    
    background_plate_path = None
    pto_path = event_dir / Settings.TEMP_PTO
    temp_stitched_video_path = event_dir / Settings.TEMP_STITCHED_VID

    try:
        if not event_dir.is_dir():
            raise FileNotFoundError(f"The specified event directory does not exist: '{event_dir}'")

        config_path = event_dir / "event.txt"
        if not config_path.is_file():
            raise FileNotFoundError(f"'event.txt' not found in directory '{event_dir}'")
        
        config = configparser.ConfigParser()
        config.read(config_path)

        # --- Common Setup Phase ---
        start_xy, end_xy = get_projection_coords(event_dir, config)
        
        gnomonic_base_pto_path = event_dir / "gnomonic.pto"
        if not gnomonic_base_pto_path.is_file():
            raise FileNotFoundError(f"Base projection PTO file not found at '{gnomonic_base_pto_path}'")

        final_w, final_h = create_fireball_pto(gnomonic_base_pto_path, pto_path, start_xy, end_xy)
        
        source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
        if not source_video_path:
            raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
        
        background_plate_path = create_background_plate(event_dir, pto_path, source_video_path)

        # --- Mode-Specific Processing ---

        if args.mode == "image":
            print("\nProcessing image to extract meteor track...")
            source_image_path = source_video_path.with_suffix(".jpg")
            if not source_image_path.is_file():
                raise FileNotFoundError(f"Could not find corresponding source image '{source_image_path.name}'")
            
            track_image = extract_meteor_track(source_image_path, pto_path, event_dir, background_plate_path)
            output_path = event_dir / Settings.OUTPUT_FILENAME
            track_image.save(filename=str(output_path))
            track_image.close()
            print(f"✅ Success! Created '{output_path.name}'")
        
        elif args.mode == "video":
            create_fireball_video(event_dir, pto_path, background_plate_path, final_w, final_h, delete_stitched_vid=True)
            
        elif args.mode == "both":
            # 1. Process video, keeping the stitched intermediate for the image step
            create_fireball_video(event_dir, pto_path, background_plate_path, final_w, final_h, delete_stitched_vid=False)
            
            # 2. Process image using the stacked frames of the stitched video
            print("\nProcessing to create final image from video stack...")
            if not temp_stitched_video_path.is_file():
                raise FileNotFoundError(f"Stitched video '{temp_stitched_video_path.name}' needed for stacking not found. Video processing may have failed.")

            # Stack frames from the stitched video into a single BGR numpy array
            stacked_bgr_array = stack_video_to_image(temp_stitched_video_path)
            
            # Save a copy of the original (pre-subtraction) stacked image
            orig_output_path = event_dir / Settings.OUTPUT_ORIG_FILENAME
            cv2.imwrite(str(orig_output_path), stacked_bgr_array)
            print(f"Saved original stacked image as '{orig_output_path.name}'")

            # Subtract the background from the stacked image array
            track_image = subtract_background_from_image_array(stacked_bgr_array, background_plate_path)
            
            output_path = event_dir / Settings.OUTPUT_FILENAME
            track_image.save(filename=str(output_path))
            track_image.close()
            print(f"✅ Success! Created '{output_path.name}'")

    except ScriptError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("An unexpected error occurred:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final cleanup of shared and temporary resources
        if background_plate_path and background_plate_path.exists():
            background_plate_path.unlink()
        # The stitched video is kept for 'both' mode, so clean it up here.
        if temp_stitched_video_path.exists():
            temp_stitched_video_path.unlink()


if __name__ == "__main__":
    main()
