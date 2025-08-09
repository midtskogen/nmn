#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts a normalized image and video of a meteor track from a gnomonic projection.

This script reads an event's configuration, determines the start and end
pixel coordinates of the meteor track on a gnomonic image, and then
performs a series of processing steps to isolate and normalize the track.

The image process (`fireball.jpg`) involves:
1.  Reading the event's start/end celestial coordinates from event.txt.
2.  Using the pto_mapper library to transform these coordinates into pixel
    positions on the corresponding gnomonic image.
3.  Finding the correct gnomonic source image based on the event timestamp.
4.  Applying background removal, rotation, and masking to isolate the track using
    the Wand (ImageMagick) library.
5.  Saving the final cropped and normalized track as 'fireball.jpg'.

The video process (`fireball.mp4`) involves a robust, multi-step workflow:
1.  A temporary PTO file is created to command the stitcher to reproject a
    large, square area around the meteor track from the original source video.
2.  The first frame of the source video is similarly reprojected, blurred with
    padding, rotated, and cropped to create a clean background plate.
3.  An intermediate video is rendered with the background's luminance subtracted.
4.  This clean video is then analyzed to detect the meteor's start and end times.
5.  A final command trims the video and applies a consistent normalization
    to produce the final, high-contrast output.
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
from typing import Tuple, List
import json
import re

import numpy as np

# Third-party libraries must be installed (e.g., pip install Wand tqdm).
# The pto_mapper.py and stitcher.py scripts should be in the same directory or Python path.
try:
    import pto_mapper
    from wand.api import library
    from wand.color import Color
    from wand.drawing import Drawing
    from wand.image import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: A required library is missing. {e}", file=sys.stderr)
    print("Please install required packages using: pip install Wand tqdm", file=sys.stderr)
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

    class VideoTrim:
        """Parameters for detecting the meteor event to trim the video."""
        ENABLED = True
        BASELINE_FRAMES = 30
        NOISE_RANGE_FACTOR = 1.2
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
    
    duration_str = '0'
    fps_str = '25/1'
    frames_str = '0'

    for line in lines:
        if '/' in line:
            fps_str = line
        elif '.' in line:
            duration_str = line
        elif line.isdigit():
            frames_str = line

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
        stdout=subprocess.DEVNULL,  # Discard stdout to prevent pipe blocking
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    # Determine the progress format based on the command
    if 'ffmpeg' in cmd[0]:
        unit = 's'
        bar_format = '{l_bar}{bar}| {n:.2f}/{total:.2f}s'
        pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
    else: # stitcher
        unit = '%'
        bar_format = '{l_bar}{bar}| {n:.1f}/{total:.1f}%'
        pattern = re.compile(r"PROGRESS:(\d+\.?\d*)")

    with tqdm(total=round(total, 2), desc=desc, unit=unit, bar_format=bar_format) as pbar:
        for line in iter(process.stderr.readline, ''):
            match = pattern.search(line)
            if match:
                if unit == 's':
                    hours, minutes, seconds, hundredths = map(int, match.groups())
                    elapsed_time = hours * 3600 + minutes * 60 + seconds + hundredths / 100
                    update_value = min(elapsed_time, total)
                    pbar.update(update_value - pbar.n)
                else: # unit == '%'
                    percent = float(match.group(1))
                    pbar.update(percent - pbar.n)
    
        if pbar.n < total:
            pbar.update(total - pbar.n)
    
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, stderr=process.stderr.read()
        )


# --- End Helper Functions ---


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


def find_gnomonic_image(event_dir: Path, config: configparser.ConfigParser) -> Path:
    """Finds the correct gnomonic source image using a series of fallbacks."""
    try:
        timestamp = int(float(config.get('trail', 'timestamps').split()[0]))
        ts_str = datetime.fromtimestamp(timestamp, UTC).strftime("%Y%m%d%H%M%S")
        recalibrated = config.getint('summary', 'recalibrated', fallback=1)
    except (configparser.NoSectionError, configparser.NoOptionError, IndexError) as e:
        raise ConfigError(f"Could not read initial timestamp from 'event.txt': {e}") from e
    
    patterns = [f'*{ts_str}-gnomonic.jpg', '*-gnomonic.jpg']
    if recalibrated == 0:
        patterns = [f'*{ts_str}-gnomonic_uncorr.jpg'] + patterns
        patterns += ['*-gnomonic_uncorr.jpg']

    for pattern in patterns:
        try:
            return next(event_dir.glob(pattern))
        except StopIteration:
            continue
            
    raise FileNotFoundError("Could not find any gnomonic source image in the event directory.")


def extract_meteor_track(image_path: Path, start_xy: List[float], end_xy: List[float], track_width: int) -> Image:
    """Isolates, rotates, and normalizes the meteor track from an image."""
    BACKGROUND_BLUR_RADIUS = 256
    TRACK_PADDING_FACTOR = 3
    CROP_WIDTH_ADJUSTMENT = 128

    with Image(filename=str(image_path)) as pic:
        with pic.clone() as background:
            background.resize(filter='box', blur=BACKGROUND_BLUR_RADIUS)
            background.modulate(brightness=100, saturation=100)
            pic.composite(background, operator='difference')

        angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
        pic.rotate(-math.degrees(angle_rad), background=Color('black'))

        with Image(width=pic.width, height=pic.height, background=Color('black')) as mask:
            with Drawing() as draw:
                track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
                padded_len = track_len + TRACK_PADDING_FACTOR * track_width
                
                draw.stroke_width = track_width
                draw.stroke_color = Color('white')
                draw.line(
                    (int((pic.width - padded_len) / 2), int(pic.height / 2)),
                    (int((pic.width + padded_len) / 2), int(pic.height / 2))
                )
                draw(mask)
            
            pic.composite(mask, operator='multiply')

        final_len = min(int(padded_len - CROP_WIDTH_ADJUSTMENT), pic.width)
        
        pic.crop(
            left=int(pic.width / 2 - final_len / 2),
            top=int(pic.height / 2 - track_width / 2),
            width=final_len,
            height=track_width
        )
        
        pic.normalize()
        return pic.clone()


def detect_meteor_activity(video_path: Path, trim_config: Settings.VideoTrim) -> Tuple[float, float] | None:
    """Analyzes a video to find the start and end time of significant brightness changes."""
    print("Step 5/6: Detecting meteor activity on subtracted video...")
    try:
        _, frame_rate, _ = get_video_info(video_path)
        if frame_rate == 0:
             print(f"Warning: Could not determine frame rate. Skipping trim.")
             return None

        ffprobe_yavg_cmd = [
            "ffprobe", "-v", "error",
            "-f", "lavfi", f"movie={video_path.as_posix()},signalstats",
            "-show_entries", "frame_tags=lavfi.signalstats.YAVG",
            "-of", "csv=p=0"
        ]
        yavg_result = subprocess.run(ffprobe_yavg_cmd, check=True, capture_output=True, text=True, timeout=120)
        
        brightness_str = yavg_result.stdout.strip().split('\n')
        brightness = [float(line.strip(',')) for line in brightness_str if line.strip()]

        if not brightness:
            print("Warning: Could not parse any brightness values. Skipping trim.")
            return None
        
        num_frames = len(brightness)
        timestamps = [i / frame_rate for i in range(num_frames)]
        
        video_duration = timestamps[-1]
        num_baseline_frames = min(trim_config.BASELINE_FRAMES, len(brightness))
        if num_baseline_frames < 2:
            return None
        
        initial_frames_brightness = brightness[:num_baseline_frames]
        baseline_brightness = np.median(initial_frames_brightness)
        noise_range = np.max(initial_frames_brightness) - np.min(initial_frames_brightness)
        threshold = baseline_brightness + (trim_config.NOISE_RANGE_FACTOR * noise_range)
        max_brightness = max(brightness)

        active_indices = [i for i, b in enumerate(brightness) if b > threshold]
        
        if not active_indices:
            print("No significant activity detected based on brightness. Skipping trim.")
            return None

        start_time = timestamps[active_indices[0]]
        end_time = timestamps[active_indices[-1]]

        if (end_time - start_time) < trim_config.MIN_EVENT_DURATION_S:
            print(f"Detected event is too short ({end_time - start_time:.2f}s). Skipping trim.")
            return None

        start_time_padded = max(0, start_time - trim_config.PADDING_S)
        end_time_padded = min(video_duration, end_time + trim_config.PADDING_S)
        
        print(f"Detected activity from {start_time_padded:.2f}s to {end_time_padded:.2f}s.")
        return start_time_padded, end_time_padded

    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Warning: A data processing step failed, cannot trim video. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred during activity detection: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def create_fireball_video(event_dir: Path, gnomonic_base_pto_path: Path, start_xy: List[float], end_xy: List[float]):
    """Creates a cropped, rotated, and background-subtracted video."""
    print("Starting advanced video creation process...")

    # --- 1. Calculate Geometry & Paths ---
    angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
    track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    
    final_crop_h = Settings.TRACK_WIDTH
    padded_len = track_len + (3 * final_crop_h)
    unaligned_width = int(padded_len - 128)
    final_crop_w = (unaligned_width + 32 - 1) & -32

    square_side = max(final_crop_w, final_crop_h) * math.sqrt(2)
    square_side = (int(square_side) + 31) & -32
    
    source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
    if not source_video_path: raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"
    if not stitcher_path.is_file(): raise FileNotFoundError("stitcher.py not found in script directory.")

    # Define all temporary and final file paths
    pto_path = event_dir / "fireball.pto"
    temp_square_video = event_dir / "fireball-square.mp4"
    temp_rotated_video = event_dir / "fireball-rotated.mp4"
    temp_subtracted_video = event_dir / "fireball-subtracted.mp4"
    first_frame_tmp = event_dir / "fireball-firstframe.jpg"
    square_frame_tmp = event_dir / "fireball-firstframe-square.jpg"
    bg_plate_final = event_dir / "fireball-bg-plate.jpg"
    final_video_path = event_dir / Settings.OUTPUT_VIDEO_FILENAME
    
    try:
        # --- 2. Create PTO for Reprojection ---
        pto_data = pto_mapper.parse_pto_file(str(gnomonic_base_pto_path))
        global_options, images = pto_data
        mid_x = (start_xy[0] + end_xy[0]) / 2
        mid_y = (start_xy[1] + end_xy[1]) / 2
        s_left = mid_x - (square_side / 2)
        s_right = mid_x + (square_side / 2)
        s_top = mid_y - (square_side / 2)
        s_bottom = mid_y + (square_side / 2)
        global_options['S'] = f"{int(s_left)},{int(s_top)},{int(s_right)},{int(s_bottom)}"
        pto_mapper.write_pto_file((global_options, images), str(pto_path))
        print(f"Generated '{pto_path}' for reprojection.")

        # --- 3. Stitch Square Video ---
        stitcher_cmd = [sys.executable, str(stitcher_path), "--pad", "0", str(pto_path), str(source_video_path), str(temp_square_video)]
        run_command_with_progress(stitcher_cmd, desc="Step 1/6: Stitching square video", total=100)

        # --- 4. Create Background Plate for Subtraction ---
        print("Step 2/6: Creating background subtraction plate...")
        background_subtraction_enabled = False
        try:
            ffmpeg_extract_cmd = ["ffmpeg", "-i", str(source_video_path), "-vframes", "1", "-q:v", "2", "-y", str(first_frame_tmp)]
            subprocess.run(ffmpeg_extract_cmd, check=True, capture_output=True, text=True)

            stitcher_img_cmd = [sys.executable, str(stitcher_path), "--pad", "0", str(pto_path), str(first_frame_tmp), str(square_frame_tmp)]
            subprocess.run(stitcher_img_cmd, check=True, capture_output=True, text=True)
            
            with Image(filename=str(square_frame_tmp)) as bg_img:
                bg_img.resize(filter='box', blur=256)
                bg_img.save(filename=str(square_frame_tmp))

            bg_filter_chain = f"rotate={-angle_rad}:c=black,crop={final_crop_w}:{final_crop_h}"
            ffmpeg_bg_cmd = ["ffmpeg", "-i", str(square_frame_tmp), "-vf", bg_filter_chain, "-y", str(bg_plate_final)]
            subprocess.run(ffmpeg_bg_cmd, check=True, capture_output=True, text=True)
            
            background_subtraction_enabled = True
        except Exception as e:
            print(f"⚠️ Warning: Could not create background plate. Proceeding without subtraction. Error: {e}", file=sys.stderr)

        # --- 5. Get Video Info ---
        print("Step 3/6: Getting video info...")
        duration_s, _, _ = get_video_info(temp_square_video)
        if duration_s == 0:
            raise ScriptError("Video appears to have zero frames or duration after stitching.")

        # --- 6. Render Temporary Background-Subtracted Video (2-Step Process) ---
        print("Step 4/6: Rendering temporary background-subtracted video...")
        
        if background_subtraction_enabled and bg_plate_final.is_file():
            desc_a = "  - Step 4a: Rotating/Cropping"
            rotate_filter = f"rotate={-angle_rad}:c=black,crop={final_crop_w}:{final_crop_h}"
            ffmpeg_rotate_cmd = ["ffmpeg", "-i", str(temp_square_video), "-vf", rotate_filter, "-y", str(temp_rotated_video)]
            run_command_with_progress(ffmpeg_rotate_cmd, total=duration_s, desc=desc_a)

            desc_b = "  - Step 4b: Subtracting Bkgnd"
            duration_rotated, _, _ = get_video_info(temp_rotated_video)
            ffmpeg_blend_cmd = [
                "ffmpeg", "-i", str(temp_rotated_video),
                "-i", str(bg_plate_final),
                "-filter_complex", "[0:v][1:v]blend=c0_mode=difference",
                "-y", str(temp_subtracted_video)
            ]
            run_command_with_progress(ffmpeg_blend_cmd, total=duration_rotated, desc=desc_b)
        else:
            rotate_filter = f"rotate={-angle_rad}:c=black,crop={final_crop_w}:{final_crop_h}"
            ffmpeg_rotate_cmd = ["ffmpeg", "-i", str(temp_square_video), "-vf", rotate_filter, "-y", str(temp_subtracted_video)]
            run_command_with_progress(ffmpeg_rotate_cmd, total=duration_s, desc="Rendering Video")
        
        # --- 7. Detect Meteor Activity ---
        trim_times = detect_meteor_activity(temp_subtracted_video, Settings.VideoTrim())
        
        # --- 8. Perform Final Trim and Normalization ---
        print("Step 6/6: Performing final trim and applying fixed normalization...")
        
        normalization_filter = "normalize"
        if trim_times:
            start_time, end_time = trim_times
            duration = end_time - start_time
            ffmpeg_trim_cmd = [
                "ffmpeg", "-ss", f"{start_time:.3f}", "-i", str(temp_subtracted_video),
                "-t", f"{duration:.3f}", "-vf", normalization_filter, "-y", str(final_video_path)
            ]
            run_command_with_progress(ffmpeg_trim_cmd, total=duration, desc="Trimming & Normalizing")
        else:
            final_duration, _, _ = get_video_info(temp_subtracted_video)
            ffmpeg_norm_cmd = ["ffmpeg", "-i", str(temp_subtracted_video), "-vf", normalization_filter, "-y", str(final_video_path)]
            run_command_with_progress(ffmpeg_norm_cmd, total=final_duration, desc="Normalizing Video")

        print(f"✅ Success! Created '{final_video_path}'")

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr or ''
        raise ScriptError(f"A processing step failed with exit code {e.returncode}:\nCOMMAND: {' '.join(e.cmd)}\nSTDERR:\n{stderr_output}")
    finally:
        # Clean up all intermediate files
        # pto_path.unlink(missing_ok=True)
        temp_square_video.unlink(missing_ok=True)
        temp_rotated_video.unlink(missing_ok=True)
        temp_subtracted_video.unlink(missing_ok=True)
        first_frame_tmp.unlink(missing_ok=True)
        square_frame_tmp.unlink(missing_ok=True)
        bg_plate_final.unlink(missing_ok=True)


def main():
    """Main execution flow of the script."""
    try:
        args = get_args()
        event_dir = args.event_dir

        if not event_dir.is_dir():
            raise FileNotFoundError(f"The specified event directory does not exist: '{event_dir}'")

        config_path = event_dir / 'event.txt'
        if not config_path.is_file():
            raise FileNotFoundError(f"'event.txt' not found in directory '{event_dir}'")
        
        config = configparser.ConfigParser()
        config.read(config_path)

        start_xy, end_xy = get_projection_coords(event_dir, config)
        gnomonic_image_path = find_gnomonic_image(event_dir, config)
        
        recalibrated = config.getint('summary', 'recalibrated', fallback=1)
        gnomonic_grid_pto_path = event_dir / ('gnomonic_corr_grid.pto' if recalibrated == 0 else 'gnomonic_grid.pto')
        gnomonic_base_pto_path = event_dir / 'gnomonic.pto'

        if not gnomonic_grid_pto_path.is_file():
            raise FileNotFoundError(f"Required grid PTO file not found at '{gnomonic_grid_pto_path}'")
        if not gnomonic_base_pto_path.is_file():
            raise FileNotFoundError(f"Base projection PTO file not found at '{gnomonic_base_pto_path}'")

        print("Processing image to extract meteor track...")
        
        track_image = extract_meteor_track(
            gnomonic_image_path, start_xy, end_xy, Settings.TRACK_WIDTH
        )
        
        output_path = event_dir / Settings.OUTPUT_FILENAME
        track_image.format = 'jpeg'
        track_image.save(filename=str(output_path))
        track_image.close()
        print(f"✅ Success! Created '{output_path}'")
        
        create_fireball_video(
            event_dir, 
            gnomonic_base_pto_path, 
            start_xy, 
            end_xy
        )

    except ScriptError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("An unexpected error occurred:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
