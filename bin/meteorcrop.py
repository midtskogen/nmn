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
2.  The resulting square video is then rotated using FFmpeg to make the meteor
    track horizontal.
3.  A final, tight crop is performed on the rotated video using FFmpeg to
    produce the final output.
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

import numpy as np

# Third-party libraries must be installed (e.g., pip install Wand).
# The pto_mapper.py and stitcher.py scripts should be in the same directory or Python path.
try:
    import pto_mapper
    from wand.api import library
    from wand.color import Color
    from wand.drawing import Drawing
    from wand.image import Image
except ImportError as e:
    print(f"Error: A required library is missing. {e}", file=sys.stderr)
    print("Please install required packages using: pip install Wand", file=sys.stderr)
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


def create_fireball_video(event_dir: Path, gnomonic_base_pto_path: Path, start_xy: List[float], end_xy: List[float]):
    """Creates a cropped and rotated video using a robust 2-step (stitch->rotate&crop) process."""
    print("Creating cropped video via 2-step (stitch -> rotate & crop) process...")

    # 1. Calculate the essential geometry
    angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
    track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    
    final_crop_h = Settings.TRACK_WIDTH
    padded_len = track_len + (3 * final_crop_h)
    unaligned_width = int(padded_len - 128)
    final_crop_w = (unaligned_width + 32 - 1) & -32

    square_side = max(final_crop_w, final_crop_h) * math.sqrt(2)
    square_side = (int(square_side) + 31) & -32

    # 2. Create the PTO for the initial SQUARE crop
    try:
        pto_data = pto_mapper.parse_pto_file(str(gnomonic_base_pto_path))
        global_options, images = pto_data
        
        mid_x = (start_xy[0] + end_xy[0]) / 2
        mid_y = (start_xy[1] + end_xy[1]) / 2
        s_left = mid_x - (square_side / 2)
        s_right = mid_x + (square_side / 2)
        s_top = mid_y - (square_side / 2)
        s_bottom = mid_y + (square_side / 2)
        global_options['S'] = f"{int(s_left)},{int(s_top)},{int(s_right)},{int(s_bottom)}"

        pto_path = event_dir / "fireball.pto"
        pto_mapper.write_pto_file((global_options, images), str(pto_path))
        print(f"Generated '{pto_path}' for the initial stitch.")

    except Exception as e:
        raise ScriptError(f"Failed to create temporary PTO file: {e}")

    # --- Find source video and stitcher script ---
    source_video_path = next((v for v in event_dir.glob("*.mp4") if "-gnomonic" not in v.name and "-grid" not in v.name and "fireball" not in v.name), None)
    if not source_video_path: raise FileNotFoundError(f"Could not find a source video in '{event_dir}'")
    stitcher_path = Path(__file__).parent.resolve() / "stitcher.py"
    if not stitcher_path.is_file(): raise FileNotFoundError("stitcher.py not found in script directory.")

    # --- Define temporary and final file paths ---
    temp_square_video = event_dir / "fireball-square.mp4"
    final_video_path = event_dir / Settings.OUTPUT_VIDEO_FILENAME
    
    try:
        # STEP 1: Stitch the larger, un-rotated square video
        stitcher_cmd = [sys.executable, str(stitcher_path), "--pad", "0", str(pto_path), str(source_video_path), str(temp_square_video)]
        print(f"Step 1/2: Stitching square video cropped around the meteor...")
        subprocess.run(stitcher_cmd, check=True, capture_output=True, text=True, timeout=300)

        # STEP 2: Rotate AND Crop the video in a single FFmpeg command
        filter_chain = f"rotate={-angle_rad}:c=black,crop={final_crop_w}:{final_crop_h}"
        ffmpeg_cmd = ["ffmpeg", "-i", str(temp_square_video), "-vf", filter_chain, "-y", str(final_video_path)]
        print(f"Step 2/2: Rotating and cropping final video...")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

        print(f"✅ Success! Created '{final_video_path}'")

    except subprocess.CalledProcessError as e:
        raise ScriptError(f"A processing step failed with exit code {e.returncode}:\nCOMMAND: {' '.join(e.cmd)}\nSTDERR:\n{e.stderr}")
    finally:
        # Clean up all intermediate files
        pto_path.unlink(missing_ok=True)
        temp_square_video.unlink(missing_ok=True)


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
        track_image = extract_meteor_track(gnomonic_image_path, start_xy, end_xy, Settings.TRACK_WIDTH)
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
