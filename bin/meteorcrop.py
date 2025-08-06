#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts a normalized image of a meteor track from a gnomonic projection.

This script reads an event's configuration, determines the start and end
pixel coordinates of the meteor track on a gnomonic image, and then
performs a series of image processing steps to isolate and normalize the track.

The process involves:
1.  Reading the event's start/end coordinates from event.txt.
2.  Using the pto_mapper library to transform these coordinates into pixel 
    positions on the corresponding gnomonic image.
3.  Finding the correct gnomonic source image based on timestamp and status.
4.  Applying background removal, rotation, and masking to isolate the track.
5.  Saving the final cropped and normalized track as 'fireball.jpg'.

Usage:
    python3 meteorcrop.py <path_to_event_directory>
"""

import argparse
import configparser
import ctypes
import math
import sys
import traceback
# FIXED: Import UTC alongside datetime for timezone-aware operations.
from datetime import datetime, UTC
from pathlib import Path
from typing import Tuple, List

# Third-party libraries must be installed (e.g., pip install Wand).
# The pto_mapper.py script should be in the same directory or Python path.
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


# --- Custom Motion Blur Binding ---
# This adds a direct ctypes binding to the MagickMotionBlurImage function from
# the ImageMagick library.
library.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,  # angle
)

class MotionBlurImage(Image):
    """
    A Wand Image subclass with a custom motion_blur method.
    
    This class would allow for applying motion blur via a more direct, low-level
    call to ImageMagick's C API. It is not currently used in the processing pipeline.
    """
    def motion_blur(self, radius: float = 0.0, sigma: float = 0.0, angle: float = 0.0):
        library.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class Settings:
    """Configuration constants for the script."""
    # Width of the extracted track in pixels.
    TRACK_WIDTH = 128
    # Output filename for the final processed image.
    OUTPUT_FILENAME = "fireball.jpg"


def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extracts a meteor track from a gnomonic image."
    )
    parser.add_argument(
        "event_dir",
        type=Path,
        help="Path to the directory containing the event.txt file.",
    )
    return parser.parse_args()


def get_projection_coords(event_dir: Path, config: configparser.ConfigParser) -> Tuple[List[float], List[float]]:
    """
    Uses pto_mapper to transform celestial coordinates (az/alt) to pixels.

    This function assumes the PTO file describes a transformation from a
    360x180 degree equirectangular panorama to the source gnomonic image.

    Args:
        event_dir: The path to the event directory.
        config: The parsed event configuration from event.txt.

    Raises:
        ConfigError: If required keys are missing from the configuration file.
        FileNotFoundError: If the required .pto projection file is not found.
        ProjectionError: If pto_mapper fails to parse the file or map coords.

    Returns:
        A tuple containing the (start_xy, end_xy) pixel coordinates as lists.
    """
    try:
        recalibrated = config.getint('summary', 'recalibrated', fallback=1)
        start_pos_str = config.get('summary', 'startpos')
        end_pos_str = config.get('summary', 'endpos')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ConfigError(f"Missing required data in 'event.txt': {e}") from e

    # Choose the appropriate projection file based on recalibration status
    pto_filename = 'gnomonic_grid.pto' if recalibrated != 0 else 'gnomonic_corr_grid.pto'
    pto_file = event_dir / pto_filename
    if not pto_file.is_file():
        raise FileNotFoundError(f"Projection file not found at '{pto_file}'")

    # Parse the PTO file using the pto_mapper library
    try:
        pto_data = pto_mapper.parse_pto_file(str(pto_file))
        global_options, _ = pto_data
    except Exception as e:
        raise ProjectionError(f"Error parsing PTO file '{pto_file}': {e}") from e

    # The projection logic relies on a full equirectangular panorama.
    if global_options.get('f') != 2:
        print("Warning: PTO file panorama (p-line) is not equirectangular (f=2). Results may be incorrect.", file=sys.stderr)
    
    pano_w = global_options.get('w')
    pano_h = global_options.get('h')
    if not pano_w or not pano_h:
        raise ProjectionError("PTO 'p' line must contain 'w' and 'h' parameters.")

    # Get start and end celestial coordinates from the event config
    start_az, start_alt = map(float, start_pos_str.split())
    end_az, end_alt = map(float, end_pos_str.split())
    
    # --- Map Az/Alt to Panorama Pixel Coordinates ---
    # This maps the spherical Az/Alt coordinates onto a flat 2D map that
    # represents the celestial sphere.
    start_pano_x = start_az * pano_w / 360.0
    start_pano_y = (90.0 - start_alt) * pano_h / 180.0
    
    end_pano_x = end_az * pano_w / 360.0
    end_pano_y = (90.0 - end_alt) * pano_h / 180.0

    # --- Map Panorama Coordinates to Source Image Coordinates ---
    start_result = pto_mapper.map_pano_to_image(pto_data, start_pano_x, start_pano_y)
    end_result = pto_mapper.map_pano_to_image(pto_data, end_pano_x, end_pano_y)

    if not start_result:
        raise ProjectionError("Could not map start coordinates to an image.")
    if not end_result:
        raise ProjectionError("Could not map end coordinates to an image.")

    # The result from pto_mapper is a tuple: (image_index, x, y)
    start_xy = [start_result[1], start_result[2]]
    end_xy = [end_result[1], end_result[2]]

    return start_xy, end_xy


def find_gnomonic_image(event_dir: Path, config: configparser.ConfigParser) -> Path:
    """
    Finds the correct gnomonic source image using a series of fallbacks.

    Args:
        event_dir: The path to the event directory.
        config: The parsed event configuration.

    Raises:
        ConfigError: If the timestamp is missing from the configuration.
        FileNotFoundError: If no suitable gnomonic image can be found.

    Returns:
        The Path to the found gnomonic image file.
    """
    try:
        # FIXED: Replaced deprecated utcfromtimestamp with fromtimestamp(ts, UTC).
        timestamp = int(float(config.get('trail', 'timestamps').split()[0]))
        ts_str = datetime.fromtimestamp(timestamp, UTC).strftime("%Y%m%d%H%M%S")
        recalibrated = config.getint('summary', 'recalibrated', fallback=1)
    except (configparser.NoSectionError, configparser.NoOptionError, IndexError) as e:
        raise ConfigError(f"Could not read initial timestamp from 'event.txt': {e}") from e
    
    # Define search patterns in order of preference. The script first looks
    # for a specific timestamped file, then falls back to any generic file.
    if recalibrated == 0:
        patterns = [
            f'*{ts_str}-gnomonic_uncorr.jpg',
            f'*{ts_str}-gnomonic.jpg',
            '*-gnomonic_uncorr.jpg',
            '*-gnomonic.jpg',
        ]
    else:
        patterns = [
            f'*{ts_str}-gnomonic.jpg',
            '*-gnomonic.jpg',
        ]

    # Iterate through patterns and return the first match
    for pattern in patterns:
        try:
            return next(event_dir.glob(pattern))
        except StopIteration:
            continue
            
    raise FileNotFoundError("Could not find any gnomonic source image in the event directory.")


def extract_meteor_track(
    image_path: Path, start_xy: List[float], end_xy: List[float], track_width: int
) -> Image:
    """
    Isolates, rotates, and normalizes the meteor track from an image.

    Args:
        image_path: Path to the source gnomonic image.
        start_xy: The [x, y] pixel coordinates of the track start.
        end_xy: The [x, y] pixel coordinates of the track end.
        track_width: The desired width of the track mask.

    Returns:
        A wand.Image object of the processed meteor track.
    """
    # Define constants for image processing steps for clarity
    BACKGROUND_BLUR_RADIUS = 256
    TRACK_PADDING_FACTOR = 3  # Multiplier for track_width to pad the mask length
    CROP_WIDTH_ADJUSTMENT = 128 # Pixels to shorten the final crop from the padded length

    with Image(filename=str(image_path)) as pic:
        # --- 1. Background Removal ---
        # Create a heavily blurred, desaturated copy of the image to represent
        # the large-scale background gradient (e.g., sky glow).
        with pic.clone() as background:
            background.resize(filter='box', blur=BACKGROUND_BLUR_RADIUS)
            background.modulate(brightness=100, saturation=100)
            
            # Subtract this background pattern from the original image. This
            # effectively removes the glow, leaving fainter objects like stars
            # and the meteor track more prominent.
            pic.composite(background, operator='difference')

        # --- 2. Rotation ---
        # Calculate the angle of the meteor track.
        angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
        # Rotate the entire image so the track becomes perfectly horizontal.
        # This simplifies the subsequent masking and cropping steps.
        pic.rotate(-math.degrees(angle_rad), background=Color('black'))

        # --- 3. Masking ---
        # Create a new black image to serve as a mask canvas.
        with Image(width=pic.width, height=pic.height, background=Color('black')) as mask:
            with Drawing() as draw:
                # Calculate the track length and add padding to ensure the mask
                # covers the entire track plus some buffer on each end.
                track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
                padded_len = track_len + TRACK_PADDING_FACTOR * track_width
                
                # Draw a thick, white horizontal line in the center of the mask.
                draw.stroke_width = track_width
                draw.stroke_color = Color('white')
                draw.line(
                    (int((pic.width - padded_len) / 2), int(pic.height / 2)),
                    (int((pic.width + padded_len) / 2), int(pic.height / 2))
                )
                draw(mask)
            
            # Apply the mask to the rotated image using a 'multiply' composite.
            # Black areas (0) of the mask will turn corresponding image pixels
            # black, while white areas (1) will leave them unchanged.
            pic.composite(mask, operator='multiply')

        # --- 4. Cropping and Normalization ---
        # Calculate the final length for the cropped image.
        final_len = min(int(padded_len - CROP_WIDTH_ADJUSTMENT), pic.width)
        
        # Crop the image to the area of the horizontal track.
        pic.crop(
            left=int(pic.width / 2 - final_len / 2),
            top=int(pic.height / 2 - track_width / 2),
            width=final_len,
            height=track_width
        )
        
        # NOTE: The following motion blur code is commented out. It can be
        # enabled to apply a blur effect along the track's axis.
        # with MotionBlurImage(track) as img:
        #    img.motion_blur(radius=24, sigma=24, angle=0)
        #    track = img.clone()
        
        # Normalize the image to enhance contrast, making the track stand out.
        pic.normalize()
        return pic.clone()


def main():
    """Main execution flow of the script."""
    try:
        args = get_args()
        event_dir = args.event_dir

        # Ensure the event directory exists
        if not event_dir.is_dir():
            raise FileNotFoundError(f"The specified event directory does not exist: '{event_dir}'")

        # Load configuration from event.txt
        config_path = event_dir / 'event.txt'
        if not config_path.is_file():
            raise FileNotFoundError(f"'event.txt' not found in directory '{event_dir}'")
        
        config = configparser.ConfigParser()
        config.read(config_path)

        # Get all necessary data
        start_xy, end_xy = get_projection_coords(event_dir, config)
        gnomonic_image_path = find_gnomonic_image(event_dir, config)

        # Process the image to extract the meteor track
        print("Processing image to extract meteor track...")
        track_image = extract_meteor_track(
            gnomonic_image_path, start_xy, end_xy, Settings.TRACK_WIDTH
        )

        # Save the final result
        output_path = event_dir / Settings.OUTPUT_FILENAME
        track_image.format = 'jpeg'
        track_image.save(filename=str(output_path))
        track_image.close()
        
        print(f"✅ Success! Created '{output_path}'")

    except ScriptError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("An unexpected error occurred:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
