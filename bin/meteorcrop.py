#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts a normalized image of a meteor track from a gnomonic projection.

This script reads an event's configuration, determines the start and end
pixel coordinates of the meteor track on a gnomonic image, and then
performs a series of image processing steps to isolate and normalize the track.

The process involves:
1.  Reading the event's start/end coordinates from event.txt.
2.  Using pto_mapper to transform these coordinates into pixel positions
    on the corresponding gnomonic image.
3.  Finding the correct gnomonic source image.
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
from datetime import datetime
from pathlib import Path

# Third-party libraries must be installed.
# pto_mapper.py should be in the same directory or Python path.
import pto_mapper
from wand.api import library
from wand.color import Color
from wand.drawing import Drawing
from wand.image import Image


# --- Custom Motion Blur Binding (Preserved from original script) ---
# This adds a direct ctypes binding to the MagickMotionBlurImage function.
library.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,  # angle
)

class MotionBlurImage(Image):
    """A Wand Image subclass with a custom motion_blur method."""
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        library.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class Settings:
    """Configuration constants for the script."""
    # Width of the extracted track in pixels.
    TRACK_WIDTH = 128


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


def get_projection_coords(event_dir: Path, config: configparser.ConfigParser) -> tuple:
    """
    Uses the pto_mapper library to transform az/alt coordinates to pixels.

    This function assumes the PTO file describes a transformation from a
    360x180 degree equirectangular panorama (representing the celestial
    sphere) to the source gnomonic image.

    Args:
        event_dir: The path to the event directory.
        config: The parsed event configuration.

    Returns:
        A tuple containing the (start_xy, end_xy) pixel coordinates as lists.
    """
    recalibrated = config.getint('summary', 'recalibrated', fallback=1)

    # Choose the appropriate projection file
    if recalibrated == 0:
        pto_file = event_dir / 'gnomonic_corr_grid.pto'
    else:
        pto_file = event_dir / 'gnomonic_grid.pto'

    if not pto_file.is_file():
        print(f"Error: Projection file not found at '{pto_file}'")
        sys.exit(1)

    # Parse the PTO file using pto_mapper
    try:
        pto_data = pto_mapper.parse_pto_file(str(pto_file))
        global_options, _ = pto_data
    except Exception as e:
        print(f"Error parsing PTO file '{pto_file}': {e}")
        sys.exit(1)

    # Check that the panorama is equirectangular (f=2)
    if not global_options.get('f') == 2:
        print("Warning: PTO file panorama (p-line) is not equirectangular (f=2). Results may be incorrect.")
    
    pano_w = global_options.get('w')
    pano_h = global_options.get('h')
    if not pano_w or not pano_h:
        print("Error: PTO 'p' line must contain 'w' and 'h' parameters.")
        sys.exit(1)

    # Get start and end coordinates from the event config
    start_az, start_alt = map(float, config.get('summary', 'startpos').split())
    end_az, end_alt = map(float, config.get('summary', 'endpos').split())
    
    # --- Map Az/Alt to Panorama Pixel Coordinates ---
    # This simulates the behavior of the original hsi-based scaling by treating
    # the celestial sphere as a flat map.
    start_pano_x = start_az * pano_w / 360.0
    start_pano_y = (90.0 - start_alt) * pano_h / 180.0
    
    end_pano_x = end_az * pano_w / 360.0
    end_pano_y = (90.0 - end_alt) * pano_h / 180.0

    # --- Map Panorama Coordinates to Source Image Coordinates ---
    start_result = pto_mapper.map_pano_to_image(pto_data, start_pano_x, start_pano_y)
    end_result = pto_mapper.map_pano_to_image(pto_data, end_pano_x, end_pano_y)

    if not start_result:
        print("Error: Could not map start coordinates to image.")
        sys.exit(1)
    if not end_result:
        print("Error: Could not map end coordinates to image.")
        sys.exit(1)

    # The result from pto_mapper is (image_index, x, y)
    start_xy = [start_result[1], start_result[2]]
    end_xy = [end_result[1], end_result[2]]

    return start_xy, end_xy


def find_gnomonic_image(event_dir: Path, config: configparser.ConfigParser) -> Path:
    """
    Finds the correct gnomonic source image using a series of fallbacks.

    Args:
        event_dir: The path to the event directory.
        config: The parsed event configuration.

    Returns:
        The Path to the gnomonic image file.
    """
    ts_str = datetime.utcfromtimestamp(
        int(float(config.get('trail', 'timestamps').split()[0]))
    ).strftime("%Y%m%d%H%M%S")
    
    recalibrated = config.getint('summary', 'recalibrated', fallback=1)

    # Define search patterns in order of preference
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
            
    print("Error: Could not find any gnomonic source image.")
    sys.exit(1)


def extract_meteor_track(
    image_path: Path, start_xy: list, end_xy: list, track_width: int
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
    with Image(filename=str(image_path)) as pic:
        # --- 1. Background Removal ---
        # Create a blurred, modulated copy to act as the background pattern
        with pic.clone() as background:
            background.resize(filter='box', blur=256)
            background.modulate(brightness=100, saturation=100)
            
            # Subtract the background pattern from the original image
            with Drawing() as draw:
                draw.composite(
                    operator='difference',
                    left=0, top=0,
                    width=pic.width, height=pic.height,
                    image=background
                )
                draw(pic)

        # --- 2. Rotation ---
        # Calculate the angle of the track and rotate the image to make it horizontal
        angle_rad = math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0])
        pic.rotate(-math.degrees(angle_rad), background=Color('black'))

        # --- 3. Masking ---
        # Create a white line on a black canvas to serve as a mask for the track
        with Image(width=pic.width, height=pic.height, background=Color('black')) as mask:
            with Drawing() as draw:
                # Calculate the length of the track plus padding
                track_len = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
                padded_len = track_len + 3 * track_width
                
                # Draw a horizontal line in the center of the mask
                draw.stroke_width = track_width
                draw.stroke_color = Color('white')
                draw.line(
                    (int((pic.width - padded_len) / 2), int(pic.height / 2)),
                    (int((pic.width + padded_len) / 2), int(pic.height / 2))
                )
                draw(mask)
            
            # Apply the mask to the rotated image
            with Drawing() as draw:
                draw.composite(
                    operator='multiply',
                    left=0, top=0,
                    width=pic.width, height=pic.height,
                    image=mask
                )
                draw(pic)

        # --- 4. Cropping and Normalization ---
        final_len = min(int(padded_len - 128), pic.width)
        pic.crop(
            left=int(pic.width / 2 - final_len / 2),
            top=int(pic.height / 2 - track_width / 2),
            width=final_len,
            height=track_width
        )
        
        # The motion blur call was commented out in the original script
        # with MotionBlurImage(track) as img:
        #    img.motion_blur(radius=24, sigma=24, angle=0)
        #    track = img.clone()
        
        pic.normalize()
        return pic.clone()


def main():
    """Main execution flow of the script."""
    args = get_args()
    event_dir = args.event_dir

    # Load configuration from event.txt
    config_path = event_dir / 'event.txt'
    if not config_path.is_file():
        print(f"Error: 'event.txt' not found in directory '{event_dir}'")
        sys.exit(1)
        
    config = configparser.ConfigParser()
    config.read(config_path)

    # Get all necessary data
    start_xy, end_xy = get_projection_coords(event_dir, config)
    gnomonic_image_path = find_gnomonic_image(event_dir, config)

    # Process the image to extract the track
    track_image = extract_meteor_track(
        gnomonic_image_path, start_xy, end_xy, Settings.TRACK_WIDTH
    )

    # Save the final result
    output_path = event_dir / 'fireball.jpg'
    track_image.format = 'jpeg'
    track_image.save(filename=str(output_path))
    track_image.close()
    
    print(f"Successfully created '{output_path}'")


if __name__ == "__main__":
    main()
