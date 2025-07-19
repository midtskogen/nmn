#!/usr/bin/env python3
"""
Applies a filter to an image to isolate a track or path.

This script processes an input image by adjusting its brightness and saturation,
applying a background removal technique, and saving the processed image to a
new file. It uses the Wand library for image manipulation.
"""

import argparse
import sys
import ctypes
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
from wand.api import library

# Define argument types for custom Wand library functions to ensure stability.
# This avoids potential issues with incorrect data types being passed to the
# underlying ImageMagick C API.

# Motion Blur arguments
library.MagickMotionBlurImage.argtypes = [
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double   # angle
]

# Selective Blur arguments
library.MagickSelectiveBlurImage.argtypes = [
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double   # threshold
]


class EnhancedImage(Image):
    """
    An enhanced Image class with custom blur methods.

    This class extends the base Wand Image class to include specialized blur
    effects like selective blur and motion blur, which are not available in the
    standard Wand API.
    """

    def selective_blur(self, radius: float = 1.0, sigma: float = 1.0, threshold: float = 0.0):
        """
        Applies a selective blur to the image.

        Args:
            radius: The radius of the blur.
            sigma: The standard deviation of the blur.
            threshold: The brightness threshold for applying the blur.
        """
        library.MagickSelectiveBlurImage(self.wand, radius, sigma, threshold)

    def motion_blur(self, radius: float = 0.0, sigma: float = 0.0, angle: float = 0.0):
        """
        Applies a motion blur to the image.

        Args:
            radius: The radius of the blur.
            sigma: The standard deviation of the blur.
            angle: The angle of the motion.
        """
        library.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def extract_track(image: Image, brightness: float, saturation: float) -> Image:
    """
    Extracts a track from an image by processing its visual properties.

    This function isolates a track by modulating brightness and saturation,
    removing the background, and normalizing the result.

    Args:
        image: The input image to process.
        brightness: The brightness modulation factor.
        saturation: The saturation modulation factor.

    Returns:
        The processed image with the track highlighted.
    """
    # Modulate the image to enhance the track
    image.modulate(brightness=brightness, saturation=saturation)

    # Create a blurred version for background subtraction
    background = image.clone()
    background.resize(filter='box', blur=256)

    # Subtract the original image from the blurred background
    with Drawing() as draw:
        draw.composite(
            operator='minus_dst',
            left=0, top=0,
            width=image.width,
            height=image.height,
            image=image
        )
        draw(background)

    # Normalize the result to enhance contrast
    background.normalize()

    # The following commented-out section shows how to apply custom blurs
    # using the EnhancedImage class. It is preserved for reference.
    # with EnhancedImage(background) as img:
    #     # Determine blur amount from image brightness
    #     with background.clone() as temp_img:
    #         temp_img.resize(1, 1)
    #         buffer = bytearray(temp_img.make_blob(format='RGB'))
    #         img_brightness = sum(buffer)
    #         blur_amount = (24 - min(img_brightness, 24)) / 2

    #     # Apply selective and motion blurs
    #     img.selective_blur(radius=8, sigma=8, threshold=6400)
    #     img.motion_blur(radius=blur_amount, sigma=blur_amount, angle=0)
    #     return img.clone()

    return background


def main():
    """
    Main function to parse arguments and run the image processing script.
    """
    parser = argparse.ArgumentParser(description="Process an image to extract a track.")
    parser.add_argument("input_file", help="The path to the input image file.")
    parser.add_argument("output_file", help="The path to save the output JPEG image.")
    args = parser.parse_args()

    try:
        with Image(filename=args.input_file) as pic:
            # Extract the track using the defined function
            track = extract_track(pic, 100, 100)

            # Save the processed image
            track.format = 'jpeg'
            track.save(filename=args.output_file)
            print(f"Processed image saved to {args.output_file}")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
