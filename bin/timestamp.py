#!/usr/bin/env python3
"""
timestamp2.py

Reads timestamps from image and video files.

This script extracts timestamps embedded in video frames or images from various
camera models. It uses Optical Character Recognition (OCR) based on predefined
font bitmaps. The core processing is accelerated with Numba.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numba
import numpy as np
import pytz
from PIL import Image

# --- Optional PyAV Import for Video Processing ---
try:
    import av
    av.logging.set_level(av.logging.ERROR)
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

# --- Type Hint Definitions ---
ImgSource = Union[str, np.ndarray, Image.Image]
Bitmap = List[int]
FontData = Dict[str, Union[int, List[int], List[Bitmap], bool]]
FontDB = Dict[str, FontData]

# --- Constants ---
# Normalization divisors help compare matching errors across fonts of different sizes.
# This is based on the number of pixels in a character (width * height).
# Some widths are adjusted to match quirks in the original C code's logic.
NORMALIZATION_DIVISORS = {
    "IP8172": 8 * 11,
    "IP9171": 8 * 16,
    "IP8151": 8 * 11,
    "IP816A": 8 * 11,
    "IMX291SD": 11 * 15,
    "IMX291HD": 11 * 15,
    "IMX307SD": 8 * 12,
    "IMX307HD_24x36": 24 * 36,
    "IMX307HD_16x24": 16 * 24,
}

# --- Font Data ---
# In a larger application, this would be better suited in a separate JSON or config file.
FONT_DATABASE: FontDB = {
    "IP8172": {"w": 8, "h": 11, "y": -18, "x": [26, 36, 51, 61, 76, 86, 101, 111, 127, 137, 153, 163], "b": [
        [60, 102, 195, 195, 195, 195, 195, 195, 195, 102, 60], [120, 216, 24, 24, 24, 24, 24, 24, 24, 24, 126],
        [124, 134, 3, 3, 3, 6, 12, 24, 48, 96, 255], [124, 135, 3, 3, 7, 62, 7, 3, 3, 135, 124],
        [15, 27, 19, 35, 67, 195, 131, 255, 3, 3, 3], [254, 192, 192, 192, 252, 134, 3, 3, 3, 134, 124],
        [60, 98, 64, 192, 252, 230, 195, 195, 195, 102, 60], [255, 3, 7, 6, 6, 12, 12, 24, 24, 56, 48],
        [126, 231, 195, 195, 231, 60, 231, 195, 195, 231, 126], [60, 102, 195, 195, 195, 103, 63, 3, 2, 70, 60]]},
    "IP9171": {"w": 8, "h": 16, "y": -42, "x": [37, 53, 85, 101, 133, 149, 173, 189, 217, 233, 261, 277], "b": [
        [28, 62, 127, 103, 227, 227, 227, 227, 227, 227, 227, 227, 103, 127, 62, 28], [24, 248, 248, 184, 56, 56, 56, 56, 56, 56, 56, 56, 56, 254, 254, 0],
        [28, 126, 127, 71, 7, 7, 7, 14, 14, 28, 56, 56, 112, 255, 255, 0], [56, 126, 126, 6, 6, 6, 30, 60, 62, 7, 7, 7, 7, 254, 254, 56],
        [0, 14, 14, 30, 30, 62, 54, 118, 230, 198, 255, 255, 6, 6, 6, 0], [0, 252, 252, 192, 192, 192, 248, 252, 30, 14, 14, 14, 14, 252, 252, 112],
        [14, 63, 63, 112, 96, 96, 254, 255, 247, 227, 227, 99, 99, 127, 62, 28], [0, 255, 255, 7, 6, 6, 14, 14, 12, 28, 28, 24, 56, 56, 48, 0],
        [28, 126, 127, 103, 99, 103, 126, 62, 126, 119, 227, 227, 227, 127, 126, 28], [28, 126, 127, 231, 227, 227, 227, 231, 127, 127, 3, 7, 7, 126, 124, 56]]},
    "IP8151": {"w": 8, "h": 11, "y": 4, "x": [26, 34, 49, 58, 73, 82, 99, 108, 120, 129, 141, 150], "b": [
        [60, 66, 66, 66, 66, 66, 66, 66, 60, 0, 0], [16, 112, 16, 16, 16, 16, 16, 16, 124, 0, 0],
        [60, 66, 66, 2, 4, 24, 32, 64, 126, 0, 0], [60, 66, 2, 2, 28, 2, 2, 66, 60, 0, 0],
        [4, 12, 20, 36, 68, 127, 4, 4, 4, 0, 0], [126, 64, 64, 124, 2, 2, 2, 66, 60, 0, 0],
        [28, 32, 64, 124, 66, 66, 66, 66, 60, 0, 0], [126, 2, 4, 4, 8, 8, 16, 16, 32, 0, 0],
        [60, 66, 66, 66, 60, 66, 66, 66, 60, 0, 0], [60, 66, 66, 66, 66, 62, 2, 4, 56, 0, 0]]},
    "IP816A": {"w": 8, "h": 11, "y": -15, "x": [26, 36, 51, 61, 76, 86, 101, 111, 127, 137, 153, 163], "b": [
        [60, 102, 195, 195, 195, 195, 195, 195, 195, 102, 60], [120, 216, 24, 24, 24, 24, 24, 24, 24, 24, 126],
        [124, 134, 3, 3, 3, 6, 12, 24, 48, 96, 255], [124, 135, 3, 3, 7, 62, 7, 3, 3, 135, 124],
        [15, 27, 19, 35, 67, 195, 131, 255, 3, 3, 3], [254, 192, 192, 192, 252, 134, 3, 3, 3, 134, 124],
        [60, 98, 64, 192, 252, 230, 195, 195, 195, 102, 60], [255, 3, 7, 6, 6, 12, 12, 24, 24, 56, 48],
        [126, 231, 195, 195, 231, 60, 231, 195, 195, 231, 126], [60, 102, 195, 195, 195, 103, 63, 3, 2, 70, 60]]},
    "IMX291SD": {"w": 12, "h": 15, "y": -28, "x": [29, 43, 65, 79, 101, 115, 137, 151, 173, 187, 209, 223], "b": [
        [3840, 8064, 12480, 12480, 24672, 24672, 24672, 24672, 24672, 24672, 24672, 28896, 12480, 8064, 3840], [768, 768, 1792, 16128, 15104, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768],
        [8064, 16320, 28896, 24672, 96, 96, 192, 192, 896, 1536, 3072, 6144, 12288, 32736, 32736], [7936, 16256, 29120, 24768, 192, 448, 1920, 1984, 192, 96, 96, 24672, 28896, 16320, 7936],
        [384, 896, 1920, 1920, 3456, 6528, 12672, 24960, 49536, 65504, 65504, 384, 384, 384, 384], [8128, 8128, 6144, 6144, 12288, 16256, 16320, 12512, 96, 96, 96, 24672, 28864, 16320, 7936],
        [3968, 8128, 14560, 12384, 24576, 28544, 32704, 28864, 24672, 24672, 24672, 12384, 14528, 8128, 3840], [32736, 32736, 96, 192, 192, 384, 384, 768, 768, 1536, 1536, 1536, 3072, 3072, 6144],
        [3840, 8064, 12480, 12480, 12480, 6528, 3840, 8064, 14784, 24672, 24672, 24672, 28896, 16320, 8064], [3840, 16256, 12736, 24768, 24672, 24672, 24672, 12512, 16352, 8032, 96, 24768, 29120, 16256, 7936]]},
    "IMX291HD": {"w": 12, "h": 15, "y": -26, "x": [29, 43, 65, 79, 101, 115, 137, 151, 173, 187, 209, 223], "b": [
        [3840, 8064, 12480, 12480, 24672, 24672, 24672, 24672, 24672, 24672, 24672, 28896, 12480, 8064, 3840], [768, 768, 1792, 16128, 15104, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768],
        [8064, 16320, 28896, 24672, 96, 96, 192, 192, 896, 1536, 3072, 6144, 12288, 32736, 32736], [7936, 16256, 29120, 24768, 192, 448, 1920, 1984, 192, 96, 96, 24672, 28896, 16320, 7936],
        [384, 896, 1920, 1920, 3456, 6528, 12672, 24960, 49536, 65504, 65504, 384, 384, 384, 384], [8128, 8128, 6144, 6144, 12288, 16256, 16320, 12512, 96, 96, 96, 24672, 28864, 16320, 7936],
        [3968, 8128, 14560, 12384, 24576, 28544, 32704, 28864, 24672, 24672, 24672, 12384, 14528, 8128, 3840], [32736, 32736, 96, 192, 192, 384, 384, 768, 768, 1536, 1536, 1536, 3072, 3072, 6144],
        [3840, 8064, 12480, 12480, 12480, 6528, 3840, 8064, 14784, 24672, 24672, 24672, 28896, 16320, 8064], [3840, 16256, 12736, 24768, 24672, 24672, 24672, 12512, 16352, 8032, 96, 24768, 29120, 16256, 7936]], "downscaled": True},
    "IMX307SD": {"w": 8, "h": 12, "y": -19, "x": [18, 27, 42, 51, 66, 75, 89, 98, 112, 121, 135, 144], "b": [
        [0x3c, 0x7e, 0x66, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xe6, 0x7e, 0x3c], [0x0c, 0x1c, 0x7c, 0x7c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c],
        [0x1c, 0x3e, 0x66, 0x66, 0x06, 0x0e, 0x0c, 0x18, 0x30, 0x70, 0x7e, 0x7e], [0x3c, 0x7e, 0x66, 0x06, 0x1c, 0x1e, 0x03, 0x03, 0x63, 0x63, 0x3e, 0x1c],
        [0x06, 0x0e, 0x0e, 0x1e, 0x36, 0x76, 0x66, 0xff, 0xff, 0x06, 0x06, 0x06], [0x7e, 0x7e, 0x60, 0x60, 0x7c, 0x7e, 0x67, 0x03, 0x63, 0x67, 0x3e, 0x1c],
        [0x1e, 0x3f, 0x33, 0x60, 0x7c, 0x7e, 0x63, 0x63, 0x63, 0x73, 0x3e, 0x1c], [0x7f, 0x7f, 0x03, 0x06, 0x06, 0x0c, 0x0c, 0x0c, 0x18, 0x18, 0x18, 0x30],
        [0x3e, 0x7f, 0x63, 0x63, 0x77, 0x3e, 0x3e, 0x73, 0x63, 0x63, 0x3e, 0x1c], [0x1c, 0x3e, 0x67, 0x63, 0x63, 0x63, 0x3f, 0x1f, 0x03, 0x66, 0x7e, 0x3c]]},
    "IMX307HD_24x36": {"w": 24, "h": 36, "y": -51, "x": [54, 81, 126, 153, 198, 225, 267, 294, 336, 363, 405, 432], "b": [
        [0x03ffc000,0x03ffc000,0x03ffc000,0x1ffff800,0x1ffff800,0x1ffff800,0x1f81f800,0x1f81f800,0x1f81f800,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xfc003f00,0xff81f800,0xff81f800,0xff81f800,0x1ffff800,0x1ffff800,0x1ffff800,0x03ffc000,0x03ffc000,0x03ffc000], [0x000fc000,0x000fc000,0x000fc000,0x007fc000,0x007fc000,0x007fc000,0x1fffc000,0x1fffc000,0x1fffc000,0x1fffc000,0x1fffc000,0x1fffc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000],
        [0x007fc000,0x007fc000,0x007fc000,0x03fff800,0x03fff800,0x03fff800,0x1f81f800,0x1f81f800,0x1f81f800,0x1f81f800,0x1f81f800,0x1f81f800,0x0001f800,0x0001f800,0x0001f800,0x000ff800,0x000ff800,0x000ff800,0x000fc000,0x000fc000,0x000fc000,0x007e0000,0x007e0000,0x007e0000,0x03f00000,0x03f00000,0x03f00000,0x1ff00000,0x1ff00000,0x1ff00000,0x1ffff800,0x1ffff800,0x1ffff800,0x1ffff800,0x1ffff800,0x1ffff800], [0x03ffc000,0x03ffc000,0x03ffc000,0x1ffff800,0x1ffff800,0x1ffff800,0x1f81f800,0x1f81f800,0x1f81f800,0x0001f800,0x0001f800,0x0001f800,0x007fc000,0x007fc000,0x007fc000,0x007ff800,0x007ff800,0x007ff800,0x00003f00,0x00003f00,0x00003f00,0x00003f00,0x00003f00,0x00003f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x03fff800,0x03fff800,0x03fff800,0x007fc000,0x007fc000,0x007fc000],
        [0x0001f800,0x0001f800,0x0001f800,0x000ff800,0x000ff800,0x000ff800,0x000ff800,0x000ff800,0x000ff800,0x007ff800,0x007ff800,0x007ff800,0x03f1f800,0x03f1f800,0x03f1f800,0x1ff1f800,0x1ff1f800,0x1ff1f800,0x1f81f800,0x1f81f800,0x1f81f800,0xffffff00,0xffffff00,0xffffff00,0xffffff00,0xffffff00,0xffffff00,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800], [0x1ffff800,0x1ffff800,0x1ffff800,0x1ffff800,0x1ffff800,0x1ffff800,0x1f800000,0x1f800000,0x1f800000,0x1f800000,0x1f800000,0x1f800000,0x1fffc000,0x1fffc000,0x1fffc000,0x1ffff800,0x1ffff800,0x1ffff800,0x1f81ff00,0x1f81ff00,0x1f81ff00,0x00003f00,0x00003f00,0x00003f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f81ff00,0x1f81ff00,0x1f81ff00,0x03fff800,0x03fff800,0x03fff800,0x007fc000,0x007fc000,0x007fc000],
        [0x007ff800,0x007ff800,0x007ff800,0x03ffff00,0x03ffff00,0x03ffff00,0x03f03f00,0x03f03f00,0x03f03f00,0x1f800000,0x1f800000,0x1f800000,0x1fffc000,0x1fffc000,0x1fffc000,0x1ffff800,0x1ffff800,0x1ffff800,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1ff03f00,0x1ff03f00,0x1ff03f00,0x03fff800,0x03fff800,0x03fff800,0x007fc000,0x007fc000,0x007fc000], [0x1fffff00,0x1fffff00,0x1fffff00,0x1fffff00,0x1fffff00,0x1fffff00,0x00003f00,0x00003f00,0x00003f00,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x0001f800,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x000fc000,0x007e0000,0x007e0000,0x007e0000,0x007e0000,0x007e0000,0x007e0000,0x007e0000,0x007e0000,0x007e0000,0x03f00000,0x03f00000,0x03f00000],
        [0x03fff800,0x03fff800,0x03fff800,0x1fffff00,0x1fffff00,0x1fffff00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1ff1ff00,0x1ff1ff00,0x1ff1ff00,0x03fff800,0x03fff800,0x03fff800,0x03fff800,0x03fff800,0x03fff800,0x1ff03f00,0x1ff03f00,0x1ff03f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x03fff800,0x03fff800,0x03fff800,0x007fc000,0x007fc000,0x007fc000], [0x007fc000,0x007fc000,0x007fc000,0x03fff800,0x03fff800,0x03fff800,0x1f81ff00,0x1f81ff00,0x1f81ff00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x1f803f00,0x03ffff00,0x03ffff00,0x03ffff00,0x007fff00,0x007fff00,0x007fff00,0x00003f00,0x00003f00,0x00003f00,0x1f81f800,0x1f81f800,0x1f81f800,0x1ffff800,0x1ffff800,0x1ffff800,0x03ffc000,0x03ffc000,0x03ffc000]]},
    "IMX307HD_16x24": {"w": 16, "h": 24, "y": -36, "x": [36, 54, 84, 102, 132, 150, 178, 196, 224, 242, 270, 288], "b": [
        [0x0ff0, 0x0ff0, 0x3ffc, 0x3ffc, 0x3c3c, 0x3c3c, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xfc3c, 0xfc3c, 0x3ffc, 0x3ffc, 0x0ff0, 0x0ff0], [0x00f0, 0x00f0, 0x03f0, 0x03f0, 0x3ff0, 0x3ff0, 0x3ff0, 0x3ff0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0],
        [0x03f0, 0x03f0, 0x0ffc, 0x0ffc, 0x3c3c, 0x3c3c, 0x3c3c, 0x3c3c, 0x003c, 0x003c, 0x00fc, 0x00fc, 0x00f0, 0x00f0, 0x03c0, 0x03c0, 0x0f00, 0x0f00, 0x3f00, 0x3f00, 0x3ffc, 0x3ffc, 0x3ffc, 0x3ffc], [0x0ff0, 0x0ff0, 0x3ffc, 0x3ffc, 0x3c3c, 0x3c3c, 0x003c, 0x003c, 0x03f0, 0x03f0, 0x03fc, 0x03fc, 0x000f, 0x000f, 0x000f, 0x000f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0],
        [0x003c, 0x003c, 0x00fc, 0x00fc, 0x00fc, 0x00fc, 0x03fc, 0x03fc, 0x0f3c, 0x0f3c, 0x3f3c, 0x3f3c, 0x3c3c, 0x3c3c, 0xffff, 0xffff, 0xffff, 0xffff, 0x003c, 0x003c, 0x003c, 0x003c, 0x003c, 0x003c], [0x3ffc, 0x3ffc, 0x3ffc, 0x3ffc, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3ff0, 0x3ff0, 0x3ffc, 0x3ffc, 0x3c3f, 0x3c3f, 0x000f, 0x000f, 0x3c0f, 0x3c0f, 0x3c3f, 0x3c3f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0],
        [0x03fc, 0x03fc, 0x0fff, 0x0fff, 0x0f0f, 0x0f0f, 0x3c00, 0x3c00, 0x3ff0, 0x3ff0, 0x3ffc, 0x3ffc, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3f0f, 0x3f0f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0], [0x3fff, 0x3fff, 0x3fff, 0x3fff, 0x000f, 0x000f, 0x003c, 0x003c, 0x003c, 0x003c, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x03c0, 0x03c0, 0x03c0, 0x03c0, 0x03c0, 0x03c0, 0x0f00, 0x0f00],
        [0x0ffc, 0x0ffc, 0x3fff, 0x3fff, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3f3f, 0x3f3f, 0x0ffc, 0x0ffc, 0x0ffc, 0x0ffc, 0x3f0f, 0x3f0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0], [0x03f0, 0x03f0, 0x0ffc, 0x0ffc, 0x3c3f, 0x3c3f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x0fff, 0x0fff, 0x03ff, 0x03ff, 0x000f, 0x000f, 0x3c3c, 0x3c3c, 0x3ffc, 0x3ffc, 0x0ff0, 0x0ff0]]},
}

# --- Data Structures ---
@dataclass(frozen=True)
class Font:
    """
    Holds the bitmap data and layout for a specific camera font.
    Instances are immutable to prevent accidental modification after creation.
    """
    name: str
    char_width: int
    char_height: int
    bitmaps: np.ndarray
    positions: np.ndarray
    use_downscaled_image: bool = False

# --- Numba JIT-compiled Core Functions ---
@numba.jit(nopython=True, cache=True)
def popcount(n: int) -> int:
    """Counts the number of set bits (1s) in a 32-bit integer's binary representation."""
    count = 0
    n_int = np.uint32(n)
    while n_int > 0:
        n_int &= (n_int - 1)
        count += 1
    return count

@numba.jit(nopython=True, cache=True)
def get_char_bitmap_from_image(img_array: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Extracts a character's bitmap from an image region.
    Converts a small patch of the image into a series of integers, where each
    integer represents the pixel pattern of one row.
    """
    img_height, img_width = img_array.shape
    bitmap = np.zeros(height, dtype=np.uint32)
    for row_idx in range(height):
        row_y = y + row_idx
        if 0 <= row_y < img_height:
            n = 0
            for col_idx in range(width):
                col_x = x + col_idx
                if 0 <= col_x < img_width:
                    # A pixel value > 192 is considered 'on' (white).
                    if img_array[row_y, col_x] > 192:
                        n |= (1 << (width - 1 - col_idx))
            bitmap[row_idx] = n
    return bitmap

@numba.jit(nopython=True, cache=True)
def find_best_match(img_bitmap: np.ndarray, font_bitmaps: np.ndarray) -> Tuple[int, int]:
    """
    Finds the best matching digit for a character bitmap.
    Compares the extracted image bitmap against each digit's template bitmap
    from the font definition. The "error" is the count of differing pixels
    (calculated via a XOR and popcount).
    """
    best_digit = -1
    min_error = int(1e9) # A large number
    num_digits = font_bitmaps.shape[0]
    char_height = img_bitmap.shape[0]

    for digit in range(num_digits):
        error = 0
        template_bitmap = font_bitmaps[digit]
        for i in range(char_height):
            # XOR finds the pixels that don't match between image and template.
            xor_val = img_bitmap[i] ^ template_bitmap[i]
            error += popcount(xor_val)

        if error < min_error:
            min_error = error
            best_digit = digit
    return best_digit, min_error

@numba.jit(nopython=True, cache=True)
def find_best_offset(img_array: np.ndarray, font_positions: np.ndarray, font_bitmaps: np.ndarray, char_width: int, char_height: int) -> Tuple[int, int, int]:
    """
    Performs a sliding window search to find the best (x, y) offset.
    This is crucial for "robust" mode, as timestamps might not be perfectly
    aligned. It checks a small area around the expected timestamp location.
    """
    search_radius = 5
    min_total_error = int(1e9)
    best_offset_x, best_offset_y = 0, 0
    # Check only a few key characters (e.g., year, minutes, seconds) for speed.
    char_indices_to_check = np.array([0, 5, 11])

    for y_offset in range(-search_radius, search_radius + 1):
        for x_offset in range(-search_radius, search_radius + 1):
            total_error = 0
            for char_index in char_indices_to_check:
                pos_x = font_positions[char_index, 0] + x_offset
                pos_y = font_positions[char_index, 1] + y_offset

                img_bitmap = get_char_bitmap_from_image(
                    img_array, pos_x, pos_y, char_width, char_height
                )
                _, error = find_best_match(img_bitmap, font_bitmaps)
                total_error += error

            if total_error < min_total_error:
                min_total_error = total_error
                best_offset_x = x_offset
                best_offset_y = y_offset

    return best_offset_x, best_offset_y, min_total_error

# --- Main Processing Logic ---

def initialize_fonts(img_width: int, img_height: int) -> Dict[str, Font]:
    """
    Loads font definitions and prepares them for processing.
    Converts raw font data into structured Font objects with NumPy arrays,
    which is required for Numba compatibility.
    """
    fonts = {}
    for name, data in FONT_DATABASE.items():
        if data["y"] >= 0:
            y_base = 0
            y_pos_scaled = y_base + data["y"]
        else:
            y_base = img_height
            y_pos_scaled = (y_base // 2) + data["y"]
        y_pos = y_base + data["y"]

        fonts[name] = Font(
            name=name,
            char_width=data["w"],
            char_height=data["h"],
            bitmaps=np.array(data["b"], dtype=np.uint32),
            positions=np.array([(x, y_pos_scaled if data.get("downscaled") else y_pos) for x in data["x"]], dtype=np.int32),
            use_downscaled_image=data.get("downscaled", False)
        )
    return fonts

def is_valid_timestamp(ts_str: str) -> bool:
    """Checks if a string represents a plausible date in the 21st century."""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        # Check for a reasonable year range.
        return 2000 <= dt.year <= datetime.now().year + 1
    except ValueError:
        return False

def _process_image_array(img_array: np.ndarray, robust: bool, debug: bool, model: Optional[str] = None) -> Optional[datetime]:
    """Core timestamp processing function for a NumPy grayscale image array."""
    img_height, img_width = img_array.shape
    all_fonts = initialize_fonts(img_width, img_height)

    fonts_to_check = {model: all_fonts[model]} if model and model in all_fonts else all_fonts
    if not fonts_to_check:
        raise ValueError(f"Model '{model}' not found. Available models: {list(all_fonts.keys())}")

    # Pre-calculate a downscaled image if any font requires it.
    img_array_half = None
    if any(font.use_downscaled_image for font in fonts_to_check.values()):
        # Simple and fast 2x2 downscaling by slicing.
        img_array_half = img_array[::2, ::2].copy()

    all_results = []
    for font_name, font in fonts_to_check.items():
        current_img_array = img_array_half if font.use_downscaled_image else img_array

        offset_x, offset_y = 0, 0
        if robust:
            offset_x, offset_y, _ = find_best_offset(current_img_array, font.positions, font.bitmaps, font.char_width, font.char_height)

        total_error, decoded_digits = 0, []
        for i in range(len(font.positions)):
            pos_x = font.positions[i, 0] + offset_x
            pos_y = font.positions[i, 1] + offset_y
            img_bitmap = get_char_bitmap_from_image(
                current_img_array, pos_x, pos_y, font.char_width, font.char_height
            )
            digit, error = find_best_match(img_bitmap, font.bitmaps)
            decoded_digits.append(str(digit))
            total_error += error

        # Normalize the error to allow fair comparison between different font sizes.
        divisor = NORMALIZATION_DIVISORS.get(font_name, font.char_width * font.char_height)
        # Multiply by 12 for the number of characters.
        normalized_error = (total_error * 256) / (divisor * 12)

        all_results.append({"font_name": font_name, "digits": decoded_digits, "error": normalized_error})

    # Try the best matches first.
    sorted_results = sorted(all_results, key=lambda k: k['error'])

    for result in sorted_results:
        digits = result['digits']
        if '-1' in digits:
            continue

        # Format: 20YY-MM-DD HH:MM:SS
        ts_str = f"20{digits[0]}{digits[1]}-{digits[2]}{digits[3]}-{digits[4]}{digits[5]} " \
                 f"{digits[6]}{digits[7]}:{digits[8]}{digits[9]}:{digits[10]}{digits[11]}"

        if not robust or is_valid_timestamp(ts_str):
            if debug:
                print(f"DEBUG: Match found: Font='{result['font_name']}', "
                      f"Normalized Error={result['error']:.2f}, Timestamp='{ts_str}'", file=sys.stderr)
            naive_datetime = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            # Assume UTC as the timezone is unknown.
            return pytz.utc.localize(naive_datetime)
        elif debug:
            print(f"DEBUG: Discarding invalid timestamp: Font='{result['font_name']}', "
                  f"Timestamp='{ts_str}'", file=sys.stderr)
    return None

def get_timestamp(source: ImgSource, robust: bool = False, debug: bool = False, model: Optional[str] = None) -> Optional[datetime]:
    """
    Main callable function. Reads a timestamp from an image or video source.

    Args:
        source: Path to an image/video, a PIL Image, or a NumPy array.
        robust: Enables position search and date validation (slower).
        debug: Prints debug information to stderr.
        model: If specified, only tests against this camera model's font.

    Returns:
        A timezone-aware datetime object (UTC) if a timestamp is found, otherwise None.
    """
    img: Optional[Image.Image] = None
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Input file not found: {source}")
        _, ext = os.path.splitext(source.lower())
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            img = Image.open(source)
        elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
            if not AV_AVAILABLE:
                raise ImportError("PyAV is required for video processing. Please install with: pip install av")
            try:
                with av.open(source) as container:
                    frame = next(container.decode(video=0))
                    img = frame.to_image()
            except StopIteration:
                raise ValueError(f"Could not decode any video frames from {source}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    elif isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if not img:
        return None

    # Convert to grayscale and then to a NumPy array for processing.
    img_gray = img.convert('L')
    img_array = np.array(img_gray, dtype=np.uint8)
    return _process_image_array(img_array, robust, debug, model)

# --- Command-Line Interface ---
def create_arg_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Read a timestamp from image or video file(s) and print the Unix time or ISO 8601 string.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_paths", nargs='+', help="Path(s) to the input image or video file(s).")
    parser.add_argument("--robust", action="store_true", help="Enable robust detection (slower) which searches for the timestamp\nand validates the date, reducing false positives.")
    parser.add_argument("--debug", action="store_true", help="Print detailed debug information to stderr.")
    parser.add_argument("--iso", action="store_true", help="Output timestamp in ISO 8601 format (e.g., 2023-10-27T10:30:00+00:00)\ninstead of the default Unix timestamp.")

    model_choices = list(FONT_DATABASE.keys())
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=model_choices,
        metavar='MODEL',
        help="Specify a single camera model to test, which is much faster.\n"
             "Available models:\n" + "\n".join(f"  - {mc}" for mc in model_choices)
    )
    return parser

def main():
    """Main function to run the script from the command line."""
    parser = create_arg_parser()
    args = parser.parse_args()

    exit_code = 0
    for path in args.input_paths:
        try:
            timestamp = get_timestamp(path, robust=args.robust, debug=args.debug, model=args.model)
            if timestamp:
                output_str = timestamp.isoformat() if args.iso else str(int(timestamp.timestamp()))
                # Add filename prefix only when processing multiple files.
                if len(args.input_paths) > 1:
                    print(f"{path}: {output_str}")
                else:
                    print(output_str)
            else:
                print(f"ERROR: Could not extract a valid timestamp from {path}.", file=sys.stderr)
                exit_code = 1
        except (FileNotFoundError, ValueError, ImportError, TypeError) as e:
            print(f"CRITICAL: An error occurred while processing {path}: {e}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
