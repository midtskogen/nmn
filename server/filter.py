#!/usr/bin/env python3

# Usage processreport.py <event dir>

import sys
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
import glob
import hsi
import math
import ctypes
from wand.api import library

# Add Motion blur
library.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                          ctypes.c_double,  # radius
                                          ctypes.c_double,  # sigma
                                          ctypes.c_double)  # angle

library.MagickSelectiveBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                             ctypes.c_double,  # radius
                                             ctypes.c_double,  # sigma
                                             ctypes.c_double)  # threshold
class MyImage(Image):
    def selective_blur(self, radius=1.0, sigma=1.0, threshold=0.0):
        library.MagickSelectiveBlurImage(self.wand, radius, sigma, threshold)
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        library.MagickMotionBlurImage(self.wand, radius, sigma, angle)

pic = Image(filename=sys.argv[1])

def extract(pic, brightness, saturation):
    # Background removal
    pic.modulate(brightness=brightness, saturation=saturation)
    pic2 = pic.clone()
    pic2.resize(filter='box', blur=256)
    with Drawing() as draw:
        draw.composite(operator='minus_dst', left=0, top=0, width=pic.width, height=pic.height, image=pic)
        draw(pic2)

        # Get brightness
        pic3 = pic2.clone()
        pic3.resize(1, 1)
        buffer = bytearray(pic3.make_blob(format='RGB'))
        brightness = buffer[0] + buffer[1] + buffer[2]
        blur = (24 - min(brightness, 24)) / 2

        pic2.normalize()
        #with MyImage(pic2) as img:
        #    img.selective_blur(radius=8, sigma=8, threshold=6400)
        #    img.motion_blur(radius=blur, sigma=blur, angle=0)
        #    pic2 = img.clone()
        return pic2

track = extract(pic, 100, 100)

track.format = 'jpeg'
track.save(filename=sys.argv[2])
