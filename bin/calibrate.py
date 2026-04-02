#!/usr/bin/python3

import sys
import tempfile
import subprocess
import re
import argparse
import ephem
import os
import datetime
import configparser
import ffmpeg
import pathlib
import math
from wand.image import Image
from astropy.io import fits

def process_image(input_path, output_path):
    with Image(filename=input_path) as input_img:
        # Crop the center 1280x720 area
        cropped_img = input_img.clone()
        offset_x = int((input_img.width - 1280) / 2)
        offset_y = int((input_img.height - 720) / 2)
        print(input_img.width)
        print(input_img.height)
        cropped_img.crop(left=offset_x, top=offset_y, width=1280, height=720)

        # Apply blur
        blurred_img = cropped_img.clone()
        blurred_img.blur(radius=0, sigma=32)

        # Apply fx "u-v" and brightness-contrast
        result_img = cropped_img.clone()
        result_img.composite(blurred_img, operator='minus_src')
        result_img.brightness_contrast(brightness=35, contrast=55)

        # Save the final result
        result_img.save(filename=output_path)

        return offset_x, offset_y

def solve_astrometry(image_path, pos, offset_x, offset_y, cpu_limit=30, scale_low=20):
    axy_file = tempfile.NamedTemporaryFile(delete=True, dir="/tmp").name
    subprocess.run(["solve-field", "-l", str(cpu_limit), "-p", "--scale-low", str(scale_low), "--odds-to-solve", "10000000", "-c", "0.1", "--sigma", "20", "--overwrite", "--axy", axy_file, image_path])
    with fits.open(axy_file) as hdul:
        stars = hdul[1].data

    print("""
# hugin project file
#hugin_ptoversion 2
p f2 w36000 h18000 v360  k0 E0 R0 n"TIFF_m c:LZW"
m i0

# image lines
#-hugin  cropFactor=1
i w1920 h1080 f3 v83.6919969245957 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r174.434046537962 p116.50726491636 y-138.193481930334 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0.00179788291455609 b-0.0273847081648436 c0.00403407139245843 d16.0245600505131 e35.0029544602904 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n""
#-hugin  cropFactor=1
i w1920 h1080 f3 v83.6606935505316 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r174.871685711496 p116.31956860298 y-136.834006497961 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0.00179788291455609 b-0.0277154325230141 c0.0054638629456752 d16.7588092263311 e33.7951851875761 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n""


# specify variables that should be optimized
v v0
v r0
v p0
v y0
v b0
v c0
v d0
v e0
v


# control points
""")
    for x, y, _, _ in stars:
        res = subprocess.run(["wcs-xy2rd", "-x", str(x), "-y", str(y), "-w", "stars.wcs"], capture_output=True, text=True)
        if res.returncode == 0:
            pattern = r"Pixel \((?P<x>[\d.]+), (?P<y>[\d.]+)\) -> RA,Dec \((?P<ra>[\d.]+), (?P<dec>[\d.]+)\)"
            for line in res.stdout.splitlines():
                match = re.search(pattern, line)
                if match:
                    x = float(match.group("x"))
                    y = float(match.group("y"))
                    ra = match.group("ra")
                    dec = match.group("dec")
                    star = ephem.FixedBody()
                    star._ra, star._dec, star._epoch = ra, dec, ephem.J2000
                    star.compute(pos)
                    print(f"c n0 N1 x{x + offset_x} y{y + offset_y} X{math.degrees(float(repr(star.az)))*100} Y{(90-math.degrees(float(repr(star.alt))))*100} t0")

def timestamp(img):
    ts = subprocess.run([str(pathlib.Path.home()) + "/bin/timestamp", img], stdout=subprocess.PIPE, text=True)
    return int(ts.stdout.rstrip().lstrip())

def read_frame(filename, output):
    out, err = (
        ffmpeg
        .input(filename)
        .filter_('select', 'gte(n,{})'.format(0))
        .output(output, vframes=1)
        .overwrite_output()
        .run(quiet=True)
    )
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Click on images to find coordinates.')
    parser.add_argument('-c', '--config', dest='config', help='meteor config file (default: /etc/meteor.cfg)', default='/etc/meteor.cfg', type=str)
    parser.add_argument(action='store', dest='input', help='input video')
    parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
    parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
    parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
    parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
    parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
    parser.add_argument('-d', '--date', dest='timestamp', help='start time (default: extracted from video))', type=str)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read([args.config, os.path.expanduser('~/meteor.cfg')])

    input_img = tempfile.NamedTemporaryFile(delete=True, suffix=".png", dir="/tmp").name
    read_frame(args.input, input_img)

    if not args.timestamp:
        args.timestamp = timestamp(args.input)
        
    pos = ephem.Observer()
    pos.lat = config.get('astronomy', 'latitude')
    pos.lon = config.get('astronomy', 'longitude')
    pos.elevation = float(config.get('astronomy', 'elevation'))
    pos.temp = float(config.get('astronomy', 'temperature'))
    pos.pressure = float(config.get('astronomy', 'pressure'))
    pos.date = args.timestamp

    if args.longitude:
        pos.lon = str(args.longitude)
    if args.latitude:
        pos.lat = str(args.latitude)
    if args.elevation:
        pos.elevation = args.elevation
    if args.temperature:
        pos.temp = args.temperature
    if args.pressure:
        pos.pressure = args.pressure

    processed_img = tempfile.NamedTemporaryFile(delete=True, dir="/tmp").name
    offset_x, offset_y = process_image(input_img, processed_img)
    solve_astrometry(processed_img, pos, offset_x, offset_y)
