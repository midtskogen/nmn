#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Makes a file grid.png from a .pto file
# Result usage example: composite -blend 40 meteor-20141010.jpg grid.png x.jpg

import pto_mapper
import ephem
import math
import argparse
import wand.image
import wand.drawing
import wand.color
import configparser
import os
import re
from datetime import datetime, UTC
from brightstar import brightstar
import numpy as np

class Vector2D:
    """A simple 2D vector class."""
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

def main():
    """
    Main function to generate a grid file from a .pto file.
    """
    parser = argparse.ArgumentParser(description='Create a grid file from a .pto file.')

    parser.add_argument('-p', '--picture', dest='image', help='which picture in the .pto file to use (default: 0)', default=0, type=int)
    parser.add_argument('-W', '--width', dest='width', help='width of output (if different from input)', default=-1, type=int)
    parser.add_argument('-H', '--height', dest='height', help='height of output (if different from input)', default=-1, type=int)
    parser.add_argument('-x', '--xscale', dest='xscale', help='X scaling factor (default: 1)', default=1, type=float)
    parser.add_argument('-y', '--yscale', dest='yscale', help='Y scaling factor (default: 1)', default=1, type=float)
    parser.add_argument('-s', '--spacing', dest='label_spacing', help='spacing between labels in degrees (default: 10)', default=10, type=int)
    parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects (default: 500)', default=500, type=int)
    parser.add_argument('-f', '--faintest', dest='faintest', help='faintest objects to include (default: 5)', default=5, type=float)
    parser.add_argument('-b', '--brightest', dest='brightest', help='brightest objects to include (default: -30)', default=-30, type=float)
    parser.add_argument('-X', '--longitude', dest='longitude', help='observer longitude', type=float)
    parser.add_argument('-Y', '--latitude', dest='latitude', help='observer latitude', type=float)
    parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
    parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
    parser.add_argument('-P', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
    parser.add_argument('-c', '--config', dest='config', help='meteor config file', type=str)
    parser.add_argument('-d', '--date', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)', type=int)
    parser.add_argument("--radec", help="use RA/DEC instead of az/alt", action="store_true")
    parser.add_argument("--verbose", help="more verbose labels", action="store_true")
    parser.add_argument("--refract", help="Apply atmospheric refraction correction (Az/Alt mode only).", action="store_true")

    parser.add_argument(action='store', dest='infile', help='input .pto file')
    parser.add_argument(action='store', dest='outfile', help='output grid file (default "grid.png")', default="grid.png", nargs='?')
    args = parser.parse_args()

    # Load PTO data using the new parser
    pto_data = pto_mapper.parse_pto_file(args.infile)
    global_opts, images_data = pto_data
    img0_data = images_data[args.image]

    def map_img_to_sph_unrotated(x, y):
        """Maps image pixel coordinates to un-rotated panorama spherical coordinates."""
        pano_coords = pto_mapper.map_image_to_pano(pto_data, args.image, x, y)
        if not pano_coords:
            return None
        
        pano_x, pano_y = pano_coords
        pano_w, pano_h = global_opts['w'], global_opts['h']
        
        # Assuming equirectangular panorama projection
        yaw_rad = (pano_x / pano_w - 0.5) * 2.0 * math.pi
        pitch_rad = -(pano_y / pano_h - 0.5) * math.pi
        
        az_deg = math.degrees(yaw_rad)
        alt_deg = math.degrees(pitch_rad)
        
        return az_deg, alt_deg

    def map_sph_to_img_unrotated(az, alt):
        """Maps un-rotated panorama spherical coordinates to image pixel coordinates."""
        yaw_rad = math.radians(az)
        pitch_rad = math.radians(alt)

        pano_w, pano_h = global_opts['w'], global_opts['h']

        # Assuming equirectangular panorama projection
        pano_x = (yaw_rad / (2.0 * math.pi) + 0.5) * pano_w
        pano_y = (-pitch_rad / math.pi + 0.5) * pano_h

        img_coords = pto_mapper.map_pano_to_image(pto_data, pano_x, pano_y, restrict_to_bounds=False)

        if img_coords and img_coords[0] == args.image:
            return img_coords[1], img_coords[2]
        
        return None

    def inverse_transform_new(dst, src):
        """Maps image pixel to grid coordinates (either Az/Alt or RA/Dec)."""
        # 1. Get the real-world Az/Alt. This is South-based due to .pto convention.
        az_alt = map_img_to_sph_unrotated(src.x, src.y)
        if az_alt is None:
            dst.x, dst.y = -1, -1
            return

        az_deg, alt_deg = az_alt

        # 2. Convert from South-based to North-based azimuth for all grid calculations.
        az_deg = (az_deg + 180) % 360

        if args.radec and pos.date:
            # 3. If radec mode, convert Az/Alt to RA/Dec
            ra_rad, dec_rad = pos.radec_of(math.radians(az_deg), math.radians(alt_deg))
            grid_x_deg = math.degrees(ra_rad)
            grid_y_deg = math.degrees(dec_rad)
        else:
            # 4. Otherwise, the grid is just Az/Alt
            grid_x_deg = az_deg
            grid_y_deg = alt_deg

        # 5. Store the final scaled grid coordinate
        dst.x = ((grid_x_deg + 360) % 360) * scale
        dst.y = (90.0 - grid_y_deg) * scale
        
    def forward_transform_new(dst, src):
        """Maps grid coordinates (Az/Alt or RA/Dec) to an image pixel."""
        # 1. Un-scale the incoming grid coordinate
        grid_x_deg = src.x / scale
        grid_y_deg = 90.0 - (src.y / scale) 

        if args.radec and pos.date:
            # 2. If radec, grid source is RA/Dec. Convert to Az/Alt.
            star = ephem.FixedBody()
            star._ra = math.radians(grid_x_deg)
            star._dec = math.radians(grid_y_deg)
            star.compute(pos)
            az_deg = math.degrees(star.az)
            alt_deg = math.degrees(star.alt)
        else:
            # 3. Otherwise, the grid source is already North-based Az/Alt
            az_deg = grid_x_deg
            alt_deg = grid_y_deg
        
        # 4. Convert North-based az back to South-based for pto_mapper
        az_to_map = (az_deg - 180) % 360

        # 5. Find the pixel that corresponds to the South-based Az/Alt coordinate
        img_xy = map_sph_to_img_unrotated(az_to_map, alt_deg)
        if img_xy is not None:
            dst.x, dst.y = img_xy
        else:
            dst.x, dst.y = -10000, -10000

    pos = ephem.Observer()
    if args.config:
        config = configparser.ConfigParser()
        config.read([args.config])
        pos.lat = config.get('astronomy', 'latitude')
        pos.lon = config.get('astronomy', 'longitude')
        pos.elevation = float(config.get('astronomy', 'elevation'))
        pos.temp = float(config.get('astronomy', 'temperature'))
        pos.pressure = float(config.get('astronomy', 'pressure'))

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
    if args.timestamp:
        pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')
    if args.radec and not args.refract:
        # Disable ephem's internal refraction correction.
        # This is done for RA/Dec mode (to prevent grid distortion) or if 
        # the user hasn't explicitly requested refraction for Az/Alt mode.
        pos.pressure = 0

    # Get image properties from pto_data
    width = img0_data['w']
    height = img0_data['h']
    # Use HFOV from pto_mapper data for scale calculation
    scale = int(global_opts['w'] / global_opts['v'])

    dst = Vector2D();
    centre = Vector2D();
    top = width/2

    if args.width > 0:
        args.xscale = float(args.width) / width
    if args.height > 0:
        args.yscale = float(args.height) / height

    # Find diagonal FOV
    inverse_transform_new(dst, Vector2D(0, 0))
    tl_az, tl_co_alt = dst.x / scale, dst.y / scale
    inverse_transform_new(dst, Vector2D(width, height))
    br_az, br_co_alt = dst.x / scale, dst.y / scale

    x1 = math.radians(tl_az)
    x2 = math.radians(br_az)
    y1 = math.radians(90.0 - tl_co_alt)
    y2 = math.radians(90.0 - br_co_alt)
    a = math.sin((y2-y1)/2) * math.sin((y2-y1)/2) + math.cos(y1) * math.cos(y2) * math.sin((x2-x1)/2) * math.sin((x2-x1)/2)
    fov = math.degrees(2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

    # --- NEW BOUNDS DETECTION ---
    # Find az and alt range by sampling the entire image area to be robust against projection distortions.
    minalt = 90.0
    maxalt = -90.0
    az_values = set()
    
    # Define a sampling step. A smaller step is more accurate but slower.
    sample_step = 30 

    for y in range(0, height + 1, sample_step):
        img_y = min(y, height)
        for x in range(0, width + 1, sample_step):
            img_x = min(x, width)
            inverse_transform_new(dst, Vector2D(img_x, img_y))
            
            # inverse_transform_new returns dst.x = -1 on failure
            if dst.x == -1:
                continue

            az, alt = dst.x / scale, 90.0 - (dst.y / scale)
            
            minalt = min(minalt, alt)
            maxalt = max(maxalt, alt)
            az_values.add(az)

    if not az_values:
        print("Warning: Could not map any image points to celestial coordinates. Grid will not be drawn.")
        minaz, maxaz = 0, 0
    else:
        # Determine the min and max azimuth, correctly handling the 0/360 wrap-around.
        sorted_az = sorted(list(az_values))
        
        if len(sorted_az) == 1:
            minaz = maxaz = sorted_az[0]
        else:
            # Find the largest gap between sorted azimuth values. A large gap indicates the 0/360 discontinuity.
            max_gap = 0
            gap_index = -1
            for i in range(len(sorted_az) - 1):
                gap = sorted_az[i+1] - sorted_az[i]
                if gap > max_gap:
                    max_gap = gap
                    gap_index = i
            
            # A gap > 180 degrees signifies that the azimuth range crosses the 0/360 boundary.
            if max_gap > 180:
                # The range is wrapped. E.g., from 340 deg to 20 deg.
                # We re-order it so the loop can run continuously, e.g., from 340 to 380.
                minaz = sorted_az[gap_index + 1]
                maxaz = sorted_az[gap_index] + 360
            else:
                # The range is contiguous, e.g., from 100 to 250 degrees.
                minaz = sorted_az[0]
                maxaz = sorted_az[-1]

    azsep = 15 if args.radec else 10
    minaz = max(int(minaz/azsep)*azsep, 0)
    maxaz = int(maxaz/azsep)*azsep+azsep
    maxalt = min(int(maxalt/10)*10+10, 90)

    if maxaz < minaz:
        maxaz += 360

    minalt = int(minalt/10)*10 if args.radec else max(int(minalt/10)*10, 0)

    # Zenith in the frame?
    forward_transform_new(dst, Vector2D(0, 0)) # az=0, co-alt=0 -> alt=90
    if -10 < dst.x < width + 10 and -10 < dst.y < height + 10:
        minalt = 0
        minaz = 0
        maxaz = 360
        maxalt = 90

    minalt = int(minalt)
    minalt3 = max(minalt, 10)

    def refract(alt):
        """Calculates atmospheric refraction offset, only if --refract is enabled and not in RA/Dec mode."""
        # In RA/Dec mode, refraction is disabled as it would distort the grid incorrectly.
        if args.radec or not args.refract:
            return 0.0
        
        # Refraction formula is not stable for altitudes at or below the horizon.
        val_for_tan = alt + (7.31 / (alt + 4.4))
        if val_for_tan <= 0:
            return 0.0
        
        return 0.016 / (math.tan(math.radians(val_for_tan)))

    def scalealt(alt, scale):
        """Calculates the y-coordinate on the image from altitude, applying refraction if enabled."""
        # Refraction makes objects appear higher; the correction is added to the true altitude.
        alt_apparent = alt + refract(alt)
        return (90.0 - alt_apparent) * scale

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    def get_azimuth_spacing(base_spacing, declination_or_altitude, thinning=80):
        """
        Calculates the appropriate azimuth/RA spacing in degrees to avoid
        lines becoming too packed at high declinations or altitudes.
        """
        thinning = int(round(thinning / 10.0)) * 10
        thinning = min(90, max(0, thinning - 10))
        
        val = abs(declination_or_altitude)
        for a in range(80, thinning - 10, -10):
            if val >= a:
                return base_spacing * (1 << int((a - thinning) / 10))
        return base_spacing

    def drawline(outermin, outermax, outerstep, innermin, innermax, innerstep, d, linelist, thinning=80):
        """
        Draws grid lines, breaking the line if any point is unmappable to avoid artifacts.
        """
        # Logic for drawing vertical lines (d=False) with thinning near poles
        if not d:
            for alt_band_start in frange(innermin, innermax, 10):
                alt_band_end = alt_band_start + 10
                current_outerstep = get_azimuth_spacing(outerstep, alt_band_start, thinning)
                
                for outer in frange(outermin, outermax + 0.01, current_outerstep):
                    line = []
                    for inner in frange(alt_band_start, min(alt_band_end, innermax) + 0.01, innerstep):
                        az = outer
                        alt = inner
                        y_coord = scalealt(alt, scale)
                        forward_transform_new(dst, Vector2D(az * scale, y_coord))

                        if dst.x < -9000 or dst.y < -9000:
                            if len(line) > 1: linelist.append(line)
                            line = []
                        else:
                            line.append((dst.x * args.xscale, dst.y * args.yscale))
                    
                    if len(line) > 1:
                        linelist.append(line)
        else:  # Logic for horizontal lines (d=True)
            for outer in frange(outermin, outermax + 0.01, outerstep):
                line = []
                for inner in frange(innermin, innermax + 0.01, innerstep):
                    az = inner
                    alt = outer
                    y_coord = scalealt(alt, scale)
                    forward_transform_new(dst, Vector2D(az * scale, y_coord))

                    if dst.x < -9000 or dst.y < -9000:
                        if len(line) > 1: linelist.append(line)
                        line = []
                    else:
                        line.append((dst.x * args.xscale, dst.y * args.yscale))

                if len(line) > 1:
                    linelist.append(line)

    tiny_list = []
    small_list = []
    big_list = []

    base_grey_step = 1.0
    drawline(minaz, maxaz, base_grey_step, minalt, maxalt, 2, False, small_list, thinning=60)
    drawline(minalt, maxalt, 1, minaz, maxaz, 1, True, small_list)

    base_yellow_az_step = 15 if args.radec else 10
    drawline(minaz, maxaz - 1, base_yellow_az_step, minalt, maxalt, 1, False, big_list)
    drawline(minalt, maxalt, 10, minaz, maxaz, 1, True, big_list)

    if not args.radec and abs(minalt) <= 10:
        drawline(minaz, maxaz - 1, 20, 0, 10, 1, False, big_list)

    textlist = []
    label_spacing_lat = args.label_spacing
    base_label_spacing_lon = int(args.label_spacing * 1.5) if args.radec else args.label_spacing

    label_loop_minalt = minalt
    if not args.radec:
        label_loop_minalt = max(minalt, 10)

    for alt in range(maxalt - label_spacing_lat, int(label_loop_minalt) - 1, -label_spacing_lat):
        current_label_spacing_lon = get_azimuth_spacing(base_label_spacing_lon, alt)

        for az in range(minaz, maxaz, current_label_spacing_lon):
            if alt == 10 and az % (30 if args.radec else 20) == 0:
                continue
            
            # Apply refraction to find the apparent altitude for the label position
            alt_apparent = alt + refract(alt)
            y_coord = (90.0 - alt_apparent) * scale
            az2 = az*scale
            
            forward_transform_new(dst, Vector2D(az2, y_coord))
            x = dst.x
            y = dst.y
            forward_transform_new(dst, Vector2D(az2, y_coord+scale))
            angle = -math.atan2(dst.x - x, dst.y - y)*180/math.pi

            forward_transform_new(dst, Vector2D(az2, y_coord))
            x2 = dst.x
            y2 = dst.y
            forward_transform_new(dst, Vector2D(az2+scale, y_coord))
            angle2 = -math.atan2(dst.x - x2, dst.y - y2)*180/math.pi

            if -100 < x < width+100 and -100 < y < height+100:
                if args.radec:
                    lontext = str(int(az / 15)) + 'h'
                    lattext = str(int(alt)) + '°'
                    textlist.append((lontext + ' ', angle+90, x+3*math.cos(angle/180*math.pi), y+3*math.sin(angle/180*math.pi), 'right'))
                    textlist.append((' ' + lattext, angle2+90, x+4*math.cos(angle2/180*math.pi), y+4*math.sin(angle2/180*math.pi), 'left'))
                else:
                    lontext = str(int(az % 360))
                    textlist.append((' ' + lontext, angle+90, x+3*math.cos(angle/180*math.pi), y+3*math.sin(angle/180*math.pi), 'left'))
                    textlist.append((str(int(alt)) + ' ', angle2+90, x+4*math.cos(angle2/180*math.pi), y+4*math.sin(angle2/180*math.pi), 'right'))

    image = wand.image.Image(width=int(round(width * args.xscale)), height=int(round(height * args.yscale)), background=wand.color.Color('transparent'))

    def writetext(draw, text, x, y, angle, alignment):
        if x < 0 or y < 0:
            return
        x = x * args.xscale
        y = y * args.yscale
        offset = 0
        draw.text_alignment = alignment
        draw.translate(x+offset, y-offset)
        draw.rotate(angle)
        draw.text(0, 0, text)
        draw.rotate(-angle)
        draw.translate(-x-offset, -y+offset)

    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('none')
        draw.stroke_color = wand.color.Color('grey50')
        draw.stroke_opacity = 0.5
        for line in small_list:
            draw.polyline(line)
        draw.stroke_opacity = 0.8
        draw.stroke_color = wand.color.Color('yellow')
        for line in big_list:
            draw.polyline(line)

        draw.font = 'helvetica'
        draw.fill_color = wand.color.Color('yellow')
        draw.stroke_color = wand.color.Color('transparent')
        draw.stroke_opacity = 0
        draw.font_size = 14
        for text, angle, x, y, alignment in textlist:
            writetext(draw, text, x, y, angle, alignment)

        if pos.date and pos.lat and pos.long and args.timestamp:
            stars = brightstar(pto_data, pos, args.faintest, args.brightest, args.objects)
            draw.stroke_color = wand.color.Color('white')
            draw.stroke_opacity = 1
            draw.fill_color = wand.color.Color('transparent')
            for s in stars:
                x, y = s[0] * args.xscale, s[1] * args.yscale
                if 0 <= x < image.width and 0 <= y < image.height:
                     draw.circle((x, y), (x + 5, y + 5))

            draw.translate(0, 0)
            draw.rotate(0)
            draw.text_alignment = 'left'
            draw.stroke_opacity = 1
            draw.stroke_color = wand.color.Color('transparent')
            for s in stars:
                text_x = int(s[0] * args.xscale + 10)
                text_y = int(s[1] * args.yscale + 6)
                if not (0 <= text_x < image.width and 0 <= text_y < image.height):
                    continue

                draw.fill_color = wand.color.Color('white')
                if args.radec:
                    ra, dec = pos.radec_of(str(s[2]), str(s[3]))
                    if args.verbose:
                        draw.text(text_x, text_y,
                                  s[4] + " " + str(round(s[5], 1)) + " [" + re.sub(r'\..*', '', str(ra)) + ", " + str(round(math.degrees(float(repr(dec))), 2)) + "]")
                    else:
                        draw.text(text_x, text_y, s[4])
                else:
                    draw.text(text_x, text_y,
                              s[4] + " " + str(round(s[5], 1)) + " [" + str(round(s[2], 2)) + ", " + str(round(s[3], 2)) + "]")
        
        draw(image)

        image.format = 'png'
        image.save(filename=args.outfile)

if __name__ == '__main__':
    main()
