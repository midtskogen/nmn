#!/usr/bin/env python3

# List the brightest visible stars and maps them to source image coordinates
# in a panorama project at a given Unix timestamp.
# Usage: brightstar.py <Unix timestamp> <pto file>
# Depends on: pto_mapper.py, ephem, stars.py

import sys
import ephem
import math
import argparse
import configparser
import os
from datetime import datetime, timezone

try:
    import pto_mapper
except ImportError:
    print("Error: 'pto_mapper.py' not found.", file=sys.stderr)
    print("Please ensure 'pto_mapper.py' is in the same directory or in the Python path.", file=sys.stderr)
    sys.exit(1)

from stars import cat

def brightstar(pto_data, pos, faintest, brightest, objects, map_to_source_image=True, include_img_idx=False):
    """
    Finds visible celestial objects.
    If map_to_source_image is True, it maps them to source image coordinates.
    Otherwise, it returns the raw celestial coordinates.
    If include_img_idx is True, the returned tuple for mapped objects will
    include the source image index as the first element.
    """
    global_options, _ = pto_data
    pano_w = global_options.get('w')
    pano_h = global_options.get('h')

    if not pano_w or not pano_h:
        raise ValueError("PTO 'p' line is missing width 'w' or height 'h'.")
    if global_options.get('f', 2) != 2:
        print("Warning: Panorama projection is not equirectangular (f=2). Results may be incorrect.", file=sys.stderr)

    def test_body(body, name, faintest, brightest):
        """Calculates a celestial body's position and finds its coordinates."""
        body.compute(pos)
        mag = body.mag
        
        if not (brightest <= mag <= faintest):
            return None

        alt_rad = float(repr(body.alt))
        alt_deg = math.degrees(alt_rad)

        if alt_deg <= -0.5: # Check if body is above the horizon
            return None
        
        az_rad = float(repr(body.az))
        az_deg = math.degrees(az_rad)
        
        if not map_to_source_image:
            return (az_deg, alt_deg, name, mag)

        try:
            alt2_deg = alt_deg + 0.006 / math.tan(math.radians(alt_deg + (7.31 / (alt_deg + 4.4))))
            alt2_rad = math.radians(alt2_deg)
        except ValueError:
            return None
        
        pano_x = (az_rad / (2 * math.pi)) * pano_w
        pano_y = (0.5 - alt2_rad / math.pi) * pano_h

        mapping = pto_mapper.map_pano_to_image(pto_data, pano_x, pano_y)

        if mapping:
            img_idx, sx, sy = mapping
            if include_img_idx:
                return (img_idx, sx, sy, az_deg, alt_deg, name, mag)
            else:
                return (sx, sy, az_deg, alt_deg, name, mag)

        return None

    # --- Main processing loop ---
    count = 0
    res = []
    solar_system_bodies = [
        (ephem.Sun(), "Sol"), (ephem.Moon(), "Moon"), (ephem.Mercury(), "Mercury"),
        (ephem.Venus(), "Venus"), (ephem.Mars(), "Mars"), (ephem.Jupiter(), "Jupiter"),
        (ephem.Saturn(), "Saturn")
    ]
    for (body, name) in solar_system_bodies:
        if count >= objects: break
        r = test_body(body, name, faintest, brightest)
        if r:
            count += 1
            res.append(r)

    if count < objects:
        for (ra, pmra, dec, pmdec, mag, name) in cat:
            if count >= objects: break
            if (mag <= faintest and mag >= brightest):
                body = ephem.FixedBody()
                body._ra, body._pmra, body._dec, body._pmdec, body._epoch = str(ra), pmra, str(dec), pmdec, ephem.J2000
                body.mag = mag
                r = test_body(body, name, faintest, -30)
                if r:
                    count += 1
                    res.append(r)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='List the brightest visible stars and their image coordinates for a Hugin .pto file.',
        epilog='Example: brightstar.py 1678886400 my_pano.pto'
    )
    parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects to find (default: 500)', default=500, type=int)
    parser.add_argument('-f', '--faintest', dest='faintest', help='faintest magnitude to include (default: 5)', default=5, type=float)
    parser.add_argument('-b', '--brightest', dest='brightest', help='brightest magnitude to include (default: -30)', default=-30, type=float)
    parser.add_argument('-c', '--config', dest='config', help='observer config file (e.g., /etc/meteor.cfg)', type=str)
    obs_group = parser.add_argument_group('Observer Location (overrides config file)')
    obs_group.add_argument('-x', '--longitude', dest='longitude', help='observer longitude (degrees)', type=float)
    obs_group.add_argument('-y', '--latitude', dest='latitude', help='observer latitude (degrees)', type=float)
    obs_group.add_argument('-a', '--altitude', dest='elevation', help='observer altitude (meters)', type=float)
    atm_group = parser.add_argument_group('Atmospheric Conditions (overrides config file)')
    atm_group.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (Celsius)', type=float)
    atm_group.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
    parser.add_argument(action='store', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)')
    parser.add_argument(action='store', dest='ptofile', help='Hugin .pto panorama project file')
    args = parser.parse_args()
    
    pos = ephem.Observer()
    config_files = []
    if args.config:
        config_files.append(args.config)
    config_files.append(os.path.expanduser('~/.config/meteor.cfg'))
    config_files.append('/etc/meteor.cfg')
    
    config = configparser.ConfigParser()
    read_configs = config.read(config_files)

    if read_configs:
        try:
            pos.lat = config.get('astronomy', 'latitude')
            pos.lon = config.get('astronomy', 'longitude')
            pos.elevation = float(config.get('astronomy', 'elevation'))
            pos.temp = float(config.get('astronomy', 'temperature', fallback=15))
            pos.pressure = float(config.get('astronomy', 'pressure', fallback=1010))
        except (configparser.NoSectionError, configparser.NoOptionError):
             print("Warning: Could not read all astronomy settings from config. Using defaults.", file=sys.stderr)
             pos.lat, pos.lon, pos.elevation, pos.temp, pos.pressure = '0', '0', 0, 15, 1010
    else:
        print("Warning: No config file found. Using defaults.", file=sys.stderr)
        pos.lat, pos.lon, pos.elevation, pos.temp, pos.pressure = '0', '0', 0, 15, 1010
        
    if args.longitude is not None: pos.lon = str(args.longitude)
    if args.latitude is not None: pos.lat = str(args.latitude)
    if args.elevation is not None: pos.elevation = args.elevation
    if args.temperature is not None: pos.temp = args.temperature
    if args.pressure is not None: pos.pressure = args.pressure

    pos.date = datetime.fromtimestamp(float(args.timestamp), timezone.utc)

    try:
        pto_data = pto_mapper.parse_pto_file(args.ptofile)
    except FileNotFoundError:
        sys.exit(f"Error: PTO file not found at '{args.ptofile}'")
    except Exception as e:
        sys.exit(f"Error reading or parsing PTO file: {e}")

    results = brightstar(pto_data, pos, args.faintest, args.brightest, args.objects, map_to_source_image=True, include_img_idx=False)
    for res_line in results:
        sx, sy, az, alt, name, mag = res_line
        print(f'{sx} {sy} {az} {alt} \'{name}\' {mag}')
