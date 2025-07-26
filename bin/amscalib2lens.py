#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Creates a Hugin .pto file from an AMS (AllSky Meteor Software)
JSON calibration file. This script generates control points by correlating
pixel coordinates of stars from the input file with their calculated
celestial positions (Azimuth/Altitude) for a given time and location.

Usage: amscalib2lens.py <AMS calibration json file> <pto file>
       amscalib2lens.py /mnt/ams2/cal/freecal/2022_05_06_00_32_20_000_011193/2022_05_06_00_32_20_000_011193-stacked-calparams.json lens.pto
"""

import argparse
import configparser
import json
import math
import os
import subprocess
from datetime import datetime, UTC
import io

import ephem  # For calculating celestial object positions
from stars import cat  # Star catalog for position cross-referencing


def setup_observer(args, config, calib_data):
    """
    Initializes and configures an ephem.Observer object with location,
    elevation, and atmospheric data.

    Args:
        args: Command-line arguments or a Namespace object.
        config: Parsed configuration from meteor.cfg.
        calib_data: Data from the AMS JSON calibration file.

    Returns:
        An ephem.Observer instance configured with the best available data.
    """
    obs = ephem.Observer()

    # Set defaults from config file, if available
    if config.has_section('astronomy'):
        obs.lat = config.get('astronomy', 'latitude')
        obs.lon = config.get('astronomy', 'longitude')
        obs.elevation = float(config.get('astronomy', 'elevation'))
        obs.temp = float(config.get('astronomy', 'temperature'))
        obs.pressure = float(config.get('astronomy', 'pressure'))

    # Override with arguments if provided (e.g., from command line or another script)
    if hasattr(args, 'latitude') and args.latitude is not None: obs.lat = str(args.latitude)
    if hasattr(args, 'longitude') and args.longitude is not None: obs.lon = str(args.longitude)
    if hasattr(args, 'elevation') and args.elevation is not None: obs.elevation = args.elevation
    if hasattr(args, 'temperature') and args.temperature is not None: obs.temp = args.temperature
    if hasattr(args, 'pressure') and args.pressure is not None: obs.pressure = args.pressure

    # Override with data from JSON file if present (highest priority)
    if 'device_lat' in calib_data: obs.lat = calib_data['device_lat']
    if 'device_lon' in calib_data: obs.lon = calib_data['device_lon']
    if 'device_alt' in calib_data: obs.elevation = int(calib_data['device_alt'])

    # Set the observation time
    timestamp = args.timestamp if hasattr(args, 'timestamp') and args.timestamp else None
    if not timestamp:
        try:
            fname = getattr(args, 'amscalib', '')
            fname_parts = os.path.basename(fname).split('_')
            dt_str = f"{fname_parts[0]}-{fname_parts[1]}-{fname_parts[2]} {fname_parts[3]}:{fname_parts[4]}:{fname_parts[5]}"
            timestamp = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except (IndexError, ValueError, AttributeError):
            timestamp = datetime.now(UTC).timestamp()
            print("Warning: Could not parse timestamp from filename. Using current time.")

    obs.date = datetime.fromtimestamp(float(timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')

    return obs


def _get_pto_scaffold(width, height, calib_data, date_str):
    """
    Generates the main panorama and image lines (p-line and i-lines) for the
    .pto file as a string.
    """
    pixel_scale = float(calib_data.get('pixel_scale', calib_data.get('pixscale', 0)))
    if pixel_scale == 0:
        raise ValueError("Could not find 'pixel_scale' or 'pixscale' in JSON file.")
        
    fov_orig_calc = width * pixel_scale / 3600
    center_az = calib_data.get('center_az', 180)
    center_el = calib_data.get('center_el', 0)

    # Using an in-memory string builder is more efficient
    with io.StringIO() as header_stream:
        header_stream.write("# hugin project file\n")
        header_stream.write("#hugin_ptoversion 2\n")
        header_stream.write('p f2 w36000 h18000 v360 E0 R0 n"TIFF_m c:LZW"\n')
        header_stream.write("m g1 i0 m2 p0.00784314\n\n")
        header_stream.write("# image lines\n")
        header_stream.write("#-hugin cropFactor=1\n")
        header_stream.write(f"i w{width} h{height} f3 v{fov_orig_calc} Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p{center_el} y{center_az - 180} TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5\n")
        header_stream.write('i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5 n"dummy.jpg"\n\n')
        header_stream.write("# specify variables that should be optimized\n")
        header_stream.write("v v0\nv r0\nv p0\nv y0\nv a0\nv b0\nv c0\nv d0\nv e0\nv\n\n")
        header_stream.write(f"# {date_str}\n")
        header_stream.write("# control points\n")
        return header_stream.getvalue()


def _get_control_points(calib_data, observer, match_dist_limit):
    """
    Generates control point lines from calibration data as a list of strings.
    """
    control_points = []
    for star_data in calib_data.get('cat_image_stars', []):
        dcname, mag, ra, dec, _, _, match_dist, _, _, _, _, _, _, six, siy, _, _ = star_data
        
        if match_dist > match_dist_limit:
            continue

        ra_hours = ra * 24 / 360

        best_match_body = None
        best_match_name = "Unknown"
        min_separation = 99999

        for (cat_ra, pmra, cat_dec, pmdec, cat_mag, name) in cat:
            if abs(mag - cat_mag) > 0.2:
                continue

            json_star = ephem.FixedBody()
            json_star._ra, json_star._dec, json_star._epoch = str(ra_hours), str(dec), ephem.J2000
            json_star.compute(observer)

            catalog_star = ephem.FixedBody()
            catalog_star._ra, catalog_star._pmra, catalog_star._dec, catalog_star._pmdec, catalog_star._epoch = str(cat_ra), pmra, str(cat_dec), pmdec, ephem.J2000
            catalog_star.compute(observer)
            
            separation = float(repr(ephem.separation(json_star, catalog_star)))
            if separation < min_separation:
                min_separation, best_match_body, best_match_name = separation, catalog_star, name

        if min_separation < 0.0001:
            best_match_body.compute(observer)
            az = math.degrees(float(repr(best_match_body.az)))
            alt = math.degrees(float(repr(best_match_body.alt)))
            
            if alt > 1:
                pano_x = az * 100
                pano_y = (90 - alt) * 100
                control_points.append(f'c n0 N1 x{six} y{siy} X{pano_x:.4f} Y{pano_y:.4f} t0 # {best_match_name}\n')
    
    return control_points


def generate_pto_from_json(calib_data, observer, width, height, match_dist_limit):
    """
    Generates the full content of a Hugin .pto file from AMS calibration data.
    This is the primary function for external use.

    Args:
        calib_data (dict): The loaded AMS JSON calibration data.
        observer (ephem.Observer): The configured observer object.
        width (int): The width of the image.
        height (int): The height of the image.
        match_dist_limit (float): The maximum allowed match distance for stars.

    Returns:
        str: The complete, unoptimized content of the .pto file.
    """
    # Generate the main structure (p, m, i, v lines)
    scaffold = _get_pto_scaffold(width, height, calib_data, observer.date)

    # Generate the control points (c lines)
    control_points = _get_control_points(calib_data, observer, match_dist_limit)

    # Combine all parts into a single string
    return scaffold + "".join(control_points)


def main():
    """
    Main execution function for standalone script usage. Parses arguments,
    reads data, generates the .pto file, and runs the optimizer.
    """
    parser = argparse.ArgumentParser(
        description='Convert AMS calibration into a Hugin/panotools pto file.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('amscalib', help='AMS calibration json file')
    parser.add_argument('ptofile', help='Output Hugin .pto file')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Image width (default: 1920)')
    parser.add_argument('-H', '--height', type=int, default=1080, help='Image height (default: 1080)')
    parser.add_argument('-d', '--match_dist', type=float, default=0.2, help='Maximum allowed match distance (default: 0.2)')
    parser.add_argument('-c', '--config', default='/etc/meteor.cfg', help='Meteor config file (default: /etc/meteor.cfg)')
    parser.add_argument('-T', '--timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show the full output from the autooptimiser.')
    parser.add_argument('-x', '--longitude', type=float, help='Observer longitude')
    parser.add_argument('-y', '--latitude', type=float, help='Observer latitude')
    parser.add_argument('-e', '--elevation', type=float, help='Observer elevation (m)')
    parser.add_argument('-t', '--temperature', type=float, help='Observer temperature (C, for refraction)')
    parser.add_argument('-p', '--pressure', type=float, help='Observer air pressure (hPa, for refraction)')
    args = parser.parse_args()

    # --- Data Loading ---
    try:
        with open(args.amscalib) as f:
            calib_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not read or decode JSON from '{args.amscalib}': {e}")
        return

    config = configparser.ConfigParser()
    read_files = config.read([args.config, os.path.expanduser('~/meteor.cfg')])
    if not read_files:
        print(f"Warning: Could not read config files. Using defaults.")
        config.add_section('astronomy')
        config.set('astronomy', 'latitude', '0'); config.set('astronomy', 'longitude', '0')
        config.set('astronomy', 'elevation', '0'); config.set('astronomy', 'temperature', '10')
        config.set('astronomy', 'pressure', '1010')

    # --- Setup ---
    width = calib_data.get('imagew', args.width)
    height = calib_data.get('imageh', args.height)
    observer = setup_observer(args, config, calib_data)

    # --- File Generation ---
    try:
        # Use the new core function to get the PTO data
        pto_content = generate_pto_from_json(calib_data, observer, width, height, args.match_dist)
        
        with open(args.ptofile, 'w') as ptofile_handle:
            ptofile_handle.write(pto_content)
        print(f"Successfully generated initial .pto file: {args.ptofile}")

    except (IOError, ValueError) as e:
        print(f"Error generating or writing PTO file: {e}")
        return

    # --- Optimization ---
    try:
        print("Running Hugin's autooptimiser...")
        # autooptimiser can optimize the file in-place
        proc = subprocess.run(['autooptimiser', '-n', args.ptofile, '-o', args.ptofile],
                              capture_output=True, text=True)
        
        if proc.returncode == 0:
            print("Optimization complete.")
            if args.verbose:
                print("\n--- Autooptimiser Output ---")
                print(proc.stdout if proc.stdout.strip() else "(No standard output)")
                if proc.stderr.strip():
                    print("\n--- Autooptimiser Errors/Warnings ---")
                    print(proc.stderr)
        else:
            print(f"\nError during optimization (exit code {proc.returncode}).")
            print(proc.stderr)

    except FileNotFoundError:
        print("\nError: 'autooptimiser' command not found.")
        print("Please ensure Hugin command-line tools are installed and in your system's PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during optimization: {e}")


if __name__ == '__main__':
    if 'cat' not in globals():
         print("Error: Could not find the 'cat' variable from 'stars.py'.")
         print("Please ensure 'stars.py' is in the same directory.")
    else:
         main()
