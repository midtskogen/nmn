#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Creates a Hugin .pto file from an AMS (AllSky Meteor Software)
JSON calibration file. This script generates control points by correlating
pixel coordinates of stars from the input file with their calculated
celestial positions (Azimuth/Altitude) for a given time and location.
"""

import argparse
import configparser
import json
import math
import os
import subprocess
from datetime import datetime, UTC

import ephem  # For calculating celestial object positions
from stars import cat  # Star catalog for position cross-referencing


def setup_observer(args, config, calib_data):
    """
    Initializes and configures an ephem.Observer object with location,
    elevation, and atmospheric data.

    Args:
        args: Command-line arguments.
        config: Parsed configuration from meteor.cfg.
        calib_data: Data from the AMS JSON calibration file.

    Returns:
        An ephem.Observer instance configured with the best available data.
    """
    obs = ephem.Observer()

    # Set defaults from config file
    obs.lat = config.get('astronomy', 'latitude')
    obs.lon = config.get('astronomy', 'longitude')
    obs.elevation = float(config.get('astronomy', 'elevation'))
    obs.temp = float(config.get('astronomy', 'temperature'))
    obs.pressure = float(config.get('astronomy', 'pressure'))

    # Override with command-line arguments if provided
    if args.latitude: obs.lat = str(args.latitude)
    if args.longitude: obs.lon = str(args.longitude)
    if args.elevation: obs.elevation = args.elevation
    if args.temperature: obs.temp = args.temperature
    if args.pressure: obs.pressure = args.pressure

    # Override with data from JSON file if present (highest priority)
    if 'device_lat' in calib_data: obs.lat = calib_data['device_lat']
    if 'device_lon' in calib_data: obs.lon = calib_data['device_lon']
    if 'device_alt' in calib_data: obs.elevation = int(calib_data['device_alt'])

    # Set the observation time
    timestamp = args.timestamp
    if not timestamp:
        try:
            fname_parts = os.path.basename(args.amscalib).split('_')
            dt_str = f"{fname_parts[0]}-{fname_parts[1]}-{fname_parts[2]} {fname_parts[3]}:{fname_parts[4]}:{fname_parts[5]}"
            timestamp = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except (IndexError, ValueError):
            # Fallback to current time if parsing filename fails
            timestamp = datetime.now(UTC).timestamp()
            print(f"Warning: Could not parse timestamp from filename. Using current time.")

    obs.date = datetime.fromtimestamp(float(timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')

    return obs


def write_pto_scaffold(output_file, width, height, calib_data, date_str):
    """
    Writes the main panorama and image lines (p-line and i-lines) to the
    .pto file, creating the basic structure for optimization.
    """
    pixel_scale = float(calib_data.get('pixel_scale', calib_data.get('pixscale', 0)))
    if pixel_scale == 0:
        raise ValueError("Could not find 'pixel_scale' or 'pixscale' in JSON file.")
        
    # Hugin's 'v' parameter is the horizontal field of view in degrees.
    # The original script used a calculation that is preserved here to maintain original behavior.
    fov_orig_calc = width * pixel_scale / 3600
    
    center_az = calib_data.get('center_az', 180)
    center_el = calib_data.get('center_el', 0)

    header = f"""# hugin project file
#hugin_ptoversion 2
p f2 w36000 h18000 v360 E0 R0 n"TIFF_m c:LZW"
m g1 i0 m2 p0.00784314

# image lines
#-hugin cropFactor=1
i w{width} h{height} f3 v{fov_orig_calc} Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p{center_el} y{center_az - 180} TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5
i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5 n"dummy.jpg"

# specify variables that should be optimized
v v0
v r0
v p0
v y0
v a0
v b0
v c0
v d0
v e0
v

# {date_str}
# control points
"""
    output_file.write(header)


def write_control_points(output_file, calib_data, observer, match_dist_limit):
    """
    Iterates through stars in the calibration data, finds their celestial
    coordinates using ephem, and writes them as control points to the file.
    """
    for star_data in calib_data.get('cat_image_stars', []):
        dcname, mag, ra, dec, _, _, match_dist, _, _, _, _, _, _, six, siy, _, _ = star_data
        
        if match_dist > match_dist_limit:
            continue

        # Convert RA from degrees (in JSON) to hours for ephem
        ra_hours = ra * 24 / 360

        # Find the matching star in the bright star catalog to get proper motion data
        best_match_body = None
        best_match_name = "Unknown"
        min_separation = 99999

        for (cat_ra, pmra, cat_dec, pmdec, cat_mag, name) in cat:
            if abs(mag - cat_mag) > 0.2:  # Quick magnitude check
                continue

            # Create ephem body for the star from the JSON file
            json_star = ephem.FixedBody()
            json_star._ra = str(ra_hours)
            json_star._dec = str(dec)
            json_star._epoch = ephem.J2000
            json_star.compute(observer)

            # Create ephem body for the star from the catalog
            catalog_star = ephem.FixedBody()
            catalog_star._ra = str(cat_ra)
            catalog_star._pmra = pmra
            catalog_star._dec = str(cat_dec)
            catalog_star._pmdec = pmdec
            catalog_star._epoch = ephem.J2000
            catalog_star.compute(observer)
            
            separation = float(repr(ephem.separation(json_star, catalog_star)))
            if separation < min_separation:
                min_separation = separation
                best_match_body = catalog_star
                best_match_name = name

        # If a close match is found, calculate its position and write control point
        if min_separation < 0.0001: # Threshold for a good match (in radians)
            best_match_body.compute(observer)
            az = math.degrees(float(repr(best_match_body.az)))
            alt = math.degrees(float(repr(best_match_body.alt)))
            
            if alt > 1:  # Only use stars above the horizon
                # The panorama 'X' is Azimuth * 100
                # The panorama 'Y' is (90 - Altitude) * 100 (i.e., Zenith distance)
                pano_x = az * 100
                pano_y = (90 - alt) * 100
                output_file.write(f'c n0 N1 x{six} y{siy} X{pano_x:.4f} Y{pano_y:.4f} t0 # {best_match_name}\n')


def main():
    """
    Main execution function. Parses arguments, reads data, generates the
    .pto file, and runs the optimizer.
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

    # Observer location arguments (can override config file)
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
    except FileNotFoundError:
        print(f"Error: Calibration file not found at '{args.amscalib}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.amscalib}'")
        return

    config = configparser.ConfigParser()
    config_path_user = os.path.expanduser('~/meteor.cfg')
    read_files = config.read([args.config, config_path_user])
    if not read_files:
        print(f"Warning: Could not read config files '{args.config}' or '{config_path_user}'. Using defaults.")
        config.add_section('astronomy')
        config.set('astronomy', 'latitude', '0')
        config.set('astronomy', 'longitude', '0')
        config.set('astronomy', 'elevation', '0')
        config.set('astronomy', 'temperature', '10')
        config.set('astronomy', 'pressure', '1010')

    # --- Setup ---
    width = calib_data.get('imagew', args.width)
    height = calib_data.get('imageh', args.height)
    observer = setup_observer(args, config, calib_data)

    # --- File Generation ---
    try:
        with open(args.ptofile, 'w') as ptofile_handle:
            write_pto_scaffold(ptofile_handle, width, height, calib_data, observer.date)
            write_control_points(ptofile_handle, calib_data, observer, args.match_dist)
        print(f"Successfully generated initial .pto file: {args.ptofile}")
    except IOError as e:
        print(f"Error writing to PTO file '{args.ptofile}': {e}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- Optimization ---
    try:
        print("Running Hugin's autooptimiser...")
        proc = subprocess.Popen(['autooptimiser', '-n', args.ptofile, '-o', args.ptofile],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        
        if proc.returncode == 0:
            print("Optimization complete.")
            if args.verbose:
                print("\n--- Autooptimiser Output ---")
                # Decode stdout and print, handling potential empty output
                out_str = stdout.decode().strip()
                if out_str:
                    print(out_str)
                else:
                    print("(No standard output from autooptimiser)")
                
                # Also print stderr in verbose mode in case there are warnings
                err_str = stderr.decode().strip()
                if err_str:
                    print("\n--- Autooptimiser Errors/Warnings ---")
                    print(err_str)

        else:
            print("\nError during optimization.")
            # Always print stderr on error
            print(stderr.decode())

    except FileNotFoundError:
        print("\nError: 'autooptimiser' command not found.")
        print("Please ensure Hugin command-line tools are installed and in your system's PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during optimization: {e}")


if __name__ == '__main__':
    # The 'stars.py' file containing the 'cat' variable must be in the same
    # directory or in the Python path for this script to run.
    if 'cat' not in globals():
         print("Error: Could not find the 'cat' variable from 'stars.py'.")
         print("Please ensure 'stars.py' is in the same directory.")
    else:
         main()
