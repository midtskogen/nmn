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


def setup_observer(args, config, calib_data, verbose=False):
    """
    Initializes and configures an ephem.Observer object with location,
    elevation, and atmospheric data, following a specific priority order.
    """
    obs = ephem.Observer()

    def warn_if_overriding(old_val, new_val, name, new_source, old_source):
        if old_val is not None and str(old_val) != str(new_val):
            print(f"Warning: {name} from {new_source} ('{new_val}') is overriding value from {old_source} ('{old_val}').")

    lat_val, lon_val, ele_val = None, None, None
    lat_source, lon_source, ele_source = "Nothing", "Nothing", "Nothing"

    # 1. Baseline from JSON data (Lowest Priority)
    if 'device_lat' in calib_data:
        lat_val, lat_source = str(calib_data['device_lat']), "JSON"
    if 'device_lon' in calib_data:
        lon_val, lon_source = str(calib_data['device_lon']), "JSON"
    if 'device_alt' in calib_data:
        ele_val, ele_source = float(calib_data['device_alt']), "JSON"

    # 2. Override with config file data (Medium Priority)
    if args.config and config.has_section('astronomy'):
        conf_lat = config.get('astronomy', 'latitude', fallback=None)
        conf_lon = config.get('astronomy', 'longitude', fallback=None)
        conf_ele = config.getfloat('astronomy', 'elevation', fallback=None)

        if conf_lat is not None:
            warn_if_overriding(lat_val, conf_lat, "Latitude", "Config File", lat_source)
            lat_val, lat_source = conf_lat, "Config File"
        if conf_lon is not None:
            warn_if_overriding(lon_val, conf_lon, "Longitude", "Config File", lon_source)
            lon_val, lon_source = conf_lon, "Config File"
        if conf_ele is not None:
            warn_if_overriding(ele_val, conf_ele, "Elevation", "Config File", ele_source)
            ele_val, ele_source = conf_ele, "Config File"

    # 3. Override with command-line arguments (Highest Priority)
    if args.latitude is not None:
        warn_if_overriding(lat_val, args.latitude, "Latitude", "Command-line", lat_source)
        lat_val = str(args.latitude)
    if args.longitude is not None:
        warn_if_overriding(lon_val, args.longitude, "Longitude", "Command-line", lon_source)
        lon_val = str(args.longitude)
    if args.elevation is not None:
        warn_if_overriding(ele_val, args.elevation, "Elevation", "Command-line", ele_source)
        ele_val = args.elevation

    # Final validation: ensure we have a complete location
    if lat_val is None or lon_val is None or ele_val is None:
        missing = []
        if lat_val is None: missing.append("Latitude")
        if lon_val is None: missing.append("Longitude")
        if ele_val is None: missing.append("Elevation")
        raise ValueError(f"Observer location is incomplete. Missing: {', '.join(missing)}. Provide the location via the JSON file, a config file (-c), or command-line arguments.")

    obs.lat = lat_val
    obs.lon = lon_val
    obs.elevation = ele_val

    # Handle temperature and pressure
    obs.temp = float(config.get('astronomy', 'temperature', fallback=10))
    obs.pressure = float(config.get('astronomy', 'pressure', fallback=1010))
    if args.temperature is not None: obs.temp = args.temperature
    if args.pressure is not None: obs.pressure = args.pressure

    # Handle timestamp
    timestamp = args.timestamp if hasattr(args, 'timestamp') and args.timestamp else None
    if not timestamp:
        try:
            fname = getattr(args, 'amscalib', '')
            fname_parts = os.path.basename(fname).split('_')
            dt_str = f"{fname_parts[0]}-{fname_parts[1]}-{fname_parts[2]} {fname_parts[3]}:{fname_parts[4]}:{fname_parts[5]}"
            
            naive_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            aware_dt = naive_dt.replace(tzinfo=UTC)
            timestamp = aware_dt.timestamp()

        except (IndexError, ValueError, AttributeError):
            timestamp = datetime.now(UTC).timestamp()
            print("Warning: Could not parse timestamp from filename. Using current time.")

    # Create a datetime object from the final timestamp
    final_dt_utc = datetime.fromtimestamp(float(timestamp), UTC)
    
    # Set the observer date for ephem calculations
    obs.date = final_dt_utc.strftime('%Y-%m-%d %H:%M:%S')

    # If verbose, print the observer details using the new formats
    if verbose:
        print("\n--- Observer Details ---")
        print(f"Timestamp for calculation: {final_dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Latitude: {math.degrees(obs.lat):.5f}")
        print(f"Longitude: {math.degrees(obs.lon):.5f}")
        print(f"Elevation: {int(obs.elevation)} m")
        
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
    position_angle = calib_data.get('position_angle', 0)

    with io.StringIO() as header_stream:
        header_stream.write("# hugin project file\n")
        header_stream.write("#hugin_ptoversion 2\n")
        header_stream.write('p f2 w36000 h18000 v360 E0 R0 n"TIFF_m c:LZW"\n')
        header_stream.write("m g1 i0 m2 p0.00784314\n\n")
        header_stream.write("# image lines\n")
        header_stream.write("#-hugin cropFactor=1\n")
        header_stream.write(f"i w{width} h{height} f3 v{fov_orig_calc} Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r{position_angle} p{center_el} y{center_az - 180} TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5\n")
        header_stream.write('i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5 n"dummy.jpg"\n\n')
        header_stream.write("# specify variables that should be optimized\n")
        header_stream.write("v v0\nv r0\nv p0\nv y0\nv a0\nv b0\nv c0\nv d0\nv e0\nv\n\n")
        header_stream.write(f"# {date_str}\n")
        header_stream.write("# control points\n")
        return header_stream.getvalue()


def _get_control_points(calib_data, observer, verbose=False):
    """
    Generates control point lines from the provided list of stars.
    """
    control_points = []
    # This function now expects a pre-filtered list of stars
    for star_data in calib_data.get('cat_image_stars', []):
        dcname, mag, ra, dec, _, _, match_dist, _, _, _, _, _, _, six, siy, _, _ = star_data
        
        if verbose:
            # The initial check is now done elsewhere, but we can still announce the re-verification
            print(f"\n- Verifying star from JSON: '{dcname if dcname else 'Unnamed'}' at pixel (X:{six}, Y:{siy})")

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

            json_star.compute(observer)
            json_az = math.degrees(float(repr(json_star.az)))
            json_alt = math.degrees(float(repr(json_star.alt)))

            if verbose:
                print(f"  - Re-verification match found: '{best_match_name}' (Separation: {min_separation:.6f} rad)")
                print(f"  - JSON Pos (X:{six}, Y:{siy} -> Az/Alt): {json_az:.4f}° / {json_alt:.4f}°")
                print(f"  - Ephem Pos (Catalog -> Az/Alt):  {az:.4f}° / {alt:.4f}°")

            if alt > 1:
                pano_x = az * 100
                pano_y = (90 - alt) * 100
                comment = f"Star: {best_match_name}, JSON_px:({six},{siy}), JSON_pos:({json_az:.2f},{json_alt:.2f})"
                control_points.append(f'c n0 N1 x{six} y{siy} X{pano_x:.4f} Y{pano_y:.4f} t0 # {comment}\n')
        
        elif verbose:
            print(f"  - No close re-verification match found in script's catalog (min separation: {min_separation:.6f} rad).")
            print("  -> REJECTED.")
    
    return control_points


def generate_pto_from_json(calib_data, observer, width, height, match_dist_limit, verbose=False):
    """
    Selects the best stars and generates the full content of a Hugin .pto file.
    """
    # --- New Star Selection Logic ---
    raw_star_list = calib_data.get('cat_image_stars', [])
    final_star_list = []

    if raw_star_list:
        # 1. Filter stars by the user-defined quality limit
        good_stars = [s for s in raw_star_list if s[6] <= match_dist_limit]

        # 2. If enough good stars are found, use them.
        if len(good_stars) >= 3:
            final_star_list = good_stars
            if verbose:
                print(f"Info: Found {len(final_star_list)} stars within the match distance limit of {match_dist_limit}.")
        # 3. Otherwise, fall back to using the absolute 3 best stars.
        else:
            raw_star_list.sort(key=lambda s: s[6])
            final_star_list = raw_star_list[:3]
            print(f"Warning: Only {len(good_stars)} stars met the quality limit of {match_dist_limit}.")
            print("Falling back to using the 3 best-matched stars to attempt a solution.")
    
    # Create a new dictionary with the final curated list of stars
    curated_calib_data = calib_data.copy()
    curated_calib_data['cat_image_stars'] = final_star_list

    # --- Generation ---
    scaffold = _get_pto_scaffold(width, height, curated_calib_data, observer.date)
    # The 'match_dist_limit' is no longer passed as filtering is complete
    control_points = _get_control_points(curated_calib_data, observer, verbose=verbose)

    # Final validation remains a good safeguard
    if len(control_points) < 3:
        raise ValueError(f"Only {len(control_points)} valid control points could be generated. At least 3 are required.")

    return scaffold + "".join(control_points)


def main():
    """
    Main execution function for standalone script usage.
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
    parser.add_argument('-c', '--config', help='Optional meteor config file for location data')
    parser.add_argument('-T', '--timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed script output.')
    parser.add_argument('-x', '--longitude', type=float, help='Observer longitude')
    parser.add_argument('-y', '--latitude', type=float, help='Observer latitude')
    parser.add_argument('-e', '--elevation', type=float, help='Observer elevation (m)')
    parser.add_argument('-t', '--temperature', type=float, help='Observer temperature (C, for refraction)')
    parser.add_argument('-p', '--pressure', type=float, help='Observer air pressure (hPa, for refraction)')
    args = parser.parse_args()

    try:
        with open(args.amscalib) as f:
            calib_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not read or decode JSON from '{args.amscalib}': {e}")
        return

    config = configparser.ConfigParser()
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return
        config.read(args.config)

    try:
        # Pass the verbose flag to the setup function
        observer = setup_observer(args, config, calib_data, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: Could not determine observer location. {e}")
        return

    # The observer details are now printed inside setup_observer()

    if 'cal_params' in calib_data:
        cal_params_data = calib_data['cal_params']
    else:
        cal_params_data = calib_data
    
    width = cal_params_data.get('imagew', args.width)
    height = cal_params_data.get('imageh', args.height)

    try:
        if args.verbose:
            print("\n--- Verifying Stars for Control Points ---")
            
        pto_content = generate_pto_from_json(cal_params_data, observer, width, height, match_dist_limit=args.match_dist, verbose=args.verbose)
        
        with open(args.ptofile, 'w') as ptofile_handle:
            ptofile_handle.write(pto_content)
        print(f"Successfully generated initial .pto file: {args.ptofile}")

        if args.verbose:
            print("\n--- Generated .pto File Content ---")
            print(pto_content.strip())
            print("-----------------------------------")

    except (IOError, ValueError) as e:
        print(f"Error generating or writing PTO file: {e}")
        return

    try:
        print("Running Hugin's autooptimiser...")
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
